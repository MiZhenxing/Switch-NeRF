# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import logging 

import torch
from torch import Tensor
from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import copy
from timm.models.layers import trunc_normal_
from typing import List, Optional

class TopKGate(torch.nn.Module):
    """General-purpose Top-K Gate for MoE
    """
 
    def __init__(
        self,
        model_dim,
        num_global_experts,
        a2a_ffn_overlap_degree=1,
        capacity_factor=1.0,
        k=2,
        batch_prioritized_routing=False,
        fp32_gate=False,
        is_postscore=True,
        input_dropout_p=0,
        use_normal_noise=False,
        ray_prioritized_droping=False,
        ray_prioritized_droping_mode="max",
        ray_prioritized_droping_factor=1.0,
        ray_random_droping=False,
        ray_random_droping_factor=1.0,
        gate_dim=None,
        gate_noise=-1.0,
        use_load_importance_loss=False,
        compute_balance_loss=False,
        dispatcher_no_score=False
    ):
        super().__init__()
        k = min(k, num_global_experts)
        self.top_k = k
        assert self.top_k > 0, "Top-k value %d is not valid." % self.top_k

        self.gate_dim = gate_dim
        if self.gate_dim is None:
            self.gate_dim = model_dim

        self.wg = torch.nn.Linear(self.gate_dim, num_global_experts, bias=False)
        self.fp32_gate = fp32_gate 

        self.num_global_experts = num_global_experts

    def forward(self, gate_input: torch.Tensor):

        # if self.fp32_gate:
        #     wg = self.wg.to(torch.float32)
        # else:
        #     wg = self.wg

        # wg = self.wg
        # with torch.cuda.amp.autocast(enabled=(not self.fp32_gate)):
        # logits = wg(gate_input.to((wg.weight).dtype))
        logits = self.wg(gate_input)

        gates = F.softmax(logits, dim=1)

        return gates

class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """

    def __init__(
        self,
        gate_type,
        model_dim: int,
        experts=None,
        scan_expert_func=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        parallel_type='auto',
        pad_samples=False,
        moe_no_batch=False,
        use_residual=False,
        return_gates=False,
        return_gate_logits=False,
        return_expert_mean_feature=False,
        use_random_balance_expert=False,
        use_scaled_dot=False,
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim

        self.moe_no_batch = moe_no_batch
        self.num_global_experts = self.num_local_experts = experts.get('count_per_node', 1)
        self.hidden_size = experts.get('hidden_size_per_expert', 'None')
        self.model_dim = model_dim
        self.is_builtin_experts = True

        self.expert_type = experts['type']
        if experts['type'] == 'seqexperts' or experts['type'] == 'multiseqexperts':
            if seeds is not None and seeds[1] is not None:
                torch.manual_seed(seeds[1])
            net = experts['net']
            self.experts = ModuleList([SeqExperts(net, local_experts=self.num_local_experts)])
        else:
            raise Exception('Builtin expert type is not recognized: %s' % experts['type'])

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in tutel.moe_layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            if single_gate_type['type'] == 'top':
                if seeds is not None and seeds[0] is not None:
                    torch.manual_seed(seeds[0] + gi)

                single_gate_type.pop('type')
                self.gates += [TopKGate(model_dim=model_dim, num_global_experts=self.num_global_experts, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, **single_gate_type)]
            else:
                raise Exception("Unrecognized gate_type: %s" % single_gate_type)

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def forward(self, input: Tensor, gate_input: Optional[torch.Tensor] = None):
        
        if gate_input is None:
            gate_input = input
        
        original_shape, original_dtype  = input.shape, input.dtype
        # self.original_shape = original_shape
        assert len(input.shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        reshaped_input = input.reshape(-1, input.shape[-1])
        reshaped_input_samples = reshaped_input.shape[0]
        reshaped_gate_input = gate_input.reshape(-1, gate_input.shape[-1])

        results = torch.empty(0)
        cluster_gates = self.gates[0](reshaped_gate_input)
        cluster_gates_mul, cluster_assignments = torch.topk(cluster_gates, k=1, dim=1)
        cluster_assignments = cluster_assignments.squeeze(1)

        for i, child in enumerate(self.experts[0].experts):
            cluster_mask = cluster_assignments == i
            sub_input = input[cluster_mask]

            if sub_input.shape[0] > 0:
                sub_result = child(sub_input)
                sub_result = sub_result * cluster_gates_mul[cluster_mask]

                if results.shape[0] == 0:
                    results = torch.zeros(input.shape[0], sub_result.shape[1], device=sub_result.device,
                                          dtype=sub_result.dtype)
                results[cluster_mask] = sub_result
        result_output = results[:reshaped_input_samples, :]
        # result_output = result_output.view(original_shape).to(original_dtype)
        result_output = result_output.view(original_shape)
        return result_output

moe_layer = MOELayer

class SeqExperts(torch.nn.Module):
    def __init__(self, expert, local_experts=1):
        super(SeqExperts, self).__init__()

        if isinstance(expert, torch.nn.ModuleList):
            self.experts = expert
        else:
            self.experts = torch.nn.ModuleList(
                [copy.deepcopy(expert) for i in range(local_experts)])
        self.local_experts = local_experts



class SingleExpert(torch.nn.Module):
    # one layer MLP with experts
    def __init__(self, model_dim, layer_num, skips=None, activation=F.relu, init_factor=1.0, norm_layer=nn.LayerNorm, use_norm=False, init_trunc_normal=False):
        super().__init__()
        self.model_dim = model_dim
        self.layer_num = layer_num
        self.hidden_dim = model_dim
        self.activation = activation
        self.skips = skips
        # self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.norms = None
        self.init_trunc_normal = init_trunc_normal
        if self.use_norm:
            self.norms = nn.ModuleDict()
            for skip in self.skips:
                self.norms[str(skip)] = norm_layer(model_dim)

        self.layers = nn.ModuleList()

        for i in range(self.layer_num):
            layer = nn.Linear(self.model_dim, self.model_dim)
            if self.init_trunc_normal:
                trunc_normal_linear(layer, std=init_factor)
            else:
                if init_factor != 1.0:
                    with torch.no_grad():
                        layer.weight.multiply_(init_factor)
                        if layer.bias is not None:
                            layer.bias.multiply_(init_factor)
            self.layers.append(layer)

    def extra_repr(self):
        return 'model_dim=%d, layer_num=%d' % (self.model_dim, self.layer_num)

    def forward(self, x: torch.Tensor):
        # x: 1, E, N, C
        h = x
        for layer_id, fc in enumerate(self.layers):
            # fc = self.layers[layer_id]
            h = fc(h)
            
            # skip connections
            if self.skips is not None:
                if layer_id in self.skips:
                    h = h + x
                    if layer_id < (self.layer_num - 1):
                        h = self.activation(h)
                    x = h
                else:
                    if layer_id < (self.layer_num - 1):
                        h = self.activation(h)
            else:
                if layer_id < (self.layer_num - 1):
                    h = self.activation(h)
        return h

def trunc_normal_linear(module, std=1.0):
    trunc_normal_(module.weight, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, layer_num, skips=None, act_fn=F.relu):
        super().__init__()

        self.act_fn = act_fn
        self.layer_num = layer_num
        self.fcs = nn.ModuleList()
        self.skips = skips

        for i in range(layer_num):
            in_ch = in_features if i == 0 else hidden_features
            out_ch = out_features if i == layer_num - 1 else hidden_features
            self.fcs.append(nn.Linear(in_ch, out_ch))

    def forward(self, x):
        h = x
        for i, fc in enumerate(self.fcs):
            # fc = self.fcs[i]
            h = fc(h)

            # skip connections
            if self.skips is not None:
                if i in self.skips:
                    h = h + x
                    if i < self.layer_num - 1:
                        h = self.act_fn(h)
                    x = h
                else:
                    if i < self.layer_num - 1:
                        h = self.act_fn(h)
            else:
                if i < self.layer_num - 1:
                    h = self.act_fn(h)
        return h
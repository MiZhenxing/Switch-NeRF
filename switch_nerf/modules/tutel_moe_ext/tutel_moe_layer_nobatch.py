# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn

from .tutel_fast_dispatch_nobatch import fast_dispatcher as fast_dispatcher_nobatch
from .tutel_fast_dispatch_nobatch import extract_critical as extract_critical_nobatch
from .tutel_fast_dispatch_nobatch import extract_critical_load_importance as extract_critical_load_importance_nobatch
from .tutel_fast_dispatch_nobatch import load_balance, one_hot_with_dtype
from .tutel_fast_dispatch import fast_dispatcher, extract_critical, extract_critical_load_importance
from tutel.impls import communicate as C
from . import tutel_communicate_nobatch as C_no
from timm.models.layers import trunc_normal_

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
        self.gate_noise = gate_noise
        self.use_load_importance_loss = use_load_importance_loss
        self.compute_balance_loss = compute_balance_loss
        self.dispatcher_no_score = dispatcher_no_score
        self.wg = torch.nn.Linear(self.gate_dim, num_global_experts, bias=False)
        

        self.fp32_gate = fp32_gate
        # if self.fp32_gate:
        #     self.wg = self.wg.float()
        
        self.dot_scale = model_dim ** -0.5 # from vit
        self.use_normal_noise = use_normal_noise

        self.capacity_factor = float(os.environ.get('CAP_FACTOR', capacity_factor))
        self.num_global_experts = num_global_experts

        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        
        self.ray_prioritized_droping = ray_prioritized_droping
        self.ray_prioritized_droping_mode = ray_prioritized_droping_mode
        self.ray_prioritized_droping_factor = ray_prioritized_droping_factor

        self.ray_random_droping = ray_random_droping
        self.ray_random_droping_factor = ray_random_droping_factor

        self.is_postscore = is_postscore
        self.input_dropout = torch.nn.Dropout(p=input_dropout_p) if input_dropout_p else None

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2d_a2a = (os.environ.get('TUTEL_ALLTOALL_ALGO', '').upper() == '2D' and os.environ.get('LOCAL_SIZE', -1) == C.get_world_size())

    def apply_on_expert_fn(self, input, ctx, gate_input=None):
        # input = input.to(next(iter(ctx.experts.parameters())).dtype)
        if self.input_dropout:
            input = self.input_dropout(input)

        group = ctx.group

        if self.fp32_gate:
            wg = self.wg.float()
        else:
            wg = self.wg
        with torch.cuda.amp.autocast(enabled=(not self.fp32_gate)):
            if gate_input is None:
                logits = wg(input.to(next(iter(wg.parameters())).dtype))
            else:
                logits = wg(gate_input.to(next(iter(wg.parameters())).dtype))

        if self.use_normal_noise and ctx.training:
            logits = logits + torch.randn_like(logits, device=logits.device) / self.num_global_experts # Scaling Vision with Sparse Mixture of Experts

        if self.training and self.gate_noise > 0:
            logits_w_noise = logits + self.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        
        if ctx.use_scaled_dot:
            logits_w_noise = logits_w_noise * self.dot_scale

        gates = F.softmax(logits_w_noise, dim=1)

        if self.use_load_importance_loss:
            critical_data, l_loss, l_balance_loss = extract_critical_load_importance(gates, F.softmax(logits, dim=1), logits_w_noise, 
                self.top_k, self.gate_noise, capacity_factor=self.capacity_factor, 
                fp32_gate=self.fp32_gate, batch_prioritized_routing=self.batch_prioritized_routing,
                compute_balance_loss=self.compute_balance_loss)
        else:
            critical_data, l_loss = extract_critical(gates, self.top_k, self.capacity_factor, self.fp32_gate, 
                self.batch_prioritized_routing)
        capacity = critical_data[-1]

        S, M, g_experts = input.size(0), input.size(1), self.num_global_experts
        world_size = C.get_world_size(group)

        # if not hasattr(self, '_fdr'):
        self._fdr = fast_dispatcher(num_global_experts=g_experts, capacity=capacity, model_dim=M, dispatch_dtype=input.dtype)

        self._fdr.update(*critical_data[1:], is_postscore=self.is_postscore)

        dispatched_input = self._fdr.encode(input)

        if ctx.auto_parallel:
            ctx.use_model_parallel = (dispatched_input.numel() < ctx.model_dim * ctx.hidden_size)

        if ctx.use_model_parallel:
            dispatched_input = dispatched_input.reshape(g_experts, -1).repeat(1, ctx.sharded_count)

        if ctx.sharded_count > 1:
            dispatched_input = dispatched_input.reshape(world_size, 1, -1, M)
        else:
            dispatched_input = dispatched_input.reshape(world_size, -1, capacity, M)

        if self.a2a_ffn_overlap_degree == -1:
            expert_output = ctx.expert_fn(dispatched_input)
            expert_output = expert_output.to(input.dtype)
        elif self.a2a_ffn_overlap_degree == 1:
            if self.use_2d_a2a:
                C.AllToAllStatus.init(group, 1, -1)
                dispatched_input = \
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(
                                    C.CurrentStreamRelease.apply(dispatched_input, 0), 0)), 0), 0)
            else:
                dispatched_input = C.all_to_all_single(dispatched_input, group=group)

            expert_output = ctx.expert_fn(dispatched_input)
            expert_output = expert_output.to(input.dtype)

            if self.use_2d_a2a:
                expert_output = \
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(
                                    C.CurrentStreamRelease.apply(expert_output, 0), 0)), 0), 0)
            else:
                expert_output = C.all_to_all_single(expert_output, group=group)
        else:
            split_dim = 2
            C.AllToAllStatus.init(group, self.a2a_ffn_overlap_degree, split_dim)

            # Implicit x.contiguous() in CurrentStreamRelease.forward() and CurrentStreamAcquire.backward()
            if self.use_2d_a2a:
                split_size = dispatched_input.shape[split_dim] // self.a2a_ffn_overlap_degree
                dispatched_input_split = dispatched_input.split(split_size, dim=split_dim)
                dispatched_input_scattered_after_a2a = [
                    C.NcclStreamRelease.apply(
                        C.AllToAll2DAsync.apply(
                            C.NcclStreamAcquire.apply(
                                C.CurrentStreamRelease.apply(x, i), i)), i)
                    for i, x in enumerate(dispatched_input_split)]
            else:
                dispatched_input_ready = C.CurrentStreamRelease.apply(dispatched_input, 0)
                dispatched_input_scattered_after_a2a = C.AllToAllScatterAsync.apply(dispatched_input_ready)

            expert_output_scattered = [
                C.CurrentStreamRelease.apply(ctx.expert_fn(C.CurrentStreamAcquire.apply(x, i)).to(input.dtype), i)
                for i, x in enumerate(dispatched_input_scattered_after_a2a)]

            if self.use_2d_a2a:
                expert_output_gathered_after_a2a = [
                    C.CurrentStreamAcquire.apply(
                        C.NcclStreamRelease.apply(
                            C.AllToAll2DAsync.apply(
                                C.NcclStreamAcquire.apply(x, i)), i), i)
                    for i, x in enumerate(expert_output_scattered)]
                expert_output = torch.cat(expert_output_gathered_after_a2a, dim=split_dim)
            else:
                expert_output_gathered_after_a2a = C.AllToAllGatherAsync.apply(*expert_output_scattered)
                expert_output = C.CurrentStreamAcquire.apply(expert_output_gathered_after_a2a, 0)

        expert_output = expert_output.reshape(-1, g_experts, capacity, M)
        if expert_output.size(0) > 1:
            expert_output = torch.sum(expert_output.view(g_experts, -1, capacity, M), dim=1)
        expert_output = expert_output.view(g_experts * capacity, M)

        result_output = self._fdr.decode(expert_output)

        extras = {}
        if ctx.return_gates:
            extras["gates"] = torch.topk(gates, self.top_k, dim=1).indices
        if ctx.return_gate_logits:
            extras["gate_logits"] = logits
        if self.compute_balance_loss:
            extras["balance_loss"] = l_balance_loss

        return result_output, l_loss, extras

    def apply_on_expert_fn_nobatch(self, input, ctx, gate_input=None):
        # input = input.to(next(iter(ctx.experts.parameters())).dtype)
        if self.input_dropout:
            input = self.input_dropout(input)

        group = ctx.group
        
        if self.fp32_gate:
            wg = self.wg.float()
        else:
            wg = self.wg
        with torch.cuda.amp.autocast(enabled=(not self.fp32_gate)):
            if gate_input is None:
                logits = wg(input.to(next(iter(wg.parameters())).dtype))
            else:
                logits = wg(gate_input.to(next(iter(wg.parameters())).dtype))
        
        if ctx.use_scaled_dot:
            logits = logits * self.dot_scale
        
        if self.use_normal_noise and ctx.training:
            logits = logits + torch.randn_like(logits, device=logits.device) / self.num_global_experts # Scaling Vision with Sparse Mixture of Experts

        if self.training and self.gate_noise > 0:
            logits_w_noise = logits + self.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        
        if ctx.use_scaled_dot:
            logits_w_noise = logits_w_noise * self.dot_scale

        gates = F.softmax(logits_w_noise, dim=1)

        if self.use_load_importance_loss:
            critical_data, l_loss, l_balance_loss = extract_critical_load_importance_nobatch(gates, F.softmax(logits, dim=1), logits_w_noise, 
                self.top_k, self.gate_noise, capacity_factor=self.capacity_factor, 
                fp32_gate=self.fp32_gate, batch_prioritized_routing=self.batch_prioritized_routing,
                compute_balance_loss=self.compute_balance_loss)
        else:
            critical_data, l_loss = extract_critical_nobatch(gates, self.top_k, self.capacity_factor, self.fp32_gate, 
                self.batch_prioritized_routing)
        capacity = critical_data[-1]
        expert_input_nums = critical_data[-2]

        S, M, g_experts = input.size(0), input.size(1), self.num_global_experts
        world_size = C.get_world_size(group)

        # if not hasattr(self, '_fdr_nobatch'):
        self._fdr_nobatch = fast_dispatcher_nobatch(num_global_experts=g_experts, capacity=capacity, model_dim=M, dispatch_dtype=input.dtype)

        self._fdr_nobatch.update(*critical_data[1:], is_postscore=self.is_postscore, dispatcher_no_score=self.dispatcher_no_score)

        dispatched_input = self._fdr_nobatch.encode(input)

        if ctx.auto_parallel:
            ctx.use_model_parallel = (dispatched_input.numel() < ctx.model_dim * ctx.hidden_size)

        if ctx.use_model_parallel:
            raise NotImplementedError

        if ctx.sharded_count > 1:
            raise NotImplementedError
        else:
            dispatched_input = dispatched_input.reshape(-1, M)
        
        if self.a2a_ffn_overlap_degree == -1:
            raise NotImplementedError
        elif self.a2a_ffn_overlap_degree == 1:
            if self.use_2d_a2a:
                raise NotImplementedError
            else:
                expert_input_nums_a2a = C.all_to_all_single(expert_input_nums, group=group)

                chunk_size = expert_input_nums_a2a.shape[0] // world_size
                expert_input_nums_chunk_sum = torch.sum(expert_input_nums.reshape(-1, chunk_size), dim=1)
                expert_input_nums_a2a_chunk_sum = torch.sum(expert_input_nums_a2a.reshape(-1, chunk_size), dim=1)

                dispatched_input = C_no.list_all_to_all_single(input=dispatched_input, input_splits=expert_input_nums_chunk_sum.tolist(), 
                    output_splits=expert_input_nums_a2a_chunk_sum.tolist(), group=group)

            chunk_size = expert_input_nums_a2a.shape[0] // world_size
            expert_input_nums_input = torch.split(expert_input_nums_a2a, chunk_size)

            if ctx.expert_type == 'seqexperts' or ctx.expert_type == 'multiseqexperts':
                dispatched_input = torch.split(dispatched_input, expert_input_nums_a2a_chunk_sum.tolist(), dim=0)
                expert_output = ctx.expert_fn([dispatched_input, expert_input_nums_input])
            else:
                expert_output = ctx.expert_fn(dispatched_input)

            expert_output = [i.to(input.dtype) for i in expert_output]

            if self.use_2d_a2a:
                raise NotImplementedError
            else:
                chunk_size = len(expert_input_nums) // world_size
                expert_input_nums_chunk_sum = torch.sum(expert_input_nums.reshape(-1, chunk_size), dim=1)

                expert_output = torch.cat(expert_output, dim=0)
                expert_output = C_no.list_all_to_all_single(input=expert_output, input_splits=expert_input_nums_a2a_chunk_sum.tolist(), 
                    output_splits=expert_input_nums_chunk_sum.tolist(), group=group)

        else:
            raise NotImplementedError

        expert_output = expert_output.contiguous()
        result_output = self._fdr_nobatch.decode(expert_output)

        extras = {}
        if ctx.return_gates:
            extras["gates"] = torch.topk(gates, self.top_k, dim=1).indices
        if ctx.return_gate_logits:
            extras["gate_logits"] = logits
        if self.compute_balance_loss:
            extras["balance_loss"] = l_balance_loss

        return result_output, l_loss, extras



    def apply_on_expert_fn_nobatch_torch(self, input, ctx, gate_input=None):
        # input = input.to(next(iter(ctx.experts.parameters())).dtype)
        if self.input_dropout:
            input = self.input_dropout(input)

        group = ctx.group

        if self.fp32_gate:
            wg = self.wg.float()
        else:
            wg = self.wg
        with torch.cuda.amp.autocast(enabled=(not self.fp32_gate)):
            if gate_input is None:
                logits = wg(input.to(next(iter(wg.parameters())).dtype))
            else:
                logits = wg(gate_input.to(next(iter(wg.parameters())).dtype))
        
        if ctx.use_scaled_dot:
            logits = logits * self.dot_scale
        
        if self.use_normal_noise and ctx.training:
            logits = logits + torch.randn_like(logits, device=logits.device) / self.num_global_experts # Scaling Vision with Sparse Mixture of Experts

        if self.training and self.gate_noise > 0:
            logits_w_noise = logits + self.gate_noise * torch.randn_like(logits) / self.num_global_experts
        else:
            logits_w_noise = logits
        
        if ctx.use_scaled_dot:
            logits_w_noise = logits_w_noise * self.dot_scale

        gates = F.softmax(logits_w_noise, dim=1)

        results = [torch.zeros_like(input) for i in range(self.top_k)]
        cluster_gates_mul, cluster_assignments = torch.topk(gates, k=self.top_k, dim=1)
        # only support top1
        # cluster_assignments = cluster_assignments.squeeze(1)

        for top_k_id in range(self.top_k):
            for expert_id, expert in enumerate(ctx.experts[0].experts):
                cluster_mask = cluster_assignments[..., top_k_id] == expert_id
                sub_input = input[cluster_mask]
                if sub_input.shape[0] == 0:
                    sub_input = input[0:1, :] # dummy input for expert
                    sub_result = expert(sub_input)
                    results[top_k_id][0:1, :] = sub_result
                else:
                    sub_result = expert(sub_input)
                    if not self.dispatcher_no_score:
                        sub_result = sub_result * cluster_gates_mul[..., top_k_id:top_k_id+1][cluster_mask]
                    results[top_k_id][cluster_mask] = sub_result

        # weight average
        if self.top_k > 1:
            result_output = sum(results) / torch.sum(cluster_gates_mul, dim=-1, keepdim=True)
        else:
            result_output = results[0]
        top1_indices = cluster_assignments[..., 0].view(-1)
        masks_se = one_hot_with_dtype(top1_indices, num_classes=self.num_global_experts, dtype=top1_indices.dtype)
        l_loss = load_balance(gates, masks_se, self.num_global_experts, self.fp32_gate)
        
        result_output = result_output.reshape(-1, result_output.shape[-1])

        extras = {}
        if ctx.return_gates:
            extras["gates"] = cluster_assignments
        if ctx.return_gate_logits:
            extras["gate_logits"] = logits

        return result_output, l_loss, extras


class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """
    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-5))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception("Invalid value of num_local_experts: %d" % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, "Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}."
        return world_size // -num_local_experts

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
        use_scaled_dot=False
    ):
        super().__init__()
        assert model_dim % 2 == 0, "Model_dim (%s) must be even value, while this Model_dim mod 2 > 0." % model_dim
        group = group or dist.group.WORLD

        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)
        self.moe_no_batch = moe_no_batch
        self.use_residual = use_residual
        self.return_gates = return_gates
        self.return_gate_logits = return_gate_logits
        self.use_scaled_dot = use_scaled_dot

        if not isinstance(experts, dict):
            self.is_builtin_experts = False
            self.num_local_experts = len(self.experts)
        else:
            self.is_builtin_experts = True
            self.num_local_experts = experts.get('count_per_node', 1)

        self.num_global_experts = MOELayer.global_expert_count(self.num_local_experts, self.group)

        num_devices = C.get_world_size(self.group)
        if self.num_global_experts < num_devices:
            sharded_count = num_devices // self.num_global_experts
            assert experts['hidden_size_per_expert'] % sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({experts['hidden_size_per_expert']}) to {sharded_count} slices"
            self.num_local_experts, experts['hidden_size_per_expert'] = 1, experts['hidden_size_per_expert'] // sharded_count
            self.ffn_zero_group = C.create_groups_from_world(group_count=self.num_global_experts).model_group
        else:
            sharded_count = 1
            self.ffn_zero_group = None

        if sharded_count == 1 or not self.is_builtin_experts:
            self.auto_parallel, self.use_model_parallel = False, False
        elif parallel_type == 'auto':
            self.auto_parallel, self.use_model_parallel = True, False
        else:
            self.auto_parallel, self.use_model_parallel = False, (parallel_type == 'model')

        self.hidden_size = experts.get('hidden_size_per_expert', 'None')
        self.model_dim = model_dim
        self.sharded_count = sharded_count

        if self.use_residual:
            self.coefficient = torch.nn.Linear(model_dim, 2)

        if not isinstance(experts, dict):
            self.experts = cast(ModuleList, experts) if type(experts) == ModuleList else ModuleList(experts)
            self.expert_type = None
        else:
            # mizhenxing
            self.expert_type = experts['type']
            if experts['type'] == 'ffn':
                ''' << Fused FFN Experts V1 >> (kernels = 5)

                    hidden[W, E, C, V] +=! input[W, E, C, M] x expert_fc1[0, E, M, V]
                    hidden[W, E, C, V]  =  hidden[W, E, C, V] + bias_fc1[E, V]
                    hidden[W, E, C, V]  =  activation_fn(hidden[W, E, C, V])
                    hidden[W, E, C, M] +=! hidden[W, E, C, V] x expert_fc2[0, E, V, M]
                    output[W, E, C, M]  =  hidden[W, E, C, M] + bias_fc2[E, M]

                    << Fused FFN Experts V2 >> (kernels = 7)

                    hidden[E, W, C, M]  =   input[W, E, C, M]
                    hidden[E, W, C, V] +=! hidden[E, W, C, M] x expert_fc1[0, E, M, V]
                    hidden[E, W, C, V]  =  hidden[E, W, C, V] + bias_fc1[E, V]
                    hidden[E, W, C, V]  =  activation_fn(hidden[E, W, C, V])
                    hidden[E, W, C, M] +=! hidden[E, W, C, V] x expert_fc2[0, E, V, M]
                    hidden[E, W, C, M]  =  hidden[E, W, C, M] + bias_fc2[E, M]
                    output[W, E, C, M]  =  hidden[E, W, C, M]
                '''

                fused_custom_fn = experts.get('fused_custom_fn')
                if fused_custom_fn is None:
                    activation_fn = experts.get('activation_fn', lambda x: F.relu(x))
                implicit_dropout_p = experts.get('implicit_dropout_p', 0)

                class FusedExpertsNetwork(torch.nn.Module):
                    def __init__(self, model_dim, hidden_size, local_experts, init_trunc_normal=False, init_factor=1.0):
                        super().__init__()
                        self.skip_expert = (int(os.environ.get('SKIP_EXPERT', '0')) != 0)

                        self.init_trunc_normal = init_trunc_normal

                        fc1_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc2_weight = torch.empty(1, local_experts, hidden_size, model_dim)
                        fc1_bias = torch.empty(1, local_experts, 1, hidden_size)
                        fc2_bias = torch.empty(1, local_experts, 1, (model_dim + sharded_count - 1) // sharded_count)                       

                        for i in range(local_experts):
                            fc1 = torch.nn.Linear(model_dim, hidden_size)
                            fc2 = torch.nn.Linear(hidden_size, model_dim)

                            if self.init_trunc_normal:
                                # https://github.com/rwightman/pytorch-image-models/blob/7cedc8d4743f2b2bbf835fc387c917461fa4911a/timm/models/vision_transformer.py#L455
                                trunc_normal_linear(fc1, std=init_factor)
                                trunc_normal_linear(fc2, std=init_factor)

                            fc1_weight[0, i, :, :], fc1_bias[0, i, :, :] = fc1.weight, fc1.bias
                            fc2_weight[0, i, :, :], fc2_bias[0, i, :, :] = fc2.weight.t(), fc2.bias[:fc2_bias.size(-1)]

                        self.model_dim, self.hidden_size, self.local_experts = model_dim, hidden_size, local_experts
                        if self.local_experts == 1:
                            fc1_weight = fc1_weight.view(self.hidden_size, self.model_dim)
                            fc2_weight = fc2_weight.view(self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.hidden_size)
                            fc2_bias = fc2_bias.view(-1)
                        else:
                            fc1_weight = fc1_weight.view(self.local_experts, self.hidden_size, self.model_dim)
                            fc2_weight = fc2_weight.view(self.local_experts, self.hidden_size, self.model_dim)
                            fc1_bias = fc1_bias.view(self.local_experts, 1, self.hidden_size)
                            fc2_bias = fc2_bias.view(self.local_experts, 1, -1)

                        self.register_parameter(name='fc1_weight', param=torch.nn.Parameter(fc1_weight))
                        self.register_parameter(name='fc2_weight', param=torch.nn.Parameter(fc2_weight))
                        self.register_parameter(name='fc1_bias', param=torch.nn.Parameter(fc1_bias))
                        self.register_parameter(name='fc2_bias', param=torch.nn.Parameter(fc2_bias))

                        if implicit_dropout_p:
                            self.dropout_fc1 = torch.nn.Dropout(p=implicit_dropout_p)
                            self.dropout_fc2 = torch.nn.Dropout(p=implicit_dropout_p)
                        else:
                            self.dropout_fc1 = self.dropout_fc2 = lambda x: x

                    def extra_repr(self):
                        return 'model_dim=%d, hidden_size=%d, local_experts=%d, bias=%s' % (self.model_dim, self.hidden_size, self.local_experts, self.fc1_bias is not None)

                    def forward(self, x, ctx):
                        if self.skip_expert:
                            return x
                        if fused_custom_fn is not None:
                            return fused_custom_fn(self, x)

                        fc1_weight, fc2_weight, fc1_bias, fc2_bias = self.fc1_weight, self.fc2_weight, self.fc1_bias, self.fc2_bias
                        if ctx.ffn_zero_group is not None:
                            if not ctx.use_model_parallel:
                                fc1_weight = C.zero_gather(self.fc1_weight, group=ctx.ffn_zero_group)
                                fc2_weight = C.zero_gather(self.fc2_weight, group=ctx.ffn_zero_group)
                                fc1_bias = C.zero_gather(self.fc1_bias, group=ctx.ffn_zero_group)

                            # Specially treat fc2_bias to make hybrid data & model parallels equivalent
                            fc2_bias = C.zero_gather(self.fc2_bias, group=ctx.ffn_zero_group)
                            if fc2_bias.size(-1) != self.model_dim:
                                fc2_bias = fc2_bias[:, :self.model_dim]

                        if self.local_experts == 1:
                            original_shape, x = x.shape, x.view(-1, self.model_dim)
                            x = torch.addmm(fc1_bias.unsqueeze(0), x, fc1_weight.t())
                            x = activation_fn(x.unsqueeze(0)).squeeze(0)
                            x = self.dropout_fc1(x)
                            if ctx.use_model_parallel:
                                fc2_bias = torch.mul(fc2_bias, 1.0 / sharded_count)
                            x = torch.addmm(fc2_bias.unsqueeze(0), x, fc2_weight)
                            x = self.dropout_fc2(x)
                            x = x.view(original_shape)
                        else:
                            x = x.permute(1, 0, 2, 3)
                            original_shape, x = x.shape, x.reshape(self.local_experts, -1, self.model_dim)
                            x = torch.matmul(x, fc1_weight.swapaxes(1, 2)) + fc1_bias
                            x = activation_fn(x)
                            x = self.dropout_fc1(x)
                            x = torch.matmul(x, fc2_weight) + fc2_bias
                            x = self.dropout_fc2(x)
                            x = x.reshape(self.local_experts, original_shape[1], original_shape[2], self.model_dim)
                            x = x.permute(1, 0, 2, 3)
                        return x

                    def to(self, *args, **kwargs):
                        self = super().to(*args, **kwargs)
                        self.fc1_weight = self.fc1_weight.to(*args, **kwargs)
                        self.fc2_weight = self.fc2_weight.to(*args, **kwargs)
                        self.fc1_bias = self.fc1_bias.to(*args, **kwargs)
                        self.fc2_bias = self.fc2_bias.to(*args, **kwargs)
                        return self

                if seeds is not None and seeds[1] is not None:
                    torch.manual_seed(seeds[1])
                self.experts = ModuleList([FusedExpertsNetwork(model_dim, self.hidden_size, self.num_local_experts, 
                    init_trunc_normal=experts.get('init_trunc_normal', False), init_factor=experts['init_factor'])])
                if self.use_residual:
                    self.residual_expert = FusedExpertsNetwork(model_dim, self.hidden_size, 1, 
                        init_trunc_normal=experts.get('init_trunc_normal', False), init_factor=experts['init_factor'])
            elif experts['type'] == 'seqexperts' or experts['type'] == 'multiseqexperts':
                if seeds is not None and seeds[1] is not None:
                    torch.manual_seed(seeds[1])
                net = experts['net']
                self.experts = ModuleList([SeqExperts(net, local_experts=self.num_local_experts)])
                if self.use_residual:
                    if isinstance(net, nn.ModuleList):
                        self.residual_expert = SeqExperts(net[0], local_experts=1)
                    else:
                        self.residual_expert = SeqExperts(net, local_experts=1)
            elif experts['type'] == 'expertmlp':
                if seeds is not None and seeds[1] is not None:
                    torch.manual_seed(seeds[1])
                layer_num = experts['layer_num']
                skips = experts['skips']
                init_factor = experts['init_factor']
                activation_fn = experts.get('activation_fn', lambda x: F.relu(x))
                init_trunc_normal = experts.get('init_trunc_normal', False)
                self.experts = ModuleList([ExpertMLP(model_dim=model_dim, 
                    local_experts=self.num_local_experts, 
                    layer_num=layer_num, skips=skips, 
                    activation=activation_fn, init_factor=init_factor, 
                    init_trunc_normal=init_trunc_normal)])
                if self.use_residual:
                    self.residual_expert = ExpertMLP(model_dim=model_dim, 
                        local_experts=1, 
                        layer_num=layer_num, skips=skips, 
                        activation=activation_fn, init_factor=init_factor, 
                        init_trunc_normal=init_trunc_normal)
            else:
                raise Exception('Builtin expert type is not recognized: %s' % experts['type'])

        if scan_expert_func is not None:
            for expert in self.experts:
                for n, p in expert.named_parameters():
                    scan_expert_func(n, p)

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

        def expert_fn(input):
            if self.is_builtin_experts:
                if self.expert_type == 'ffn':
                    expert_output = self.experts[0](input, self)
                elif self.expert_type == 'seqexperts' or self.expert_type == 'multiseqexperts':
                    if self.moe_no_batch:
                        # input should be a list
                        expert_output = self.experts[0](input[0], input[1])
                    else:
                        expert_output = self.experts[0](input)
                elif self.expert_type == 'expertmlp':
                    expert_output = self.experts[0](input, self)
            else:
                chunks = input.chunk(self.num_local_experts, dim=1)
                expert_output = torch.cat([expert(chunk) for chunk, expert in zip(chunks, self.experts)], dim=1)
            return expert_output

        self.expert_fn = expert_fn
        self.expected_sample_size = 0 if pad_samples else -1

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception("Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts." % param_type)

    def forward(self, input: Tensor, gate_index=0, **kwargs):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output
        
        if "gate_input" in kwargs:
            gate_input = kwargs["gate_input"]
        else:
            gate_input = None

        original_shape, original_dtype  = input.shape, input.dtype
        # self.original_shape = original_shape
        assert len(input.shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
        reshaped_input = input.reshape(-1, input.shape[-1])
        reshaped_input_samples = reshaped_input.shape[0]

        if gate_input is not None:
            original_gate_shape, original_gate_dtype  = gate_input.shape, gate_input.dtype
            assert len(gate_input.shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"
            reshaped_gate_input = gate_input.reshape(-1, gate_input.shape[-1])
        else:
            reshaped_gate_input = None

        if self.expected_sample_size >= 0:
            self.expected_sample_size = self.expected_sample_size or reshaped_input.size(0)
            if reshaped_input.size(0) != self.expected_sample_size:
                if C.get_world_rank(self.group) == 0:
                    logging.warning('MoE is scaled to work on sample size = %s, while receiving sample size = %s (will slow down this forward step)' % (self.expected_sample_size, reshaped_input.size(0)))
                if reshaped_input.size(0) > self.expected_sample_size:
                    self.expected_sample_size = reshaped_input.size(0)
                pad_input = torch.zeros([self.expected_sample_size, self.model_dim], dtype=reshaped_input.dtype, layout=reshaped_input.layout, device=reshaped_input.device)
                pad_input[:reshaped_input.size(0)] = reshaped_input
                reshaped_input = pad_input

        if ("apply_on_expert_fn_name" in kwargs) and (kwargs["apply_on_expert_fn_name"] is not None):
            apply_on_expert_fn_call = getattr(self.gates[gate_index], kwargs["apply_on_expert_fn_name"])
            result_output, l_aux, gate_extras = apply_on_expert_fn_call(reshaped_input, self, gate_input=reshaped_gate_input)
        else:
            if self.moe_no_batch:
                result_output, l_aux, gate_extras = self.gates[gate_index].apply_on_expert_fn_nobatch(reshaped_input, self, gate_input=reshaped_gate_input)
            else:
                result_output, l_aux, gate_extras = self.gates[gate_index].apply_on_expert_fn(reshaped_input, self, gate_input=reshaped_gate_input)

        # from deepspeed
        if self.use_residual:
            reshaped_residual_input = reshaped_input.reshape(1, 1, -1, input.shape[-1])
            # Residual MoE
            output_residual = self.residual_expert(reshaped_residual_input, self)
            if type(output_residual) is tuple:
                output_residual = output_residual[0]
            output_residual = output_residual.reshape(-1, output_residual.shape[-1])
            
            coef = self.coefficient(reshaped_input)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            result_output = result_output * coef[..., 0:1] + output_residual * coef[..., 1:]

        result_output = result_output[:reshaped_input_samples, :]
        result_output = result_output.view(original_shape).to(original_dtype)
        self.l_aux = result_output.l_aux = l_aux
        if self.return_gates or self.return_gate_logits \
            or self.gates[gate_index].compute_balance_loss:

            self.gate_extras = result_output.gate_extras = gate_extras
        return self.result_func(result_output) if self.result_func is not None else result_output

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

    def forward(self, inputs, expert_input_nums=None):
        # inputs: list of [N, M] or B, E, Cap, C
        # expert_input_nums: list of tensors of numbers for each input
        if expert_input_nums is None:
            chunks = torch.split(inputs, 1, dim=1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                out = expert(chunk)
                expert_outputs += [out]
            output = torch.cat(expert_outputs, dim=1)
        else:
            output = []
            for input_i, input_d in enumerate(inputs):
                chunks = torch.split(input_d, expert_input_nums[input_i].tolist(), dim=0)
                expert_outputs = []
                for chunk, expert in zip(chunks, self.experts):
                    out = expert(chunk)
                    expert_outputs += [out]
                expert_output = torch.cat(expert_outputs, dim=0)
                output.append(expert_output)

        return output


class ExpertMLP(torch.nn.Module):
    # one layer MLP with experts
    def __init__(self, model_dim, local_experts, layer_num, skips=None, activation=F.relu, init_factor=1.0, init_trunc_normal=False):
        super().__init__()
        self.model_dim = model_dim
        self.local_experts = local_experts
        self.layer_num = layer_num
        self.hidden_dim = model_dim
        self.activation = activation
        self.skips = skips
        self.init_factor = init_factor
        self.init_trunc_normal = init_trunc_normal

        self.weights = nn.ParameterList()
        self.bias = nn.ParameterList()

        for j in range(self.layer_num):
            fc1_weight = nn.Parameter(torch.zeros(local_experts, model_dim, self.hidden_dim))
            fc1_bias = nn.Parameter(torch.zeros(local_experts, 1, self.hidden_dim))
            for i in range(local_experts):
                fc1 = nn.Linear(model_dim, self.hidden_dim)
                if self.init_trunc_normal:
                    trunc_normal_linear(fc1, std=init_factor)
                    with torch.no_grad():
                        fc1_weight[i, :, :], fc1_bias[i, :, :] = fc1.weight.t(), fc1.bias
                else:
                    with torch.no_grad():
                        fc1_weight[i, :, :], fc1_bias[i, :, :] = fc1.weight.t() * init_factor, fc1.bias * init_factor
            
            self.weights.append(fc1_weight)
            self.bias.append(fc1_bias)

    def extra_repr(self):
        return 'model_dim=%d, local_experts=%d, layer_num=%d' % (self.model_dim, self.local_experts, self.layer_num)

    @torch.no_grad()
    def reset_parameters(self):
        for j in range(self.layer_num):
            fc1_weight = self.weights[j]
            fc1_bias = self.bias[j]
            for i in range(self.local_experts):
                fc1 = nn.Linear(self.model_dim, self.hidden_dim)
                if self.init_trunc_normal:
                    trunc_normal_linear(fc1, std=self.init_factor)
                    with torch.no_grad():
                        fc1_weight[i, :, :], fc1_bias[i, :, :] = fc1.weight.t(), fc1.bias
                else:
                    with torch.no_grad():
                        fc1_weight[i, :, :], fc1_bias[i, :, :] = fc1.weight.t() * self.init_factor, fc1.bias * self.init_factor

    def forward(self, x, ctx):
        # x: 1, E, N, C
        x = x.squeeze(0)
        h = x
        for layer_id in range(self.layer_num):
            fc1_weight, fc1_bias = self.weights[layer_id], self.bias[layer_id]
            if ctx.ffn_zero_group is not None:
                if not ctx.use_model_parallel:
                    fc1_weight = C.PrimAllgather.apply(ctx.ffn_zero_group, self.weights[layer_id], fused=True)
                    if layer_id < (self.layer_num - 1):
                        fc1_bias = C.PrimAllgather.apply(ctx.ffn_zero_group, self.bias[layer_id], fused=True)

                # Specially treat fc1_bias to make hybrid data & model parallels equivalent
                if layer_id == (self.layer_num - 1):
                    fc1_bias = C.PrimAllgather.apply(ctx.ffn_zero_group, self.bias[layer_id], fused=True)
                    if fc1_bias.size(-1) != self.model_dim:
                        fc1_bias = fc1_bias[:, :self.model_dim]

            # h = torch.addmm(fc1_bias, h, fc1_weight)
            # h = torch.matmul(h, fc1_weight) + fc1_bias
            # h = torch.einsum('benc,ecd->bend', (h, fc1_weight)) + fc1_bias
            h = torch.baddbmm(fc1_bias, h, fc1_weight)
            
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

        return h.unsqueeze(0)


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

    def forward(self, x):
        # x: 1, E, N, C
        h = x
        for layer_id in range(self.layer_num):
            fc = self.layers[layer_id]
            h = fc(h)
            
            # skip connections
            if self.skips is not None:
                if layer_id in self.skips:
                    h = h + x
                    if self.use_norm:
                        norm = self.norms[str(layer_id)]
                        h = norm(h)
                    elif layer_id < (self.layer_num - 1):
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
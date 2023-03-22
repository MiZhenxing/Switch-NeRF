from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from switch_nerf.models.nerf import Embedding, ShiftedSoftplus, MipEmbedder, trunc_exp
from modules.tutel_moe_ext.tutel_moe_layer_nobatch import MOELayer
from modules.tutel_moe_ext.tutel_moe_nobatch import moe_layer, SingleExpert

from modules.tutel_moe_ext.torch_moe_layer_nobatch import MOELayer as MOELayer_torch
from modules.tutel_moe_ext.torch_moe_layer_nobatch import SingleExpert as SingleExpert_torch
from modules.tutel_moe_ext.torch_moe_layer_nobatch import Mlp as Mlp_torch
import copy

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
        for i in range(self.layer_num):
            fc = self.fcs[i]
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

class NormMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, layer_num, skips=None, act_fn=F.relu, norm_name="none"):
        super().__init__()

        self.act_fn = act_fn
        self.layer_num = layer_num
        self.fcs = nn.ModuleList()
        self.skips = skips
        self.norms = nn.ModuleList()

        for i in range(layer_num):
            in_ch = in_features if i == 0 else hidden_features
            out_ch = out_features if i == layer_num - 1 else hidden_features
            self.fcs.append(nn.Linear(in_ch, out_ch))
            if i < layer_num - 1:
                if norm_name == "batchnorm":
                    self.norms.append(nn.BatchNorm1d(out_ch))
                if norm_name == "layernorm":
                    self.norms.append(nn.LayerNorm(out_ch))
                elif norm_name == "none":
                    pass
                else:
                    raise NotImplementedError

    def forward(self, x):
        use_norm = len(self.norms) > 0
        h = x
        for i in range(self.layer_num):
            fc = self.fcs[i]
            h = fc(h)

            # skip connections
            if self.skips is not None:
                if i in self.skips:
                    h = h + x
                    if i < self.layer_num - 1:
                        if use_norm:
                            h = self.norms[i](h)
                        h = self.act_fn(h)
                    x = h
                else:
                    if i < self.layer_num - 1:
                        if use_norm:
                            h = self.norms[i](h)
                        h = self.act_fn(h)
            else:
                if i < self.layer_num - 1:
                    if use_norm:
                        h = self.norms[i](h)
                    h = self.act_fn(h)
        return h
        
class NeRFMoE(nn.Module):
    def __init__(self, args, pos_xyz_dim: int, pos_dir_dim: int,
                 appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int,
                 sigma_activation: nn.Module):
        super().__init__()
        self.args = args
        self.layer_cfg = args.layer_cfg
        self.layer_num_main = self.layer_cfg["layer_num_main"]
        self.sigma_tag = str(self.layer_cfg["sigma_tag"])
        # self.affine_tag = str(self.layer_cfg["affine_tag"])
        self.dir_tag = str(self.layer_cfg["dir_tag"])
        self.color_tag = str(self.layer_cfg["color_tag"])
        # self.skip_tag = [str(i) for i in self.layer_cfg["skip_tag"]]

        # if args.no_expert_parallel:
        #     from tutel import net
        #     self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        # else:
        #     self.single_data_group = None

        # self.xyz_ch = self.layer_cfg["xyz_ch"]
        if rgb_dim > 3:
            assert pos_dir_dim == 0
        # self.dir_ch = self.layer_cfg["dir_ch"]

        self.xyz_dim = xyz_dim
        self.embedding_xyz = Embedding(pos_xyz_dim)
        self.in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2

        self.layers = nn.ModuleDict()

        self.layer_tags_main = [i for i in range(self.layer_num_main)]
        self.layer_tags_all = self.layer_tags_main + ["xyz", "sigma"]
        if pos_dir_dim > 0:
            self.layer_tags_all += ["color"]
        
        self.scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True)

        if pos_dir_dim > 0:
            self.embedding_dir = Embedding(pos_dir_dim)
            self.in_channels_dir = 3 + 3 * pos_dir_dim * 2
        else:
            self.embedding_dir = None
            self.in_channels_dir = 0
        
        if appearance_dim > 0:
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None

        if affine_appearance:
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None

        if pos_dir_dim > 0:
            # self.layer_tags_all += ["dir"]
            self.dir_ch = self.in_channels_dir + (appearance_dim if not affine_appearance else 0)

        self.sigma_activation = sigma_activation
        
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()
        else:
            self.rgb_activation = None  # We're using spherical harmonics and will convert to sigmoid in rendering.py

        self.pos_dir_dim = pos_dir_dim
        self._ddp_params_and_buffers_to_ignore = []
        
        if self.args.use_moe_external_gate:
            self.layer_tags_all += ["moe_external_gate"]
        if self.args.use_gate_input_norm:
            self.layer_tags_all += ["gate_input_norm"]

        # xyz encoding layers
        for i in self.layer_tags_all:
            i_tag = str(i)
            i_cfg = self.layer_cfg["layers"][i_tag]
            in_ch = i_cfg["in_ch"]
            h_ch = i_cfg["h_ch"]
            out_ch = i_cfg["out_ch"]
            if i == "sigma":
                if pos_dir_dim > 0:
                    assert out_ch == 1
                else:
                    assert out_ch == 4

            if i == "color":
                assert out_ch == 3
            
            # if i == "dir":
            #     assert in_ch == self.dir_ch
            
            i_num = i_cfg["num"]

            if i_cfg["type"] == "mlp":
                self.layers[i_tag] = Mlp(in_features=in_ch, hidden_features=h_ch, out_features=out_ch, layer_num=i_num, skips=i_cfg.get("skips", None))
                if "requires_grad" in i_cfg:
                    for tmp_fc in self.layers[i_tag].fcs:
                        tmp_fc.weight.requires_grad = i_cfg["requires_grad"]
                        if hasattr(tmp_fc, "bias"):
                            tmp_fc.bias.requires_grad = i_cfg["requires_grad"]
            elif i_cfg["type"] == "moe":
                assert in_ch == out_ch
                if i_num > 1:
                    assert h_ch == in_ch
                
                gate_type = {'type': i_cfg["gate_type"], 'k': i_cfg["k"], 
                    'fp32_gate': i_cfg["fp32_gate"], 
                    "capacity_factor": args.moe_capacity_factor,
                    "batch_prioritized_routing": args.batch_prioritized_routing,
                    "gate_noise": args.gate_noise,
                    "compute_balance_loss": args.compute_balance_loss,
                    "dispatcher_no_score": args.dispatcher_no_score,
                    "is_postscore": not args.dispatcher_no_postscore}

                if 'gate_dim' in i_cfg:
                    gate_type["gate_dim"] = i_cfg["gate_dim"]

                # gate_type['record_expert_hist'] = i_cfg["record_expert_hist"]
                # gate_type['return_topk_indices'] = i_cfg["return_topk_indices"]

                if hasattr(args, "moe_expert_type"):
                    expert_type = args.moe_expert_type
                else:
                    expert_type = 'expertmlp'

                local_expert_num = i_cfg.get("local_expert_num", None)
                if local_expert_num is None:
                    local_expert_num = args.moe_local_expert_num
                experts = {
                    'type': expert_type, 
                    'count_per_node': local_expert_num,
                    'hidden_size_per_expert': in_ch
                }

                if "init_trunc_normal" in i_cfg:
                    experts['init_trunc_normal'] = i_cfg["init_trunc_normal"]
                else:
                    experts['init_trunc_normal'] = False
                
                if expert_type == 'expertmlp':
                    experts['layer_num'] = i_num
                    experts['skips'] = i_cfg["skips"]
                    experts['init_factor'] = i_cfg["init_factor"]
                
                if expert_type == 'seqexperts':
                    fcs = nn.ModuleList()
                    for j in range(local_expert_num):
                        init_trunc_normal = False
                        if "init_trunc_normal" in i_cfg:
                            init_trunc_normal=i_cfg["init_trunc_normal"]
                        fcs.append(SingleExpert(model_dim=in_ch, layer_num=i_num, skips=i_cfg["skips"], 
                        init_factor=i_cfg["init_factor"], init_trunc_normal=init_trunc_normal))
                    experts["net"] = fcs
                
                if expert_type == 'multiseqexperts':
                    fcs = nn.ModuleList()
                    for expert_id in range(local_expert_num):
                        expert_cfg = i_cfg[str(expert_id)]
                        expert_init_trunc_normal = False
                        if "init_trunc_normal" in expert_cfg:
                            expert_init_trunc_normal=expert_cfg["init_trunc_normal"]
                        expert_in_ch = expert_cfg["in_ch"]
                        expert_i_num = expert_cfg["num"]
                        expert_skips = expert_cfg["skips"]
                        expert_init_factor = expert_cfg["init_factor"]
                        fcs.append(SingleExpert(model_dim=expert_in_ch, layer_num=expert_i_num, skips=expert_skips, 
                        init_factor=expert_init_factor, init_trunc_normal=expert_init_trunc_normal))
                    experts["net"] = fcs

                parallel_env = args.parallel_env
                dist_rank = parallel_env.global_rank

                self.layers[i_tag] = moe_layer(
                    gate_type=gate_type, 
                    model_dim=in_ch,
                    experts=experts,
                    scan_expert_func=None if args.no_expert_parallel else self.scan_expert_func,
                    result_func=None,
                    seeds=(1, dist_rank + 1, 1),
                    a2a_ffn_overlap_degree=1,
                    parallel_type='auto',
                    pad_samples=False,
                    moe_no_batch=False,
                    group=args.single_data_group,
                    return_gates=args.moe_return_gates,
                    return_gate_logits=args.moe_return_gate_logits
                )
            elif i_cfg["type"] == "normmlp":
                self.layers[i_tag] = NormMlp(in_features=in_ch, hidden_features=h_ch, out_features=out_ch, 
                    layer_num=i_num, skips=i_cfg.get("skips", None), norm_name=i_cfg.get("norm_name", "none"))
                if "requires_grad" in i_cfg:
                    for tmp_fc in self.layers[i_tag].fcs:
                        tmp_fc.weight.requires_grad = i_cfg["requires_grad"]
                        if hasattr(tmp_fc, "bias"):
                            tmp_fc.bias.requires_grad = i_cfg["requires_grad"]
            elif i_cfg["type"] == "layernorm":
                self.layers[i_tag] = nn.LayerNorm(in_ch)
            elif i_cfg["type"] == "batchnorm":
                self.layers[i_tag] = nn.BatchNorm1d(in_ch)
            elif i_cfg["type"] == "groupnorm":
                group_num = i_cfg["group_num"]
                self.layers[i_tag] = nn.GroupNorm(group_num, in_ch)
            elif i_cfg["type"] == "dropout":
                self.layers[i_tag] = nn.Dropout(i_cfg["prob"])

    
    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)

    def set_no_batch(self, mode=True):
        for net in self.modules():
            if type(net) == MOELayer:
                net.moe_no_batch = mode

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        expected = self.xyz_dim \
                   + (0 if (sigma_only or self.embedding_dir is None) else 3) \
                   + (0 if (sigma_only or self.embedding_a is None) else 1)

        if x.shape[1] != expected:
            raise Exception(
                'Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        # xyz mapping
        xyz_fc = self.layers["xyz"]
        h = xyz_fc(input_xyz)
        # if not self.args.no_feature_mapping_relu:
        xyz_cfg = self.layer_cfg["layers"]["xyz"]
        if "act" in xyz_cfg:
            if xyz_cfg["act"] == "relu":
                h = F.relu(h)
            elif xyz_cfg["act"] == "none":
                pass
            else:
                raise NotImplementedError
        
        xyz_h = h

        if self.args.use_moe_external_gate:
            moe_external_gate_fc = self.layers["moe_external_gate"]
            moe_external_gate_h = moe_external_gate_fc(xyz_h)
            moe_external_gate_cfg = self.layer_cfg["layers"]["moe_external_gate"]

            if "act" in moe_external_gate_cfg:
                if moe_external_gate_cfg["act"] == "relu":
                    moe_external_gate_h = F.relu(moe_external_gate_h)
                elif moe_external_gate_cfg["act"] == "none":
                    pass
                else:
                    raise NotImplementedError
        
        moe_loss = []
        moe_gates = []

        for i in self.layer_tags_main:
            i_tag = str(i)
            i_cfg = self.layer_cfg["layers"][i_tag]

            fc = self.layers[i_tag]
            if i_cfg["type"] == "moe":
                if self.args.use_moe_external_gate:
                    gate_input = moe_external_gate_h
                    if self.args.use_gate_input_norm:
                        gate_input_norm = self.layers["gate_input_norm"]
                        gate_input = gate_input_norm(gate_input)
                    
                    h = fc(h, gate_input=gate_input)
                else:
                    h = fc(h)
                l_aux = h.l_aux
                moe_loss.append(l_aux)
                if self.args.moe_return_gates:
                    moe_gates.append(h.gate_extras["gates"])
            else:
                h = fc(h)
                
            if "act" in i_cfg:
                if i_cfg["act"] == "relu":
                    h = F.relu(h)
                elif i_cfg["act"] == "none":
                    pass
                else:
                    raise NotImplementedError
                      
            if i_tag == self.sigma_tag:
                sigma = h
                fc = self.layers["sigma"]
                
                if self.args.amp_use_bfloat16:
                    sigma = fc(sigma)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        sigma = fc(sigma.float())

                if self.pos_dir_dim <= 0:
                    rgb, sigma = torch.tensor_split(sigma, [3], dim=-1)

                    rgb = self.rgb_activation(rgb) if self.rgb_activation is not None else rgb
                    
                    if sigma_noise is not None:
                        sigma += sigma_noise
                    sigma = self.sigma_activation(sigma)

                    outputs = torch.cat([rgb, sigma], -1)
                    break
                else:
                    if sigma_noise is not None:
                        sigma += sigma_noise
                    sigma = self.sigma_activation(sigma)
            
            # add direction
            if (i_tag == self.dir_tag) and (self.pos_dir_dim > 0):
                dir_a_encoding_input = []
                dir_a_encoding_input.append(h)
                if self.embedding_dir is not None:
                    # dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))
                    dir_a_encoding_input.append(self.embedding_dir(x[:, self.xyz_dim:self.xyz_dim + 3]))

                if self.embedding_a is not None and self.affine is None:
                    dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
                
                h = torch.cat(dir_a_encoding_input, -1)
            
            if i_tag == self.color_tag:
                rgb = h
                fc = self.layers["color"]
                rgb = fc(rgb)

                if self.affine is not None and self.embedding_a is not None:
                    affine_transform = self.affine(self.embedding_a(x[:, -1].long())).view(-1, 3, 4)
                    rgb = (affine_transform[:, :, :3] @ rgb.unsqueeze(-1) + affine_transform[:, :, 3:]).squeeze(-1)

                rgb = self.rgb_activation(rgb) if self.rgb_activation is not None else rgb
                outputs = torch.cat([rgb, sigma], -1)
                break

        extras = {}
        if self.args.moe_return_gates:
            extras["moe_gates"] = moe_gates
        
        if len(moe_loss) != 0:
            moe_loss = torch.stack(moe_loss)
            extras["moe_loss"] = moe_loss
        
        return {
            "outputs": outputs,
            "extras": extras
        }


class MipNeRFMoE(nn.Module):
    def __init__(self, args, pos_xyz_dim: int, pos_dir_dim: int,
                 appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int,
                 sigma_activation: nn.Module):
        super().__init__()
        self.args = args
        self.layer_cfg = args.layer_cfg
        self.layer_num_main = self.layer_cfg["layer_num_main"]
        self.sigma_tag = str(self.layer_cfg["sigma_tag"])
        # self.affine_tag = str(self.layer_cfg["affine_tag"])
        self.dir_tag = str(self.layer_cfg["dir_tag"])
        self.color_tag = str(self.layer_cfg["color_tag"])
        # self.skip_tag = [str(i) for i in self.layer_cfg["skip_tag"]]

        # if args.no_expert_parallel:
        #     from tutel import net
        #     self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        # else:
        #     self.single_data_group = None

        # self.xyz_ch = self.layer_cfg["xyz_ch"]
        if rgb_dim > 3:
            assert pos_dir_dim == 0
        # self.dir_ch = self.layer_cfg["dir_ch"]

        self.xyz_dim = xyz_dim
        self.embedding_xyz = MipEmbedder(pos_xyz_dim, input_dims=xyz_dim)
        self.in_channels_xyz = self.embedding_xyz.out_dim

        self.layers = nn.ModuleDict()

        self.layer_tags_main = [i for i in range(self.layer_num_main)]
        self.layer_tags_all = self.layer_tags_main + ["xyz", "sigma"]
        if pos_dir_dim > 0:
            self.layer_tags_all += ["color"]
        
        self.scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True)

        if pos_dir_dim > 0:
            self.embedding_dir = Embedding(pos_dir_dim)
            self.in_channels_dir = 3 + 3 * pos_dir_dim * 2
        else:
            self.embedding_dir = None
            self.in_channels_dir = 0
        
        if appearance_dim > 0:
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None

        if affine_appearance:
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None

        if pos_dir_dim > 0:
            # self.layer_tags_all += ["dir"]
            self.dir_ch = self.in_channels_dir + (appearance_dim if not affine_appearance else 0)

        self.sigma_activation = sigma_activation
        
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()
        else:
            self.rgb_activation = None  # We're using spherical harmonics and will convert to sigmoid in rendering.py

        self.pos_dir_dim = pos_dir_dim
        self._ddp_params_and_buffers_to_ignore = []
        
        if self.args.use_moe_external_gate:
            self.layer_tags_all += ["moe_external_gate"]
        if self.args.use_gate_input_norm:
            self.layer_tags_all += ["gate_input_norm"]

        # xyz encoding layers
        for i in self.layer_tags_all:
            i_tag = str(i)
            i_cfg = self.layer_cfg["layers"][i_tag]
            in_ch = i_cfg["in_ch"]
            h_ch = i_cfg["h_ch"]
            out_ch = i_cfg["out_ch"]
            if i == "sigma":
                if pos_dir_dim > 0:
                    assert out_ch == 1
                else:
                    assert out_ch == 4

            if i == "color":
                assert out_ch == 3
            
            # if i == "dir":
            #     assert in_ch == self.dir_ch
            
            i_num = i_cfg["num"]

            if i_cfg["type"] == "mlp":
                self.layers[i_tag] = Mlp(in_features=in_ch, hidden_features=h_ch, out_features=out_ch, layer_num=i_num, skips=i_cfg.get("skips", None))
                if "requires_grad" in i_cfg:
                    for tmp_fc in self.layers[i_tag].fcs:
                        tmp_fc.weight.requires_grad = i_cfg["requires_grad"]
                        if hasattr(tmp_fc, "bias"):
                            tmp_fc.bias.requires_grad = i_cfg["requires_grad"]
            elif i_cfg["type"] == "moe":
                assert in_ch == out_ch
                if i_num > 1:
                    assert h_ch == in_ch
                
                gate_type = {'type': i_cfg["gate_type"], 'k': i_cfg["k"], 
                    'fp32_gate': i_cfg["fp32_gate"], 
                    "capacity_factor": args.moe_capacity_factor,
                    "batch_prioritized_routing": args.batch_prioritized_routing,
                    "gate_noise": args.gate_noise,
                    "compute_balance_loss": args.compute_balance_loss,
                    "dispatcher_no_score": args.dispatcher_no_score,
                    "is_postscore": not args.dispatcher_no_postscore}

                if 'gate_dim' in i_cfg:
                    gate_type["gate_dim"] = i_cfg["gate_dim"]

                # gate_type['record_expert_hist'] = i_cfg["record_expert_hist"]
                # gate_type['return_topk_indices'] = i_cfg["return_topk_indices"]

                if hasattr(args, "moe_expert_type"):
                    expert_type = args.moe_expert_type
                else:
                    expert_type = 'expertmlp'

                local_expert_num = i_cfg.get("local_expert_num", None)
                if local_expert_num is None:
                    local_expert_num = args.moe_local_expert_num
                experts = {
                    'type': expert_type, 
                    'count_per_node': local_expert_num,
                    'hidden_size_per_expert': in_ch
                }

                if "init_trunc_normal" in i_cfg:
                    experts['init_trunc_normal'] = i_cfg["init_trunc_normal"]
                else:
                    experts['init_trunc_normal'] = False
                
                if expert_type == 'expertmlp':
                    experts['layer_num'] = i_num
                    experts['skips'] = i_cfg["skips"]
                    experts['init_factor'] = i_cfg["init_factor"]
                
                if expert_type == 'seqexperts':
                    fcs = nn.ModuleList()
                    for j in range(local_expert_num):
                        init_trunc_normal = False
                        if "init_trunc_normal" in i_cfg:
                            init_trunc_normal=i_cfg["init_trunc_normal"]
                        fcs.append(SingleExpert(model_dim=in_ch, layer_num=i_num, skips=i_cfg["skips"], 
                        init_factor=i_cfg["init_factor"], init_trunc_normal=init_trunc_normal))
                    experts["net"] = fcs
                
                if expert_type == 'multiseqexperts':
                    fcs = nn.ModuleList()
                    for expert_id in range(local_expert_num):
                        expert_cfg = i_cfg[str(expert_id)]
                        expert_init_trunc_normal = False
                        if "init_trunc_normal" in expert_cfg:
                            expert_init_trunc_normal=expert_cfg["init_trunc_normal"]
                        expert_in_ch = expert_cfg["in_ch"]
                        expert_i_num = expert_cfg["num"]
                        expert_skips = expert_cfg["skips"]
                        expert_init_factor = expert_cfg["init_factor"]
                        fcs.append(SingleExpert(model_dim=expert_in_ch, layer_num=expert_i_num, skips=expert_skips, 
                        init_factor=expert_init_factor, init_trunc_normal=expert_init_trunc_normal))
                    experts["net"] = fcs

                parallel_env = args.parallel_env
                dist_rank = parallel_env.global_rank

                self.layers[i_tag] = moe_layer(
                    gate_type=gate_type, 
                    model_dim=in_ch,
                    experts=experts,
                    scan_expert_func=None if args.no_expert_parallel else self.scan_expert_func,
                    result_func=None,
                    seeds=(1, dist_rank + 1, 1),
                    a2a_ffn_overlap_degree=1,
                    parallel_type='auto',
                    pad_samples=False,
                    moe_no_batch=False,
                    group=args.single_data_group,
                    return_gates=args.moe_return_gates,
                    return_gate_logits=args.moe_return_gate_logits
                )
            elif i_cfg["type"] == "normmlp":
                self.layers[i_tag] = NormMlp(in_features=in_ch, hidden_features=h_ch, out_features=out_ch, 
                    layer_num=i_num, skips=i_cfg.get("skips", None), norm_name=i_cfg.get("norm_name", "none"))
                if "requires_grad" in i_cfg:
                    for tmp_fc in self.layers[i_tag].fcs:
                        tmp_fc.weight.requires_grad = i_cfg["requires_grad"]
                        if hasattr(tmp_fc, "bias"):
                            tmp_fc.bias.requires_grad = i_cfg["requires_grad"]
            elif i_cfg["type"] == "layernorm":
                self.layers[i_tag] = nn.LayerNorm(in_ch)
            elif i_cfg["type"] == "batchnorm":
                self.layers[i_tag] = nn.BatchNorm1d(in_ch)
            elif i_cfg["type"] == "groupnorm":
                group_num = i_cfg["group_num"]
                self.layers[i_tag] = nn.GroupNorm(group_num, in_ch)
            elif i_cfg["type"] == "dropout":
                self.layers[i_tag] = nn.Dropout(i_cfg["prob"])

    
    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)

    def set_no_batch(self, mode=True):
        for net in self.modules():
            if type(net) == MOELayer:
                net.moe_no_batch = mode  

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        expected = self.xyz_dim * 2 \
                   + (0 if (sigma_only or self.embedding_dir is None) else 3) \
                   + (0 if (sigma_only or self.embedding_a is None) else 1)

        if x.shape[1] != expected:
            raise Exception(
                'Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim * 2])
        # xyz mapping
        xyz_fc = self.layers["xyz"]
        h = xyz_fc(input_xyz)
        # if not self.args.no_feature_mapping_relu:
        xyz_cfg = self.layer_cfg["layers"]["xyz"]
        if "act" in xyz_cfg:
            if xyz_cfg["act"] == "relu":
                h = F.relu(h)
            elif xyz_cfg["act"] == "none":
                pass
            else:
                raise NotImplementedError
        
        xyz_h = h

        if self.args.use_moe_external_gate:
            moe_external_gate_fc = self.layers["moe_external_gate"]
            moe_external_gate_h = moe_external_gate_fc(xyz_h)
            moe_external_gate_cfg = self.layer_cfg["layers"]["moe_external_gate"]

            if "act" in moe_external_gate_cfg:
                if moe_external_gate_cfg["act"] == "relu":
                    moe_external_gate_h = F.relu(moe_external_gate_h)
                elif moe_external_gate_cfg["act"] == "none":
                    pass
                else:
                    raise NotImplementedError
        
        moe_loss = []
        moe_gates = []

        for i in self.layer_tags_main:
            i_tag = str(i)
            i_cfg = self.layer_cfg["layers"][i_tag]

            fc = self.layers[i_tag]
            if i_cfg["type"] == "moe":
                if self.args.use_moe_external_gate:
                    gate_input = moe_external_gate_h
                    if self.args.use_gate_input_norm:
                        gate_input_norm = self.layers["gate_input_norm"]
                        gate_input = gate_input_norm(gate_input)
                    
                    h = fc(h, gate_input=gate_input)
                else:
                    h = fc(h)
                l_aux = h.l_aux
                moe_loss.append(l_aux)
                if self.args.moe_return_gates:
                    moe_gates.append(h.gate_extras["gates"])
            else:
                h = fc(h)

            if "act" in i_cfg:
                if i_cfg["act"] == "relu":
                    h = F.relu(h)
                elif i_cfg["act"] == "none":
                    pass
                else:
                    raise NotImplementedError
                      
            if i_tag == self.sigma_tag:
                sigma = h
                fc = self.layers["sigma"]
                
                if self.args.amp_use_bfloat16:
                    sigma = fc(sigma)
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        sigma = fc(sigma.float())

                if self.pos_dir_dim <= 0:
                    rgb, sigma = torch.tensor_split(sigma, [3], dim=-1)

                    rgb = self.rgb_activation(rgb) if self.rgb_activation is not None else rgb
                    
                    if sigma_noise is not None:
                        sigma += sigma_noise
                    sigma = self.sigma_activation(sigma)

                    outputs = torch.cat([rgb, sigma], -1)
                    break
                else:
                    if sigma_noise is not None:
                        sigma += sigma_noise
                    sigma = self.sigma_activation(sigma)
            
            # add direction
            if (i_tag == self.dir_tag) and (self.pos_dir_dim > 0):
                dir_a_encoding_input = []
                dir_a_encoding_input.append(h)
                if self.embedding_dir is not None:
                    # dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))
                    dir_a_encoding_input.append(self.embedding_dir(x[:, self.xyz_dim * 2:self.xyz_dim * 2 + 3]))

                if self.embedding_a is not None and self.affine is None:
                    dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
                
                h = torch.cat(dir_a_encoding_input, -1)
            
            if i_tag == self.color_tag:
                rgb = h
                fc = self.layers["color"]
                rgb = fc(rgb)

                if self.affine is not None and self.embedding_a is not None:
                    affine_transform = self.affine(self.embedding_a(x[:, -1].long())).view(-1, 3, 4)
                    rgb = (affine_transform[:, :, :3] @ rgb.unsqueeze(-1) + affine_transform[:, :, 3:]).squeeze(-1)

                rgb = self.rgb_activation(rgb) if self.rgb_activation is not None else rgb
                outputs = torch.cat([rgb, sigma], -1)
                break

        extras = {}
        if self.args.moe_return_gates:
            extras["moe_gates"] = moe_gates
        
        if len(moe_loss) != 0:
            moe_loss = torch.stack(moe_loss)
            extras["moe_loss"] = moe_loss
        
        return {
            "outputs": outputs,
            "extras": extras
        }
  


class NeRFMoETorch(nn.Module):
    def __init__(self, args, pos_xyz_dim: int, pos_dir_dim: int,
                 appearance_dim: int, affine_appearance: bool, appearance_count: int, rgb_dim: int, xyz_dim: int,
                 sigma_activation: nn.Module):
        super().__init__()
        if rgb_dim > 3:
            assert pos_dir_dim == 0

        self.xyz_dim = xyz_dim
        self.embedding_xyz = Embedding(pos_xyz_dim)
        self.in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2
        
        if pos_dir_dim > 0:
            self.embedding_dir = Embedding(pos_dir_dim)
            self.in_channels_dir = 3 + 3 * pos_dir_dim * 2
        else:
            self.embedding_dir = None
            self.in_channels_dir = 0
        
        if appearance_dim > 0:
            self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        else:
            self.embedding_a = None

        if affine_appearance:
            assert appearance_dim > 0
            self.affine = nn.Linear(appearance_dim, 12)
        else:
            self.affine = None

        if pos_dir_dim > 0:
            # self.layer_tags_all += ["dir"]
            self.dir_ch = self.in_channels_dir + (appearance_dim if not affine_appearance else 0)

        self.sigma_activation = sigma_activation
        
        if rgb_dim == 3:
            self.rgb_activation = nn.Sigmoid()
        else:
            self.rgb_activation = None  # We're using spherical harmonics and will convert to sigmoid in rendering.py

        self.pos_dir_dim = pos_dir_dim

        self.layers = nn.ModuleDict()
        
        # xyz
        # self.layer_xyz = Mlp_torch(in_features=75, hidden_features=0, out_features=256, layer_num=1, skips=None)
        self.layers["xyz"] = Mlp_torch(in_features=75, hidden_features=0, out_features=256, layer_num=1, skips=None)
        
        # 0
        in_ch = 256
        out_ch = 256
        i_num = 7
        skips = [3]
        init_factor = 1.0
        gate_type = {'type': "top", 'k': 1, 
            'fp32_gate': True, 
            "capacity_factor": args.moe_capacity_factor,
            "batch_prioritized_routing": args.batch_prioritized_routing,
            "gate_noise": args.gate_noise,
            "use_load_importance_loss": args.use_load_importance_loss,
            "compute_balance_loss": args.compute_balance_loss,
            "dispatcher_no_score": args.dispatcher_no_score,
            "is_postscore": not args.dispatcher_no_postscore}

        gate_type["gate_dim"] = 256

        if hasattr(args, "moe_expert_type"):
            expert_type = args.moe_expert_type
        else:
            expert_type = 'expertmlp'

        local_expert_num = args.moe_local_expert_num
        experts = {
            'type': expert_type, 
            'count_per_node': local_expert_num,
            'hidden_size_per_expert': in_ch
        }

        experts['init_trunc_normal'] = False
        
        if expert_type == 'expertmlp':
            experts['layer_num'] = i_num
            experts['skips'] = skips
            experts['init_factor'] = init_factor
        
        if expert_type == 'seqexperts':
            fcs = nn.ModuleList()
            for j in range(local_expert_num):
                init_trunc_normal = False
                fcs.append(SingleExpert_torch(model_dim=in_ch, layer_num=i_num, skips=skips, 
                init_factor=init_factor, init_trunc_normal=init_trunc_normal))
            experts["net"] = fcs
        
        dist_rank = 0

        # self.layer_0 = MOELayer_torch(
        self.layers["0"] = MOELayer_torch(
            gate_type=gate_type, 
            model_dim=in_ch,
            experts=experts,
            scan_expert_func=None,
            result_func=None,
            seeds=(1, dist_rank + 1, 1),
            a2a_ffn_overlap_degree=1,
            parallel_type='auto',
            pad_samples=False,
            moe_no_batch=False,
            group=args.single_data_group,
            use_residual=args.moe_use_residual,
            return_gates=args.moe_return_gates,
            return_gate_logits=args.moe_return_gate_logits
        )

        # 1
        # self.layer_1 = Mlp_torch(in_features=256, hidden_features=0, out_features=256, layer_num=1, skips=None)
        self.layers["1"] = Mlp_torch(in_features=256, hidden_features=0, out_features=256, layer_num=1, skips=None)
        # 2
        self.layers["2"] = Mlp_torch(in_features=331, hidden_features=0, out_features=128, layer_num=1, skips=None)
        # sigma
        self.layers["sigma"] = Mlp_torch(in_features=256, hidden_features=0, out_features=1, layer_num=1, skips=None)
        # color
        self.layers["color"] = Mlp_torch(in_features=128, hidden_features=0, out_features=3, layer_num=1, skips=None)
        # moe_external_gate
        self.layers["moe_external_gate"] = Mlp_torch(in_features=256, hidden_features=256, out_features=256, layer_num=2, skips=None)
        # gate_input_norm
        self.layers["gate_input_norm"] = nn.LayerNorm(256)

    def set_no_batch(self, mode=True):
        for net in self.modules():
            if type(net) == MOELayer_torch:
                net.moe_no_batch = mode

    def forward(self, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        expected = self.xyz_dim \
                   + (0 if (sigma_only or self.embedding_dir is None) else 3) \
                   + (0 if (sigma_only or self.embedding_a is None) else 1)

        if x.shape[1] != expected:
            raise Exception(
                'Unexpected input shape: {} (expected: {}, xyz_dim: {})'.format(x.shape, expected, self.xyz_dim))

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        # xyz mapping
        h = self.layers["xyz"](input_xyz)        
        xyz_h = h

        moe_external_gate_h = self.layers["moe_external_gate"](xyz_h)

        # 0
        gate_input = moe_external_gate_h
        gate_input = self.layers["gate_input_norm"](gate_input)
        h = self.layers["0"](h, gate_input=gate_input)
        h = F.relu(h)

        # sigma
        sigma = h
        sigma = self.layers["sigma"](sigma)
        if sigma_noise is not None:
            sigma += sigma_noise
        sigma = self.sigma_activation(sigma)
        if sigma_only:
            return sigma

        # 1
        h = self.layers["1"](h)

        # 2
        dir_a_encoding_input = []
        dir_a_encoding_input.append(h)
        if self.embedding_dir is not None:
            # dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))
            dir_a_encoding_input.append(self.embedding_dir(x[:, self.xyz_dim:self.xyz_dim + 3]))

        if self.embedding_a is not None and self.affine is None:
            dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
        h = torch.cat(dir_a_encoding_input, -1)

        h = self.layers["2"](h)
        h = F.relu(h)

        # color
        rgb = h
        rgb = self.layers["color"](rgb)
        rgb = self.rgb_activation(rgb) if self.rgb_activation is not None else rgb
        outputs = torch.cat([rgb, sigma], -1)
        
        return outputs

def get_nerf_moe_inner(hparams, appearance_count: int, xyz_dim: int, model_cfg_name="model") -> nn.Module:
    rgb_dim = 3 * ((hparams.sh_deg + 1) ** 2) if hparams.sh_deg is not None else 3

    model_cfg = getattr(hparams, model_cfg_name)
    layer_cfg = {}

    nerfmoe_class_name = getattr(hparams, "nerfmoe_class_name", "NeRFMoE")
    layer_cfg["layer_num_main"] = model_cfg["layer_num_main"]
    layer_cfg["sigma_tag"] = model_cfg["sigma_tag"]
    layer_cfg["dir_tag"] = model_cfg["dir_tag"]
    layer_cfg["color_tag"] = model_cfg["color_tag"]

    layer_cfg["layers"] = model_cfg["layers"]
    hparams.layer_cfg = layer_cfg

    if nerfmoe_class_name == "MipNeRFMoE":
        nerf_class = MipNeRFMoE
    elif nerfmoe_class_name == "NeRFMoETorch":
        nerf_class = NeRFMoETorch
    else:
        nerf_class = NeRFMoE
    
    sigma_activation = ShiftedSoftplus() if hparams.shifted_softplus else nn.ReLU()

    nerf_model = nerf_class(args=hparams,
        pos_xyz_dim=hparams.pos_xyz_dim,
        pos_dir_dim=hparams.pos_dir_dim,
        appearance_dim=hparams.appearance_dim,
        affine_appearance=hparams.affine_appearance,
        appearance_count=appearance_count,
        rgb_dim=rgb_dim, xyz_dim=xyz_dim,
        sigma_activation=sigma_activation)

    for name, param in nerf_model.named_parameters():
        if hasattr(param, 'skip_allreduce'):
            nerf_model.add_param_to_skip_allreduce(name)
    
    return nerf_model
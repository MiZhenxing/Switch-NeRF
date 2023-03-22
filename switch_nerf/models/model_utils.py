from argparse import Namespace

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from switch_nerf.models.cascade import Cascade
from switch_nerf.models.mega_nerf import MegaNeRF
from switch_nerf.models.nerf import NeRF, ShiftedSoftplus
from switch_nerf.models.nerf_moe import get_nerf_moe_inner

def convert_to_seqexperts(state_dict):
    keys = list(state_dict.keys())
    for key in keys:
        if "layers.0.experts.0." in key:
            if "weight" in key or "bias" in key:
                para_type = "weight" if "weight" in key else "bias"
                layer_id = int(key[-1])
                v = state_dict.pop(key)
                v = torch.unbind(v, dim=0)
                for expert_id, expert_v in enumerate(v):
                    new_key = f'module.layers.0.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                    if para_type == "weight":
                        new_v = expert_v.t().contiguous()
                    if para_type == "bias":
                        new_v = expert_v.squeeze(0)
                    state_dict[new_key] = new_v
    return state_dict

def convert_to_seqexperts1(state_dict, moe_layer_num):
    keys = list(state_dict.keys())
    for key in keys:
        for moe_layer_id in range(moe_layer_num):
            if f"layers.{moe_layer_id}.experts.0." in key:
                if "weight" in key or "bias" in key:
                    para_type = "weight" if "weight" in key else "bias"
                    layer_id = int(key[-1])
                    v = state_dict.pop(key)
                    v = torch.unbind(v, dim=0)
                    for expert_id, expert_v in enumerate(v):
                        new_key = f'module.layers.{moe_layer_id}.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                        if para_type == "weight":
                            new_v = expert_v.t().contiguous()
                        if para_type == "bias":
                            new_v = expert_v.squeeze(0)
                        state_dict[new_key] = new_v
    return state_dict


def convert_to_seqexperts2(state_dict, moe_layer_ids):
    keys = list(state_dict.keys())
    for key in keys:
        for moe_layer_id in moe_layer_ids:
            if f"layers.{moe_layer_id}.experts.0." in key:
                if "weight" in key or "bias" in key:
                    para_type = "weight" if "weight" in key else "bias"
                    layer_id = int(key[-1])
                    v = state_dict.pop(key)
                    v = torch.unbind(v, dim=0)
                    for expert_id, expert_v in enumerate(v):
                        new_key = f'module.layers.{moe_layer_id}.experts.0.experts.{expert_id}.layers.{layer_id}.{para_type}'
                        if para_type == "weight":
                            new_v = expert_v.t().contiguous()
                        if para_type == "bias":
                            new_v = expert_v.squeeze(0)
                        state_dict[new_key] = new_v
    return state_dict

def get_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.layer_dim, 3, 'model_state_dict')


def get_bg_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    if hparams.bg_use_cfg:
        tmp_use_moe = hparams.use_moe
        hparams.use_moe = hparams.bg_use_moe
        bg_nerf = _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict')
        hparams.use_moe = tmp_use_moe
    else:
        tmp_use_moe = hparams.use_moe
        hparams.use_moe = False
        bg_nerf = _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict')
        hparams.use_moe = tmp_use_moe
    return bg_nerf


def _get_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int,
                    weight_key: str) -> nn.Module:
    if hparams.container_path is not None:
        container = torch.jit.load(hparams.container_path, map_location='cpu')
        if xyz_dim == 3:
            return MegaNeRF([getattr(container, 'sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, False, container.cluster_2d)
        else:
            return MegaNeRF([getattr(container, 'bg_sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, True, container.cluster_2d)
    elif hparams.use_cascade:
        if hparams.use_moe:
            if weight_key == "model_state_dict":
                model_cfg_name = "model"
            elif weight_key == "bg_model_state_dict":
                model_cfg_name = "model_bg"
            else:
                model_cfg_name = None
                raise NotImplementedError
            nerf = Cascade(
                get_nerf_moe_inner(hparams, appearance_count, 
                    xyz_dim, model_cfg_name=model_cfg_name),
                get_nerf_moe_inner(hparams, appearance_count, 
                    xyz_dim, model_cfg_name=model_cfg_name)
                    if hparams.fine_samples > 0 else None)
        else:
            nerf = Cascade(
                _get_single_nerf_inner(hparams, appearance_count,
                                    layer_dim if xyz_dim == 4 else layer_dim,
                                    xyz_dim),
                _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim) if hparams.fine_samples > 0 else None)
    elif hparams.train_mega_nerf is not None:
        centroid_metadata = torch.load(hparams.train_mega_nerf, map_location='cpu')
        centroids = centroid_metadata['centroids']
        nerf = MegaNeRF(
            [_get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim) for _ in
             range(len(centroids))], centroids, 1, xyz_dim == 4, centroid_metadata['cluster_2d'], True)
    elif hparams.use_moe:
        if weight_key == "model_state_dict":
            model_cfg_name = "model"
        elif weight_key == "bg_model_state_dict":
            model_cfg_name = "model_bg"
        else:
            model_cfg_name = None
            raise NotImplementedError
        nerf = get_nerf_moe_inner(hparams, appearance_count, xyz_dim, model_cfg_name=model_cfg_name)
    else:
        nerf = _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim)

    if hparams.ckpt_path is not None:
        state_dict = torch.load(hparams.ckpt_path, map_location='cpu')[weight_key]

        if hparams.expertmlp2seqexperts and hparams.use_moe:
            if getattr(hparams, "moe_layer_num", 1) > 1:
                state_dict = convert_to_seqexperts1(state_dict, hparams.moe_layer_num)
            elif getattr(hparams, "moe_layer_ids", None) is not None:
                state_dict = convert_to_seqexperts2(state_dict, hparams.moe_layer_ids)
            else:
                state_dict = convert_to_seqexperts(state_dict)

        consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')

        model_dict = nerf.state_dict()
        model_dict.update(state_dict)
        nerf.load_state_dict(model_dict)

    return nerf


def _get_single_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int) -> nn.Module:
    rgb_dim = 3 * ((hparams.sh_deg + 1) ** 2) if hparams.sh_deg is not None else 3

    return NeRF(hparams.pos_xyz_dim,
                hparams.pos_dir_dim,
                hparams.layers,
                hparams.skip_layers,
                layer_dim,
                hparams.appearance_dim,
                hparams.affine_appearance,
                appearance_count,
                rgb_dim,
                xyz_dim,
                ShiftedSoftplus() if hparams.shifted_softplus else nn.ReLU())

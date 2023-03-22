from argparse import Namespace
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from switch_nerf.models.mega_nerf import MegaNeRF
from switch_nerf.models.mega_nerf_container import MegaNeRFContainer
from switch_nerf.models.model_utils import get_nerf, get_bg_nerf
from switch_nerf.opts import get_opts_base


def _get_merge_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--centroid_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    return parser.parse_known_args()[0]


@torch.inference_mode()
def main(hparams: Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = Path(hparams.exp_name)
    exp_name.mkdir(parents=True, exist_ok=True)
    output = exp_name / hparams.output
    hparams.moe_local_expert_num = hparams.moe_expert_num
    hparams.single_data_group = None
    centroid_metadata = torch.load(hparams.centroid_path, map_location='cpu')
    centroids = centroid_metadata['centroids']

    loaded = torch.load(hparams.ckpt_path, map_location='cpu')
    consume_prefix_in_state_dict_if_present(loaded['model_state_dict'], prefix='module.')

    if hparams.appearance_dim > 0:
        appearance_count = len(loaded['model_state_dict']['embedding_a.weight'])
    else:
        appearance_count = 0

    sub_module = get_nerf(hparams, appearance_count)

    if 'bg_model_state_dict' in loaded:
        bg_sub_module = get_bg_nerf(hparams, appearance_count)
    
    container = MegaNeRFContainer([sub_module], [bg_sub_module] if 'bg_model_state_dict' in loaded else [], centroids,
        torch.IntTensor(centroid_metadata['grid_dim']),
        centroid_metadata['min_position'],
        centroid_metadata['max_position'],
        hparams.pos_dir_dim > 0,
        hparams.appearance_dim > 0,
        centroid_metadata['cluster_2d'])
    torch.jit.save(torch.jit.script(container.eval()), output)
    container = torch.jit.load(output, map_location='cpu')

    # Test container
    nerf = getattr(container, 'sub_module_{}'.format(0)).to(device)

    width = 3
    if hparams.pos_dir_dim > 0:
        width += 3
    if hparams.appearance_dim > 0:
        width += 1

    print('fg test eval: {}'.format(nerf(torch.ones(1, width, device=device))))
    sub_module = sub_module.to(device)
    print('fg sub_module test eval: {}'.format(sub_module(torch.ones(1, width, device=device))))

    if 'bg_model_state_dict' in loaded:
        bg_nerf = getattr(container, 'bg_sub_module_{}'.format(0)).to(device)

        width = 8
        print('bg test eval: {}'.format(bg_nerf(torch.ones(1, width, device=device))))
        bg_sub_module = bg_sub_module.to(device)
        print('bg bg_sub_module test eval: {}'.format(bg_nerf(torch.ones(1, width, device=device))))


if __name__ == '__main__':
    main(_get_merge_opts())

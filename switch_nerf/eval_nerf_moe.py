from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from switch_nerf.opts_nerf import get_opts_base
from switch_nerf.runner_nerf import Runner


def _get_eval_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None

    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).eval_nerf()
    else:
        Runner(hparams).eval_nerf()


if __name__ == '__main__':
    main(_get_eval_opts())

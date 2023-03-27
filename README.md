# Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-scale Neural Radiance Fields 

### [Openreview](https://openreview.net/forum?id=PQ2zoIZqvm) | [Project Page](https://mizhenxing.github.io/switchnerf) | [Checkpoints]() | [Visualizer]()

## Warning
This is only a beta release and certainly contains bugs. We release it in response to the requests of the community. We will continue improving the codes, writing detailed documents and add trained models. It will take about 1 week.

## Installation

The main dependencies are in the `requirements.txt`. We use [this version](https://github.com/microsoft/tutel/tree/56dbd664341cf6485c9fa292955f77d3ac918a65) of Tutel in for MoE layers. The Tutel has changed a lot so make sure to install the version of the correct commit. Please follow the instructions in Tutel to install it. We will also add detailed documentation about the Tutel installation.

## Dataset
We have performed experiments in the datasets from the Mega-NeRF, Block-NeRF and Bungee-NeRF.

### Mega-NeRF

Please follow the instructions in the code of [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf) to download and process the Mill 19 and UrbanScene 3D datasets.

### Block-NeRF

Please follow the website of [Block-NeRF](https://waymo.com/intl/zh-cn/research/block-nerf) to download the raw Mission Bay dataset.

## Configure

<!-- We use [yaml](https://yaml.org/) file to set options in our codes. Several key options are explained below. Other options are self-explanatory in the codes. Before running our codes, you may need to change the `true_gpu`, `data: root_dir` and `model_path` (only for testing). -->

<!-- * `output_dir` A relative or absolute folder path for writing logs, depthmaps. -->

## Training

### Mega-NeRF scenes
We provide the example commands to train the model on Building scene.

We should first generate data chunks. The `dataset_path` should be set to the scene folder processed above. The `exp_name` is used for logging results. If it does not exit, the program will make a new one. The `chunk_paths` is used to store the generate the data chunks. The chunks will be reused in later experiments.

Generate chunks.
```sh
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=1 -m switch_nerf.train --config=switch_nerf/configs/switch_nerf/building.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/building-pixsfm --chunk_paths=/your/absolute/chunk/path/building_chunk_factor_1_bg --generate_chunk
```

Train the model on the Building scene and the generated chunks.
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=8 -m switch_nerf.train --config=switch_nerf/configs/switch_nerf/building.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/building-pixsfm --chunk_paths=/your/absolute/chunk/path/building_chunk_factor_1_bg --use_balance_loss --i_print=1000 --batch_size=8192 --moe_expert_type=expertmlp --moe_train_batch --moe_test_batch --model_chunk_size=131072 --moe_capacity_factor=1.0 --batch_prioritized_routing --moe_l_aux_wt=0.0005 --amp_use_bfloat16 --use_moe_external_gate --use_gate_input_norm --use_sigma_noise --sigma_noise_std=1.0
```

### Block-NeRF scenes

We adapt a data interface mainly based on the [UnboundedNeRFPytorch](https://github.com/sjtuytc/UnboundedNeRFPytorch). We first generate data chunks from the raw `tf_records` in Block-NeRF dataset.

```sh
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=1 -m switch_nerf.train --config=switch_nerf/configs/switch_nerf/mission_bay.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/Mission_Bay/v1.0 --block_train_list_path=switch_nerf/datasets/lists/block_nerf_train_val.txt --block_image_hash_id_map_path=switch_nerf/datasets/lists/block_nerf_id_map.json --chunk_paths=/your/absolute/chunk/path/mission_bay_chunk_radii_1 --no_bg_nerf --near=0.01 --far=10.0 --generate_chunk
```

Then we train the model on the Mission Bay scene and the generated chunks. The `batch_size` is set according to the memory of RTX 3090.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=8 -m switch_nerf.train --config=switch_nerf/configs/switch_nerf/mission_bay.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/Mission_Bay/v1.0 --block_train_list_path=switch_nerf/datasets/lists/block_nerf_train_val.txt --block_image_hash_id_map_path=switch_nerf/datasets/lists/block_nerf_id_map.json --chunk_paths=/your/absolute/chunk/path/mission_bay_chunk_radii_1 --no_bg_nerf --near=0.01 --far=10.0 --use_balance_loss --i_print=1000 --batch_size=13312 --moe_expert_type=expertmlp --moe_train_batch --moe_test_batch --model_chunk_size=212992 --coarse_samples=257 --fine_samples=257 --moe_capacity_factor=1.0 --batch_prioritized_routing --moe_l_aux_wt=0.0005 --amp_use_bfloat16 --use_moe_external_gate --use_gate_input_norm --use_sigma_noise --sigma_noise_std=1.0
```


## Testing

We provide checkpoints in this [repository]().

Test on the Building scene in Mega-NeRF dataset.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=8 -m switch_nerf.eval_image --config=switch_nerf/configs/switch_nerf/building.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/building-pixsfm --i_print=1000 --moe_expert_type=seqexperts --model_chunk_size=131072 --ckpt_path=/your/absolute/ckpt/path/building.pt --expertmlp2seqexperts --use_moe_external_gate --use_gate_input_norm
```

Test on the the Mission Bay scene in Block-NeRF dataset.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=8 -m switch_nerf.eval_image_blocknerf --config=switch_nerf/configs/switch_nerf/mission_bay.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/Mission_Bay/v1.0 --block_val_list_path=switch_nerf/datasets/lists/block_nerf_val.txt --block_train_list_path=switch_nerf/datasets/lists/block_nerf_train_val.txt --block_image_hash_id_map_path=switch_nerf/datasets/lists/block_nerf_id_map.json --i_print=1000 --near=0.01 --far=10.0 --moe_expert_type=seqexperts --model_chunk_size=212992 --coarse_samples=513 --fine_samples=513 --ckpt_path=/your/absolute/ckpt/path/mission_bay.pt --expertmlp2seqexperts --use_moe_external_gate --use_gate_input_norm --set_timeout --image_pixel_batch_size=8192
```

You can also use less GPUs.

## Visualization

We provide a simple point cloud visualizer in this [repository](). You can use the commands below to create point clouds and visualize them with transparency. You can use [Meshlab](https://www.meshlab.net) to visualize the point clouds without transparency. Meshlab can also visualize the transparency with "Shading: Dot Decorator" selected but the visualization is not clear enough.

Generate point clouds:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use_env --master_port=12345 --nproc_per_node=8 -m switch_nerf.eval_points --config=switch_nerf/configs/switch_nerf/building.yaml --use_moe --exp_name=/your/absolute/experiment/path --dataset_path=/your/absolute/scene/path/building-pixsfm --i_print=1000 --moe_expert_type=seqexperts --model_chunk_size=131072 --ckpt_path=/your/absolute/ckpt/path/500000.pt --expertmlp2seqexperts --use_moe_external_gate --use_gate_input_norm --moe_return_gates --return_pts --return_pts_rgb --return_pts_alpha --render_test_points_sample_skip=4 --val_scale_factor=8 --render_test_points_image_num=20
```

Other scenes in Mega-NeRF use `--render_test_points_image_num=21`.

Merge point clouds from different validation images.

```sh
python -m switch_nerf.scripts.merge_points --data_path=/your/absolute/experiment/path/0/eval_points --merge_all --image_num=20 --model_type=switch -r=0.2
```

Other scenes in Mega-NeRF use `--image_num=21`. `-r` is used to randomly downsample point clouds by a ratio so that it can be visualized on our desktop.


## License

Our code is distributed under the MIT License. See `LICENSE` file for more information.

## Citation

```bibtex
@inproceedings{mi2023switchnerf,
  title={Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-scale Neural Radiance Fields},
  author={Zhenxing Mi and Dan Xu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=PQ2zoIZqvm}
}
```

## Contact

If you have any questions, please raise an issue or email to Zhenxing Mi (`zmiaa@connect.ust.hk`).

## Acknowledgments

Our code follows several awesome repositories. We appreciate them for making their codes available to public.

* [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf)
* [Tutel](https://github.com/xy-guo/MVSNet_pytorch)
* [UnboundedNeRFPytorch](https://github.com/sjtuytc/UnboundedNeRFPytorch)
* [xrnerf](https://github.com/openxrlab/xrnerf)
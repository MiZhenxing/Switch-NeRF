import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, cast
from xml.dom.expatbuilder import parseString

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from switch_nerf.datasets.filesystem_dataset import FilesystemDataset
from switch_nerf.datasets.memory_dataset import MemoryDataset
from switch_nerf.image_metadata import ImageMetadata
from switch_nerf.datasets.nerf_data.nerf_loader import NeRFDataset, NeRFDatasetTest, NeRFDatasetTrain, NeRFDatasetVal
from switch_nerf.opts_nerf import get_nerf_dataset_args
from switch_nerf.metrics import psnr, ssim, lpips, psnr_mask, ssim_mask
from switch_nerf.misc_utils import main_print, main_tqdm, main_log, count_parameters
from switch_nerf.models.model_utils import get_nerf, get_bg_nerf, model_scale_lecun_normal_init
from switch_nerf.ray_utils import get_rays, get_ray_directions
from switch_nerf.rendering import render_rays
from modules.tutel_moe_ext import tutel_system
from utils.logger import setup_logger
import subprocess
from utils.functions import DictAverageMeter, voc_palette, DictAverageMeter1
import time
from modules.tutel_moe_ext.tutel_moe_layer_nobatch import MOELayer

from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.optim import OSS

from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from contextlib import nullcontext
from plyfile import PlyData, PlyElement

from torch.utils.data import Subset
import json
from switch_nerf.rendering_mip import render_rays as render_rays_mip

from modules.tutel_moe_ext.tutel_fast_dispatch_nobatch import one_hot_with_dtype

class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.data_type == "block_nerf":
            self.block_init(hparams, set_experiment_path)
            return
        elif hparams.data_type == "nerf":
            self.init_nerf(hparams, set_experiment_path)
            return

        # setup tutel
        parallel_env = tutel_system.init_data_model_parallel(use_slurm=hparams.use_slurm)
        if hparams.use_slurm:
            os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

        if hparams.no_expert_parallel:
            from tutel import net
            self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        else:
            self.single_data_group = None

        hparams.single_data_group = self.single_data_group
        hparams.parallel_env = parallel_env
        hparams.local_rank = parallel_env.local_device.index
        hparams.dist_rank = parallel_env.global_rank
        dist_rank = hparams.dist_rank
        self.device = parallel_env.local_device
        hparams.device = self.device
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.set_default_dtype(torch.float32)

        self.is_master = (int(os.environ['RANK']) == 0)
        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        world_size = parallel_env.global_size
        if hparams.no_expert_parallel:
            hparams.moe_local_expert_num = hparams.moe_expert_num
        else:
            hparams.moe_local_expert_num = hparams.moe_expert_num // world_size
            assert hparams.moe_local_expert_num * world_size == hparams.moe_expert_num

        # dir
        self.hparams = hparams
        self.experiment_path = self._get_experiment_path() if self.is_master else None
        self.model_path = self.experiment_path / 'models' if self.is_master else None

        hparams.experiment_path = self.experiment_path

        self.train_items, self.val_items = self._get_image_metadata()
        self._setup_experiment_dir()

        # logger
        if self.is_master:
            log_dir = self.experiment_path
            hparams.logdir = log_dir
            if hparams.config_file is not None:
                shutil.copy(hparams.config_file, log_dir)
            self.logger = setup_logger(None, log_dir, timestamp=False) # root logger
        else:
            self.logger = None
        hparams.logger = self.logger
        
        try:
            git_commit_id = \
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
            git_branch_name = \
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]
        except:
            main_log("No git founded")
            git_commit_id = ""
            git_branch_name = ""

        self.hparams = hparams
        
        main_log("Branch " + git_branch_name)
        main_log("Commit ID " + git_commit_id)
        main_log(" ".join(sys.argv))
        main_log("Loaded configuration file {}".format(hparams.config_file))
        main_log("Running with config:\n{}".format(hparams))     

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            torch.cuda.manual_seed_all(hparams.random_seed)
            random.seed(hparams.random_seed)

        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
        self.origin_drb = torch.tensor(coordinate_info['origin_drb'])
        self.pose_scale_factor = coordinate_info['pose_scale_factor']
        main_log('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        self.near = hparams.near / self.pose_scale_factor

        if self.hparams.far is not None:
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf:
            self.far = 1e5
        else:
            self.far = 2

        main_log('Ray bounds: {}, {}'.format(self.near, self.far))

        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in
                                   hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
        main_log('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
        main_log('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

        if self.ray_altitude_range is not None:
            assert self.ray_altitude_range[0] < self.ray_altitude_range[1]

        if self.hparams.cluster_mask_path is not None:
            cluster_params = torch.load(Path(self.hparams.cluster_mask_path).parent / 'params.pt', map_location='cpu')
            assert cluster_params['near'] == self.near
            assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
            assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

            if self.ray_altitude_range is not None:
                assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                       torch.FloatTensor(self.ray_altitude_range))), \
                    '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

        # self.train_items, self.val_items = self._get_image_metadata()
        main_log('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_log('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                max_position * self.pose_scale_factor + self.origin_drb))

        main_log('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)
        self.model_parameter_num = count_parameters(self.nerf)

        if 'RANK' in os.environ:
            self.nerf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.nerf)
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                output_device=int(os.environ['LOCAL_RANK']),
                                                                find_unused_parameters=hparams.find_unused_parameters)

        if hparams.bg_nerf:
            self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
            self.model_parameter_num += count_parameters(self.bg_nerf)

            if 'RANK' in os.environ:
                self.bg_nerf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.bg_nerf)
                self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                        device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                        output_device=int(os.environ['LOCAL_RANK']),
                                                                        find_unused_parameters=hparams.find_unused_parameters)

            if hparams.ellipse_bounds:
                assert hparams.ray_altitude_range is not None

                if self.ray_altitude_range is not None:
                    ground_poses = camera_positions.clone()
                    ground_poses[:, 0] = self.ray_altitude_range[1]
                    air_poses = camera_positions.clone()
                    air_poses[:, 0] = self.ray_altitude_range[0]
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses])
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1]
                main_log('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                                 max_position))

                self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
                self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
                scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                    dim=-1).max()

                self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
            else:
                self.sphere_center = None
                self.sphere_radius = None

            main_log('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
        else:
            self.bg_nerf = None
            self.sphere_center = None
            self.sphere_radius = None
        
        main_log(f"Total parameters number is {self.model_parameter_num / 1024.0 / 1024.0:.4f} M")
    

    def block_init(self, hparams: Namespace, set_experiment_path: bool = True):
        # faulthandler.register(signal.SIGUSR1)

        # setup tutel
        if hparams.set_timeout:
            timeout = datetime.timedelta(days=1)
        else:
            timeout = None
        parallel_env = tutel_system.init_data_model_parallel(use_slurm=hparams.use_slurm, timeout=timeout)
        if hparams.use_slurm:
            os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

        if hparams.no_expert_parallel:
            from tutel import net
            self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        else:
            self.single_data_group = None

        hparams.single_data_group = self.single_data_group
        hparams.parallel_env = parallel_env
        hparams.local_rank = parallel_env.local_device.index
        hparams.dist_rank = parallel_env.global_rank
        dist_rank = hparams.dist_rank
        self.device = parallel_env.local_device
        hparams.device = self.device
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.set_default_dtype(torch.float32)

        self.is_master = (int(os.environ['RANK']) == 0)
        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        world_size = parallel_env.global_size
        if hparams.no_expert_parallel:
            hparams.moe_local_expert_num = hparams.moe_expert_num
        else:
            hparams.moe_local_expert_num = hparams.moe_expert_num // world_size
            assert hparams.moe_local_expert_num * world_size == hparams.moe_expert_num

        # dir
        self.hparams = hparams
        self.experiment_path = self._get_experiment_path() if self.is_master else None
        self.model_path = self.experiment_path / 'models' if self.is_master else None

        hparams.experiment_path = self.experiment_path

        with open(self.hparams.block_image_hash_id_map_path) as f:
            self.image_hash_id_map = json.load(f)

        self.train_items, self.val_items = [], []
        self._setup_experiment_dir()

        # logger
        if self.is_master:
            log_dir = self.experiment_path
            hparams.logdir = log_dir
            if hparams.config_file is not None:
                shutil.copy(hparams.config_file, log_dir)
            self.logger = setup_logger(None, log_dir, timestamp=False) # root logger
        else:
            self.logger = None
        hparams.logger = self.logger
        
        try:
            git_commit_id = \
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
            git_branch_name = \
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]
        except:
            main_log("No git founded")
            git_commit_id = ""
            git_branch_name = ""

        self.hparams = hparams
        
        main_log("Branch " + git_branch_name)
        main_log("Commit ID " + git_commit_id)
        main_log(" ".join(sys.argv))
        main_log("Loaded configuration file {}".format(hparams.config_file))
        main_log("Running with config:\n{}".format(hparams))     

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            torch.cuda.manual_seed_all(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.near = hparams.near
        self.far = hparams.far

        main_log('Ray bounds: {}, {}'.format(self.near, self.far))
        
        main_log('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        self.nerf = get_nerf(hparams, self.image_hash_id_map["image_num"]).to(self.device)

        if 'RANK' in os.environ:
            self.nerf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.nerf)
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                output_device=int(os.environ['LOCAL_RANK']),
                                                                find_unused_parameters=hparams.find_unused_parameters)

        assert not hparams.bg_nerf
        if hparams.bg_nerf:
            pass
        else:
            self.bg_nerf = None
            self.sphere_center = None
            self.sphere_radius = None


    def init_nerf(self, hparams: Namespace, set_experiment_path: bool = True):
        # setup tutel
        parallel_env = tutel_system.init_data_model_parallel(use_slurm=hparams.use_slurm)
        if hparams.use_slurm:
            os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        
        if hparams.no_expert_parallel:
            from tutel import net
            self.single_data_group = net.create_groups_from_world(group_count=1).data_group
        else:
            self.single_data_group = None
        
        hparams.single_data_group = self.single_data_group
        hparams.parallel_env = parallel_env
        hparams.local_rank = parallel_env.local_device.index
        hparams.dist_rank = parallel_env.global_rank
        dist_rank = hparams.dist_rank
        self.device = parallel_env.local_device
        hparams.device = self.device
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.set_default_dtype(torch.float32)

        if 'RANK' in os.environ:
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True
        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        world_size = parallel_env.global_size
        if hparams.no_expert_parallel:
            hparams.moe_local_expert_num = hparams.moe_expert_num
        else:
            hparams.moe_local_expert_num = hparams.moe_expert_num // world_size
            assert hparams.moe_local_expert_num * world_size == hparams.moe_expert_num

        # dir
        self.hparams = hparams
        self.experiment_path = self._get_experiment_path() if self.is_master else None
        self.model_path = self.experiment_path / 'models' if self.is_master else None

        hparams.experiment_path = self.experiment_path
        self._setup_experiment_dir_nerf()

        # logger
        if self.is_master:
            log_dir = self.experiment_path
            hparams.logdir = log_dir
            if hparams.config_file is not None:
                shutil.copy(hparams.config_file, log_dir)
            self.logger = setup_logger(None, log_dir, timestamp=False) # root logger
        else:
            self.logger = None
        hparams.logger = self.logger
        
        try:
            git_commit_id = \
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('UTF-8')[0:-1]
            git_branch_name = \
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('UTF-8')[0:-1]
        except:
            main_log("No git founded")
            git_commit_id = ""
            git_branch_name = ""

        self.hparams = hparams
        
        main_log("Branch " + git_branch_name)
        main_log("Commit ID " + git_commit_id)
        main_log(" ".join(sys.argv))
        main_log("Loaded configuration file {}".format(hparams.config_file))
        main_log("Running with config:\n{}".format(hparams))     

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            torch.cuda.manual_seed_all(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.nerf = get_nerf(hparams, 0).to(self.device)
        self.model_parameter_num = count_parameters(self.nerf)
        self.bg_nerf = None

        if 'RANK' in os.environ:
            self.nerf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.nerf)
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                output_device=int(os.environ['LOCAL_RANK']),
                                                                find_unused_parameters=hparams.find_unused_parameters)
        
        dataset_args = get_nerf_dataset_args(hparams=hparams)
        self.dataset = NeRFDataset(dataset_args)
        self.train_dataset = NeRFDatasetTrain(self.dataset)
        self.val_dataset = NeRFDatasetVal(self.dataset)
        self.test_dataset = NeRFDatasetTest(self.dataset)
        
        main_log(f"Total parameters number is {self.model_parameter_num / 1024.0 / 1024.0:.4f} M")


    def train(self):
        # self._setup_experiment_dir()
        
        if not self.hparams.moe_train_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']

            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        if self.hparams.no_optimizer_schedulers:
            pass
        else:
            for key, optimizer in optimizers.items():
                schedulers[key] = ExponentialLR(optimizer,
                                                gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
                                                last_epoch=train_iterations - 1)

        if self.hparams.dataset_type == 'filesystem':
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()
            dataset = FilesystemDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size, self.hparams.shuffle_chunk)

            
            if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device)
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        train_meter = DictAverageMeter1()
        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                main_log(f"Loading chunk {dataset.get_state()}")
                chunk_time = time.time()
                dataset.load_chunk()
                chunk_time = time.time() - chunk_time
                main_log(f"Chunk {dataset.get_state()} loaded by {chunk_time:.2f} s")

            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=self.hparams.data_loader_num_workers, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.data_loader_num_workers,
                                         pin_memory=True)

            data_sample_time = time.time()
            for dataset_index, item in enumerate(data_loader):
                data_sample_time = time.time() - data_sample_time
                if dataset_index <= discard_index:
                    data_sample_time = time.time()
                    continue

                discard_index = -1
                should_accumulate = ((train_iterations + 1) % self.hparams.accumulation_steps != 0)
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
                fwd_bwd_time = time.time()
                fwd_time = time.time()
                amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float16
                self.hparams.amp_dtype = amp_dtype
                
                if self.hparams.compute_memory:
                    torch.cuda.reset_peak_memory_stats()
                with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
                    if self.hparams.appearance_dim > 0:
                        image_indices = item['image_indices'].to(self.device, non_blocking=True)
                    else:
                        image_indices = None

                    if self.hparams.training_step_fn is not None:
                        training_step_fn = getattr(self, self.hparams.training_step_fn)
                    else:
                        training_step_fn = self._training_step
                    if self.hparams.training_step_fn == "_training_step_mip":
                        metrics, _ = training_step_fn(
                            item['rgbs'].to(self.device, non_blocking=True),
                            item['rays'].to(self.device, non_blocking=True),
                            item['radii'].to(self.device, non_blocking=True),
                            image_indices=image_indices)
                    else:
                        metrics, bg_nerf_rays_present = training_step_fn(
                            item['rgbs'].to(self.device, non_blocking=True),
                            item['rays'].to(self.device, non_blocking=True),
                            image_indices)

                    if self.hparams.disable_check_finite:
                        pass
                    else:
                        check_finite = torch.tensor(1, device=self.device)
                        with torch.no_grad():
                            for key, val in metrics.items():
                                if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                    continue
                                if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                                    continue

                                if not math.isfinite(val):
                                    # raise Exception('Train metrics not finite: {}'.format(metrics))
                                    check_finite -= 1
                                    main_log(f"{key} is infinite")
                                    break
                        
                        # dist.all_reduce(check_finite)
                        check_finite_gather = [torch.tensor(0, device=self.device) for _ in range(dist.get_world_size())]
                        dist.all_gather(check_finite_gather, check_finite)
                        check_finite_gather = torch.stack(check_finite_gather)
                        # check_finite = (check_finite.item() == dist.get_world_size())

                # for optimizer in optimizers.values():
                #     optimizer.zero_grad(set_to_none=True)

                all_loss = metrics['loss']
                if (self.hparams.use_moe or self.hparams.bg_use_moe) and self.hparams.use_balance_loss:

                    moe_l_aux_wt = self.hparams.moe_l_aux_wt
                    if "gate_loss" in metrics:
                        all_loss = all_loss + moe_l_aux_wt * metrics['gate_loss']

                    if self.hparams.bg_use_moe:
                        assert self.hparams.bg_use_cfg
                        all_loss = all_loss + moe_l_aux_wt * metrics['bg_gate_loss']

                all_loss = all_loss / self.hparams.accumulation_steps
                metrics['all_loss'] = all_loss

                if self.hparams.disable_check_finite:
                    pass
                else:
                    check_finite_flag = (check_finite_gather.sum().item() == dist.get_world_size())
                    if not check_finite_flag:
                        with self.nerf.no_sync():
                            with self.bg_nerf.no_sync() if self.bg_nerf is not None else nullcontext():
                                scaler.scale(all_loss).backward()
                        for optimizer in optimizers.values():
                            optimizer.zero_grad(set_to_none=True)
                        
                        main_log(f"skip step {train_iterations} due to inf")
                        main_log(f"check_finite of GPUs {check_finite_gather}")
                        continue

                fwd_time = time.time() - fwd_time

                with self.nerf.no_sync() if should_accumulate else nullcontext():
                    with self.bg_nerf.no_sync() if (should_accumulate and self.bg_nerf is not None) else nullcontext():
                        scaler.scale(all_loss).backward()

                if not should_accumulate:
                    for key, optimizer in optimizers.items():
                        if key == 'bg_nerf' and (not bg_nerf_rays_present):
                            continue
                        else:
                            scaler.step(optimizer)

                    scaler.update()
                    for optimizer in optimizers.values():
                        optimizer.zero_grad(set_to_none=True)

                for scheduler in schedulers.values():
                    scheduler.step()

                fwd_bwd_time = time.time() - fwd_bwd_time

                if self.hparams.compute_memory:
                    forward_backward_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                train_iterations += 1
                train_meter_dict = {}
                for k, v in metrics.items():
                    train_meter_dict[k] = v.item() if isinstance(v, torch.Tensor) and len(v.shape) == 0 else v

                train_meter.update(train_meter_dict)
                if self.is_master:
                    train_meter_mean = train_meter.mean()
                    pbar.set_postfix(psnr=f"{train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})")
                    pbar.update(1)
                    for key, value in metrics.items():
                        if not isinstance(value, torch.Tensor):
                            self.writer.add_scalar('train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        tmp_optimizers = optimizers
                        self._save_checkpoint(tmp_optimizers, scaler, train_iterations, dataset_index,
                                              dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
                    if train_iterations % self.hparams.i_print==0 or train_iterations == self.hparams.train_iterations:
                        train_meter_mean = train_meter.mean()
                        train_print_str = [
                            f"[TRAIN] Iter: {train_iterations} all_loss: {train_meter_dict['all_loss']:.5f} ({train_meter_mean['all_loss']:.5f})",
                            f"PSNR: {train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})",
                            f"img_loss: {train_meter_dict['loss']:.5f} ({train_meter_mean['loss']:.5f})",
                            # f"gate_loss: {self.hparams.moe_l_aux_wt * train_meter_dict['gate_loss']:.7f} ({self.hparams.moe_l_aux_wt * train_meter_mean['gate_loss']:.7f})",
                            # f"real_gate_loss: {train_meter_dict['gate_loss']:.7f} ({train_meter_mean['gate_loss']:.7f})",
                            f"lr: {optimizers['nerf'].param_groups[0]['lr']:.5f}",
                            f"moe_l_aux_wt: {moe_l_aux_wt:.5f}",
                            f"data time: {data_sample_time:.5f}",
                            f"fwd_bwd time: {fwd_bwd_time:.5f}",
                            f"fwd_time: {fwd_time:.5f}",
                            f"bwd_time: {fwd_bwd_time - fwd_time:.5f}",
                            f"fwd_bwd memory: {forward_backward_memory:.2f}" if self.hparams.compute_memory else ""
                        ]

                        if self.hparams.use_balance_loss:
                            if "gate_loss" in metrics:
                                train_print_str.append(f"gate_loss: {self.hparams.moe_l_aux_wt * train_meter_dict['gate_loss']:.7f} ({self.hparams.moe_l_aux_wt * train_meter_mean['gate_loss']:.7f})")
                                train_print_str.append(f"real_gate_loss: {train_meter_dict['gate_loss']:.7f} ({train_meter_mean['gate_loss']:.7f})")
                        
                        train_print_str = " ".join(train_print_str)
                        main_log(train_print_str)


                data_sample_time = time.time()

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            tmp_optimizers = optimizers
            self._save_checkpoint(tmp_optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

        # if self.hparams.cluster_mask_path is None:
        #     val_metrics = self._run_validation(train_iterations)
        #     self._write_final_metrics(val_metrics)


    def train_nerf(self):
        if not self.hparams.moe_train_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)

        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            
            discard_epoch = checkpoint['epoch_id'] if self.hparams.resume_ckpt_state else -1
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1
            discard_epoch = -1

        schedulers = {}
        for key, optimizer in optimizers.items():
            schedulers[key] = ExponentialLR(optimizer,
                                            gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
                                            last_epoch=train_iterations - 1)

        if 'RANK' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            train_sampler = DistributedSampler(self.train_dataset, world_size, int(os.environ['RANK']))
            assert self.hparams.batch_size % world_size == 0
            data_loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size // world_size, sampler=train_sampler,
                                        num_workers=0, pin_memory=True)
        else:
            data_loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                        pin_memory=True)
            train_sampler = None

        # if the epoch is not fully used
        if -1 < discard_index < (len(data_loader) - 1):
            discard_epoch = discard_epoch - 1
        if discard_index == (len(data_loader) - 1):
            discard_index = -1

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        # while train_iterations < self.hparams.train_iterations:
        train_meter = DictAverageMeter()
        for epoch_id in range(self.hparams.num_epochs):
            if epoch_id <= discard_epoch:
                main_log(f"discard_epoch is {discard_epoch}, skip epoch: {epoch_id}")
                continue
            if train_iterations >= self.hparams.train_iterations:
                break

            if train_sampler is not None:
                train_sampler.set_epoch(epoch_id)

            data_sample_time = time.time()
            for dataset_index, item in enumerate(data_loader):
                data_sample_time = time.time() - data_sample_time
                if dataset_index <= discard_index:
                    main_log(f"discard_index is {discard_index}, skip dataset_index: {dataset_index}")
                    continue

                discard_index = -1
                fwd_bwd_time = time.time()
                if self.hparams.compute_memory:
                    torch.cuda.reset_peak_memory_stats()

                if self.hparams.training_step_fn is not None:
                    training_step_fn = getattr(self, self.hparams.training_step_fn)
                else:
                    training_step_fn = self._training_step_nerf
                if self.hparams.training_step_fn == "_training_step_nerf_mip":
                    metrics, _ = training_step_fn(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                        item['radii'].to(self.device, non_blocking=True),
                        image_indices=None)
                else:
                    metrics, _ = training_step_fn(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                        image_indices=None)

                with torch.no_grad():
                    for key, val in metrics.items():
                        if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                            continue

                        if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                            continue
                        elif not math.isfinite(val):
                            raise Exception('Train metrics not finite: {}'.format(metrics))

                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                all_loss = metrics['loss']
                if self.hparams.use_moe and self.hparams.use_balance_loss:
                    if "gate_loss" in metrics:
                        all_loss = all_loss + self.hparams.moe_l_aux_wt * metrics['gate_loss']

                metrics['all_loss'] = all_loss
                all_loss.backward()

                for optimizer in optimizers.values():
                    optimizer.step()

                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1

                fwd_bwd_time = time.time() - fwd_bwd_time
                if self.hparams.compute_memory:
                    fwd_bwd_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                # update meters
                train_meter_dict = {}
                for k, v in metrics.items():
                    train_meter_dict[k] = v.item() if isinstance(v, torch.Tensor) and len(v.shape) == 0 else v
                train_meter_dict["fwd_bwd_time"] = fwd_bwd_time
                train_meter_dict["data_sample_time"] = data_sample_time
                if self.hparams.compute_memory:
                    train_meter_dict["fwd_bwd_memory"] = fwd_bwd_memory
                train_meter.update(train_meter_dict)

                if self.is_master:
                    train_meter_mean = train_meter.mean()
                    pbar.set_postfix(psnr=f"{train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})")
                    pbar.update(1)
                    for key, value in metrics.items():
                        if not isinstance(value, torch.Tensor):
                            self.writer.add_scalar('train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        main_log(f"save checkpoint after step {train_iterations}")
                        tmp_optimizers = optimizers
                        self._save_checkpoint_nerf(tmp_optimizers, train_iterations, dataset_index, epoch_id=epoch_id)

                    if train_iterations % self.hparams.i_print==0 or train_iterations == self.hparams.train_iterations:
                        train_print_str = [
                            f"[TRAIN] Iter: {train_iterations} all_loss: {train_meter_dict['all_loss']:.5f} ({train_meter_mean['all_loss']:.5f})",
                            f"PSNR: {train_meter_dict['psnr']:.2f} ({train_meter_mean['psnr']:.2f})",
                            f"img_loss: {train_meter_dict['loss']:.5f} ({train_meter_mean['loss']:.5f})",
                            f"lr: {optimizers['nerf'].param_groups[0]['lr']:.5f}",
                            f"data_time: {data_sample_time:.5f}, ({train_meter_mean['data_sample_time']:.5f})",
                            f"fwd_bwd_time: {fwd_bwd_time:.5f}, ({train_meter_mean['fwd_bwd_time']:.5f})",
                            f"fwd_bwd_memory: {fwd_bwd_memory:.2f}, ({train_meter_mean['fwd_bwd_memory']:.5f})" if self.hparams.compute_memory else ""
                        ]
                        if self.hparams.use_balance_loss:
                            if "gate_loss" in metrics:
                                train_print_str.append(f"gate_loss: {self.hparams.moe_l_aux_wt * train_meter_dict['gate_loss']:.7f} ({self.hparams.moe_l_aux_wt * train_meter_mean['gate_loss']:.7f})")
                                train_print_str.append(f"real_gate_loss: {train_meter_dict['gate_loss']:.7f} ({train_meter_mean['gate_loss']:.7f})")
                        
                        train_print_str = " ".join(train_print_str)
                        main_log(train_print_str)


                if train_iterations >= self.hparams.train_iterations:
                    break
                data_sample_time = time.time()

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            main_log(f"save checkpoint after step {train_iterations}")
            tmp_optimizers = optimizers
            self._save_checkpoint_nerf(tmp_optimizers, train_iterations, dataset_index, epoch_id=epoch_id)



    def set_no_batch(self, mode=True):
        for net in self.nerf.modules():
            # if type(net) == MOELayer:
            if hasattr(net, "moe_no_batch"):
                net.moe_no_batch = mode
        if self.hparams.bg_nerf:
            for net in self.bg_nerf.modules():
                # if type(net) == MOELayer:
                if hasattr(net, "moe_no_batch"):
                    net.moe_no_batch = mode
  
    def eval(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        val_metrics = self._run_validation(0)
        self._write_final_metrics(val_metrics)  

    def eval_image(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        val_metrics = self._run_validation_image(0)
        self._write_final_metrics(val_metrics)  

    def eval_image_blocknerf(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        val_metrics = self._run_validation_image_blocknerf(0)
        # self._write_final_metrics(val_metrics)  

    def eval_points(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        val_metrics = self._run_validation_points(0)
        self._write_final_metrics(val_metrics)

    def eval_ckpt(self):
        # self._setup_experiment_dir()
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        dict = {
            'model_state_dict': self.nerf.state_dict()
        }
        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(0))

    def eval_nerf(self):
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        self._run_validation_nerf(0, mode="test")

    def eval_points_nerf(self):
        if not self.hparams.moe_test_batch:
            self.set_no_batch(mode=True)
        else:
            self.set_no_batch(mode=False)
        self._run_validation_points_nerf(0, mode="test")

    def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
        if self.is_master:
            with (self.experiment_path / 'metrics.txt').open('w') as f:
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    message = 'Average {}: {}'.format(key, avg_val)
                    main_log(message)
                    f.write('{}\n'.format(message))

            self.writer.flush()
            self.writer.close()

    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != "1":
            dist.barrier()
        
    def _setup_experiment_dir_nerf(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != "1":
            dist.barrier()

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                    bg_nerf=self.bg_nerf,
                                                    rays=rays,
                                                    image_indices=image_indices,
                                                    hparams=self.hparams,
                                                    sphere_center=self.sphere_center,
                                                    sphere_radius=self.sphere_radius,
                                                    get_depth=False,
                                                    get_depth_variance=True,
                                                    get_bg_fg_rgb=False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if (self.hparams.use_moe or self.hparams.bg_use_moe) and self.hparams.use_balance_loss:
            gate_loss = torch.mean(results[f'gate_loss_{typ}'])
            metrics[f'{typ}_gate_loss'] = gate_loss
            metrics['gate_loss'] = gate_loss
            if typ == 'fine':
                coarse_gate_loss = torch.mean(results[f'gate_loss_coarse'])
                metrics['coarse_gate_loss'] = coarse_gate_loss
                metrics['gate_loss'] = (metrics['gate_loss'] + coarse_gate_loss) / 2.0
            
            if self.hparams.bg_use_moe:
                assert self.hparams.bg_use_cfg
                bg_gate_loss = torch.mean(results[f'bg_gate_loss_{typ}'])
                metrics[f'bg_{typ}_gate_loss'] = bg_gate_loss
                metrics['bg_gate_loss'] = bg_gate_loss
                if typ == 'fine':
                    bg_coarse_gate_loss = torch.mean(results[f'bg_gate_loss_coarse'])
                    metrics['bg_coarse_gate_loss'] = bg_coarse_gate_loss
                    metrics['bg_gate_loss'] = (metrics['bg_gate_loss'] + bg_coarse_gate_loss) / 2.0

        return metrics, bg_nerf_rays_present


    def _training_step_mip(self, rgbs: torch.Tensor, rays: torch.Tensor, radii: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, _ = render_rays_mip(
            nerf=self.nerf,
            rays=rays,
            radii=radii,
            image_indices=image_indices,
            hparams=self.hparams,
            get_depth=False,
            get_depth_variance=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if typ != 'coarse':
            coarse_loss = F.mse_loss(results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += coarse_loss
            metrics['loss'] /= 2

        if self.hparams.use_moe and self.hparams.use_balance_loss:
            gate_loss = torch.mean(results[f'gate_loss_{typ}'])
            metrics[f'{typ}_gate_loss'] = gate_loss
            metrics['gate_loss'] = gate_loss
            if typ == 'fine':
                coarse_gate_loss = torch.mean(results[f'gate_loss_coarse'])
                metrics['coarse_gate_loss'] = coarse_gate_loss
                metrics['gate_loss'] = (metrics['gate_loss'] + coarse_gate_loss) / 2.0

        return metrics, False



    def _training_step_nerf(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, _ = render_rays(
            nerf=self.nerf,
            bg_nerf=self.bg_nerf,
            rays=rays,
            image_indices=None,
            hparams=self.hparams,
            sphere_center=None,
            sphere_radius=None,
            get_depth=False,
            get_depth_variance=True,
            get_bg_fg_rgb=False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if self.hparams.use_cascade and typ != 'coarse':
            coarse_loss = F.mse_loss(results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += coarse_loss
            metrics['loss'] /= 2

        if self.hparams.use_moe and self.hparams.use_balance_loss:
            gate_loss = torch.mean(results[f'gate_loss_{typ}'])
            metrics[f'{typ}_gate_loss'] = gate_loss
            metrics['gate_loss'] = gate_loss
            if typ == 'fine':
                coarse_gate_loss = torch.mean(results[f'gate_loss_coarse'])
                metrics['coarse_gate_loss'] = coarse_gate_loss
                metrics['gate_loss'] = (metrics['gate_loss'] + coarse_gate_loss) / 2.0

        return metrics, False


    def _training_step_nerf_mip(self, rgbs: torch.Tensor, rays: torch.Tensor, radii: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, _ = render_rays_mip(
            nerf=self.nerf,
            rays=rays,
            radii=radii,
            image_indices=None,
            hparams=self.hparams,
            get_depth=False,
            get_depth_variance=True)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
        metrics['photo_loss'] = photo_loss
        metrics['loss'] = photo_loss

        if typ != 'coarse':
            coarse_loss = F.mse_loss(results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += coarse_loss
            metrics['loss'] /= 2

        if self.hparams.use_moe and self.hparams.use_balance_loss:
            gate_loss = torch.mean(results[f'gate_loss_{typ}'])
            metrics[f'{typ}_gate_loss'] = gate_loss
            metrics['gate_loss'] = gate_loss
            if typ == 'fine':
                coarse_gate_loss = torch.mean(results[f'gate_loss_coarse'])
                metrics['coarse_gate_loss'] = coarse_gate_loss
                metrics['gate_loss'] = (metrics['gate_loss'] + coarse_gate_loss) / 2.0

        return metrics, False



    def _run_validation(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                    dist.barrier()
                else:
                    indices_to_eval = np.arange(len(self.val_items))

                for i in main_tqdm(indices_to_eval):
                    metadata_item = self.val_items[i]
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    if self.hparams.render_image_fn_name is not None:
                        render_image_fn = getattr(self, self.hparams.render_image_fn_name)
                        results, _ = render_image_fn(metadata_item)
                    else:
                        results, _ = self.render_image(metadata_item)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_psnr, train_index)
                    else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_ssim, train_index)
                    else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        metric_key = '{}/{}'.format(agg_key, i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                        val_metrics[agg_key] += val_lpips_metrics[network]

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                    if self.writer is not None:
                        self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))

                    if self.hparams.bg_nerf:
                        if (typ == "fine") and (f'bg_rgb_{typ}' not in results):
                            bg_typ = "coarse"
                        else:
                            bg_typ = typ
                        if f'bg_rgb_{bg_typ}' in results:
                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'bg_rgb_{bg_typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'bg_depth_{bg_typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))

                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'fg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics

    def _run_validation_image(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                        base_img_path = Path(self.experiment_path) / "images"
                        base_img_path.mkdir()
                        base_img_path_broadcast = [base_img_path]
                    if not self.is_master:
                        base_img_path_broadcast = [None]
                    torch.distributed.broadcast_object_list(base_img_path_broadcast, src=0)
                    dist.barrier()
                    base_img_path = base_img_path_broadcast[0]
                else:
                    indices_to_eval = np.arange(len(self.val_items))
                    base_img_path = Path(self.experiment_path) / "images"
                    base_img_path.mkdir()

                for i in main_tqdm(indices_to_eval):
                    metadata_item = self.val_items[i]

                    torch.cuda.reset_peak_memory_stats()
                    time_end = time.time()
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    if self.hparams.render_image_fn_name is not None:
                        render_image_fn = getattr(self, self.hparams.render_image_fn_name)
                        results, _ = render_image_fn(metadata_item)
                    else:
                        results, _ = self.render_image(metadata_item)
                    forward_time = time.time() - time_end
                    forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                    metric_key = 'val/time/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, forward_time, train_index)
                    else:
                        torch.save({'value': forward_time, 'metric_key': metric_key, 'agg_key': 'val/time'},
                                   metric_path / 'time-{}.pt'.format(i))
                    val_metrics['val/time'] += forward_time                    

                    metric_key = 'val/memory/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, forward_max_memory_allocated, train_index)
                    else:
                        torch.save({'value': forward_max_memory_allocated, 'metric_key': metric_key, 'agg_key': 'val/memory'},
                                   metric_path / 'memory-{}.pt'.format(i))
                    val_metrics['val/memory'] += forward_max_memory_allocated                    

                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_psnr, train_index)
                    else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_ssim, train_index)
                    else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        metric_key = '{}/{}'.format(agg_key, i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                        val_metrics[agg_key] += val_lpips_metrics[network]
                    
                    val_metrics_txt = {"psnr": val_psnr, "ssim": val_ssim}
                    for tmp_network in val_lpips_metrics:
                        val_metrics_txt['lpips-{}'.format(tmp_network)] = val_lpips_metrics[tmp_network]
                    val_metrics_txt["time"] = forward_time
                    val_metrics_txt["memory"] = forward_max_memory_allocated

                    with (base_img_path / f'metrics_{i}.txt').open('w') as f:
                        for key in val_metrics_txt:
                            message = '{}: {}'.format(key, val_metrics_txt[key])
                            f.write('{}\n'.format(message))

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                    for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                        # left, upper, right, lower
                        img_w, img_h = img.size
                        img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                        img.crop(img_box).save(str(base_img_path / '{}_{}.jpg'.format(i, img_suf)))

                    if self.writer is not None:
                        self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))

                    if self.hparams.bg_nerf:
                        if (typ == "fine") and (f'bg_rgb_{typ}' not in results):
                            bg_typ = "coarse"
                        else:
                            bg_typ = typ
                        if f'bg_rgb_{bg_typ}' in results:
                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'bg_rgb_{bg_typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'bg_depth_{bg_typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))
                            
                            for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                                # left, upper, right, lower
                                img_w, img_h = img.size
                                img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                                img.crop(img_box).save(str(base_img_path / '{}_{}_bg.jpg'.format(i, img_suf)))

                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'fg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                            for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                                # left, upper, right, lower
                                img_w, img_h = img.size
                                img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                                img.crop(img_box).save(str(base_img_path / '{}_{}_fg.jpg'.format(i, img_suf)))

                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics

    
    def _run_validation_image_blocknerf(self, train_index: int) -> Dict[str, float]:
        from switch_nerf.datasets.block_filesystem_dataset import load_tfrecord
        id_map_path = self.hparams.block_image_hash_id_map_path
        with open(id_map_path) as f:
            image_hash_id_map = json.load(f)
        
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None

            data_path = Path(self.hparams.dataset_path)
            with open(self.hparams.block_val_list_path) as f:
                lines = f.readlines()
                tfrecord_paths = [data_path / line.rstrip() for line in lines]
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name)
                    # base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'val_metrics'
                    image_path = base_tmp_path / 'val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(tfrecord_paths), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir(exist_ok=True)
                        metric_path.mkdir(exist_ok=True)
                        image_path.mkdir(exist_ok=True)
                        base_img_path = Path(self.hparams.exp_name) / "images"
                        base_img_path.mkdir(exist_ok=True)
                        base_img_path_broadcast = [base_img_path]
                    if not self.is_master:
                        base_img_path_broadcast = [None]
                    torch.distributed.broadcast_object_list(base_img_path_broadcast, src=0)
                    dist.barrier()
                    base_img_path = base_img_path_broadcast[0]
                else:
                    indices_to_eval = np.arange(len(tfrecord_paths))
                    base_img_path = Path(self.hparams.exp_name) / "images"
                    base_img_path.mkdir(exist_ok=True)
                
                local_tfrecord_paths = [tfrecord_paths[tmp_i] for tmp_i in indices_to_eval]
                for tfrecord_path in main_tqdm(local_tfrecord_paths):
                    # assert "val" in str(tfrecord_path)
                    hash_id_map = image_hash_id_map[os.path.basename(tfrecord_path)]
                    tfrecord_data_dicts = load_tfrecord(tfrecord_path, hash_id_map, near=self.near, far=self.far, load_mask=True)
                    
                    for tfrecord_data_dict in tfrecord_data_dicts:
                        i = tfrecord_data_dict["image_hash"]
                        if (image_path / '{}.jpg'.format(i)).exists():
                            continue

                        torch.cuda.reset_peak_memory_stats()
                        time_end = time.time()
                        viz_rgbs = tfrecord_data_dict['rgbs']
                        rays = tfrecord_data_dict['rays']
                        image_indices = tfrecord_data_dict['image_indices']
                        valid_mask = tfrecord_data_dict['mask'] < 0.5 # 0 is valid
                        if self.hparams.use_mip:
                            radii = tfrecord_data_dict['radii']
                        else:
                            radii = None

                        results, _ = self.render_image_blocknerf(
                            rays=rays,
                            radii=radii,
                            image_indices=image_indices)
                        forward_time = time.time() - time_end
                        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                        metric_key = 'val/time/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, forward_time, train_index)
                        # else:
                        torch.save({'value': forward_time, 'metric_key': metric_key, 'agg_key': 'val/time'},
                                metric_path / 'time-{}.pt'.format(i))
                        # val_metrics['val/time'] += forward_time                    

                        metric_key = 'val/memory/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, forward_max_memory_allocated, train_index)
                        # else:
                        torch.save({'value': forward_max_memory_allocated, 'metric_key': metric_key, 'agg_key': 'val/memory'},
                                metric_path / 'memory-{}.pt'.format(i))
                        # val_metrics['val/memory'] += forward_max_memory_allocated                    

                        typ = 'fine' if 'rgb_fine' in results else 'coarse'
                        viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                        eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                        eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                        eval_valid_mask = valid_mask[:, viz_rgbs.shape[1] // 2:].contiguous()

                        val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                        metric_key = 'val/psnr/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_psnr, train_index)
                        # else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                metric_path / 'psnr-{}.pt'.format(i))

                        # val_metrics['val/psnr'] += val_psnr

                        val_psnr_mask = psnr_mask(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3), valid_mask=eval_valid_mask.view(-1))

                        metric_key = 'val/psnr_mask/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_psnr_mask, train_index)
                        # else:
                        torch.save({'value': val_psnr_mask, 'metric_key': metric_key, 'agg_key': 'val/psnr_mask'},
                                metric_path / 'psnr_mask-{}.pt'.format(i))

                        # val_metrics['val/psnr_mask'] += val_psnr_mask

                        val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                        metric_key = 'val/ssim/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_ssim, train_index)
                        # else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                metric_path / 'ssim-{}.pt'.format(i))

                        # val_metrics['val/ssim'] += val_ssim

                        val_ssim_mask = ssim_mask(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1, valid_mask=eval_valid_mask.view(*eval_rgbs.shape[0:-1]))

                        metric_key = 'val/ssim_mask/{}'.format(i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_ssim, train_index)
                        # else:
                        torch.save({'value': val_ssim_mask, 'metric_key': metric_key, 'agg_key': 'val/ssim_mask'},
                                metric_path / 'ssim_mask-{}.pt'.format(i))

                        # val_metrics['val/ssim_mask'] += val_ssim_mask

                        val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                        for network in val_lpips_metrics:
                            agg_key = 'val/lpips/{}'.format(network)
                            metric_key = '{}/{}'.format(agg_key, i)
                            # if self.writer is not None:
                            #     self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                            # else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                            # val_metrics[agg_key] += val_lpips_metrics[network]
                        
                        val_metrics_txt = {"psnr": val_psnr, "ssim": val_ssim, "psnr_mask": val_psnr_mask, "ssim_mask": val_ssim_mask}
                        for tmp_network in val_lpips_metrics:
                            val_metrics_txt['lpips-{}'.format(tmp_network)] = val_lpips_metrics[tmp_network]
                        val_metrics_txt["time"] = forward_time
                        val_metrics_txt["memory"] = forward_max_memory_allocated

                        with (base_img_path / f'metrics_{i}.txt').open('w') as f:
                            for key in val_metrics_txt:
                                message = '{}: {}'.format(key, val_metrics_txt[key])
                                f.write('{}\n'.format(message))

                        viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                        viz_depth = results[f'depth_{typ}']
                        if f'fg_depth_{typ}' in results:
                            to_use = results[f'fg_depth_{typ}'].view(-1)
                            while to_use.shape[0] > 2 ** 24:
                                to_use = to_use[::2]
                            ma = torch.quantile(to_use, 0.95)

                            viz_depth = viz_depth.clamp_max(ma)

                        img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                        for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                            # left, upper, right, lower
                            img_w, img_h = img.size
                            img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                            img.crop(img_box).save(str(base_img_path / '{}_{}.jpg'.format(i, img_suf)))

                        # if self.writer is not None:
                        #     self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                        # else:
                        img.save(str(image_path / '{}.jpg'.format(i)))

                        if self.hparams.bg_nerf:
                            if f'bg_rgb_{typ}' in results:
                                img = Runner._create_result_image(viz_rgbs,
                                                                results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                                viz_rgbs.shape[1],
                                                                                                3).cpu(),
                                                                results[f'bg_depth_{typ}'])

                                # if self.writer is not None:
                                #     self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                                # else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))
                                
                                for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                                    # left, upper, right, lower
                                    img_w, img_h = img.size
                                    img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                                    img.crop(img_box).save(str(base_img_path / '{}_{}_bg.jpg'.format(i, img_suf)))

                                img = Runner._create_result_image(viz_rgbs,
                                                                results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                                viz_rgbs.shape[1],
                                                                                                3).cpu(),
                                                                results[f'fg_depth_{typ}'])

                                # if self.writer is not None:
                                #     self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                                # else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                                for img_i, img_suf in enumerate(["gt", "pred", "depth"]):
                                    # left, upper, right, lower
                                    img_w, img_h = img.size
                                    img_box = [img_w // 3 * img_i, 0, img_w // 3 * (img_i + 1), img_h]
                                    img.crop(img_box).save(str(base_img_path / '{}_{}_fg.jpg'.format(i, img_suf)))

                        del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    image_num = image_hash_id_map["val_image_num"]
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        # for image_file in image_path.iterdir():
                        #     img = Image.open(str(image_file))
                        #     self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / image_num
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)
                    
                    if self.is_master:
                        with (self.experiment_path / 'metrics.txt').open('w') as f:
                            for key in val_metrics:
                                avg_val = val_metrics[key] / image_num
                                message = 'Average {}: {}'.format(key, avg_val)
                                main_log(message)
                                f.write('{}\n'.format(message))

                        self.writer.flush()
                        self.writer.close()

                    dist.barrier()

                self.nerf.train()
            finally:
                # if self.is_master and base_tmp_path is not None:
                #     shutil.rmtree(base_tmp_path)
                pass

            return val_metrics

    def _run_validation_points(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            val_items_len = min(self.hparams.render_test_points_image_num, len(self.val_items))
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), val_items_len, world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                        eval_points_base_dir = Path(self.experiment_path) / "eval_points"
                        eval_points_base_dir.mkdir()
                        eval_points_base_dir_broadcast = [eval_points_base_dir]
                    if not self.is_master:
                        eval_points_base_dir_broadcast = [None]
                    torch.distributed.broadcast_object_list(eval_points_base_dir_broadcast, src=0)
                    dist.barrier()
                    eval_points_base_dir = eval_points_base_dir_broadcast[0]
                else:
                    indices_to_eval = np.arange(val_items_len)
                    eval_points_base_dir = Path(self.experiment_path) / "eval_points"
                    eval_points_base_dir.mkdir()

                for i in main_tqdm(indices_to_eval):
                    metadata_item = self.val_items[i]
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    if self.hparams.render_image_fn_name is not None:
                        render_image_fn = getattr(self, self.hparams.render_image_fn_name)
                        results, _ = render_image_fn(metadata_item)
                    else:
                        results, _ = self.render_image(metadata_item)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_psnr, train_index)
                    else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_ssim, train_index)
                    else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        metric_key = '{}/{}'.format(agg_key, i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                        val_metrics[agg_key] += val_lpips_metrics[network]

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                    if self.writer is not None:
                        self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))
                    
                    # save color points
                    
                    eval_points_save_dir = eval_points_base_dir / str(i)
                    eval_points_save_dir.mkdir(parents=True, exist_ok=True)
                    img.save(str(eval_points_save_dir / '{}.jpg'.format(i)))

                    if self.hparams.bg_nerf:
                        if (typ == "fine") and (f'bg_rgb_{typ}' not in results):
                            bg_typ = "coarse"
                        else:
                            bg_typ = typ
                        if f'bg_rgb_{bg_typ}' in results:
                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'bg_rgb_{bg_typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'bg_depth_{bg_typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))

                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              results[f'fg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                    
                    # save color points
                    # eval_points_save_dir = self.experiment_path / "eval_points" / str(i)
                    # eval_points_save_dir.mkdir(parents=True, exist_ok=True)
                    # img.save(str(eval_points_save_dir / '{}.jpg'.format(i)))

                    if self.hparams.use_moe:
                        for typ in self.hparams.render_test_points_typ:
                            moe_gates = results[f'moe_gates_{typ}'] # rays, samples, layer_num, top_k
                            moe_gates = moe_gates.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *moe_gates.shape[1:]) # H, W, samples, layer_num, top_k
                            
                            pts = results[f"pts_{typ}"] # rays, samples, 3
                            pts = pts.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts.shape[1:]) # H, W, samples, 3

                            pts_rgb = results[f"pts_rgb_{typ}"] # rays, samples, 3
                            pts_rgb = pts_rgb.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_rgb.shape[1:]) # H, W, samples, 3

                            pts_alpha = results[f"pts_alpha_{typ}"] # rays, samples
                            pts_alpha = pts_alpha.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_alpha.shape[1:]) # H, W, samples, 3

                            moe_gates = moe_gates[:, :, ::self.hparams.render_test_points_sample_skip, 0] # H, W, sample, top_k
                            pts = pts[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_rgb = pts_rgb[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_alpha = pts_alpha[:, :, ::self.hparams.render_test_points_sample_skip]

                            vertexs = pts.reshape(-1, 3).cpu().numpy()
                            vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

                            # write
                            # plyfilename = os.path.join(testsavedir, '{:03d}_{}_pts_rgba.ply'.format(img_i, typ))
                            pts_rgba = torch.cat([pts_rgb, pts_alpha.unsqueeze(3)], dim=3)
                            vertex_colors = (pts_rgba * 255).to(torch.uint8).reshape(-1, 4).cpu().numpy()
                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

                            # vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
                            # for prop in vertexs.dtype.names:
                            #     vertex_all[prop] = vertexs[prop]
                            # for prop in vertex_colors.dtype.names:
                            #     vertex_all[prop] = vertex_colors[prop]

                            # el = PlyElement.describe(vertex_all, 'vertex')
                            # PlyData([el]).write(plyfilename)

                            expert_num = self.hparams.moe_expert_num
                            moe_k_value = moe_gates.shape[-1]
                            for tmp_k in range(moe_k_value):
                                moe_index_k = moe_gates[:, :, :, tmp_k] # H, W, sample
                                vertex_colors = vertex_colors

                                out_ids = list(range(expert_num))
                                if tmp_k == 0:
                                    out_ids += ["all"]

                                for expert_id in out_ids:

                                    plyfilename = '{:03d}_{}_pts_rgba.ply'.format(i, typ) if expert_id == "all" else \
                                        '{:03d}_{}_pts_rgba_top_{:01d}_exp_{}.ply'.format(i, typ, tmp_k, expert_id)
                                    plyfilename = os.path.join(eval_points_save_dir, plyfilename)

                                    if expert_id != "all":
                                        expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                        vertexs_expert = vertexs[expert_pts_idx]
                                        vertex_colors_expert = vertex_colors[expert_pts_idx]
                                    else:
                                        vertexs_expert = vertexs
                                        vertex_colors_expert = vertex_colors
                                    vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                    for prop in vertexs_expert.dtype.names:
                                        vertex_all[prop] = vertexs_expert[prop]
                                    for prop in vertex_colors_expert.dtype.names:
                                        vertex_all[prop] = vertex_colors_expert[prop]

                                    el = PlyElement.describe(vertex_all, 'vertex')
                                    PlyData([el]).write(plyfilename)

                            
                            if self.hparams.return_pts_class_seg:
                                palette = torch.tensor(voc_palette(), dtype=torch.uint8, device=moe_gates.device)[1:, ]
                                for tmp_k in range(moe_k_value):
                                    moe_index_k = moe_gates[:, :, :, tmp_k] # H, W, sample
                                    color_seg_k = torch.zeros((list(pts.shape[0:3]) + [3]), dtype=torch.uint8, device=moe_index_k.device)
                                    for expert_id in range(expert_num):
                                        expert_color = palette[expert_id]
                                        color_seg_k[moe_index_k == expert_id, :] = expert_color
                                    
                                    for name_suffix in ["alpha", ""]:

                                        if name_suffix == "alpha":
                                            color_seg_ka = (pts_alpha * 255).to(torch.uint8)
                                            color_seg_ka = torch.cat([color_seg_k, color_seg_ka.unsqueeze(3)], dim=3)
                                            vertex_colors = color_seg_ka.reshape(-1, 4).cpu().numpy()
                                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

                                            for expert_id in list(range(expert_num)) + ["all"]:

                                                plyfilename = '{:03d}_{}_top_{:01d}_{}.ply'.format(i, typ, tmp_k, name_suffix) if expert_id == "all" else \
                                                    '{:03d}_{}_top_{:01d}_{}_exp_{}.ply'.format(i, typ, tmp_k, name_suffix, expert_id)
                                                plyfilename = os.path.join(eval_points_save_dir, plyfilename)

                                                if expert_id != "all":
                                                    expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                                    vertexs_expert = vertexs[expert_pts_idx]
                                                    vertex_colors_expert = vertex_colors[expert_pts_idx]
                                                else:
                                                    vertexs_expert = vertexs
                                                    vertex_colors_expert = vertex_colors
                                                vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                                for prop in vertexs_expert.dtype.names:
                                                    vertex_all[prop] = vertexs_expert[prop]
                                                for prop in vertex_colors_expert.dtype.names:
                                                    vertex_all[prop] = vertex_colors_expert[prop]

                                                el = PlyElement.describe(vertex_all, 'vertex')
                                                PlyData([el]).write(plyfilename)                            
                                        
                                        else:
                                            color_seg_k[:, :, -1, :] = (viz_result_rgbs * 255).to(torch.uint8)
                                            vertex_colors = color_seg_k.reshape(-1, 3).cpu().numpy()
                                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

                                            for expert_id in list(range(expert_num)) + ["all"]:    
                                                plyfilename = '{:03d}_{}_top_{:01d}.ply'.format(i, typ, tmp_k) if expert_id == "all" else \
                                                    '{:03d}_{}_top_{:01d}_exp_{}.ply'.format(i, typ, tmp_k, expert_id)
                                                plyfilename = os.path.join(eval_points_save_dir, plyfilename)
                                                
                                                if expert_id != "all":
                                                    expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                                    vertexs_expert = vertexs[expert_pts_idx]
                                                    vertex_colors_expert = vertex_colors[expert_pts_idx]
                                                else:
                                                    vertexs_expert = vertexs
                                                    vertex_colors_expert = vertex_colors
                                                vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                                for prop in vertexs_expert.dtype.names:
                                                    vertex_all[prop] = vertexs_expert[prop]
                                                for prop in vertex_colors_expert.dtype.names:
                                                    vertex_all[prop] = vertex_colors_expert[prop]

                                                el = PlyElement.describe(vertex_all, 'vertex')
                                                PlyData([el]).write(plyfilename)
                    else:
                        for typ in self.hparams.render_test_points_typ:
                            pts = results[f"pts_{typ}"] # rays, samples, 3
                            pts = pts.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts.shape[1:]) # H, W, samples, 3

                            pts_rgb = results[f"pts_rgb_{typ}"] # rays, samples, 3
                            pts_rgb = pts_rgb.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_rgb.shape[1:]) # H, W, samples, 3

                            pts_alpha = results[f"pts_alpha_{typ}"] # rays, samples
                            pts_alpha = pts_alpha.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_alpha.shape[1:]) # H, W, samples, 3

                            pts = pts[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_rgb = pts_rgb[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_alpha = pts_alpha[:, :, ::self.hparams.render_test_points_sample_skip]

                            vertexs = pts.reshape(-1, 3).cpu().numpy()
                            vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

                            pts_rgba = torch.cat([pts_rgb, pts_alpha.unsqueeze(3)], dim=3)
                            vertex_colors = (pts_rgba * 255).to(torch.uint8).reshape(-1, 4).cpu().numpy()
                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

                            plyfilename = '{:03d}_{}_pts_rgba.ply'.format(i, typ)
                            plyfilename = os.path.join(eval_points_save_dir, plyfilename)

                            vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
                            for prop in vertexs.dtype.names:
                                vertex_all[prop] = vertexs[prop]
                            for prop in vertex_colors.dtype.names:
                                vertex_all[prop] = vertex_colors[prop]

                            el = PlyElement.describe(vertex_all, 'vertex')
                            PlyData([el]).write(plyfilename)
                    
                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / val_items_len
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics
 

    def _run_validation_nerf(self, train_index: int, mode="val") -> Dict[str, float]:
        assert mode in ["test", "val"]
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    experiment_path_broadcast = [self.experiment_path]
                    torch.distributed.broadcast_object_list(experiment_path_broadcast, src=0)
                    self.experiment_path = experiment_path_broadcast[0]

                    base_tmp_path = Path(self.experiment_path) / "tmp_metrics" / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'
                    base_img_path = Path(self.experiment_path) / f"{mode}_images_{train_index}"

                    world_size = int(os.environ['WORLD_SIZE'])
                    if self.is_master:
                        base_tmp_path.mkdir(parents=True)
                        metric_path.mkdir()
                        image_path.mkdir()
                        base_img_path.mkdir()
                    dist.barrier()
                else:
                    base_img_path = Path(self.experiment_path) / f"{mode}_images_{train_index}"
                    base_img_path.mkdir()
                
                if mode == "val":
                    real_dataset = self.val_dataset
                elif mode == "test":
                    real_dataset = self.test_dataset
                else:
                    raise NotImplementedError
                if 'RANK' in os.environ:
                    world_size = int(os.environ['WORLD_SIZE'])
                    data_sampler = DistributedSampler(real_dataset, world_size, int(os.environ['RANK']))
                    data_loader = DataLoader(real_dataset, batch_size=1, sampler=data_sampler,
                                                num_workers=0, pin_memory=True)
                else:
                    data_loader = DataLoader(real_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                pin_memory=True)
                    data_sampler = None

                for data_item in main_tqdm(data_loader):
                    rays = data_item['rays'].squeeze(0)
                    viz_rgbs = data_item['rgbs'].squeeze(0)
                    img_i = data_item['img_i'][0].item()

                    torch.cuda.reset_peak_memory_stats()
                    time_end = time.time()
                    if self.hparams.render_image_fn_name is not None:
                        render_image_fn = getattr(self, self.hparams.render_image_fn_name)
                        results, _ = render_image_fn(rays)
                    else:
                        if self.hparams.use_mip:
                            radii = data_item['radii'].squeeze(0)
                            results = self.render_image_nerf_mip(rays, radii)
                        else:
                            results = self.render_image_nerf(rays)
                    forward_time = time.time() - time_end
                    forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                    metric_key = '{}/time/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, forward_time, train_index)
                    # else:
                    torch.save({'value': forward_time, 'metric_key': metric_key, 'agg_key': f'{mode}/time'},
                                metric_path / 'time-{}.pt'.format(img_i))
                    # val_metrics[f'{mode}/time'] += forward_time                    

                    metric_key = '{}/memory/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, forward_max_memory_allocated, train_index)
                    # else:
                    torch.save({'value': forward_max_memory_allocated, 'metric_key': metric_key, 'agg_key': f'{mode}/memory'},
                                metric_path / 'memory-{}.pt'.format(img_i))
                    # val_metrics[f'{mode}/memory'] += forward_max_memory_allocated


                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    # eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    # eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    eval_rgbs = viz_rgbs
                    eval_result_rgbs = viz_result_rgbs

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = '{}/psnr/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, val_psnr, train_index)
                    # else:
                    torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': f'{mode}/psnr'},
                                metric_path / 'psnr-{}.pt'.format(img_i))

                    # val_metrics[f'{mode}/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = '{}/ssim/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, val_ssim, train_index)
                    # else:
                    torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': f'{mode}/ssim'},
                                metric_path / 'ssim-{}.pt'.format(img_i))

                    # val_metrics[f'{mode}/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = '{}/lpips/{}'.format(mode, network)
                        metric_key = '{}/{}'.format(agg_key, img_i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        # else:
                        torch.save(
                            {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                            metric_path / 'lpips-{}-{}.pt'.format(network, img_i))

                        # val_metrics[agg_key] += val_lpips_metrics[network]

                    val_metrics_txt = {"psnr": val_psnr, "ssim": val_ssim}
                    for tmp_network in val_lpips_metrics:
                        val_metrics_txt['lpips-{}'.format(tmp_network)] = val_lpips_metrics[tmp_network]
                    val_metrics_txt["time"] = forward_time
                    val_metrics_txt["memory"] = forward_max_memory_allocated

                    with (base_img_path / f'metrics_{img_i}.txt').open('w') as f:
                        for key in val_metrics_txt:
                            message = '{}: {}'.format(key, val_metrics_txt[key])
                            f.write('{}\n'.format(message))

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img, imgs = Runner._create_result_image_nerf(viz_rgbs, viz_result_rgbs, viz_depth, colormap=self.hparams.colormap)
                    # imgs = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                    # if self.writer is not None:
                    #     self.writer.add_image('{}/{}'.format(mode, img_i), T.ToTensor()(img), train_index)
                    # else:
                    img.save(str(image_path / '{}.jpg'.format(img_i)))
                    
                    for write_img_i, write_img_suf in enumerate(["gt", "pred", "depth"]):
                        imgs[write_img_i].save(str(base_img_path / '{}_{}.jpg'.format(img_i, write_img_suf)))

                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('{}/{}'.format(mode, image_file.stem), T.ToTensor()(img), train_index)

                        main_log(f'step {train_index} {mode}\n')
                        with (base_img_path / 'metrics.txt').open('w') as f:
                            f.write(f'step {train_index} {mode}\n')
                            for key in val_metrics:
                                avg_val = val_metrics[key] / len(real_dataset)
                                self.writer.add_scalar('{}/avg'.format(key), avg_val, train_index)
                                message = 'Average {}: {}'.format(key, avg_val)
                                main_log(message)
                                f.write('{}\n'.format(message))
                        
                        self.writer.flush()
                        self.writer.close()

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            # if self.is_master:
            #     main_log(f'step {train_index} {mode}\n')
            #     with (base_img_path / 'metrics.txt').open('w') as f:
            #         for key in val_metrics:
            #             avg_val = val_metrics[key] / len(real_dataset)
            #             message = 'Average {}: {}'.format(key, avg_val)
            #             main_log(message)
            #             f.write('{}\n'.format(message))

            #     self.writer.flush()
            #     self.writer.close()

            return val_metrics


    def _run_validation_points_nerf(self, train_index: int, mode="val") -> Dict[str, float]:
        assert mode in ["test", "val"]
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    experiment_path_broadcast = [self.experiment_path]
                    torch.distributed.broadcast_object_list(experiment_path_broadcast, src=0)
                    self.experiment_path = experiment_path_broadcast[0]

                    base_tmp_path = Path(self.experiment_path) / "tmp_metrics" / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'
                    base_img_path = Path(self.experiment_path) / f"{mode}_images_{train_index}"
                    eval_points_base_dir = Path(self.experiment_path) / f"{mode}_points_{train_index}"

                    world_size = int(os.environ['WORLD_SIZE'])
                    if self.is_master:
                        base_tmp_path.mkdir(parents=True)
                        eval_points_base_dir.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                        base_img_path.mkdir()
                    dist.barrier()
                else:
                    base_img_path = Path(self.experiment_path) / f"{mode}_images_{train_index}"
                    base_img_path.mkdir()
                    eval_points_base_dir = Path(self.experiment_path) / f"{mode}_points_{train_index}"
                    eval_points_base_dir.mkdir()
                
                if mode == "val":
                    real_dataset = self.val_dataset
                elif mode == "test":
                    real_dataset = self.test_dataset
                else:
                    raise NotImplementedError
                if 'RANK' in os.environ:
                    world_size = int(os.environ['WORLD_SIZE'])
                    data_sampler = DistributedSampler(real_dataset, world_size, int(os.environ['RANK']))
                    data_loader = DataLoader(real_dataset, batch_size=1, sampler=data_sampler,
                                                num_workers=0, pin_memory=True)
                else:
                    data_loader = DataLoader(real_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                pin_memory=True)
                    data_sampler = None

                for data_item in main_tqdm(data_loader):
                    rays = data_item['rays'].squeeze(0)
                    viz_rgbs = data_item['rgbs'].squeeze(0)
                    img_i = data_item['img_i'][0].item()

                    torch.cuda.reset_peak_memory_stats()
                    time_end = time.time()
                    if self.hparams.render_image_fn_name is not None:
                        render_image_fn = getattr(self, self.hparams.render_image_fn_name)
                        results, _ = render_image_fn(rays)
                    else:
                        if self.hparams.use_mip:
                            radii = data_item['radii'].squeeze(0)
                            results = self.render_image_nerf_mip(rays, radii)
                        elif self.hparams.use_occupancy:
                            results = self.render_image_nerf_occupancy(rays)
                        else:
                            results = self.render_image_nerf(rays)
                    forward_time = time.time() - time_end
                    forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

                    metric_key = '{}/time/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, forward_time, train_index)
                    # else:
                    torch.save({'value': forward_time, 'metric_key': metric_key, 'agg_key': f'{mode}/time'},
                                metric_path / 'time-{}.pt'.format(img_i))
                    # val_metrics[f'{mode}/time'] += forward_time                    

                    metric_key = '{}/memory/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, forward_max_memory_allocated, train_index)
                    # else:
                    torch.save({'value': forward_max_memory_allocated, 'metric_key': metric_key, 'agg_key': f'{mode}/memory'},
                                metric_path / 'memory-{}.pt'.format(img_i))
                    # val_metrics[f'{mode}/memory'] += forward_max_memory_allocated


                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    # eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    # eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    eval_rgbs = viz_rgbs
                    eval_result_rgbs = viz_result_rgbs

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = '{}/psnr/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, val_psnr, train_index)
                    # else:
                    torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': f'{mode}/psnr'},
                                metric_path / 'psnr-{}.pt'.format(img_i))

                    # val_metrics[f'{mode}/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = '{}/ssim/{}'.format(mode, img_i)
                    # if self.writer is not None:
                    #     self.writer.add_scalar(metric_key, val_ssim, train_index)
                    # else:
                    torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': f'{mode}/ssim'},
                                metric_path / 'ssim-{}.pt'.format(img_i))

                    # val_metrics[f'{mode}/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = '{}/lpips/{}'.format(mode, network)
                        metric_key = '{}/{}'.format(agg_key, img_i)
                        # if self.writer is not None:
                        #     self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        # else:
                        torch.save(
                            {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                            metric_path / 'lpips-{}-{}.pt'.format(network, img_i))

                        # val_metrics[agg_key] += val_lpips_metrics[network]

                    val_metrics_txt = {"psnr": val_psnr, "ssim": val_ssim}
                    for tmp_network in val_lpips_metrics:
                        val_metrics_txt['lpips-{}'.format(tmp_network)] = val_lpips_metrics[tmp_network]
                    val_metrics_txt["time"] = forward_time
                    val_metrics_txt["memory"] = forward_max_memory_allocated

                    with (base_img_path / f'metrics_{img_i}.txt').open('w') as f:
                        for key in val_metrics_txt:
                            message = '{}: {}'.format(key, val_metrics_txt[key])
                            f.write('{}\n'.format(message))

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img, imgs = Runner._create_result_image_nerf(viz_rgbs, viz_result_rgbs, viz_depth, colormap=self.hparams.colormap)
                    # imgs = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_depth)

                    # if self.writer is not None:
                    #     self.writer.add_image('{}/{}'.format(mode, img_i), T.ToTensor()(img), train_index)
                    # else:
                    img.save(str(image_path / '{}.jpg'.format(img_i)))
                    
                    for write_img_i, write_img_suf in enumerate(["gt", "pred", "depth"]):
                        imgs[write_img_i].save(str(base_img_path / '{}_{}.jpg'.format(img_i, write_img_suf)))

                    
                    # save color points
                    # eval_points_save_dir = self.experiment_path / "eval_points" / str(i)
                    # eval_points_save_dir.mkdir(parents=True, exist_ok=True)
                    # img.save(str(eval_points_save_dir / '{}.jpg'.format(i)))
                    i = img_i
                    eval_points_save_dir = eval_points_base_dir / str(i)
                    eval_points_save_dir.mkdir(parents=True, exist_ok=True)
                    if self.hparams.use_moe:
                        for typ in self.hparams.render_test_points_typ:
                            _, moe_gates = results[f'moe_gates_{typ}'] # rays, samples, layer_num, top_k
                            moe_gates = moe_gates.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *moe_gates.shape[1:]) # H, W, samples, layer_num, top_k
                            
                            pts = results[f"pts_{typ}"] # rays, samples, 3
                            pts = pts.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts.shape[1:]) # H, W, samples, 3

                            pts_rgb = results[f"pts_rgb_{typ}"] # rays, samples, 3
                            pts_rgb = pts_rgb.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_rgb.shape[1:]) # H, W, samples, 3

                            pts_alpha = results[f"pts_alpha_{typ}"] # rays, samples
                            pts_alpha = pts_alpha.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_alpha.shape[1:]) # H, W, samples, 3

                            moe_gates = moe_gates[:, :, ::self.hparams.render_test_points_sample_skip, 0] # H, W, sample, top_k
                            pts = pts[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_rgb = pts_rgb[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_alpha = pts_alpha[:, :, ::self.hparams.render_test_points_sample_skip]

                            vertexs = pts.reshape(-1, 3).cpu().numpy()
                            vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

                            # write
                            # plyfilename = os.path.join(testsavedir, '{:03d}_{}_pts_rgba.ply'.format(img_i, typ))
                            pts_rgba = torch.cat([pts_rgb, pts_alpha.unsqueeze(3)], dim=3)
                            vertex_colors = (pts_rgba * 255).to(torch.uint8).reshape(-1, 4).cpu().numpy()
                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

                            # vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
                            # for prop in vertexs.dtype.names:
                            #     vertex_all[prop] = vertexs[prop]
                            # for prop in vertex_colors.dtype.names:
                            #     vertex_all[prop] = vertex_colors[prop]

                            # el = PlyElement.describe(vertex_all, 'vertex')
                            # PlyData([el]).write(plyfilename)
                            
                            expert_num = self.hparams.moe_expert_num
                            moe_k_value = moe_gates.shape[-1]
                            for tmp_k in range(moe_k_value):
                                moe_index_k = moe_gates[:, :, :, tmp_k] # H, W, sample
                                vertex_colors = vertex_colors

                                out_ids = list(range(expert_num))
                                if tmp_k == 0:
                                    out_ids += ["all"]

                                for expert_id in out_ids:

                                    plyfilename = '{:03d}_{}_pts_rgba.ply'.format(i, typ) if expert_id == "all" else \
                                        '{:03d}_{}_pts_rgba_top_{:01d}_exp_{}.ply'.format(i, typ, tmp_k, expert_id)
                                    plyfilename = os.path.join(eval_points_save_dir, plyfilename)

                                    if expert_id != "all":
                                        expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                        vertexs_expert = vertexs[expert_pts_idx]
                                        vertex_colors_expert = vertex_colors[expert_pts_idx]
                                    else:
                                        vertexs_expert = vertexs
                                        vertex_colors_expert = vertex_colors
                                    vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                    for prop in vertexs_expert.dtype.names:
                                        vertex_all[prop] = vertexs_expert[prop]
                                    for prop in vertex_colors_expert.dtype.names:
                                        vertex_all[prop] = vertex_colors_expert[prop]

                                    el = PlyElement.describe(vertex_all, 'vertex')
                                    PlyData([el]).write(plyfilename)

                            
                            if self.hparams.return_pts_class_seg:
                                palette = torch.tensor(voc_palette(), dtype=torch.uint8, device=moe_gates.device)[1:, ]
                                for tmp_k in range(moe_k_value):
                                    moe_index_k = moe_gates[:, :, :, tmp_k] # H, W, sample
                                    color_seg_k = torch.zeros((list(pts.shape[0:3]) + [3]), dtype=torch.uint8, device=moe_index_k.device)
                                    for expert_id in range(expert_num):
                                        expert_color = palette[expert_id]
                                        color_seg_k[moe_index_k == expert_id, :] = expert_color
                                    
                                    for name_suffix in ["alpha", ""]:

                                        if name_suffix == "alpha":
                                            color_seg_ka = (pts_alpha * 255).to(torch.uint8)
                                            color_seg_ka = torch.cat([color_seg_k, color_seg_ka.unsqueeze(3)], dim=3)
                                            vertex_colors = color_seg_ka.reshape(-1, 4).cpu().numpy()
                                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])

                                            for expert_id in list(range(expert_num)) + ["all"]:

                                                plyfilename = '{:03d}_{}_top_{:01d}_{}.ply'.format(i, typ, tmp_k, name_suffix) if expert_id == "all" else \
                                                    '{:03d}_{}_top_{:01d}_{}_exp_{}.ply'.format(i, typ, tmp_k, name_suffix, expert_id)
                                                plyfilename = os.path.join(eval_points_save_dir, plyfilename)

                                                if expert_id != "all":
                                                    expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                                    vertexs_expert = vertexs[expert_pts_idx]
                                                    vertex_colors_expert = vertex_colors[expert_pts_idx]
                                                else:
                                                    vertexs_expert = vertexs
                                                    vertex_colors_expert = vertex_colors
                                                vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                                for prop in vertexs_expert.dtype.names:
                                                    vertex_all[prop] = vertexs_expert[prop]
                                                for prop in vertex_colors_expert.dtype.names:
                                                    vertex_all[prop] = vertex_colors_expert[prop]

                                                el = PlyElement.describe(vertex_all, 'vertex')
                                                PlyData([el]).write(plyfilename)                            
                                        
                                        else:
                                            color_seg_k[:, :, -1, :] = (viz_result_rgbs * 255).to(torch.uint8)
                                            vertex_colors = color_seg_k.reshape(-1, 3).cpu().numpy()
                                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

                                            for expert_id in list(range(expert_num)) + ["all"]:    
                                                plyfilename = '{:03d}_{}_top_{:01d}.ply'.format(i, typ, tmp_k) if expert_id == "all" else \
                                                    '{:03d}_{}_top_{:01d}_exp_{}.ply'.format(i, typ, tmp_k, expert_id)
                                                plyfilename = os.path.join(eval_points_save_dir, plyfilename)
                                                
                                                if expert_id != "all":
                                                    expert_pts_idx = (moe_index_k == expert_id).reshape(-1).cpu().numpy()
                                                    vertexs_expert = vertexs[expert_pts_idx]
                                                    vertex_colors_expert = vertex_colors[expert_pts_idx]
                                                else:
                                                    vertexs_expert = vertexs
                                                    vertex_colors_expert = vertex_colors
                                                vertex_all = np.empty(len(vertexs_expert), vertexs_expert.dtype.descr + vertex_colors_expert.dtype.descr)
                                                for prop in vertexs_expert.dtype.names:
                                                    vertex_all[prop] = vertexs_expert[prop]
                                                for prop in vertex_colors_expert.dtype.names:
                                                    vertex_all[prop] = vertex_colors_expert[prop]

                                                el = PlyElement.describe(vertex_all, 'vertex')
                                                PlyData([el]).write(plyfilename)
  
                    else:
                        for typ in self.hparams.render_test_points_typ:
                            pts = results[f"pts_{typ}"] # rays, samples, 3
                            pts = pts.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts.shape[1:]) # H, W, samples, 3

                            pts_rgb = results[f"pts_rgb_{typ}"] # rays, samples, 3
                            pts_rgb = pts_rgb.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_rgb.shape[1:]) # H, W, samples, 3

                            pts_alpha = results[f"pts_alpha_{typ}"] # rays, samples
                            pts_alpha = pts_alpha.view(viz_rgbs.shape[0], viz_rgbs.shape[1], *pts_alpha.shape[1:]) # H, W, samples, 3

                            pts = pts[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_rgb = pts_rgb[:, :, ::self.hparams.render_test_points_sample_skip]
                            pts_alpha = pts_alpha[:, :, ::self.hparams.render_test_points_sample_skip]

                            vertexs = pts.reshape(-1, 3).cpu().numpy()
                            vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

                            plyfilename = '{:03d}_{}_pts_rgba.ply'.format(i, typ)
                            plyfilename = os.path.join(eval_points_save_dir, plyfilename)
                            
                            pts_rgba = torch.cat([pts_rgb, pts_alpha.unsqueeze(3)], dim=3)
                            vertex_colors = (pts_rgba * 255).to(torch.uint8).reshape(-1, 4).cpu().numpy()
                            vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])
                            vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)

                            for prop in vertexs.dtype.names:
                                vertex_all[prop] = vertexs[prop]
                            for prop in vertex_colors.dtype.names:
                                vertex_all[prop] = vertex_colors[prop]

                            el = PlyElement.describe(vertex_all, 'vertex')
                            PlyData([el]).write(plyfilename)
                    
                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('{}/{}'.format(mode, image_file.stem), T.ToTensor()(img), train_index)

                        main_log(f'step {train_index} {mode}\n')
                        with (base_img_path / 'metrics.txt').open('w') as f:
                            f.write(f'step {train_index} {mode}\n')
                            for key in val_metrics:
                                avg_val = val_metrics[key] / len(real_dataset)
                                self.writer.add_scalar('{}/avg'.format(key), avg_val, train_index)
                                message = 'Average {}: {}'.format(key, avg_val)
                                main_log(message)
                                f.write('{}\n'.format(message))
                        
                        self.writer.flush()
                        self.writer.close()

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            # if self.is_master:
            #     main_log(f'step {train_index} {mode}\n')
            #     with (base_img_path / 'metrics.txt').open('w') as f:
            #         for key in val_metrics:
            #             avg_val = val_metrics[key] / len(real_dataset)
            #             message = 'Average {}: {}'.format(key, avg_val)
            #             main_log(message)
            #             f.write('{}\n'.format(message))

            #     self.writer.flush()
            #     self.writer.close()

            return val_metrics
         
    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    
    def _save_checkpoint_nerf(self, optimizers: Dict[str, any], train_index: int, dataset_index: int, epoch_id: int) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index,
            "epoch_id": epoch_id
        }

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        directions = get_ray_directions(metadata.W,
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)

        amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float16
        with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.appearance_dim > 0 else None
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True)

                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []

                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)

            return results, rays

    def render_image_blocknerf(self, rays, radii=None, image_indices=None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        amp_dtype = torch.bfloat16 if self.hparams.amp_use_bfloat16 else torch.float16
        with torch.cuda.amp.autocast(enabled=self.hparams.amp, dtype=amp_dtype):
            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            if radii is not None:
                radii = radii.view(-1, 1).to(self.device, non_blocking=True)  # (H*W, 1)
            if image_indices is not None:
                image_indices = image_indices.view(-1).to(self.device, non_blocking=True) # (H*W)
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                if radii is None:
                    result_batch, _ = render_rays(
                        nerf=nerf, bg_nerf=bg_nerf,
                        rays=rays[i:i + self.hparams.image_pixel_batch_size],
                        image_indices=image_indices[
                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                        hparams=self.hparams,
                        sphere_center=self.sphere_center,
                        sphere_radius=self.sphere_radius,
                        get_depth=True,
                        get_depth_variance=False,
                        get_bg_fg_rgb=False)
                else:
                    result_batch, _ = render_rays_mip(
                        nerf=nerf,
                        rays=rays[i:i + self.hparams.image_pixel_batch_size],
                        radii=radii[i:i + self.hparams.image_pixel_batch_size],
                        image_indices=image_indices[i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                        hparams=self.hparams,
                        get_depth=True,
                        get_depth_variance=False)

                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []

                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)

            return results, rays

    
    def render_image_nerf(self, rays) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
        image_indices = None
        results = {}

        if 'RANK' in os.environ:
            nerf = self.nerf.module
        else:
            nerf = self.nerf

        for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
            result_batch, _ = render_rays(nerf=nerf, bg_nerf=None,
                                            rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                            image_indices=None,
                                            hparams=self.hparams,
                                            sphere_center=None,
                                            sphere_radius=None,
                                            get_depth=True,
                                            get_depth_variance=False,
                                            get_bg_fg_rgb=False)

            for key, value in result_batch.items():
                if key not in results:
                    results[key] = []

                results[key].append(value.cpu())

        for key, value in results.items():
            results[key] = torch.cat(value)

        return results


    def render_image_nerf_mip(self, rays, radii) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
        radii = radii.view(-1, 1).to(self.device, non_blocking=True)  # (H*W, 1)
        image_indices = None
        results = {}

        if 'RANK' in os.environ:
            nerf = self.nerf.module
        else:
            nerf = self.nerf

        for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
            result_batch, _ = render_rays_mip(
                nerf=nerf,
                rays=rays[i:i + self.hparams.image_pixel_batch_size],
                radii=radii[i:i + self.hparams.image_pixel_batch_size],
                image_indices=None,
                hparams=self.hparams,
                get_depth=True,
                get_depth_variance=False)

            for key, value in result_batch.items():
                if key not in results:
                    results[key] = []

                results[key].append(value.cpu())

        for key, value in results.items():
            results[key] = torch.cat(value)

        return results


    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor) -> Image:
        depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
        images = (rgbs * 255, result_rgbs * 255, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))

    
    @staticmethod
    def _create_result_image_nerf(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor, colormap=cv2.COLORMAP_INFERNO) -> Image:
        depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu(), colormap=colormap)
        images = (rgbs.numpy() * 255, result_rgbs.numpy() * 255, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8)), [Image.fromarray(i.astype(np.uint8)) for i in images]

    @staticmethod
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        dataset_path = Path(self.hparams.dataset_path)

        train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
        train_paths = [train_path_candidates[i] for i in
                       range(0, len(train_path_candidates), self.hparams.train_every)]

        val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))
        train_paths += val_paths
        train_paths.sort(key=lambda x: x.name)
        val_paths_set = set(val_paths)

        image_indices = {}
        for i, train_path in enumerate(train_paths):
            image_indices[train_path.name] = i

        train_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set) for x
            in train_paths]
        val_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, True) for x in val_paths]

        return train_items, val_items

    def _get_metadata_item(self, metadata_path: Path, image_index: int, scale_factor: int,
                           is_val: bool) -> ImageMetadata:
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        assert image_path.exists()

        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        assert metadata['W'] % scale_factor == 0
        assert metadata['H'] % scale_factor == 0

        dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_log('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
            mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
        elif dataset_mask.exists():
            if image_index == 0:
                main_log('Using dataset mask path: {}'.format(dataset_mask.parent))
            mask_path = dataset_mask
        else:
            mask_path = None
            main_log('No mask path')

        return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                             intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir() if x.name.isdigit()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path

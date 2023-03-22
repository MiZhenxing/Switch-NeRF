from enum import EnumMeta
import math
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from npy_append_array import NpyAppendArray
from torch.utils.data import Dataset

from switch_nerf.misc_utils import main_tqdm, main_log, process_log
import random
from functools import partial
import tensorflow as tf
import cv2
import json

RAY_CHUNK_SIZE = 64 * 1024

# https://github.com/dvlab-research/BlockNeRFPytorch/blob/main/data_preprocess/fetch_data_from_tf_record.py
def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        {
            "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
            "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # 0~12
            "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "ray_origins": tf.io.VarLenFeature(tf.float32),
            "ray_dirs": tf.io.VarLenFeature(tf.float32),
            "intrinsics": tf.io.VarLenFeature(tf.float32),
        }
    )

def decode_mask_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        {
            "image_hash": tf.io.FixedLenFeature([], dtype=tf.int64),
            "cam_idx": tf.io.FixedLenFeature([], dtype=tf.int64),  # 0~12
            "equivalent_exposure": tf.io.FixedLenFeature([], dtype=tf.float32),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "ray_origins": tf.io.VarLenFeature(tf.float32),
            "ray_dirs": tf.io.VarLenFeature(tf.float32),
            "intrinsics": tf.io.VarLenFeature(tf.float32),
            "mask": tf.io.VarLenFeature(tf.int64)
        }
    )

class BlockFilesystemDataset(Dataset):

    def __init__(self, data_path: Path, near: float, far: float, 
                device: torch.device, scale_factor: int,
                list_path, id_map_path, chunk_paths: List[Path], num_chunks: int,
                disk_flush_size: int, shuffle_chunk=False):
        super(BlockFilesystemDataset, self).__init__()
        self._device = device
        self._near = near
        self._far = far

        self._tfrecord_paths = self._get_tfrecord_paths(data_path, list_path=list_path)
        with open(id_map_path) as f:
            self._image_hash_id_map = json.load(f)
        
        append_arrays = self._check_existing_paths(chunk_paths)
        if append_arrays is not None:
            main_log('Reusing {} chunks from previous run'.format(len(append_arrays[0])))
            self._rgb_arrays = append_arrays[0]
            self._ray_arrays = append_arrays[1]
            self._img_arrays = append_arrays[2]
        else:
            self._rgb_arrays = []
            self._ray_arrays = []
            self._img_arrays = []
            self._write_chunks(chunk_paths, num_chunks, disk_flush_size)

        self._rgb_arrays.sort(key=lambda x: x.name)
        self._ray_arrays.sort(key=lambda x: x.name)
        self._img_arrays.sort(key=lambda x: x.name)


        if shuffle_chunk:
            process_log("Shuffle chunk")
            chunk_indices = list(range(len(self._rgb_arrays)))
            random.shuffle(chunk_indices)
            process_log(f"Using chunk order:")
            process_log(chunk_indices)
            self._chunk_index = cycle(chunk_indices)
        else:
            self._chunk_index = cycle(range(len(self._rgb_arrays)))
        self._loaded_rgbs = None
        self._loaded_rays = None
        self._loaded_image_indices = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
        self._chosen = None

    def load_chunk(self) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        self._chosen = chosen
        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
    
    def load_chunk_chosen(self, chosen_) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        main_log(f"Loaded {chosen}")
        while str(chosen) != chosen_:
            next_chunk_index = next(self._chunk_index)
            chosen = self._rgb_arrays[next_chunk_index]
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        main_log(f"Loaded {chosen}")
        self._chosen = chosen

        next_chunk_index = next(self._chunk_index)
        self._chunk_future = self._chunk_load_executor.submit(partial(self._load_chunk_inner, next_chunk_index))

    def get_state(self) -> str:
        return self._chosen

    def set_state(self, chosen: str) -> None:
        # while self._chosen != chosen:
        self.load_chunk_chosen(chosen)

    def __len__(self) -> int:
        return self._loaded_rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._loaded_rgbs[idx],
            'rays': self._loaded_rays[idx][1:],
            'radii': self._loaded_rays[idx][0:1],
            'image_indices': self._loaded_image_indices[idx]
        }

    def _load_chunk_inner(self, next_index) -> Tuple[
        str, torch.FloatTensor, torch.FloatTensor, torch.ShortTensor]:
        if 'RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        # next_index = next(self._chunk_index)
        chosen = self._rgb_arrays[next_index]
        loaded_rgbs = torch.FloatTensor(np.load(chosen)) / 255.
        loaded_img_indices = torch.ShortTensor(np.load(self._img_arrays[next_index]))
        loaded_rays = torch.FloatTensor(np.load(self._ray_arrays[next_index]))
        near = torch.tensor(self._near).expand([*loaded_rays.shape[0:-1], 1])
        far = torch.tensor(self._far).expand([*loaded_rays.shape[0:-1], 1])
        loaded_rays = torch.cat([loaded_rays, near, far], dim=-1)

        return str(chosen), loaded_rgbs, loaded_rays, loaded_img_indices
        
    def _load_tfrecord_inner(self, next_index) -> Tuple[
        str, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # if 'RANK' in os.environ:
        #     torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        # next_index = next(self._chunk_index)
        chosen = self._tfrecord_paths[next_index]
        tfrecord_data_dicts = handle_one_record(chosen, hash_id_map=self._image_hash_id_map[os.path.basename(chosen)])

        if "validation" in str(chosen):
            # keep left half
            loaded_rgbs = [tfrecord_data_dict["image"][:, :tfrecord_data_dict["width"] // 2].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_origins = [tfrecord_data_dict["ray_origins"][:, :tfrecord_data_dict["width"] // 2].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_dirs = [tfrecord_data_dict["ray_dirs"][:, :tfrecord_data_dict["width"] // 2].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_radii = [compute_radii(tfrecord_data_dict["ray_dirs"])[:, :tfrecord_data_dict["width"] // 2].reshape([-1, 1]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_img_indices = [tfrecord_data_dict["image_ids"][:, :tfrecord_data_dict["width"] // 2].reshape([-1]) for tfrecord_data_dict in tfrecord_data_dicts]
        else:
            loaded_rgbs = [tfrecord_data_dict["image"].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_origins = [tfrecord_data_dict["ray_origins"].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_dirs = [tfrecord_data_dict["ray_dirs"].reshape([-1, 3]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_ray_radii = [compute_radii(tfrecord_data_dict["ray_dirs"]).reshape([-1, 1]) for tfrecord_data_dict in tfrecord_data_dicts]
            loaded_img_indices = [tfrecord_data_dict["image_ids"].reshape([-1]) for tfrecord_data_dict in tfrecord_data_dicts]

        loaded_rgbs = torch.cat(loaded_rgbs, dim=0) # uint8
        loaded_ray_origins = torch.cat(loaded_ray_origins, dim=0)
        loaded_ray_dirs = torch.cat(loaded_ray_dirs, dim=0)
        loaded_ray_radii = torch.cat(loaded_ray_radii, dim=0)
        loaded_img_indices = torch.cat(loaded_img_indices, dim=0).to(torch.short)

        # near = torch.tensor(self._near).expand([*loaded_ray_origins.shape[0:-1], 1])
        # far = torch.tensor(self._far).expand([*loaded_ray_origins.shape[0:-1], 1])
        # loaded_rays = torch.cat([loaded_ray_origins, loaded_ray_dirs, near, far], dim=-1)
        loaded_rays = torch.cat([loaded_ray_radii, loaded_ray_origins, loaded_ray_dirs], dim=-1)

        return str(chosen), loaded_rgbs, loaded_rays, loaded_img_indices

    def _get_tfrecord_paths(self, data_path, list_path) -> \
            Optional[List[Path]]:
        
        data_path = Path(data_path)
        with open(list_path) as f:
            lines = f.readlines()
            tfrecord_paths = [data_path / line.rstrip() for line in lines]
        
        return tfrecord_paths

    def _write_chunks(self, chunk_paths: List[Path], num_chunks: int, disk_flush_size: int) -> None:
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        path_frees = []
        total_free = 0

        for chunk_path in chunk_paths:
            (chunk_path / 'rgb-chunks').mkdir(parents=True)
            (chunk_path / 'ray-chunks').mkdir(parents=True)
            (chunk_path / 'img-chunks').mkdir(parents=True)

            _, _, free = shutil.disk_usage(chunk_path)
            total_free += free
            path_frees.append(free)

        rgb_append_arrays = []
        ray_append_arrays = []
        img_append_arrays = []

        index = 0
        for chunk_path, path_free in zip(chunk_paths, path_frees):
            allocated = int(path_free / total_free * num_chunks)
            main_log('Allocating {} chunks to dataset path {}'.format(allocated, chunk_path))
            for j in range(allocated):
                rgb_array_path = chunk_path / 'rgb-chunks' / '{}.npy'.format(index)
                self._rgb_arrays.append(rgb_array_path)
                rgb_append_arrays.append(NpyAppendArray(str(rgb_array_path)))

                ray_array_path = chunk_path / 'ray-chunks' / '{}.npy'.format(index)
                self._ray_arrays.append(ray_array_path)
                ray_append_arrays.append(NpyAppendArray(str(ray_array_path)))

                img_array_path = chunk_path / 'img-chunks' / '{}.npy'.format(index)
                self._img_arrays.append(img_array_path)
                img_append_arrays.append(NpyAppendArray(str(img_array_path)))
                index += 1
        main_log('{} chunks allocated'.format(index))

        write_futures = []
        rgbs = []
        rays = []
        indices = []
        in_memory_count = 0
        tfrecord_num = len(self._tfrecord_paths)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for tfrecord_id in main_tqdm(list(range(tfrecord_num))):
                _, image_rgbs, image_rays, image_indices = self._load_tfrecord_inner(tfrecord_id)
                rgbs.append(image_rgbs)
                indices.append(image_indices)
                rays.append(image_rays)
                in_memory_count += len(image_rgbs)

                if in_memory_count >= disk_flush_size:
                    for write_future in write_futures:
                        write_future.result()

                    write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                        rgb_append_arrays, ray_append_arrays, img_append_arrays)

                    rgbs = []
                    rays = []
                    indices = []
                    in_memory_count = 0

            for write_future in write_futures:
                write_future.result()

            if in_memory_count > 0:
                write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                    rgb_append_arrays, ray_append_arrays, img_append_arrays)

                for write_future in write_futures:
                    write_future.result()

        for source in [rgb_append_arrays, ray_append_arrays, img_append_arrays]:
            for append_array in source:
                append_array.close()

        main_log('Finished writing chunks to dataset paths')


    def _check_existing_paths(self, chunk_paths: List[Path]) -> \
            Optional[Tuple[List[Path], List[Path], List[Path]]]:
        rgb_arrays = []
        ray_arrays = []
        img_arrays = []

        num_exist = 0
        for chunk_path in chunk_paths:
            if chunk_path.exists():

                for child in list((chunk_path / 'rgb-chunks').iterdir()):
                    rgb_arrays.append(child)
                    ray_arrays.append(child.parent.parent / 'ray-chunks' / child.name)
                    img_arrays.append(child.parent.parent / 'img-chunks' / child.name)
                num_exist += 1

        if num_exist > 0:
            assert num_exist == len(chunk_paths)
            return rgb_arrays, ray_arrays, img_arrays
        else:
            return None

    @staticmethod
    def _write_to_disk(executor: ThreadPoolExecutor, rgbs: torch.Tensor, rays: torch.FloatTensor,
                       image_indices: torch.Tensor, rgb_append_arrays, ray_append_arrays, img_append_arrays):
        indices = torch.randperm(rgbs.shape[0])
        num_chunks = len(rgb_append_arrays)
        chunk_size = math.ceil(rgbs.shape[0] / num_chunks)

        futures = []

        def append(index: int) -> None:
            rgb_append_arrays[index].append(rgbs[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            ray_append_arrays[index].append(rays[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            img_append_arrays[index].append(image_indices[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())

        for i in range(num_chunks):
            future = executor.submit(append, i)
            futures.append(future)

        return futures

def compute_radii(rays_d):
    # rays_d: H, W, 3
    dx = torch.sqrt(
        torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[-2:-1, :]], 0)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii # H, W, 1

# https://github.com/dvlab-research/BlockNeRFPytorch/blob/main/data_preprocess/fetch_data_from_tf_record.py
def handle_one_record(tfrecord, hash_id_map, load_mask=False):
    tf.config.set_visible_devices([], 'GPU')
    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="GZIP",
    )
    if load_mask:
        dataset_map = dataset.map(decode_mask_fn)
    else:
        dataset_map = dataset.map(decode_fn)
    tfrecord_data_dicts = []

    for idx, batch in enumerate(dataset_map):
        # main_log(f"\tLoading the {idx + 1}th image...")
        image_hash = str(int(batch["image_hash"]))
        imagestr = batch["image"]
        image = tf.io.decode_png(imagestr, channels=0, dtype=tf.dtypes.uint8, name=None)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cam_idx = int(batch["cam_idx"])
        equivalent_exposure = float(batch["equivalent_exposure"])
        height, width = int(batch["height"]), int(batch["width"])
        intrinsics = tf.sparse.to_dense(batch["intrinsics"]).numpy()

        ray_origins = tf.sparse.to_dense(batch["ray_origins"]).numpy().reshape(height, width, 3)
        ray_dirs = tf.sparse.to_dense(batch["ray_dirs"]).numpy().reshape(height, width, 3)
        if load_mask:
            # 0 means not move, valid, 1 means move, invalid
            mask = tf.sparse.to_dense(batch["mask"]).numpy().reshape(height, width, 1)

        tfrecord_data_dict = {
            "image_hash": image_hash,
            "cam_idx": cam_idx,
            "equivalent_exposure": equivalent_exposure,
            "height": height,
            "width": width,
            "intrinsics": torch.tensor(intrinsics),
            "image": torch.tensor(image, dtype=torch.uint8),
            "ray_origins": torch.tensor(ray_origins),
            "ray_dirs": torch.tensor(ray_dirs),
            "image_ids": torch.tensor(hash_id_map[image_hash]).expand(ray_origins.shape[0:2])
        }
        if load_mask:
            tfrecord_data_dict["mask"] = torch.tensor(mask, dtype=torch.float32)
        tfrecord_data_dicts.append(tfrecord_data_dict)

    return tfrecord_data_dicts

def load_tfrecord(tfrecord_path, hash_id_map, near, far, load_mask=False):
    tfrecord_data_dicts = handle_one_record(tfrecord_path, hash_id_map=hash_id_map, load_mask=load_mask)

    for tfrecord_data_dict in tfrecord_data_dicts:
        loaded_rgbs = tfrecord_data_dict["image"].float() / 255.
        loaded_ray_origins = tfrecord_data_dict["ray_origins"]
        loaded_ray_dirs = tfrecord_data_dict["ray_dirs"]
        loaded_img_indices = tfrecord_data_dict["image_ids"].to(torch.short)
        loaded_ray_radii = compute_radii(loaded_ray_dirs)

        tmp_near = torch.tensor(near).expand([*loaded_ray_origins.shape[0:-1], 1])
        tmp_far = torch.tensor(far).expand([*loaded_ray_origins.shape[0:-1], 1])
        loaded_rays = torch.cat([loaded_ray_origins, loaded_ray_dirs, tmp_near, tmp_far], dim=-1)

        tfrecord_data_dict['rgbs'] = loaded_rgbs
        tfrecord_data_dict['rays'] = loaded_rays
        tfrecord_data_dict['radii'] = loaded_ray_radii
        tfrecord_data_dict['image_indices'] = loaded_img_indices

    return tfrecord_data_dicts

def test_BlockFilesystemDataset():
    import time

    data_path = "/ssddata/datasets/nerf/block-nerf/Mission_Bay/v1.0"
    near = 0.01
    far = 10.0
    device = "cpu"
    scale_factor = 1.0
    list_path = "mega_nerf/datasets/lists/block_nerf_train_val_dummy.txt"
    id_map_path = "mega_nerf/datasets/lists/block_nerf_id_map.json"
    shuffle_tfrecord = True

    chunk_paths = [Path("/ssddata/datasets/nerf/mega-nerf/bay_chunk_debug_1")]
    num_chunks = 200
    disk_flush_size = 10000000
    shuffle_chunk = False

    dataset = BlockFilesystemDataset(data_path=data_path, near=near, far=far, 
        device=device, scale_factor=scale_factor,
        list_path=list_path, id_map_path=id_map_path,
        chunk_paths=chunk_paths, num_chunks=num_chunks, 
        disk_flush_size=disk_flush_size,
        shuffle_chunk=shuffle_chunk)

    main_log(f"Loading tfrecord {dataset.get_state()}")
    chunk_time = time.time()
    dataset.load_chunk()
    chunk_time = time.time() - chunk_time
    main_log(f"Chunk {dataset.get_state()} loaded by {chunk_time:.2f} s")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=1, pin_memory=False)
    
    data_sample_time = time.time()
    for dataset_index, item in enumerate(data_loader):
        data_sample_time = time.time() - data_sample_time
        with torch.cuda.amp.autocast(enabled=True):
            image_indices = item['image_indices']
            rgbs = item['rgbs']
            rays = item['rays']
            radii = item['radii']
            pass

        print(data_sample_time)
        data_sample_time = time.time()

def test_handle_one_record():
    tfrecord = "/data2/datasets/nerf/block-nerf/Mission_Bay/v1.0/waymo_block_nerf_mission_bay_validation.tfrecord-00197-of-00373"
    id_map_path = "mega_nerf/datasets/lists/block_nerf_id_map.json"
    with open(id_map_path) as f:
        image_hash_id_map = json.load(f)
    
    hash_id_map = image_hash_id_map[os.path.basename(tfrecord)]
    handle_one_record(tfrecord, hash_id_map, load_mask=True)


if __name__ == "__main__":
    test_handle_one_record()
    # test_BlockFilesystemDataset2()
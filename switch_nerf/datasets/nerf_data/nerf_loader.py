import torch
from torch.utils.data import Dataset
from .load_llff import load_llff_data
from .load_deepvoxels import load_dv_data
from .load_blender import load_blender_data
from .load_LINEMOD import load_LINEMOD_data
from .load_bungee import load_bungee_multiscale_data, get_bungee_nearfar_radii
from .ray_utils import get_rays, ndc_rays

import numpy as np
import cv2
from switch_nerf.misc_utils import main_log

class NeRFDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        # self.split = split
        self.K = None
        if args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            main_log(f'Loaded llff {images.shape}, {render_poses.shape}, {hwf}, {args.datadir}')
            if not isinstance(i_test, list):
                i_test = [i_test]

            if args.llffhold > 0:
                main_log(f'Auto LLFF holdout, {args.llffhold}')
                i_test = np.arange(images.shape[0])[::args.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

            main_log('DEFINING BOUNDS')
            if args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.
                
            else:
                near = 0.
                far = 1.
            main_log(f'NEAR {near} FAR {far}')

        elif args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
            main_log(f'Loaded blender {images.shape}, {render_poses.shape}, {hwf}, {args.datadir}')
            i_train, i_val, i_test = i_split

            near = 2.
            far = 6.

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        elif args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
            main_log(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            main_log(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        elif args.dataset_type == 'deepvoxels':

            images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                    basedir=args.datadir,
                                                                    testskip=args.testskip)

            main_log(f'Loaded deepvoxels {images.shape}, {render_poses.shape}, {hwf}, {args.datadir}')
            i_train, i_val, i_test = i_split

            hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
            near = hemi_R-1.
            far = hemi_R+1.
        
        elif args.dataset_type == 'bungee':
            images, poses, scene_scaling_factor, scene_origin, scale_split = load_bungee_multiscale_data(args.datadir, args.factor)
            self.scene_origin = scene_origin
            self.scale_split = scale_split
            self.scene_scaling_factor = scene_scaling_factor
            # if args.llffhold > 0:
            print('Auto holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test)])
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            render_poses = poses # no use
            near = 0. # no use
            far = 1. # no use
        else:
            main_log(f'Unknown dataset type {args.dataset_type}, exiting')
            raise NotImplementedError
        
        self.poses = torch.tensor(poses)
        self.images = images
        # self.bds = bds
        self.render_poses = torch.tensor(render_poses)
        self.i_test = i_test
        self.i_train = i_train
        self.i_val = i_val

        self.near = near
        self.far = far

        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if self.K is None:
            self.K = torch.tensor([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
        
        self.hwf = hwf
        self.H = H
        self.W = W

        if hasattr(args, "scale_factor") and args.scale_factor > 1.0:
            assert (self.H % args.scale_factor == 0.0 and self.W % args.scale_factor == 0.0)
            self.H = int(self.H // args.scale_factor)
            self.W = int(self.W // args.scale_factor)
            H = int(H // args.scale_factor)
            W = int(W // args.scale_factor)

            self.hwf = [self.H, self.W, focal / args.scale_factor]
            
            # self.K[:2, 2] = self.K[:2, 2] / args.scale_factor
            # self.K[0, 0] = self.K[0, 0] / args.scale_factor
            # self.K[1, 1] = self.K[1, 1] / args.scale_factor
            self.K[:2, :] = self.K[:2, :] / args.scale_factor

            tmp_images = []
            for i, img in enumerate(self.images):
                tmp_images.append(cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA))
                # tmp_images.append(torch.nn.functional.interpolate(img, [self.H, self.W], mode='area'))
            self.images = np.array(tmp_images) # actually stack

        self.images = torch.tensor(self.images)
        
        main_log('get rays')
        # rays = torch.stack([torch.cat(get_rays(self.H, self.W, self.K, p), -1) for p in self.poses[:,:3,:4]], 0) # [N, H, W, 6]
        rays = []
        for p in self.poses[:,:3,:4]:
            rays_o, rays_d = get_rays(self.H, self.W, self.K, p)
            if not args.no_ndc:
                rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)
            else:
                rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays.append(torch.cat([rays_o, rays_d], -1))
        rays = torch.stack(rays, 0) # [N, H, W, 6]

        main_log('done, concats')

        # rays = torch.permute(rays, [0,2,3,1,4]) # [N, H, W, 2, 3]
        # rays = torch.reshape(rays, list(rays.shape[0:3]) + [6]) # [N, H, W, 6]
        if args.dataset_type == 'bungee':
            rays, radii = get_bungee_nearfar_radii(rays=rays, scene_scaling_factor=scene_scaling_factor, 
                scene_origin=scene_origin, ray_nearfar=args.bungee_ray_nearfar)
            self.radii = radii # N, H, W, 1
        else:
            rays = torch.cat([rays, self.near*torch.ones_like(rays[..., :1]), self.far*torch.ones_like(rays[..., :1])], -1) # [N, H, W, 8]
            self.radii = None
        self.rays = rays.to(torch.float32)
        if self.radii is not None:
            self.radii = self.radii.to(torch.float32)

        self.rgbs = self.images # N, H, W, 3

        self.rays_train = self.rays[i_train]
        self.rays_train = torch.reshape(self.rays_train, [-1,8])
        self.rgbs_train = self.rgbs[i_train]
        self.rgbs_train = torch.reshape(self.rgbs_train, [-1,3])
        if self.radii is not None:
            self.radii_train = self.radii[i_train]
            self.radii_train = torch.reshape(self.radii_train, [-1,1])

        self.rays_val = self.rays[i_val] # N, H, W, 8
        self.rgbs_val = self.rgbs[i_val] # N, H, W, 3
        if self.radii is not None:
            self.radii_val = self.radii[i_val] # N, H, W, 1

        self.rays_test = self.rays[i_test] # N, H, W, 8
        self.rgbs_test = self.rgbs[i_test] # N, H, W, 3
        if self.radii is not None:
            self.radii_test = self.radii[i_test] # N, H, W, 1

        self.args = args

class NeRFDatasetTrain(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return self.dataset.rays_train.shape[0]
    
    def __getitem__(self, idx):
        sample = {
            'rays': self.dataset.rays_train[idx],
            'rgbs': self.dataset.rgbs_train[idx]}
        if self.dataset.args.dataset_type == 'bungee':
            sample["radii"] = self.dataset.radii_train[idx]
        return sample

class NeRFDatasetVal(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset.i_val)
    
    def __getitem__(self, idx):
        img_i = self.dataset.i_val[idx]
        sample = {
            'rays': self.dataset.rays_val[idx],
            'rgbs': self.dataset.rgbs_val[idx],
            "img_i": img_i}
        if self.dataset.args.dataset_type == 'bungee':
            sample["radii"] = self.dataset.radii_val[idx]
        return sample

class NeRFDatasetTest(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset.i_test)
    
    def __getitem__(self, idx):
        img_i = self.dataset.i_test[idx]
        sample = {
            'rays': self.dataset.rays_test[idx],
            'rgbs': self.dataset.rgbs_test[idx],
            "img_i": img_i}
        if self.dataset.args.dataset_type == 'bungee':
            sample["radii"] = self.dataset.radii_test[idx]
        return sample
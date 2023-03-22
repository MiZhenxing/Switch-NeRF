import numpy as np
import os
import json
import cv2
import imageio
import torch

def _load_google_data(basedir, factor=None):
    img_basedir = basedir
    img_folder = 'images'
    imgdir = os.path.join(img_basedir, img_folder)
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('jpeg')]
    sh = np.array(cv2.imread(imgfiles[0]).shape)
    imgs = []
    for f in imgfiles:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im.shape[-1] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        im = cv2.resize(im, (sh[1]//factor, sh[0]//factor), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255
        imgs.append(im)
    imgs = np.stack(imgs, -1) 
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)

    data = json.load(open(os.path.join(basedir, 'poses_enu.json')))
    poses = np.array(data['poses'])[:, :-2].reshape([-1, 3, 5])
    poses[:, :2, 4] = np.array(sh[:2]//factor).reshape([1, 2])
    poses[:, 2, 4] = poses[:,2, 4] * 1./factor 

    scene_scaling_factor = data['scene_scale']
    scene_origin = np.array(data['scene_origin'])
    scale_split = data['scale_split']

    return imgs, poses, scene_scaling_factor, scene_origin, scale_split

def load_bungee_multiscale_data(basedir, factor=3):
    imgs, poses, scene_scaling_factor, scene_origin, scale_split = _load_google_data(basedir, factor=factor)
    print('Loaded image data shape:', imgs.shape, ' hwf:', poses[0,:,-1])
    return imgs, poses, scene_scaling_factor, scene_origin, scale_split

def get_bungee_nearfar_radii(rays, scene_scaling_factor, scene_origin, ray_nearfar):
    rays_o = rays[..., 0:3]
    rays_d = rays[..., 3:6]
    # rays_shape = rays.shape[0:-1]
        
    if ray_nearfar == 'sphere': ## treats earth as a sphere and computes the intersection of a ray and a sphere
        globe_center = torch.tensor(np.array(scene_origin) * scene_scaling_factor).float()
       
        # 6371011 is earth radius, 250 is the assumed height limitation of buildings in the scene
        earth_radius = 6371011 * scene_scaling_factor
        earth_radius_plus_bldg = (6371011+250) * scene_scaling_factor
        
        ## intersect with building upper limit sphere
        delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius_plus_bldg)**2)
        d_near = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
        rays_start = rays_o + (d_near[...,None]*rays_d)
        
        ## intersect with earth
        delta = (2*torch.sum((rays_o-globe_center) * rays_d, dim=-1))**2 - 4*torch.norm(rays_d, dim=-1)**2 * (torch.norm((rays_o-globe_center), dim=-1)**2 - (earth_radius)**2)
        d_far = (-2*torch.sum((rays_o-globe_center) * rays_d, dim=-1) - delta**0.5) / (2*torch.norm(rays_d, dim=-1)**2)
        rays_end = rays_o + (d_far[...,None]*rays_d)

        ## compute near and far for each ray
        new_near = torch.norm(rays_o - rays_start, dim=-1, keepdim=True)
        near = new_near * 0.9
        
        new_far = torch.norm(rays_o - rays_end, dim=-1, keepdim=True)
        far = new_far * 1.1

    elif ray_nearfar == 'flat': ## treats earth as a flat surface and computes the intersection of a ray and a plane
        normal = torch.tensor([0, 0, 1]).to(rays_o) * scene_scaling_factor
        p0_far = torch.tensor([0, 0, 0]).to(rays_o) * scene_scaling_factor
        p0_near = torch.tensor([0, 0, 250]).to(rays_o) * scene_scaling_factor

        near = (p0_near - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        far = (p0_far - rays_o * normal).sum(-1) / (rays_d * normal).sum(-1)
        near = near.clamp(min=1e-6)
        near, far = near.unsqueeze(-1), far.unsqueeze(-1)
    
    new_rays = torch.cat([rays, near, far], dim=-1)
    # new_rays = new_rays.reshape(list(rays_shape) + [8])
    # rays_d: N, H, W, 3
    dx = torch.sqrt(
        torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return new_rays, radii
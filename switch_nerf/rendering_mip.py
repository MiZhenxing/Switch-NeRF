import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from switch_nerf.spherical_harmonics import eval_sh
import numpy as np

TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse'}

def mip_cast_rays(origin, direction, radius, t): 
    t0, t1 = t[..., :-1], t[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2)) / (3*c**2 + d**2)**2)
    r_var = radius**2 * ((c**2)/4 + (5/12) * d**2 - (4/15) * (d**4) / (3*c**2 + d**2))
    mean = origin[...,None,:] + direction[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (direction**2) / torch.sum(direction**2, -1, keepdims=True)
    cov_diag = (t_var[..., None] * (direction**2)[..., None, :] + r_var[..., None] * null_outer_diag[..., None, :])
    
    return mean, cov_diag

# from https://github.com/kakaobrain/NeRF-Factory/blob/ac10296a39d8e5f1940b590ca18e3689e17eadf4/src/model/mipnerf/helper.py#L88
def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], axis=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        axis=-1,
    )

    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=weights.device) * s
        u += torch.rand_like(u) * (s - float_min_eps)
        u = torch.fmin(u, torch.ones_like(u) * (1.0 - float_min_eps))
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


# from https://github.com/openxrlab/xrnerf/blob/c42049bfc99d79204ff3a8f9c99f89eb9ec3f88d/xrnerf/models/networks/utils/mip.py#L7
def sorted_piecewise_constant_pdf1(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins."""
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    device = weights.device
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.tensor(0).to(device), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.minimum(
        torch.tensor(1).to(device), torch.cumsum(pdf[..., :-1], dim=-1))
    cdf = torch.cat([
        torch.zeros(list(cdf.shape[:-1]) + [1]).to(device), cdf,
        torch.ones(list(cdf.shape[:-1]) + [1]).to(device)
    ], -1)

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples) * s

        u = u + torch.rand(list(cdf.shape[:-1]) + [num_samples]) * (
            s - torch.finfo(torch.float32).eps)

        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.tensor(1. - torch.finfo(torch.float32).eps))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps,
                           num_samples)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    u = u.to(device)
    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.

    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]),
                       -2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]),
                       -2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def render_rays(nerf: nn.Module,
                rays: torch.Tensor,
                radii: torch.Tensor,
                image_indices: Optional[torch.Tensor],
                hparams: Namespace,
                get_depth: bool,
                get_depth_variance: bool) -> Tuple[Dict[str, torch.Tensor], bool]:
    N_rays = rays.shape[0]

    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    if image_indices is not None:
        image_indices = image_indices.unsqueeze(-1).unsqueeze(-1)

    perturb = hparams.perturb if nerf.training else 0
    last_delta = 1e10 * torch.ones(N_rays, 1, device=rays.device)
    
    rays_o = rays_o.view(rays_o.shape[0], 1, rays_o.shape[1])
    rays_d = rays_d.view(rays_d.shape[0], 1, rays_d.shape[1])

    # Sample depth points
    z_steps = torch.linspace(0, 1, hparams.coarse_samples, device=rays.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + far * z_steps
  
    z_vals = _expand_and_perturb_z_vals(z_vals, hparams.coarse_samples, perturb, N_rays)

    xyz_coarse = rays_o + rays_d * (.5 * (z_vals[...,1:] + z_vals[...,:-1])).unsqueeze(-1)
    results = _get_results(nerf=nerf,
                           rays_o=rays_o,
                           rays_d=rays_d,
                           radii=radii,
                           image_indices=image_indices,
                           hparams=hparams,
                           xyz_coarse=xyz_coarse,
                           z_vals=z_vals,
                           last_delta=last_delta,
                           get_depth=get_depth,
                           get_depth_variance=get_depth_variance,
                           flip=False,
                           xyz_fine_fn=lambda fine_z_vals: (rays_o + rays_d * fine_z_vals.unsqueeze(-1), None))

    return results, False


def _get_results(nerf: nn.Module,
                 rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 radii: torch.Tensor,
                 image_indices: Optional[torch.Tensor],
                 hparams: Namespace,
                 xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 flip: bool,
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]) \
        -> Dict[str, torch.Tensor]:
    results = {}

    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

    mip_means, mip_cov_diags = mip_cast_rays(rays_o.squeeze(1), rays_d.squeeze(1), radii, z_vals)
    _inference(results=results,
               typ='coarse',
               nerf=nerf,
               rays_d=rays_d,
               mip_means=mip_means,
               mip_cov_diags=mip_cov_diags,
               image_indices=image_indices,
               hparams=hparams,
               xyz=xyz_coarse,
               z_vals=z_vals,
               last_delta=last_delta - last_delta_diff,
               composite_rgb=True,
               get_depth=hparams.fine_samples == 0 and get_depth,
               get_depth_variance=hparams.fine_samples == 0 and get_depth_variance,
               get_weights=hparams.fine_samples > 0,
               flip=flip,
               white_bkgd=hparams.white_bkgd)

    if hparams.fine_samples > 0:  # sample points for fine model
        weights = results[f'weights_coarse']
        weights_pad = torch.cat([
            weights[..., :1],
            weights,
            weights[..., -1:],
        ], axis=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        weights_prime = weights_blur + hparams.weights_resample_padding
        z_samples = sorted_piecewise_constant_pdf1(z_vals, weights_prime, hparams.fine_samples, randomized=hparams.perturb)
        if hparams.stop_level_grad:
            z_samples = z_samples.detach()
        z_vals, _ = torch.sort(z_samples, -1)
        mip_means, mip_cov_diags = mip_cast_rays(rays_o.squeeze(1), rays_d.squeeze(1), radii, z_vals)
        fine_z_vals = z_vals

        del results['weights_coarse']

        xyz_fine, depth_real_fine = xyz_fine_fn((.5 * (fine_z_vals[...,1:] + fine_z_vals[...,:-1])))
        last_delta_diff = torch.zeros_like(last_delta)
        last_delta_diff[last_delta.squeeze() < 1e10, 0] = fine_z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

        _inference(results=results,
                   typ='fine',
                   nerf=nerf,
                   rays_d=rays_d,
                   mip_means=mip_means,
                   mip_cov_diags=mip_cov_diags,
                   image_indices=image_indices,
                   hparams=hparams,
                   xyz=xyz_fine,
                   z_vals=fine_z_vals,
                   last_delta=last_delta - last_delta_diff,
                   composite_rgb=True,
                   get_depth=get_depth,
                   get_depth_variance=get_depth_variance,
                   get_weights=False,
                   flip=flip,
                   white_bkgd=hparams.white_bkgd)

        for key in INTERMEDIATE_KEYS:
            if key in results:
                del results[key]

    return results


def _inference(results: Dict[str, torch.Tensor],
               typ: str,
               nerf: nn.Module,
               rays_d: torch.Tensor,
               mip_means: torch.Tensor,
               mip_cov_diags: torch.Tensor,
               image_indices: Optional[torch.Tensor],
               hparams: Namespace,
               xyz: torch.Tensor,
               z_vals: torch.Tensor,
               last_delta: torch.Tensor,
               composite_rgb: bool,
               get_depth: bool,
               get_depth_variance: bool,
               get_weights: bool,
               flip: bool,
               white_bkgd=False):
    N_rays_ = mip_means.shape[0]
    N_samples_ = mip_means.shape[1]

    if hparams.return_pts:
        results[f"pts_{typ}"] = xyz

    mip_values = torch.cat([mip_means, mip_cov_diags], dim=-1)
    xyz_ = mip_values.view(-1, mip_values.shape[-1])

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    rays_d_ = rays_d.repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])

    if image_indices is not None:
        image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)

    if hparams.pos_dir_dim == 0:
        if hparams.sh_deg is not None:
            rgb_dim = 3 * ((hparams.sh_deg + 1) ** 2)

        for i in range(0, B, hparams.model_chunk_size):
            xyz_chunk = xyz_[i:i + hparams.model_chunk_size]
            if image_indices is not None:
                xyz_chunk = torch.cat([xyz_chunk, image_indices_[i:i + hparams.model_chunk_size]], 1)
            
            if hparams.use_sigma_noise:
                if hparams.sigma_noise_std > 0.0:
                    sigma_noise = torch.randn(len(xyz_chunk), 1, device=xyz_chunk.device) * hparams.sigma_noise_std if nerf.training else None
                else:
                    sigma_noise = None
            else:
                sigma_noise = None

            if hparams.use_cascade:
                model_chunk = nerf(typ == 'coarse', xyz_chunk, sigma_noise=sigma_noise)
            else:
                model_chunk = nerf(xyz_chunk, sigma_noise=sigma_noise)

            if hparams.sh_deg is not None:
                rgb = torch.sigmoid(
                    eval_sh(hparams.sh_deg, model_chunk[:, :rgb_dim].view(-1, 3, (hparams.sh_deg + 1) ** 2),
                            rays_d_[i:i + hparams.model_chunk_size]))

                out_chunks += [torch.cat([rgb, model_chunk[:, rgb_dim:]], -1)]
            else:
                out_chunks += [model_chunk]
    else:
        # (N_rays*N_samples_, embed_dir_channels)
        for i in range(0, B, hparams.model_chunk_size):
            xyz_chunk = xyz_[i:i + hparams.model_chunk_size]

            if image_indices is not None:
                xyz_chunk = torch.cat([xyz_chunk,
                                       rays_d_[i:i + hparams.model_chunk_size],
                                       image_indices_[i:i + hparams.model_chunk_size]], 1)
            else:
                xyz_chunk = torch.cat([xyz_chunk, rays_d_[i:i + hparams.model_chunk_size]], 1)

            if hparams.use_sigma_noise:
                if hparams.sigma_noise_std > 0.0:
                    sigma_noise = torch.randn(len(xyz_chunk), 1, device=xyz_chunk.device) * hparams.sigma_noise_std if nerf.training else None
                else:
                    sigma_noise = None
            else:
                sigma_noise = None

            if hparams.use_cascade:
                model_chunk = nerf(typ == 'coarse', xyz_chunk, sigma_noise=sigma_noise)
            else:
                model_chunk = nerf(xyz_chunk, sigma_noise=sigma_noise)

            out_chunks += [model_chunk]
    
    if (hparams.use_moe or hparams.bg_use_moe) and isinstance(out_chunks[0], dict):
        assert "extras" in out_chunks[0]
        assert "moe_loss" in out_chunks[0]["extras"]
        gate_loss = [i["extras"]["moe_loss"] for i in out_chunks]
        results[f'gate_loss_{typ}'] = torch.cat(gate_loss, 0)

        if hparams.moe_return_gates:
            moe_gates = [torch.stack(i["extras"]["moe_gates"], dim=1) for i in out_chunks]
            moe_gates = torch.cat(moe_gates, 0) # points, layer_num, top k
            results[f'moe_gates_{typ}'] = moe_gates.view(N_rays_, N_samples_, moe_gates.shape[1], moe_gates.shape[2])

        if hparams.use_load_importance_loss and hparams.compute_balance_loss:
            balance_loss = [i["extras"]["balance_loss"] for i in out_chunks]
            results[f'balance_loss_{typ}'] = torch.cat(balance_loss, 0)
        
        out = [i["outputs"] for i in out_chunks]
    elif isinstance(out_chunks[0], dict):
        out = [i["outputs"] for i in out_chunks]
    else:
        # out = [i["outputs"] for i in out_chunks]
        out = out_chunks
    
    out = torch.cat(out, 0)
    out = out.view(N_rays_, N_samples_, out.shape[-1])

    rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples_)

    if hparams.rgb_padding is not None:
        rgbs = rgbs * (1 + 2 * hparams.rgb_padding) - hparams.rgb_padding

    z_vals = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)

    deltas = torch.cat([deltas, last_delta], -1)  # (N_rays, N_samples_)
    alphas = 1 - torch.exp(-deltas * sigmas)  # (N_rays, N_samples_)
    
    if hparams.return_pts_rgb:
        results[f"pts_rgb_{typ}"] = rgbs
    if hparams.return_pts_alpha:
        results[f"pts_alpha_{typ}"] = alphas

    T = torch.cumprod(1 - alphas + 1e-8, -1)

    T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]

    weights = alphas * T  # (N_rays, N_samples_)

    if get_weights:
        results[f'weights_{typ}'] = weights

    if composite_rgb:
        results[f'rgb_{typ}'] = (weights.unsqueeze(-1) * rgbs).sum(dim=1)  # n1 n2 c -> n1 c
        if white_bkgd:
            acc_map = torch.sum(weights, -1)
            results[f'rgb_{typ}'] = results[f'rgb_{typ}'] + (1.-acc_map[...,None])
    else:
        results[f'zvals_{typ}'] = z_vals
        results[f'raw_rgb_{typ}'] = rgbs
        results[f'raw_sigma_{typ}'] = sigmas

    with torch.no_grad():
        if get_depth or get_depth_variance:
            depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1

        if get_depth:
            results[f'depth_{typ}'] = depth_map

        if get_depth_variance:
            results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                axis=-1)


def _intersect_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                      sphere_radius: torch.Tensor) -> torch.Tensor:
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def _depth2pts_outside(rays_o: torch.Tensor, rays_d: torch.Tensor, depth: torch.Tensor, sphere_center: torch.Tensor,
                       sphere_radius: torch.Tensor, include_xyz_real: bool, cluster_2d: bool):
    '''
    rays_o, rays_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    rays_o_orig = rays_o
    rays_d_orig = rays_d
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p_mid = rays_o + d1.unsqueeze(-1) * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_norm = rays_d.norm(dim=-1)
    ray_d_cos = 1. / ray_d_norm
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = rays_o + (d1 + d2).unsqueeze(-1) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / (torch.norm(rot_axis, dim=-1, keepdim=True) + 1e-8)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-8) * torch.cos(theta) + d1

    if include_xyz_real:
        if cluster_2d:
            pts = torch.cat(
                (rays_o_orig + rays_d_orig * depth_real.unsqueeze(-1), p_sphere_new, depth.unsqueeze(-1)),
                dim=-1)
        else:
            boundary = rays_o_orig + rays_d_orig * (d1 + d2).unsqueeze(-1)
            pts = torch.cat((boundary.repeat(1, p_sphere_new.shape[1], 1), p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    else:
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts, depth_real


def _expand_and_perturb_z_vals(z_vals, samples, perturb, N_rays):
    z_vals = z_vals.expand(N_rays, samples)
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals


def _sample_pdf(bins: torch.Tensor, weights: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        fine_samples: the number of samples to draw from the distribution
        det: deterministic or not
    Outputs:
        samples: the sampled samples
    """
    weights = weights + 1e-8  # prevent division by zero (don't do inplace op!)

    pdf = weights / weights.sum(-1).unsqueeze(-1)  # (N_rays, N_samples_)

    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    return _sample_cdf(bins, cdf, fine_samples, det)


def _sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    N_rays, N_samples_ = cdf.shape

    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, fine_samples, device=bins.device)
        u = u.expand(N_rays, fine_samples)
    else:
        u = torch.rand(N_rays, fine_samples, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1)
    inds_sampled = inds_sampled.view(inds_sampled.shape[0], -1)  # n1 n2 2 -> n1 (n2 2)

    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(cdf_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    bins_g = torch.gather(bins, 1, inds_sampled)
    bins_g = bins_g.view(bins_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-8] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Renderer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
from common import get_rays, sample_pdf, normalize_3d_coordinate

#照这里添加参数
def perturbation(z_vals):
    """
    Add perturbation to sampled depth values on the rays.
    Args:
        z_vals (tensor): sampled depth values on the rays.
    Returns:
        z_vals (tensor): perturbed depth values on the rays.
    """
    # get intervals between samples
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape, device=z_vals.device)

    return lower + (upper - lower) * t_rand

def render_batch_ray(cfg,all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
    """
    Render depth and color for a batch of rays.
    Args:
        all_planes (Tuple): all feature planes.
        decoders (torch.nn.Module): decoders for TSDF and color.
        rays_d (tensor): ray directions.
        rays_o (tensor): ray origins.
        device (torch.device): device to run on.
        truncation (float): truncation threshold.
        gt_depth (tensor): ground truth depth.
    Returns:
        depth_map (tensor): depth map.
        color_map (tensor): color map.
        volume_densities (tensor): volume densities for sampled points.
        z_vals (tensor): sampled depth values on the rays.

    """
    n_stratified = cfg.n_stratified
    n_importance = cfg.n_importance
    n_rays = rays_o.shape[0]#7569

    z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
    near = 0.0
    t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
    t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

    ### pixels with gt depth:
    gt_depth = gt_depth.reshape(-1, 1)#2280
    gt_mask = (gt_depth > 0).squeeze()
    gt_nonezero = gt_depth[gt_mask]

    ## Sampling points around the gt depth (surface)
    gt_depth_surface = gt_nonezero.expand(-1, n_importance)
    z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

    gt_depth_free = gt_nonezero.expand(-1, n_stratified)
    z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

    z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
    if cfg.rendering_perturb:
        z_vals_nonzero = perturbation(z_vals_nonzero)#2280,40
    z_vals[gt_mask] = z_vals_nonzero #gt_mask:2800

    ### pixels without gt depth (importance sampling):
    if not gt_mask.all():
        with torch.no_grad():
            rays_o_uni = rays_o[~gt_mask].detach()
            rays_d_uni = rays_d[~gt_mask].detach()
            det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
            bound = cfg.bound.unsqueeze(0).to(cfg.data_device)
            t = (bound - det_rays_o)/det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

            z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
            if cfg.rendering_perturb:
                z_vals_uni = perturbation(z_vals_uni)
            pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

            pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), cfg.bound)
            sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes)
            sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
            alpha_uni = sdf2alpha(sdf_uni, decoders.beta)
            weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                    , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

            z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
            z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
            z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
            z_vals[~gt_mask] = z_vals_uni

    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

    raw = decoders(pts, all_planes)
    alpha = sdf2alpha(raw[..., -1], decoders.beta)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                            , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

    rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
    rendered_depth = torch.sum(weights * z_vals, -1)

    return rendered_depth, rendered_rgb, raw[..., -1], z_vals

def sdf2alpha(sdf, beta=10):
    """

    """
    return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

def render_img(cfg, all_planes, decoders, c2w, truncation, device, gt_depth=None):
    """
    Renders out depth and color images.
    Args:
        all_planes (Tuple): feature planes
        decoders (torch.nn.Module): decoders for TSDF and color.
        c2w (tensor, 4*4): camera pose.
        truncation (float): truncation distance.
        device (torch.device): device to run on.
        gt_depth (tensor, H*W): ground truth depth image.
    Returns:
        rendered_depth (tensor, H*W): rendered depth image.
        rendered_rgb (tensor, H*W*3): rendered color image.

    """
    with torch.no_grad():
        H =cfg.H
        W = cfg.W
        rays_o, rays_d = get_rays(H, W, cfg.fx, cfg.fy, cfg.cx, cfg.cy,  c2w, device)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        depth_list = []
        color_list = []

        ray_batch_size = cfg.ray_batch_size
        gt_depth = gt_depth.reshape(-1)

        for i in range(0, rays_d.shape[0], ray_batch_size):
            rays_d_batch = rays_d[i:i+ray_batch_size]
            rays_o_batch = rays_o[i:i+ray_batch_size]
            if gt_depth is None:
                ret = render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                            device, truncation, gt_depth=None)
            else:
                gt_depth_batch = gt_depth[i:i+ray_batch_size]
                ret = render_batch_ray(all_planes, decoders, rays_d_batch, rays_o_batch,
                                            device, truncation, gt_depth=gt_depth_batch)

            depth, color, _, _ = ret
            depth_list.append(depth.double())
            color_list.append(color)

        depth = torch.cat(depth_list, dim=0)
        color = torch.cat(color_list, dim=0)

        depth = depth.reshape(H, W)
        color = color.reshape(H, W, 3)

        return depth, color
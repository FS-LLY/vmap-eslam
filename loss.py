import torch
import render_rays
import torch.nn.functional as F
import torch.nn as nn

def step_batch_loss(cfg,sdf, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0):
    """
    apply depth where depth are valid                                       -> mask_depth
    apply depth, color loss on this_obj & unkown_obj == (~other_obj)        -> mask_obj
    apply occupancy/opacity loss on this_obj & other_obj == (~unknown_obj)  -> mask_sem

    output:
    loss for training
    loss_all for per sample, could be used for active sampling, replay buffer
    """
    '''
    vmap:
    alpha = alpha.squeeze(dim=-1)
    color = color.squeeze(dim=-1)

    occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]

    render_depth = render_rays.render(termination, z_vals)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    var = render_rays.render(termination, diff_sq).detach()  # must detach here!
    render_color = render_rays.render(termination[..., None], color, dim=-2)
    render_opacity = torch.sum(termination, dim=-1)     # similar to obj-nerf opacity loss

    # 2D depth loss: only on valid depth & mask
    # [mask_depth & mask_obj]
    # loss_all = torch.zeros_like(render_depth)
    loss_depth_raw = render_rays.render_loss(render_depth, gt_depth, loss="L1", normalise=False) #torch.abs(render - gt)
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    # loss_all += loss_depth
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    # [mask_obj]
    loss_col_raw = render_rays.render_loss(render_color, gt_color, loss="L1", normalise=False)
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    # loss_all += loss_col / 3. * color_scaling
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)

    # 2D occupancy/opacity loss: apply except unknown area
    # [mask_sem]
    # loss_opacity_raw = F.mse_loss(torch.clamp(render_opacity, 0, 1), mask_obj.float().detach()) # encourage other_obj to be empty, while this_obj to be solid
    # print("opacity max ", torch.max(render_opacity.max()))
    # print("opacity min ", torch.max(render_opacity.min()))
    loss_opacity_raw = render_rays.render_loss(render_opacity, mask_obj.float(), loss="L1", normalise=False)
    loss_opacity = torch.mul(loss_opacity_raw, mask_sem)  # but ignore -1 unkown area e.g., mask edges
    # loss_all += loss_opacity * opacity_scaling
    loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem)   # todo var

    # loss for bp
    l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling
    loss = l_batch.sum()
    '''

    mask_obj = sem_labels != 0
    mask_obj = mask_obj.detach()
    mask_sem = sem_labels != 2
    mask_sem = mask_sem.detach()
    mask_depth = (mask_depth > 0)

    sdf = sdf.view(-1,10)
    color = color.view(-1,10,3)
    gt_depth = gt_depth.reshape(gt_depth.shape[0]*gt_depth.shape[1])
    gt_color = gt_color.reshape(gt_color.shape[0]*gt_color.shape[1],3)
    z_vals = z_vals.reshape(z_vals.shape[0]*z_vals.shape[1],10)

    mask_obj = mask_obj.reshape(mask_obj.shape[0]*mask_obj.shape[1])
    mask_sem = mask_sem.reshape(mask_sem.shape[0]*mask_sem.shape[1])
    mask_depth = mask_depth.reshape(mask_depth.shape[0]*mask_depth.shape[1])

    beta = nn.Parameter(10 * torch.ones(1)).to("cuda:0")
    alpha = sdf2alpha(sdf, beta)#beta=10
    #print(alpha.shape)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device="cuda:0")
                                            , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

    render_color = torch.sum(weights[..., None] *color, -2)
    render_depth = torch.sum(weights * z_vals, -1)
   
    
    loss = sdf_losses(cfg,sdf[mask_depth & mask_sem], z_vals[mask_depth & mask_sem], gt_depth[mask_depth & mask_sem])
    loss = loss + 5 * torch.square(render_color[mask_obj] - gt_color[mask_obj]).mean() #w_color = 5
    loss = loss + 1 * torch.square(render_depth[mask_depth & mask_obj] - gt_depth[mask_depth & mask_obj]).mean()


    return loss, None       # return loss, loss_all.detach()
def sdf2alpha(sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

def sdf_losses(cfg, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        truncation = cfg.truncation
        w_sdf_fs = cfg.w_sdf_fs
        w_sdf_center = cfg.w_sdf_center
        w_sdf_tail = cfg.w_sdf_tail

        front_mask = torch.where(z_vals < (gt_depth[:, None] - truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = w_sdf_fs * fs_loss + w_sdf_center * center_loss + w_sdf_tail * tail_loss

        return sdf_losses


def bg_loss(cfg, depth, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0):
    """
    apply depth where depth are valid                                       -> mask_depth
    apply depth, color loss on this_obj & unkown_obj == (~other_obj)        -> mask_obj
    apply occupancy/opacity loss on this_obj & other_obj == (~unknown_obj)  -> mask_sem

    output:
    loss for training
    loss_all for per sample, could be used for active sampling, replay buffer
    """

    # sdf = torch.reshape(sdf, (cfg.n_iter_per_frame, cfg.n_per_optim_bg, -1)) # (24000,40) -> (20,1200,40)
    depth = torch.reshape(depth, (cfg.n_iter_per_frame, cfg.n_per_optim_bg))
    color = torch.reshape(color, (cfg.n_iter_per_frame, cfg.n_per_optim_bg, -1)) # (24000,3) -> (20,1200,3)
    # color = color.unsqueeze(-2).expand(-1, -1, cfg.n_stratified + cfg.n_importance, -1) # (20,1200,3) -> (20,1200,40,3)
    gt_depth = torch.reshape(gt_depth, (cfg.n_iter_per_frame, cfg.n_per_optim_bg))
    gt_color = torch.reshape(gt_color, (cfg.n_iter_per_frame, cfg.n_per_optim_bg, -1))
    sem_labels = torch.reshape(sem_labels, (cfg.n_iter_per_frame, cfg.n_per_optim_bg))
    mask_depth = torch.reshape(mask_depth, (cfg.n_iter_per_frame, cfg.n_per_optim_bg))
    # z_vals = torch.reshape(z_vals, (cfg.n_iter_per_frame, cfg.n_per_optim_bg,-1))

    # beta = nn.Parameter(10 * torch.ones(1)).to("cuda:0")
    # alpha = sdf2alpha(sdf, beta)#beta=10 # 20*120*40
    
    mask_obj = sem_labels != 0 # 0: other object
    mask_obj = mask_obj.detach()
    mask_sem = sem_labels != 2 # 2: unknown object
    mask_sem = mask_sem.detach()
    '''
    occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]

    render_depth = render_rays.render(termination, z_vals)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    var = render_rays.render(termination, diff_sq).detach()  # must detach here!
    render_color = render_rays.render(termination[..., None], color, dim=-2)
    render_opacity = torch.sum(termination, dim=-1)     # similar to obj-nerf opacity loss

    loss_depth_raw = render_rays.render_loss(depth, gt_depth, loss="L1", normalise=False)
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    loss_col_raw = render_rays.render_loss(color, gt_color, loss="L1", normalise=False)
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)

    # 2D occupancy/opacity loss: apply except unknown area
    # [mask_sem]
    loss_opacity_raw = render_rays.render_loss(render_opacity, mask_obj.float(), loss="L1", normalise=False)
    loss_opacity = torch.mul(loss_opacity_raw, mask_sem)  # but ignore -1 unkown area e.g., mask edges
    loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem)   # todo var
    '''

    loss_depth_raw = render_rays.render_loss(depth, gt_depth, loss="L1", normalise=False)
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=None, avg=True, mask=mask_depth & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    loss_col_raw = render_rays.render_loss(color, gt_color, loss="L1", normalise=False)
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)
    # loss for bp
    l_batch = loss_depth + loss_col * color_scaling 
    # + loss_opacity * opacity_scaling
    loss = l_batch.sum()

    return loss, None       # return loss, loss_all.detach()

import random
import numpy as np
import torch
import time
from time import perf_counter_ns
from tqdm import tqdm
import cv2
import trainer
import open3d 
import trimesh
import scipy
from bidict import bidict
import copy
import os
import torch.nn as nn
from packaging import version

import utils


class performance_measure:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, type, value, tb):
        self.end_time = perf_counter_ns()
        self.exec_time = self.end_time - self.start_time

        print(f"{self.name} excution time: {(self.exec_time)/1000000:.2f} ms")

def origin_dirs_W(T_WC, dirs_C):#T_WC: camera pose

    assert T_WC.shape[0] == dirs_C.shape[0]
    assert T_WC.shape[1:] == (4, 4)
    assert dirs_C.shape[2] == 3

    dirs_W = (T_WC[:, None, :3, :3] @ dirs_C[..., None]).squeeze()

    origins = T_WC[:, :3, -1]

    return origins, dirs_W


# @torch.jit.script 分层采样
def stratified_bins(min_depth, max_depth, n_bins, n_rays, type=torch.float32, device = "cuda:0"):
    # type: (Tensor, Tensor, int, int, type, device) -> Tensor

    bin_limits_scale = torch.linspace(0, 1, n_bins+1, dtype=type, device=device)

    if not torch.is_tensor(min_depth):
        min_depth = torch.ones(n_rays, dtype=type, device=device) * min_depth
    
    if not torch.is_tensor(max_depth):
        max_depth = torch.ones(n_rays, dtype=type, device=device) * max_depth

    depth_range = max_depth - min_depth
  
    lower_limits_scale = depth_range[..., None] * bin_limits_scale + min_depth[..., None]#n_bin+1个min_depth到max_depth的均匀点
    lower_limits_scale = lower_limits_scale[:, :-1]

    assert lower_limits_scale.shape == (n_rays, n_bins)

    bin_length_scale = depth_range / n_bins
    increments_scale = torch.rand(
        n_rays, n_bins, device=device,
        dtype=torch.float32) * bin_length_scale[..., None]#随机噪声

    z_vals_scale = lower_limits_scale + increments_scale

    assert z_vals_scale.shape == (n_rays, n_bins)

    return z_vals_scale

# @torch.jit.script
def normal_bins_sampling(depth, n_bins, n_rays, delta, device = "cuda:0"):
    # type: (Tensor, int, int, float, device) -> Tensor

    # device = "cpu"
    # bins = torch.normal(0.0, delta / 3., size=[n_rays, n_bins], devi
        # self.keyframes_batch = torch.empty(self.n_keyframes,ce=device).sort().values
    bins = torch.empty(n_rays, n_bins, dtype=torch.float32, device=device).normal_(mean=0.,std=delta / 3.).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth[:, None] + bins

    assert z_vals.shape == (n_rays, n_bins)

    return z_vals


class sceneObject:
    """
    object instance mapping,
    updating keyframes, get training samples, optimizing MLP map
    """

    def __init__(self, cfg, obj_id, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, live_frame_id) -> None:
        self.do_bg = cfg.do_bg
        self.obj_id = obj_id
        self.data_device = cfg.data_device
        self.training_device = cfg.training_device
        self.bound = cfg.bound
        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)

        if self.do_bg and self.obj_id == 0: # do seperate bg, we use ESLAM character planes here
            self.obj_scale = cfg.bg_scale
            self.hidden_feature_size = cfg.hidden_feature_size_bg
            self.n_bins_cam2surface = cfg.n_bins_cam2surface_bg
            self.keyframe_step = cfg.keyframe_step_bg
        else:
            self.obj_scale = cfg.obj_scale
            self.hidden_feature_size = cfg.hidden_feature_size
            self.n_bins_cam2surface = cfg.n_bins_cam2surface
            self.keyframe_step = cfg.keyframe_step

        self.frames_width = rgb.shape[0]
        self.frames_height = rgb.shape[1]

        self.min_bound = cfg.min_depth
        self.max_bound = cfg.max_depth
        self.n_bins = cfg.n_bins
        self.n_unidir_funcs = cfg.n_unidir_funcs

        self.surface_eps = cfg.surface_eps
        self.stop_eps = cfg.stop_eps

        self.n_keyframes = 1  # Number of keyframes
        self.kf_pointer = None
        self.keyframe_buffer_size = cfg.keyframe_buffer_size
        if obj_id == 0:
            self.keyframe_buffer_size*=25
        self.kf_id_dict = bidict({live_frame_id:0})
        self.kf_buffer_full = False
        self.frame_cnt = 0  # number of frames taken in
        self.lastest_kf_queue = []

        self.bbox = torch.empty(  # obj bounding bounding box in the frame
            self.keyframe_buffer_size,
            4,
            device=self.data_device)  # [u low, u high, v low, v high]
        self.bbox[0] = bbox_2d

        # RGB + pixel state batch
        self.rgb_idx = slice(0, 3)
        self.state_idx = slice(3, 4)
        self.rgbs_batch = torch.empty(self.keyframe_buffer_size,
                                      self.frames_width,
                                      self.frames_height,
                                      4,
                                      dtype=torch.uint8,
                                      device=self.data_device)

        # Pixel states:
        self.other_obj = 0  # pixel doesn't belong to obj
        self.this_obj = 1  # pixel belong to obj 
        self.unknown_obj = 2  # pixel state is unknown

        # Initialize first frame rgb and pixel state
        self.rgbs_batch[0, :, :, self.rgb_idx] = rgb
        self.rgbs_batch[0, :, :, self.state_idx] = mask[..., None]
        

        self.depth_batch = torch.empty(self.keyframe_buffer_size,
                                       self.frames_width,
                                       self.frames_height,
                                       dtype=torch.float32,
                                       device=self.data_device)

        # Initialize first frame's depth 
        self.depth_batch[0] = depth
        self.t_wc_batch = torch.empty(
            self.keyframe_buffer_size, 4, 4,
            dtype=torch.float32,
            device=self.data_device)  # world to camera transform

        # Initialize first frame's world2cam transform
        self.t_wc_batch[0] = t_wc

        # neural field map
        trainer_cfg = copy.deepcopy(cfg)
        trainer_cfg.obj_id = self.obj_id
        trainer_cfg.hidden_feature_size = self.hidden_feature_size
        trainer_cfg.obj_scale = self.obj_scale
        # 3D boundary
        self.bbox3d = None
        self.pc = []

        # init  obj local frame
        # self.obj_center = self.init_obj_center(intrinsic, depth, mask, t_wc)
        self.obj_center = torch.tensor(0.0) # shouldn't make any difference because of frequency embedding


    def init_obj_center(self, intrinsic_open3d, depth, mask, t_wc):
        obj_depth = depth.cpu().clone()
        obj_depth[mask!=self.this_obj] = 0
        T_CW = np.linalg.inv(t_wc.cpu().numpy())
        pc_obj_init = open3d.geometry.PointCloud.create_from_depth_image(
            depth=open3d.geometry.Image(np.asarray(obj_depth.permute(1,0).numpy(), order="C")),
            intrinsic=intrinsic_open3d,#内参
            extrinsic=T_CW,#外参
            depth_trunc=self.max_bound,
            depth_scale=1.0)
        obj_center = torch.from_numpy(np.mean(pc_obj_init.points, axis=0)).float()
        return obj_center

    # @profile
    def append_keyframe(self, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, frame_id:np.uint8=1):
        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)
        assert self.n_keyframes <= self.keyframe_buffer_size - 1
        assert rgb.dtype == torch.uint8
        assert mask.dtype == torch.uint8
        assert depth.dtype == torch.float32

        # every kf_step choose one kf
        is_kf = (self.frame_cnt % self.keyframe_step == 0) or self.n_keyframes == 1
        # print("---------------------")
        # print("self.kf_id_dict ", self.kf_id_dict)
        # print("live frame id ", frame_id)
        # print("n_frames ", self.n_keyframes)
        if self.n_keyframes == self.keyframe_buffer_size - 1:  # kf buffer full, need to prune
            self.kf_buffer_full = True
            if self.kf_pointer is None:
                self.kf_pointer = self.n_keyframes

            self.rgbs_batch[self.kf_pointer, :, :, self.rgb_idx] = rgb
            self.rgbs_batch[self.kf_pointer, :, :, self.state_idx] = mask[..., None]
            self.depth_batch[self.kf_pointer, ...] = depth
            self.t_wc_batch[self.kf_pointer, ...] = t_wc
            self.bbox[self.kf_pointer, ...] = bbox_2d
            self.kf_id_dict.inv[self.kf_pointer] = frame_id

            if is_kf:
                self.lastest_kf_queue.append(self.kf_pointer)
                pruned_frame_id, pruned_kf_id = self.prune_keyframe() #ramdomly choose one kf
                self.kf_pointer = pruned_kf_id
                print("pruned kf id ", self.kf_pointer)

        else:
            if not is_kf:   # not kf, replace
                self.rgbs_batch[self.n_keyframes-1, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes-1, :, :, self.state_idx] = mask[..., None]
                self.depth_batch[self.n_keyframes-1, ...] = depth
                self.t_wc_batch[self.n_keyframes-1, ...] = t_wc
                self.bbox[self.n_keyframes-1, ...] = bbox_2d
                self.kf_id_dict.inv[self.n_keyframes-1] = frame_id
            else:   # is kf, add new kf
                self.kf_id_dict[frame_id] = self.n_keyframes
                self.rgbs_batch[self.n_keyframes, :, :, self.rgb_idx] = rgb
                self.rgbs_batch[self.n_keyframes, :, :, self.state_idx] = mask[..., None]
                self.depth_batch[self.n_keyframes, ...] = depth
                self.t_wc_batch[self.n_keyframes, ...] = t_wc
                self.bbox[self.n_keyframes, ...] = bbox_2d
                self.lastest_kf_queue.append(self.n_keyframes)
                self.n_keyframes += 1

        # print("self.kf_id_dic ", self.kf_id_dict)
        self.frame_cnt += 1
        if len(self.lastest_kf_queue) > 2:  # keep latest two frames
            self.lastest_kf_queue = self.lastest_kf_queue[-2:]

    def prune_keyframe(self):
        # simple strategy to prune, randomly choose
        key, value = random.choice(list(self.kf_id_dict.items())[:-2])  # do not prune latest two frames
        return key, value

    def get_bound(self, intrinsic_open3d):
        # get 3D boundary from posed depth img   todo update sparse pc when append frame
        pcs = open3d.geometry.PointCloud()
        for kf_id in range(self.n_keyframes) :
            mask = self.rgbs_batch[kf_id, :, :, self.state_idx].squeeze() == self.this_obj
            depth = self.depth_batch[kf_id].cpu().clone()
            twc = self.t_wc_batch[kf_id].cpu().clone().numpy()
            if self.obj_id == 0:
                mask = mask.transpose(0,1)
                depth = depth.transpose(0,1)
                twc[:3,1]*= -1
                twc[:3,2]*= -1
            depth[~mask] = 0#只对mask为0的值进行操作
            depth = depth.permute(1,0).numpy().astype(np.float32)
            T_CW = np.linalg.inv(twc)
            pc = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(np.asarray(depth, order="C")), intrinsic=intrinsic_open3d, extrinsic=T_CW)
            # self.pc += pc
            pcs += pc

        # # get minimal oriented 3d bbox
        # try:
        #     bbox3d = open3d.geometry.OrientedBoundingBox.create_from_points(pcs.points)
        # except RuntimeError:
        #     print("too few pcs obj ")
        #     return None
        # trimesh has a better minimal bbox implementation than open3d
        try:
            transform, extents = trimesh.bounds.oriented_bounds(np.array(pcs.points))  # pc
            transform = np.linalg.inv(transform)
        except scipy.spatial._qhull.QhullError:
            print("too few pcs obj ")
            return None

        for i in range(extents.shape[0]):
            extents[i] = np.maximum(extents[i], 0.10)  # at least rendering 10cm
        bbox = utils.BoundingBox()
        bbox.center = transform[:3, 3]
        bbox.R = transform[:3, :3]
        bbox.extent = extents
        bbox3d = open3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)

        min_extent = 0.05
        bbox3d.extent = np.maximum(min_extent, bbox3d.extent)
        bbox3d.color = (255,0,0)
        self.bbox3d = utils.bbox_open3d2bbox(bbox_o3d=bbox3d)
        # self.pc = []
        print("obj ", self.obj_id)
        print("bound ", bbox3d)
        print("kf id dict ", self.kf_id_dict)
        # open3d.visualization.draw_geometries([bbox3d, pcs])
        return bbox3d
    
    def get_bound_from_frames(self,cfg, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """
        start_time = time.time()
        H, W, fx, fy, cx, cy = cfg.H, cfg.W, cfg.fx, cfg.fy, cfg.cx, cfg.cy
        self.mesh_bound_scale =cfg.mesh_bound_scale

        if version.parse(open3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = open3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = open3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=open3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for kf_id in range(self.n_keyframes):
            c2w = self.t_wc_batch[kf_id].cpu().clone().numpy()
            # convert to open3d camera pose #############
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            mask = self.rgbs_batch[kf_id, :, :, self.state_idx].squeeze() == self.this_obj
            mask= mask.cpu().clone()
            depth = self.depth_batch[kf_id].cpu().clone().numpy()
            color = self.rgbs_batch[kf_id,:,:,:3].cpu().clone().numpy() /255.0
            #print("mask device:", mask.device)
            #print("depth device:", depth.device)
            depth[~mask] = 0#只对mask为0的值进行操作
            depth = open3d.geometry.Image(depth.astype(np.float32))
            color = open3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = open3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(open3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        endtime = time.time()
        print("!!!Time to get mesh bound: ",endtime-start_time, "s", "keyframe length:", self.n_keyframes)
        return return_mesh



    def get_training_samples(self, n_frames, n_samples, cached_rays_dir):
        # Sample pixels
        if self.n_keyframes > 2: # make sure latest 2 frames are sampled    todo if kf pruned, this is not the latest frame
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames - 2,),
                                         dtype=torch.long,
                                         device=self.data_device)
            # if self.kf_buffer_full:
            # latest_frame_ids = list(self.kf_id_dict.values())[-2:]
            latest_frame_ids = self.lastest_kf_queue[-2:]
            keyframe_ids = torch.cat([keyframe_ids,
                                          torch.tensor(latest_frame_ids, device=keyframe_ids.device)])
            # print("latest_frame_ids", latest_frame_ids)
            # else:   # sample last 2 frames
            #     keyframe_ids = torch.cat([keyframe_ids,
            #                               torch.tensor([self.n_keyframes-2, self.n_keyframes-1], device=keyframe_ids.device)])
        else:
            keyframe_ids = torch.randint(low=0,
                                         high=self.n_keyframes,
                                         size=(n_frames,),
                                         dtype=torch.long,
                                         device=self.data_device)
        keyframe_ids = torch.unsqueeze(keyframe_ids, dim=-1)
        
        idx_w = torch.rand(n_frames, n_samples, device=self.data_device)
        idx_h = torch.rand(n_frames, n_samples, device=self.data_device)

        # resizing idx_w and idx_h to be in the bbox range
        idx_w = idx_w * (self.bbox[keyframe_ids, 1] - self.bbox[keyframe_ids, 0]) + self.bbox[keyframe_ids, 0]
        idx_h = idx_h * (self.bbox[keyframe_ids, 3] - self.bbox[keyframe_ids, 2]) + self.bbox[keyframe_ids, 2]
        idx_w = idx_w.long()
        idx_h = idx_h.long()
        sampled_rgbs = self.rgbs_batch[keyframe_ids, idx_w, idx_h]
        sampled_depth = self.depth_batch[keyframe_ids, idx_w, idx_h]

        # Get ray directions for sampled pixels
        if self.do_bg and self.obj_id == 0:
            sampled_ray_dirs = cached_rays_dir[idx_h, idx_w]
        else:
            sampled_ray_dirs = cached_rays_dir[idx_w, idx_h]

        # Get sampled keyframe poses
        sampled_twc = self.t_wc_batch[keyframe_ids[:, 0], :, :]

        origins, dirs_w = origin_dirs_W(sampled_twc, sampled_ray_dirs)

        return self.sample_3d_points(sampled_rgbs, sampled_depth, origins, dirs_w)

    def sample_3d_points(self, sampled_rgbs, sampled_depth, origins, dirs_w):
        """
        3D sampling strategy

        * For pixels with invalid depth:
            - N+M from minimum bound to max (stratified)
        
        * For pixels with valid depth:
            # Pixel belongs to this object
                - N from cam to surface (stratified)
                - M around surface (stratified/normal)
            # Pixel belongs that don't belong to this object
                - N from cam to surface (stratified)
                - M around surface (stratified)
            # Pixel with unknown state
                - Do nothing!
        """

        n_bins_cam2surface = self.n_bins_cam2surface
        n_bins = self.n_bins
        eps = self.surface_eps
        other_objs_max_eps = self.stop_eps #0.05   # todo 0.02
        # print("max depth ", torch.max(sampled_depth))
        sampled_z = torch.zeros(
            sampled_rgbs.shape[0] * sampled_rgbs.shape[1],
            n_bins_cam2surface + n_bins,
            dtype=self.depth_batch.dtype,
            device=self.data_device)  # shape (N*n_rays, n_bins_cam2surface + n_bins)

        invalid_depth_mask = (sampled_depth <= self.min_bound).view(-1)
        # max_bound = self.max_bound
        max_bound = torch.max(sampled_depth)
        # sampling for points with invalid depth
        invalid_depth_count = invalid_depth_mask.count_nonzero()
        if invalid_depth_count:
            sampled_z[invalid_depth_mask, :] = stratified_bins(
                self.min_bound, max_bound,
                n_bins_cam2surface + n_bins, invalid_depth_count,
                device=self.data_device)

        # sampling for valid depth rays
        valid_depth_mask = ~invalid_depth_mask
        valid_depth_count = valid_depth_mask.count_nonzero()


        if valid_depth_count:
            # Sample between min bound and depth for all pixels with valid depth
            sampled_z[valid_depth_mask, :n_bins_cam2surface] = stratified_bins(
                self.min_bound, sampled_depth.view(-1)[valid_depth_mask]-eps,
                n_bins_cam2surface, valid_depth_count, device=self.data_device)

            # sampling around depth for this object
            obj_mask = (sampled_rgbs[..., -1] == self.this_obj).view(-1) & valid_depth_mask # todo obj_mask
            assert sampled_z.shape[0] == obj_mask.shape[0]
            obj_count = obj_mask.count_nonzero()

            if obj_count:
                sampling_method = "normal"  # stratified or normal
                if sampling_method == "stratified":
                    sampled_z[obj_mask, n_bins_cam2surface:] = stratified_bins(
                        sampled_depth.view(-1)[obj_mask] - eps, sampled_depth.view(-1)[obj_mask] + eps,
                        n_bins, obj_count, device=self.data_device)

                elif sampling_method == "normal":
                    sampled_z[obj_mask, n_bins_cam2surface:] = normal_bins_sampling(
                        sampled_depth.view(-1)[obj_mask],
                        n_bins,
                        obj_count,
                        delta=eps,
                        device=self.data_device)

                else:
                    raise (
                        f"sampling method not implemented {sampling_method}, \
                            stratified and normal sampling only currenty implemented."
                    )

            # sampling around depth of other objects
            other_obj_mask = (sampled_rgbs[..., -1] != self.this_obj).view(-1) & valid_depth_mask
            other_objs_count = other_obj_mask.count_nonzero()
            if other_objs_count:
                sampled_z[other_obj_mask, n_bins_cam2surface:] = stratified_bins(
                    sampled_depth.view(-1)[other_obj_mask] - eps,
                    sampled_depth.view(-1)[other_obj_mask] + other_objs_max_eps,
                    n_bins, other_objs_count, device=self.data_device)

        sampled_z = sampled_z.view(sampled_rgbs.shape[0],
                                   sampled_rgbs.shape[1],
                                   -1)  # view as (n_rays, n_samples, 10)
        input_pcs = origins[..., None, None, :] + (dirs_w[:, :, None, :] *
                                                   sampled_z[..., None])
        input_pcs -= self.obj_center
        obj_labels = sampled_rgbs[..., -1].view(-1)
        return sampled_rgbs[..., :3], sampled_depth, valid_depth_mask, obj_labels, input_pcs, sampled_z, origins, dirs_w

    def save_checkpoints(self, path, epoch):
        obj_id = self.obj_id
        chechpoint_load_file = (path + "/obj_" + str(obj_id) + "_frame_" + str(epoch) + ".pth")
        if self.obj_id == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "Decoder": self.trainer.decoders.state_dict(),
                    "eslam": self.trainer.eslam,
                    "decoders_para_list": self.trainer.decoders_para_list,
                    "planes_para": self.trainer.planes_para,
                    "c_planes_para": self.trainer.c_planes_para,
                    "obj_id": self.obj_id,
                    "bbox": self.bbox3d,
                    "obj_scale": self.trainer.obj_scale
                },
                chechpoint_load_file,
            )
        else :
            torch.save(
                {
                    "epoch": epoch,
                    "FC_state_dict": self.trainer.fc_occ_map.state_dict(),
                    "PE_state_dict": self.trainer.pe.state_dict(),
                    "obj_id": self.obj_id,
                    "bbox": self.bbox3d,
                    "obj_scale": self.trainer.obj_scale
                },
                chechpoint_load_file,
            )
        # optimiser?

    def load_checkpoints(self, ckpt_file):
        checkpoint_load_file = (ckpt_file)
        if not os.path.exists(checkpoint_load_file):
            print("ckpt not exist ", checkpoint_load_file)
            return
        checkpoint = torch.load(checkpoint_load_file)
        self.obj_id = checkpoint["obj_id"]
        if self.obj_id == 0:
            self.trainer.decoders.load_state_dict(checkpoint["Decoder"])
            self.trainer.eslam = checkpoint["eslam"]
            self.trainer.decoders_para_list = checkpoint["decoders_para_list"]
            self.trainer.planes_para = checkpoint["planes_para"]
            self.trainer.c_planes_para = checkpoint["c_planes_para"]
        else:
            self.trainer.fc_occ_map.load_state_dict(checkpoint["FC_state_dict"])
            self.trainer.pe.load_state_dict(checkpoint["PE_state_dict"])
            self.trainer.fc_occ_map.to(self.training_device)
            self.trainer.pe.to(self.training_device)
            
        self.bbox3d = checkpoint["bbox"]
        self.trainer.obj_scale = checkpoint["obj_scale"]

        


class cameraInfo:

    def __init__(self, cfg) -> None:
        self.device = cfg.data_device
        self.width = cfg.W  # Frame width
        self.height = cfg.H  # Frame height

        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy

        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self, depth_type="z"):
        idx_w = torch.arange(end=self.width, device=self.device)
        idx_h = torch.arange(end=self.height, device=self.device)

        dirs = torch.ones((self.width, self.height, 3), device=self.device)

        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None]
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy)

        if depth_type == "euclidean":
            raise Exception(
                "Get camera rays directions with euclidean depth not yet implemented"
            )
            norm = torch.norm(dirs, dim=-1)
            dirs = dirs * (1. / norm)[:, :, :, None]

        return dirs

class ESLAM():
    """
    ESLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking processes.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """
    def __init__(self, cfg):
        #目前只保留特征平面
        self.coarse_planes_res = cfg.coarse_planes_res
        self.fine_planes_res = cfg.fine_planes_res
        self.scale =cfg.scale
        
        self.coarse_c_planes_res = cfg.coarse_c_planes_res
        self.fine_c_planes_res = cfg.fine_c_planes_res
        self.bound = torch.from_numpy(np.array(cfg.bound)*self.scale).float()
        bound_dividable = cfg.bound_dividable
        # enlarge the bound a bit to allow it dividable by bound_dividable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]

        c_dim = cfg.c_dim
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        ####### Initializing Planes ############
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        #the resolutions of the planes in the xy, xz, and yz directions for both coarse and fine planes.
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in c_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        self.shared_planes_xy = planes_xy
        self.shared_planes_xz = planes_xz
        self.shared_planes_yz = planes_yz

        self.shared_c_planes_xy = c_planes_xy
        self.shared_c_planes_xz = c_planes_xz
        self.shared_c_planes_yz = c_planes_yz
        
        self.all_planes = (self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz, self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz)

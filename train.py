import time
import loss
from vmap import *
import utils
import open3d
import dataset
import vis
from functorch import vmap
import argparse
from cfg import Config
import shutil
import Renderer
from pynvml import *
import matplotlib
import cv2

if __name__ == "__main__":
    #############################################
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    matplotlib.interactive(False)

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--logdir', default='./logs/debug',
                        type=str)#target
    parser.add_argument('--config',#data 
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    parser.add_argument('--save_ckpt',
                        default=False,
                        type=bool)
    args = parser.parse_args()

    log_dir = args.logdir
    config_file = args.config
    save_ckpt = args.save_ckpt
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    n_sample_per_step = cfg.n_per_optim
    n_sample_per_step_bg = cfg.n_per_optim_bg

    # param for vis
    vis3d = open3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D mesh vis",
                        width=cfg.W,
                        height=cfg.H,
                        left=600, top=50)#原本left=600, top=50
    view_ctl = vis3d.get_view_control()
    view_ctl.set_constant_z_far(10.)#sets the constant value for the far clipping plane distance. 

    # set camera
    cam_info = cameraInfo(cfg)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)

    # init obj_dict
    obj_dict = {}   # only objs
    vis_dict = {}   # including bg

    #  for ESLAM
    
    
    
    # init for training
    AMP = False#whether to use automatic mixed precision training.
    if AMP:
        scaler = torch.cuda.amp.GradScaler()  # amp https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


    #############################################
    # init data stream
    if not cfg.live_mode:#default:not
        # load dataset
        dataloader = dataset.init_loader(cfg)
        dataloader_iterator = iter(dataloader)
        dataset_len = len(dataloader)
    else:
        dataset_len = 1000000
        # # init ros node
        # torch.multiprocessing.set_start_method('spawn')  # spawn
        # import ros_nodes
        # track_to_map_Buffer = torch.multiprocessing.Queue(maxsize=5)
        # # track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
        # kfs_que = torch.multiprocessing.Queue(maxsize=5)  # to store one more buffer
        # track_p = torch.multiprocessing.Process(target=ros_nodes.Tracking,
        #                                              args=(
        #                                              (cfg), (track_to_map_Buffer), (None),
        #                                              (kfs_que), (True),))
        # track_p.start()
    # init vmap
    fc_models, pe_models = [], []
    scene_bg = None
    estimate_c2w_list = torch.zeros((dataset_len, 4, 4), device=cfg.device)
    estimate_c2w_list.share_memory_()
    #nvmlInit()  
    #handle = nvmlDeviceGetHandleByIndex(0)
    #info_past = nvmlDeviceGetMemoryInfo(handle)
    for frame_id in tqdm(range(dataset_len)):
        print("*********************************************")
        #info = nvmlDeviceGetMemoryInfo(handle)
        #if info.used - info_past.used > 1024*1024:
        #    print(f"Used GPU Memory in frame_id {frame_id}: {info.used / (1024 * 1024)} MB")
        #info_past = info
        # get new frame data
        with performance_measure(f"getting next data"):
            if not cfg.live_mode:
                # get data from dataloader
                sample = next(dataloader_iterator)#sample is coming from class replica or scannet
            else:
                pass
            
        if sample is not None:  # new frame
            last_frame_time = time.time()
            with performance_measure(f"Appending data"):
                rgb = sample["image"].to(cfg.data_device)
                depth = sample["depth"].to(cfg.data_device)
                twc = sample["T"].to(cfg.data_device)#camera pose
                bbox_dict = sample["bbox_dict"]
                eslam_rgb = rgb.clone().transpose(0, 1)
                eslam_depth = depth.clone().transpose(0, 1)
                eslam_twc = twc.clone()
                eslam_twc[:3,1] = -1*eslam_twc[:3,1]
                eslam_twc[:3,2] = -1*eslam_twc[:3,2]
                estimate_c2w_list[frame_id] = twc
                #gt_color_np = (eslam_rgb).cpu().clone().numpy()#for test
                #file_path = "color_image_"+str(frame_id)+ ".jpg"
                #cv2.imwrite(file_path, gt_color_np)
                
                if "frame_id" in sample.keys():
                    live_frame_id = sample["frame_id"]
                else:
                    live_frame_id = frame_id
                if not cfg.live_mode:
                    inst = sample["obj"].to(cfg.data_device)
                    obj_ids = torch.unique(inst)
                else:
                    inst_data_dict = sample["obj"]
                    obj_ids = inst_data_dict.keys()
                # append new frame info to objs in current view
                for obj_id in obj_ids:
                    if obj_id == -1:    # unsured area
                        continue
                    obj_id = int(obj_id)
                    # convert inst mask to state
                    if not cfg.live_mode:
                        state = torch.zeros_like(inst, dtype=torch.uint8, device=cfg.data_device)
                        state[inst == obj_id] = 1 #belonging to object
                        state[inst == -1] = 2 # error?
                    else:
                        inst_mask = inst_data_dict[obj_id].permute(1,0)
                        label_list = torch.unique(inst_mask).tolist()
                        state = torch.zeros_like(inst_mask, dtype=torch.uint8, device=cfg.data_device)
                        state[inst_mask == obj_id] = 1
                        state[inst_mask == -1] = 2
                    bbox = bbox_dict[obj_id]
                    eslam_state = state.clone().transpose(0, 1)
                    eslam_bbox = bbox.clone()
                    eslam_bbox[[0,2]] = eslam_bbox[[2,0]]
                    eslam_bbox[[1,3]] = eslam_bbox[[3,1]]
                    if obj_id in vis_dict.keys():
                        scene_obj = vis_dict[obj_id]
                        if cfg.do_bg and obj_id == 0:
                            scene_obj.append_keyframe(eslam_rgb, eslam_depth, eslam_state, eslam_bbox, eslam_twc, live_frame_id)
                        else:
                            scene_obj.append_keyframe(rgb, depth, state, bbox, twc, live_frame_id)#In vmap.py
                    else: # init new scene_obj
                        if len(obj_dict.keys()) >= cfg.max_n_models:
                            print("models full!!!! current num ", len(obj_dict.keys()))
                            continue
                        print("init new obj ", obj_id)
                        #for bg
                        if cfg.do_bg and obj_id == 0:   # todo param, Here we use ESLAM
                            #三个扔进去的参数要改
                            # 参数转换
                            scene_bg = sceneObject(cfg, obj_id, eslam_rgb, eslam_depth, eslam_state, eslam_bbox, eslam_twc, live_frame_id)
                            lr_factor = cfg.lr_factor
                            # scene_bg.init_obj_center(intrinsic_open3d, depth, state, twc)
                            #optimiser.add_param_group({"params": scene_bg.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            #optimiser.add_param_group({"params": scene_bg.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            #double optimiser
                            decoders_para_list = []
                            decoders_para_list += list(trainer.decoders.parameters())
                            planes_para = []
                            for planes in [trainer.eslam.shared_planes_xy, trainer.eslam.shared_planes_xz, trainer.eslam.shared_planes_yz]:
                                for i, plane in enumerate(planes):
                                    plane = nn.Parameter(plane)
                                    planes_para.append(plane)
                                    planes[i] = plane

                            c_planes_para = []
                            for c_planes in [trainer.eslam.shared_c_planes_xy,trainer.eslam.shared_c_planes_xz, trainer.eslam.shared_c_planes_yz]:
                                for i, c_plane in enumerate(c_planes):
                                    c_plane = nn.Parameter(c_plane)
                                    c_planes_para.append(c_plane)
                                    c_planes[i] = c_plane
                            bg_optimiser = torch.optim.Adam([{'params': scene_bg.trainer.decoders_para_list, 'lr': 0},
                                          {'params': planes_para, 'lr': 0},
                                          {'params': c_planes_para, 'lr': 0}])
                            
                            bg_optimiser.param_groups[0]['lr'] = cfg.decoders_lr*lr_factor
                            bg_optimiser.param_groups[1]['lr'] = cfg.planes_lr*lr_factor
                            bg_optimiser.param_groups[2]['lr'] = cfg.c_planes_lr*lr_factor
                            
                            vis_dict.update({obj_id: scene_bg})
                        else:
                            scene_obj = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id)
                            # scene_obj.init_obj_center(intrinsic_open3d, depth, state, twc)
                            obj_dict.update({obj_id: scene_obj})
                            vis_dict.update({obj_id: scene_obj})
                            # params = [scene_obj.trainer.fc_occ_map.parameters(), scene_obj.trainer.pe.parameters()]
                            optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            if cfg.training_strategy == "vmap":
                                update_vmap_model = True
                                fc_models.append(obj_dict[obj_id].trainer.fc_occ_map)
                                pe_models.append(obj_dict[obj_id].trainer.pe)

                        # ###################################
                        # # measure trainable params in total
                        # total_params = 0
                        # obj_k = obj_dict[obj_id]
                        # for p in obj_k.trainer.fc_occ_map.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # for p in obj_k.trainer.pe.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # print("total param ", total_params)
        # dynamically add vmap
        with performance_measure(f"add vmap"):
            if cfg.training_strategy == "vmap" and update_vmap_model == True:
                fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimiser)#used in training later
                pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimiser)
                update_vmap_model = False


        ##################################################################
        # training data preperation, get training data for all objs
        Batch_N_gt_depth = []
        Batch_N_gt_rgb = []
        Batch_N_depth_mask = []
        Batch_N_obj_mask = []
        Batch_N_input_pcs = []
        Batch_N_sampled_z = []

        with performance_measure(f"Sampling over {len(obj_dict.keys())} objects,"):
            if cfg.do_bg and scene_bg is not None:
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, origins, dirs_W \
                    = scene_bg.get_training_samples(cfg.n_iter_per_frame * cfg.win_size_bg, cfg.n_samples_per_frame_bg,
                                                    cam_info.rays_dir_cache)#(20*10,120,--)???
                bg_gt_depth = gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]])
                bg_gt_rgb = gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]])
                bg_valid_depth_mask = valid_depth_mask
                bg_obj_mask = obj_mask
                bg_input_pcs = input_pcs.reshape(
                    [input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]])
                bg_sampled_z = sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]])
                bg_origins = origins.unsqueeze(1).expand(cfg.n_iter_per_frame * cfg.win_size_bg, cfg.n_samples_per_frame_bg, -1)########################
                bg_origins = bg_origins.reshape(bg_origins.shape[0]*bg_origins.shape[1],bg_origins.shape[2])
                #200*3->24000*3
                bg_dirs_W = dirs_W.reshape(dirs_W.shape[0]*dirs_W.shape[1],dirs_W.shape[2])
                #200*120*3->24000*3

            for obj_id, obj_k in obj_dict.items():
                #input_pcs 相机信息
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, _, _ \
                    = obj_k.get_training_samples(cfg.n_iter_per_frame * cfg.win_size, cfg.n_samples_per_frame,
                                                 cam_info.rays_dir_cache)
                # merge first two dims, sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(input_pcs.reshape([input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]]))

                # # vis sampled points in open3D
                # # sampled pcs
                # pc = open3d.geometry.PointCloud()
                # pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                # open3d.visualization.draw_geometries([pc])
                # rgb_np = rgb.cpu().numpy().astype(np.uint8).transpose(1,0,2)
                # # print("rgb ", rgb_np.shape)
                # # print(rgb_np)
                # # cv2.imshow("rgb", rgb_np)
                # # cv2.waitKey(1)
                # depth_np = depth.cpu().numpy().astype(np.float32).transpose(1,0)
                # twc_np = twc.cpu().numpy()
                # rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                #     open3d.geometry.Image(rgb_np),
                #     open3d.geometry.Image(depth_np),
                #     depth_trunc=max_depth,
                #     depth_scale=1,
                #     convert_rgb_to_intensity=False,
                # )
                # T_CW = np.linalg.inv(twc_np)
                # # input image pc
                # input_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
                #     image=rgbd,
                #     intrinsic=intrinsic_open3d,
                #     extrinsic=T_CW)
                # input_pc.points = open3d.utility.Vector3dVector(np.array(input_pc.points) - obj_k.obj_center.cpu().numpy())
                # open3d.visualization.draw_geometries([pc, input_pc])
        ####################################################
        # training
        assert len(Batch_N_input_pcs) > 0
        # move data to GPU  (n_obj, n_iter_per_frame, win_size*num_per_frame, 3)
        with performance_measure(f"stacking and moving to gpu: "):

            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(cfg.training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(cfg.training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(cfg.training_device) / 255. # todo
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(cfg.training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(cfg.training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(cfg.training_device)
            if cfg.do_bg:     
                          
                bg_input_pcs = bg_input_pcs.to(cfg.training_device)
                bg_gt_depth = bg_gt_depth.to(cfg.training_device)
                bg_gt_rgb = bg_gt_rgb.to(cfg.training_device)/255.
                bg_valid_depth_mask = bg_valid_depth_mask.to(cfg.training_device)
                bg_obj_mask = bg_obj_mask.to(cfg.training_device)
                bg_sampled_z = bg_sampled_z.to(cfg.training_device) 
                
                with torch.no_grad():
                    bound = scene_bg.trainer.eslam.bound
                    det_rays_o = bg_origins.clone().detach().unsqueeze(-1)
                    det_rays_d = bg_dirs_W.clone().detach().unsqueeze(-1)
                    t = (bound.unsqueeze(0).to(cfg.data_device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= bg_gt_depth
                bg_origins = bg_origins[inside_mask,:]
                bg_dirs_W = bg_dirs_W[inside_mask,:]
                bg_gt_depth = bg_gt_depth[inside_mask]
                bg_gt_rgb = bg_gt_rgb[inside_mask,:]


        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):
            for iter_step in range(cfg.n_iter_per_frame):
                data_idx = slice(iter_step*n_sample_per_step, (iter_step+1)*n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]
                depth, color, sdf, z_vals = Renderer.render_batch_ray(cfg,scene_bg.trainer.eslam.all_planes, scene_bg.trainer.decoders,bg_dirs_W,
                                                        bg_origins, cfg.data_device, cfg.truncation,
                                                        gt_depth=bg_gt_depth)
                if cfg.training_strategy == "forloop":
                    # for loop training
                    batch_alpha = []
                    batch_color = []
                    for k, obj_id in enumerate(obj_dict.keys()):
                        obj_k = obj_dict[obj_id]
                        embedding_k = obj_k.trainer.pe(batch_input_pcs[k])#编码
                        alpha_k, color_k = obj_k.trainer.fc_occ_map(embedding_k)#MLP
                        batch_alpha.append(alpha_k)
                        batch_color.append(color_k)

                    batch_alpha = torch.stack(batch_alpha)
                    batch_color = torch.stack(batch_color)
                elif cfg.training_strategy == "vmap":
                    # batched training
                    batch_embedding = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_embedding)
                    # print("batch alpha ", batch_alpha.shape)
                else:
                    print("training strategy {} is not implemented ".format(cfg.training_strategy))
                    exit(-1)
            # step loss
            with performance_measure(f"Batch LOSS"):
                batch_loss, _ = loss.step_batch_loss(cfg,batch_alpha, batch_color,
                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                     batch_sampled_z.detach())

                if cfg.do_bg:
                    #eslam loss
                    depth_mask = (bg_gt_depth > 0)
                    # SDF losses
                    bg_loss = loss.sdf_losses(cfg, sdf[depth_mask], z_vals[depth_mask], bg_gt_depth[depth_mask])
                    # Color loss
                    bg_loss = bg_loss + cfg.w_color * torch.square(bg_gt_rgb - color).mean()
                    # Depth loss
                    bg_loss = bg_loss + cfg.w_depth * torch.square(bg_gt_depth[depth_mask] - depth[depth_mask]).mean() 
                    
            # with performance_measure(f"Backward"):
                if AMP:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:###############################
                    batch_loss.backward()
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                bg_optimiser.zero_grad()
                bg_loss.backward(retain_graph=False)
                bg_optimiser.step()
                # print("loss ", batch_loss.item())
        # update each origin model params
        # todo find a better way    # https://github.com/pytorch/functorch/issues/280
        with performance_measure(f"updating vmap param"):
            if cfg.training_strategy == "vmap":
                with torch.no_grad():
                    for model_id, (obj_id, obj_k) in enumerate(obj_dict.items()):
                        for i, param in enumerate(obj_k.trainer.fc_occ_map.parameters()):
                            param.copy_(fc_param[i][model_id])
                        for i, param in enumerate(obj_k.trainer.pe.parameters()):
                            param.copy_(pe_param[i][model_id])
        ####################################################################
        # live vis mesh
        if (((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len-1) or
            (cfg.live_mode and time.time()-last_frame_time>cfg.keep_live_time)) and frame_id >= 10:
            vis3d.clear_geometries()
            for obj_id, obj_k in vis_dict.items():
                if obj_id == 0 and frame_id == dataset_len-1:
                    mesh_bound = obj_k.get_bound_from_frames(cfg)
                    #adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
                    mesh = obj_k.trainer.meshing(mesh_bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
                    pass
                elif obj_id == 0:
                    continue
                else:
                    bound = obj_k.get_bound(intrinsic_open3d)#相机参数
                    adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
                    mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
                if bound is None:
                    print("get bound failed obj ", obj_id)
                    continue
                if mesh is None:
                    print("meshing failed obj ", obj_id)
                    continue
                # save to dir
                
                obj_mesh_output = os.path.join(log_dir, "scene_mesh")
                os.makedirs(obj_mesh_output, exist_ok=True)

                if obj_id == 0: 
                    mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.ply".format(frame_id, str(obj_id))))
                    mesh_out_file = os.path.join(obj_mesh_output, "frame_{}_obj{}.ply".format(frame_id, str(obj_id)))
                    obj_k.trainer.cull_mesh(mesh_out_file, cfg, args, cfg.device, estimate_c2w_list=estimate_c2w_list)
                else:
                    mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(frame_id, str(obj_id))))
                if obj_id != 0:
                    # live vis
                    open3d_mesh = vis.trimesh_to_open3d(mesh)
                    vis3d.add_geometry(open3d_mesh)
                    vis3d.add_geometry(bound)
                    # update vis3d
                    vis3d.poll_events()
                    vis3d.update_renderer()
        if False:    # follow cam
            cam = view_ctl.convert_to_pinhole_camera_parameters()
            T_CW_np = np.linalg.inv(twc.cpu().numpy())
            cam.extrinsic = T_CW_np
            view_ctl.convert_from_pinhole_camera_parameters(cam)
            vis3d.poll_events()
            vis3d.update_renderer()

        with performance_measure("saving ckpt"):
            if save_ckpt and ((((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len - 1) or
                               (cfg.live_mode and time.time() - last_frame_time > cfg.keep_live_time)) and frame_id >= 10):
                for obj_id, obj_k in vis_dict.items():
                    continue
                    ckpt_dir = os.path.join(log_dir, "ckpt", str(obj_id))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    bound = obj_k.get_bound(intrinsic_open3d)   # update bound
                    obj_k.save_checkpoints(ckpt_dir, frame_id)
                # save current cam pose
                cam_dir = os.path.join(log_dir, "cam_pose")
                os.makedirs(cam_dir, exist_ok=True)
                torch.save({"twc": twc,}, os.path.join(cam_dir, "twc_frame_{}".format(frame_id) + ".pth"))



import torch
import model
import embedding
import render_rays
import numpy as np
import vis
import ESLAM
import torch.nn as nn
from tqdm import tqdm
from pynvml import *
import trimesh
import skimage
from packaging import version
import numpy as np
import open3d as o3d
import time 
from dataset import Replica,ScanNet
from Renderer import sdf2alpha
import cProfile

class Trainer:
    def __init__(self, cfg):
        self.obj_id = cfg.obj_id
        self.cfg =cfg
        self.bound = cfg.bound
        self.device = cfg.training_device
        self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
        self.obj_scale = cfg.obj_scale # 10 for bg and iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs
        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        self.resolution = cfg.resolution
        self.marching_cubes_bound = cfg.marching_cubes_bound
        self.level_set = cfg.level_set
        self.points_batch_size=500000
        self.scale = cfg.scale

        self.load_network(cfg)

        if self.obj_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9

    def load_network(self,cfg):
        #这里，开始为背景配置ESLAM特征平面：
        if cfg.do_bg and self.obj_id == 0:
            eslam = ESLAM.ESLAM(cfg)
            self.decoders = model.ESLAMdecoder(c_dim=cfg.c_dim, truncation=cfg.truncation, learnable_beta=cfg.learnable_beta,bound = cfg.bound)
            self.decoders = self.decoders.to(cfg.data_device)
            decoders_para_list = []
            decoders_para_list += list(self.decoders.parameters())
            
            self.eslam = eslam
        else:
            self.fc_occ_map = model.OccupancyMap(
                self.emb_size1,
                self.emb_size2,
                hidden_size=self.hidden_feature_size
            )
            self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)
    #渲染3D模型
    def meshing(self, bound, obj_center, grid_dim=256):
        with torch.no_grad():
            if self.obj_id == 0:
                return self.bg_get_mesh(self.eslam.all_planes,self.decoders,bound)
            occ_range = [-1., 1.]
            range_dist = occ_range[1] - occ_range[0]
            scene_scale_np = bound.extent / (range_dist * self.bound_extent)
            scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
            transform_np = np.eye(4, dtype=np.float32)
            transform_np[:3, 3] = bound.center
            transform_np[:3, :3] = bound.R
            

            # transform_np = np.linalg.inv(transform_np)  #
            transform = torch.from_numpy(transform_np).to(self.device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                                scale=scene_scale, transform=transform).view(-1, 3)
            #print("grid_pc:",grid_pc)
            #print("scene_scale:",scene_scale)
            #print("obj_center",obj_center)
            grid_pc -= obj_center.to(grid_pc.device)
            scale = abs(grid_pc[1]-grid_pc[0]+grid_pc[grid_dim]-grid_pc[0]+grid_pc[grid_dim*grid_dim]-grid_pc[0])
            print("scale:",scale)
            ret = self.eval_points(grid_pc,bound)#内存爆了
            if ret is None:
                return None

            occ, _ = ret
            scale = scale.cpu().numpy()
            mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy(),scale)
            if mesh is None:
                print("marching cube failed")
                return None
            # Transform to [-1, 1] range
            mesh.apply_translation([-0.5, -0.5, -0.5])
            mesh.apply_scale(2)

            # Transform to scene coordinates
            mesh.apply_scale(scene_scale_np)
            mesh.apply_transform(transform_np)
            vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
            ret = self.eval_points(vertices_pts,bound)
            if ret is None:
                return None
            _, color = ret
            mesh_color = color * 255
            vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
            mesh.visual.vertex_colors = vertex_colors
            return mesh

    def eval_points(self, points,bound,chunk_size=100000):
        # 256^3 = 16777216
        if self.obj_id == 0 and self.cfg.do_bg:
            self.points_batch_size = 500000
            p_split = torch.split(points, self.points_batch_size)
            bound = self.bound##################################
            #bg_bound =  torch.zeros(3,2)
            #bg_bound[:,0] = torch.from_numpy(bound.center - bound.extent)
            #bg_bound[:,1] = torch.from_numpy(bound.center + bound.extent)
            #bound =bg_bound
            rets = []
            for pi in p_split:
                # mask for points out of bound
                mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
                mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
                mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
                mask = mask_x & mask_y & mask_z

                ret = self.decoders(pi, self.eslam.all_planes)

                ret[~mask, -1] = -1
                rets.append(ret)

            ret = torch.cat(rets, dim=0)
            #alpha = self.sdf2alpha(ret[..., -1], self.decoders.beta)#volume densities
            
            #occ = 1 - torch.exp(-alpha)
            return (ret[..., -1],ret[...,:3])#sdf,rgb
            #return (ret[...,-1],ret[...,:3])##?S
        else:
            alpha, color = [], []
            n_chunks = int(np.ceil(points.shape[0] / chunk_size))
            with torch.no_grad():
                for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                    chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                    embedding_k = self.pe(points[chunk_idx, ...])#位置编码
                    alpha_k, color_k = self.fc_occ_map(embedding_k)#MLP
                    alpha.extend(alpha_k.detach().squeeze())
                    color.extend(color_k.detach().squeeze())
            alpha = torch.stack(alpha)
            color = torch.stack(color)
            #把点按sigmoid 投到[-1,1]
            #occ = render_rays.occupancy_activation(alpha).detach()#这里有个detach
            #beta = nn.Parameter(10 * torch.ones(1)).to("cuda:0")
            #alpha = sdf2alpha(alpha,beta).detach
            if alpha.max() == 0:
                print("no occ")
                return None
            return (alpha, color)

    # def bg_get_mesh(self, all_planes ,decoders, device='cuda:0', color=True):
    def bg_get_mesh(self, all_planes , decoders, mesh_bound, device='cuda:0', color=True):
        """
        Get mesh from keyframes and feature planes and save to file.
        Args:
            mesh_out_file (str): output mesh file.
            all_planes (Tuple): all feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
            keyframe_dict (dict): keyframe dictionary.
            device (str): device to run the model.
            color (bool): whether to use color.
        Returns:
            None

        """
        # mesh_bound = trimesh.load('./mesh_bound.obj')

        with torch.no_grad():
            start_time = time.time()
            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            end_time = time.time()
            print("time to get grid_point:",end_time-start_time, "s" )
            start_time = time.time()

            z = []
            mask = []

            '''
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
            mask = np.concatenate(mask, axis=0)

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                z.append(self.eval_points(pnts.to(device), all_planes, decoders).cpu().numpy()[:, -1])
            z = np.concatenate(z, axis=0)
            z[~mask] = -1
            '''
            '''
            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                print("index:",i)
                start = time.time()
                temp_pnts = pnts.cpu().numpy()
                end = time.time()
                #print("move: ", end - start, "s")

                start = time.time()
                temp = mesh_bound.contains(temp_pnts)
                end = time.time()
                #print("!!!contain: ", end - start, "s")
                #print("pnts: ", pnts, pnts.shape)

                start = time.time()
                mask.append(temp)
                end = time.time()
                #print("append: ", end - start, "s")

            print("mesh_bound:", mesh_bound)
            print("Number of vertices:", mesh_bound.vertices.shape[0])
            print("Number of faces:", mesh_bound.faces.shape[0])
            print("Vertices:", mesh_bound.vertices)
            print("Faces:", mesh_bound.faces)

            start = time.time()
            mask = np.concatenate(mask, axis=0)
            end = time.time()
            print("concat: ", end - start, "s")

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                print("index:",i)
                start = time.time()
                pnts_temp = pnts.to(device)
                end = time.time()
                print("move_pnts: ", end - start, "s")
                
                start = time.time()
                sdf,_ =self.eval_points(pnts_temp, all_planes, decoders)
                end = time.time()
                print("eval_pnts: ", end - start, "s")

                start = time.time()
                sdf_temp= sdf.cpu().numpy()
                end = time.time()
                print("move_sdf: ", end - start, "s")

                start = time.time()
                z.append(sdf_temp)
                end = time.time()
                print("append:", end - start, "s")

            start = time.time()
            z = np.concatenate(z, axis=0)
            end = time.time()
            print("concat:", end - start, "s")

            print("z:", len(z), z)
            print("mask:", len(mask), mask)
            '''

            for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):

                # 操作1:判断条件并将结果存储在 mask 中
                mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                
                # 操作2:计算值并将结果存储在 z 中
                sdf,_ =self.eval_points(pnts.to(device), all_planes, decoders)
                z.append(sdf.cpu().numpy())


            # 合并结果
            mask = np.concatenate(mask, axis=0)
            z = np.concatenate(z, axis=0)

            end_time = time.time()
            print("!!! time to get z:",end_time-start_time, "s")
            
            
            # 使用布尔索引将不满足条件的元素设置为 -1
            
            start_time = time.time()
            z[~mask] = -1 
            end_time = time.time()
            print("!!! z[~mask]:",end_time-start_time, "s")
            #print("z:", len(z), z) # 329868000
            #print("mask:", len(mask), mask) # 329868000

            start_time = time.time()
            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                    print("spacing:",grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1])
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print('marching_cubes error. Possibly no surface extracted from the level set.')
                return

            # convert back to world coordinates
            vertices = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
            end_time = time.time()
            print("time to get vertice of point ",end_time-start_time, "s" )
            start_time = time.time()
            if color:
                # color is extracted by passing the coordinates of mesh vertices through the network
                points = torch.from_numpy(vertices)
                z = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    _, z_color = self.eval_points(pnts.to(device).float(), all_planes, decoders)
                    z.append(z_color.cpu())
                z = torch.cat(z, dim=0)
                vertex_colors = z.numpy()
            else:
                vertex_colors = None
            end_time = time.time()
            print("time to get color ",end_time-start_time, "s" )
            start_time = time.time()
            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            end_time = time.time()
            print("time to mesh bg ",end_time-start_time, "s" )
            start_time = time.time()
            return mesh

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05

        nsteps_x = ((bound[0][1] - bound[0][0] + 2 * padding) / resolution).round().int().item()
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding, nsteps_x)
        
        nsteps_y = ((bound[1][1] - bound[1][0] + 2 * padding) / resolution).round().int().item()
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding, nsteps_y)
        
        nsteps_z = ((bound[2][1] - bound[2][0] + 2 * padding) / resolution).round().int().item()
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding, nsteps_z)

        x_t, y_t, z_t = torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(z).float()
        grid_x, grid_y, grid_z = torch.meshgrid(x_t, y_t, z_t, indexing='xy')
        grid_points_t = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)

        return {"grid_points": grid_points_t, "xyz": [x, y, z]}
    
    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))
    
    def cull_mesh(mesh_file, cfg, args, device, estimate_c2w_list=None):
        """
        Cull the mesh by removing the points that are not visible in any of the frames.
        The output mesh file will be saved in the same directory as the input mesh file.
        Args:
            mesh_file (str): path to the mesh file
            cfg (dict): configuration
            args (argparse.Namespace): arguments
            device (torch.device): device
            estimate_c2w_list (list): list of estimated camera poses, if None, it uses the ground truth camera poses
        Returns:
            None

        """
        if cfg.dataset_format == "Replica":
            frame_reader = Replica(cfg)
        elif cfg.dataset_format == "ScanNet":
            frame_reader = ScanNet(cfg)

        eval_rec = cfg['meshing']['eval_rec']
        truncation = cfg['model']['truncation']
        H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        if estimate_c2w_list is not None:
            n_imgs = len(estimate_c2w_list)
        else:
            n_imgs = len(frame_reader)

        mesh = trimesh.load(mesh_file, process=False)
        pc = mesh.vertices

        whole_mask = np.ones(pc.shape[0]).astype('bool')
        for i in tqdm(range(0, n_imgs, 1)):
            _, _, depth, c2w = frame_reader[i]
            depth, c2w = depth.to(device), c2w.to(device)

            if not estimate_c2w_list is None:
                c2w = estimate_c2w_list[i].to(device)

            points = pc.copy()
            points = torch.from_numpy(points).to(device)

            w2c = torch.inverse(c2w)
            K = torch.from_numpy(
                np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).to(device)
            ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
            homo_points = torch.cat(
                [points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
            cam_cord_homo = w2c@homo_points
            cam_cord = cam_cord_homo[:, :3]

            cam_cord[:, 0] *= -1
            uv = K.float()@cam_cord.float()
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.squeeze(-1)

            grid = uv[None, None].clone()
            grid[..., 0] = grid[..., 0] / W
            grid[..., 1] = grid[..., 1] / H
            grid = 2 * grid - 1
            depth_samples = F.grid_sample(depth[None, None], grid, padding_mode='zeros', align_corners=True).squeeze()

            edge = 0
            if eval_rec:
                mask = (depth_samples + truncation >= -z[:, 0, 0]) & (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
            else:
                mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W -edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)

            mask = mask.cpu().numpy()

            whole_mask &= ~mask

        face_mask = whole_mask[mesh.faces].all(axis=1)
        mesh.update_faces(~face_mask)
        mesh.remove_unreferenced_vertices()
        mesh.process(validate=False)

        mesh_ext = mesh_file.split('.')[-1]
        output_file = mesh_file[:-len(mesh_ext) - 1] + '_culled.' + mesh_ext

        mesh.export(output_file)
        












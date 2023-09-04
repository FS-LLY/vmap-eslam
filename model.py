import torch
import torch.nn as nn
from common import normalize_3d_coordinate
import torch.nn.functional as F

def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_fn(m.weight)


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )


class OccupancyMap(torch.nn.Module):
    def __init__(
        self,
        emb_size1,
        emb_size2,
        hidden_size=256,
        do_color=True,
        hidden_layers_block=1
    ):
        super(OccupancyMap, self).__init__()
        self.do_color = do_color
        self.embedding_size1 = emb_size1
        self.in_layer = fc_block(self.embedding_size1, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)
        # self.embedding_size2 = 21*(5+1)+3 - self.embedding_size # 129-66=63 32
        self.embedding_size2 = emb_size2
        self.cat_layer = fc_block(
            hidden_size + self.embedding_size1, hidden_size)

        # self.cat_layer = fc_block(
        #     hidden_size , hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        if self.do_color:
            self.color_linear = fc_block(self.embedding_size2 + hidden_size, hidden_size)
            self.out_color = torch.nn.Linear(hidden_size, 3)

        # self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x,
                noise_std=None,
                do_alpha=True,
                do_color=True,
                do_cat=True):
        fc1 = self.in_layer(x[...,:self.embedding_size1])
        fc2 = self.mid1(fc1)
        # fc3 = self.cat_layer(fc2)
        if do_cat:
            fc2_x = torch.cat((fc2, x[...,:self.embedding_size1]), dim=-1)
            fc3 = self.cat_layer(fc2_x)
        else:
            fc3 = fc2
        fc4 = self.mid2(fc3)

        alpha = None
        if do_alpha:
            raw = self.out_alpha(fc4)   # todo ignore noise
            if noise_std is not None:
                noise = torch.randn(raw.shape, device=x.device) * noise_std
                raw = raw + noise

            # alpha = self.relu(raw) * scale    # nerf
            alpha = raw * 10. #self.scale     # unisurf

        color = None
        if self.do_color and do_color:
            fc4_cat = self.color_linear(torch.cat((fc4, x[..., self.embedding_size1:]), dim=-1))
            raw_color = self.out_color(fc4_cat)
            color = self.sigmoid(raw_color)

        return alpha, color

class ESLAMdecoder(torch.nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(self, c_dim=32, hidden_size=16, truncation=0.08, n_blocks=2, learnable_beta=True, bound=None):
        super().__init__()

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks
        self.bound = bound

        ## layers for SDF decoder
        self.linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        ## layers for RGB decoder
        self.c_linears = nn.ModuleList(
            [nn.Linear(2 * c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)#out of memory
        feat = torch.cat(feat, dim=-1)

        return feat

    def get_raw_sdf(self, p_nor, all_planes):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            sdf (tensor): raw SDF
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        feat = self.sample_plane_feature(p_nor, planes_xy, planes_xz, planes_yz)

        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, p_nor, all_planes):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB
        """
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        h = c_feat
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def forward(self, p, all_planes):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf = self.get_raw_sdf(p_nor, all_planes)
        rgb = self.get_raw_rgb(p_nor, all_planes)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)

        return raw
        #return sdf,rgb
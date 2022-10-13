import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .camera import (
    get_rotation_matrix,
    get_camera_mat,
    get_random_pose,
    uvr_to_pose
)
import torch.nn as nn
import torch.nn.functional as F
import torch
from .common import (
    arange_pixels, image_points_to_world, origin_to_world
)
import math
from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
import random


############################### GIRAFFE ###############################

def from_euler(rval):
    device = rval.device
    angle = rval * 2 * np.pi
    cos = torch.cos(angle).to(device)
    sin = torch.sin(angle).to(device)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


class BoundingBoxGenerator(nn.Module):
    ''' Bounding box generator class

    Args:
        n_boxes (int): number of bounding boxes (excluding background)
        scale_range_min (list): min scale values for x, y, z
        scale_range_max (list): max scale values for x, y, z
        translation_range_min (list): min values for x, y, z translation
        translation_range_max (list): max values for x, y, z translation
        rotation_range (list): min and max rotation value (between 0 and 1)
        fix_scale_ratio (bool): whether the x/y/z scale ratio should be fixed
    '''

    def __init__(
        self,
        device,
        scale_range_min=[0.5, 0.5, 0.5],
        scale_range_max=[0.5, 0.5, 0.5],
        translation_range_min=[-0.75, -0.75, 0.],
        translation_range_max=[0.75, 0.75, 0.],
        rotation_range=[0., 1.],
        fix_scale_ratio=True,
        **kwargs
        ):
        super().__init__()

        self.n_boxes = 1
        self.device = device
        self.scale_min = torch.tensor(scale_range_min).reshape(1, 1, 3)
        self.scale_range = (torch.tensor(scale_range_max) - torch.tensor(scale_range_min)).reshape(1, 1, 3)

        self.translation_min = torch.tensor(translation_range_min).reshape(1, 1, 3)
        self.translation_range = (torch.tensor(translation_range_max) - torch.tensor(translation_range_min)).reshape(1, 1, 3)

        self.rotation_range = rotation_range
        self.fix_scale_ratio = fix_scale_ratio

    def get_random_offset(self, batch_size):
        n_boxes = self.n_boxes
        # Sample sizes
        if self.fix_scale_ratio:
            s_rand = torch.rand(batch_size, n_boxes, 1)
        else:
            s_rand = torch.rand(batch_size, n_boxes, 3)
        s = self.scale_min + s_rand * self.scale_range

        # Sample translations
        t = self.translation_min + torch.rand(batch_size, n_boxes, 3) * self.translation_range

        def r_val(): return self.rotation_range[0] + np.random.rand() * (
            self.rotation_range[1] - self.rotation_range[0])

        rval = torch.tensor([r_val() for _ in range(batch_size * self.n_boxes)]).float().view(-1, 1).to(self.device)
        R = from_euler(rval)
        return s, t, R, rval

    def forward(self, batch_size, return_rval=False):
        s, t, R, rval = self.get_random_offset(batch_size)

        if return_rval:
            return s, t, R, rval
        return s, t, R


class Decoder(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
        use_z_app (bool): whether use z_app/z_shape for feature
    '''

    def __init__(
        self,
        hidden_size=128,
        n_blocks=8,
        n_blocks_view=1,
        skips=[4],
        use_viewdirs=False,
        n_freq_posenc=10,
        n_freq_posenc_views=4,
        z_dim=64,
        rgb_out_dim=128,
        final_sigmoid_activation=False,
        downscale_p_by=2.,
        positional_encoding='normal',
        gauss_dim_pos=10,
        gauss_dim_view=4,
        gauss_std=4.,
        use_z_app=False,
        **kwargs
    ):

        super().__init__()
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.n_blocks = n_blocks
        self.use_viewdirs = use_viewdirs
        self.use_z_app = use_z_app

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Density Prediction Layers
        self.dense_layers = DenseLayers(
            z_dim, hidden_size, dim_embed, n_blocks, skips)

        # Feature Prediction Layers
        self.feat_layers = FeatLayers(
            z_dim, hidden_size, dim_embed_view, rgb_out_dim, use_viewdirs, n_blocks_view, final_sigmoid_activation)


    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (np.pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * np.pi * p),
                 torch.cos((2 ** i) * np.pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d=None, z_shape=None, z_app=None, **kwargs):
        p = self.transform_points(p_in)

        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)

        sigma_out, net = self.dense_layers(p, z_shape)

        if self.use_z_app:
            feat_out = self.feat_layers(net, z_app, ray_d)
        else:
            feat_out = self.feat_layers(net, z_shape, ray_d)

        return feat_out, sigma_out


class DenseLayers(nn.Module):
    def __init__(self, z_dim, hidden_size, dim_embed, n_blocks, skips):
        super().__init__()
        self.skips = skips

        self.fc_in = nn.Linear(dim_embed, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)
        ])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(dim_embed, hidden_size) for i in range(n_skips)
            ])
        self.sigma_out = nn.Linear(hidden_size, 1)

    def forward(self, p, z):
        net = self.fc_in(p)
        net = net + self.fc_z(z).unsqueeze(1)
        net = F.relu(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = F.relu(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z).unsqueeze(1)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net).squeeze(-1)
        return sigma_out, net


class FeatLayers(nn.Module):
    def __init__(self, z_dim, hidden_size, dim_embed_view, rgb_out_dim, use_viewdirs, n_blocks_view, final_sigmoid_activation):
        super().__init__()
        self.final_sigmoid_activation = final_sigmoid_activation
        self.use_viewdirs = use_viewdirs
        self.n_blocks_view = n_blocks_view

        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(dim_embed_view + hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def forward(self, net, z, ray_d):
        net = self.feat_view(net)
        net = net + self.fc_z_view(z).unsqueeze(1)

        if self.use_viewdirs and ray_d is not None:
            net = net + self.fc_view(ray_d)
            net = F.relu(net)

            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = F.relu(layer(net))

        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)
        return feat_out


class GIRAFFEGenerator(nn.Module):
    ''' GIRAFFE Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        z_dim_bg (int): dimension of background latent code z_bg
        resolution_vol (int): resolution of volume-rendered image
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        fov (float): field of view
        bg_translation_range_min (list): min values for bg x, y, z translation
        bg_translation_range_max (list): max values for bg x, y, z translation
        bg_rotation_range (list): background rotation range (0 - 1)
        use_max_composition (bool): whether to use the max
            composition operator instead
        pos_share (bool): whether enable position sharing
    '''

    def __init__(
        self,
        device,
        z_dim=256,
        z_dim_bg=128,
        resolution_vol=16,
        rgb_out_dim=3,
        range_u=(0, 0),
        range_v=(0.25, 0.25),
        n_ray_samples=64,
        range_radius=(2.732, 2.732),
        depth_range=[0.5, 6.],
        fov=49.13,
        use_max_composition=False,
        scale_range_min=[0.5, 0.5, 0.5],
        scale_range_max=[0.5, 0.5, 0.5],
        translation_range_min=[-0.75, -0.75, 0.],
        translation_range_max=[0.75, 0.75, 0.],
        rotation_range=[0, 1],
        bg_translation_range_min=[-0.75, -0.75, 0.],
        bg_translation_range_max=[0.75, 0.75, 0.],
        bg_rotation_range=[0, 0],
        pos_share=False,
        use_viewdirs=False,
        use_z_app=False,
        **kwargs
    ):

        super().__init__()
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.fov = fov
        self.bg_rotation_range = bg_rotation_range
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.use_max_composition = use_max_composition
        self.device = device
        self.pos_share = pos_share

        self.bg_translation_min = torch.tensor(
            bg_translation_range_min).reshape(1, 3)
        self.bg_translation_range = (torch.tensor(
            bg_translation_range_max) - torch.tensor(bg_translation_range_min)
        ).reshape(1, 3)

        self.camera_matrix = get_camera_mat(fov=fov)
        self.decoder = Decoder(
            z_dim=z_dim,
            rgb_out_dim=rgb_out_dim,
            use_viewdirs=use_viewdirs,
            use_z_app=use_z_app
        )

        self.background_generator = Decoder(
            z_dim=z_dim_bg,
            hidden_size=64,
            rgb_out_dim=rgb_out_dim,
            n_blocks=4,
            downscale_p_by=12,
            skips=[],
            use_viewdirs=use_viewdirs,
            use_z_app=use_z_app
        )

        self.bounding_box_generator = BoundingBoxGenerator(
            device=device,
            z_dim=z_dim,
            scale_range_max=scale_range_max,
            scale_range_min=scale_range_min,
            translation_range_max=translation_range_max,
            translation_range_min=translation_range_min,
            rotation_range=rotation_range,
        )

    def reform_representation(self, img_rep):
        '''
        compose latent_codes, camera_matrices, transformations out of representation list
        representation list: [z_s_fg, z_a_fg, z_s_bg, z_a_bg, u, v, radius, s, t, rval]
        '''
        batch = img_rep[0].size(0)
        device = img_rep[0].device

        z_s_fg, z_a_fg, z_s_bg, z_a_bg, u, v, radius, s, t, rval = img_rep

        R = from_euler(rval)

        world_mat = uvr_to_pose((u, v, radius))

        latent_codes = (
            z_s_fg.unsqueeze(1), z_a_fg.unsqueeze(1), z_s_bg, z_a_bg)
        camera_matrices = (
            self.camera_matrix.repeat(batch, 1, 1).to(device), world_mat)
        transformations = (s.unsqueeze(1), t.unsqueeze(1), R)
        uvr = (u, v, radius)
        return latent_codes, camera_matrices, transformations, uvr, rval

    def img_representation(self, latent_codes, uvr, transformations, rval):
        '''
        transform latent_codes, transformations, uvr, rval into representation list
        '''
        device = latent_codes[0].device
        z_s_fg, z_a_fg, z_s_bg, z_a_bg = latent_codes
        s, t, _ = transformations

        u, v, radius = uvr
        if u.device != device:
            u = u.to(device)
            v = v.to(device)
            radius = radius.to(device)

        batch_size = z_s_fg.size(0)
        return [r.view(batch_size, -1) for r in [
            z_s_fg, z_a_fg, z_s_bg, z_a_bg, u, v, radius, s, t, rval]]

    def get_latent_codes(self, batch_size, tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg
        n_boxes = 1
        z_shape_obj = self.sample_z((batch_size, n_boxes, z_dim), tmp=tmp)
        z_app_obj = self.sample_z((batch_size, n_boxes, z_dim), tmp=tmp)
        z_shape_bg = self.sample_z((batch_size, z_dim_bg), tmp=tmp)
        z_app_bg = self.sample_z((batch_size, z_dim_bg), tmp=tmp)
        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def sample_z(self, size, tmp=1.):
        z = torch.randn(*size) * tmp
        z = z.to(self.device)
        return z

    def get_random_camera(self, batch_size):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat, uvr = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)

        world_mat = world_mat.to(self.device)
        camera_mat = camera_mat.to(self.device)
        return (camera_mat, world_mat), uvr

    def get_random_transformations(self, batch_size):
        device = self.device
        s, t, R, rval = self.bounding_box_generator(
            batch_size, return_rval=True)
        s, t, R = s.to(device), t.to(device), R.to(device)
        return (s, t, R), rval

    def get_rand_rep(self, batch_size):
        latent_codes = self.get_latent_codes(batch_size)
        _, uvr = self.get_random_camera(batch_size)
        transformations, rval = self.get_random_transformations(batch_size)
        img_rep = self.img_representation(latent_codes, uvr, transformations, rval)
        return img_rep

    def get_rep_from_code(self, latent_codes):
        batch_size = latent_codes[0].size(0)
        _, uvr = self.get_random_camera(batch_size)
        transformations, rval = self.get_random_transformations(batch_size)
        img_rep = self.img_representation(latent_codes, uvr, transformations, rval)
        return img_rep

    def get_random_bg_rotation(self, batch_size):
        if self.bg_rotation_range != [0., 0.]:
            bg_r = self.bg_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(Rot.from_euler(
                    'z', r_random * 2 * np.pi).as_dcm()
                ) for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()

        R_bg = R_bg.to(self.device)
        return R_bg

    def get_random_bg_transformations(self, batch_size):
        bg_t = self.bg_translation_min + torch.rand(batch_size, 3) * self.bg_translation_range
        bg_t = bg_t.to(self.device)
        return bg_t

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0,
                                scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)
                                     ).permute(0, 2, 1)).permute(
            0, 2, 1) / bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def transform_points_to_box_bg(self, p, transformations):
        bb_t, bb_R = transformations
        p_box = (bb_R @ (p - bb_t.unsqueeze(1)
                         ).permute(0, 2, 1)).permute(0, 2, 1)
        return p_box

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 bg_transformations):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_bg = self.transform_points_to_box_bg(
            pixels_world, bg_transformations)
        camera_world_bg = self.transform_points_to_box_bg(
            camera_world, bg_transformations)

        ray_bg = pixels_world_bg - camera_world_bg

        p = camera_world_bg.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_bg.unsqueeze(-2).contiguous()
        r = ray_bg.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations, i):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations, i)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations, i)
        ray_i = pixels_world_i - camera_world_i

        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p_i.shape == ray_i.shape)
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)
        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            if self.use_max_composition:
                bs, rs, ns = sigma.shape[1:]
                sigma_sum, ind = torch.max(sigma, dim=0)
                feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                     torch.arange(rs).reshape(
                                         1, -1, 1), torch.arange(ns).reshape(
                                             1, 1, -1)]
            else:
                denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
                denom_sigma[denom_sigma == 0] = 1e-4
                w_sigma = sigma / denom_sigma
                sigma_sum = torch.sum(sigma, dim=0)
                feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def get_2Dbbox(self, img_rep, return_size, n_steps=256, render_size=256, padd=0.1):
        '''
        get 3D bounding box projected 2D bounding box
        '''
        device = self.device
        res = return_size
        if return_size > render_size:
            res = render_size
        n_points = res * res
        depth_range = self.depth_range

        camera_matrices, transformations = self.reform_representation(img_rep)[1:3]
        batch_size = camera_matrices[0].size(0)

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size,
                                invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.

        pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        camera_world = origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        p_i = self.get_evaluation_points(
            pixels_world, camera_world, di, transformations, 0)[0]

        # Mask out values outside
        # padd = 0.1
        mask_box = torch.all(
            p_i <= 1. + padd, dim=-1) & torch.all(
                p_i >= -1. - padd, dim=-1)

        # Get 2d bbox
        bbox_sigma = torch.ones(
            batch_size, n_points*n_steps).to(device) * 100
        bbox_sigma[mask_box == 0] = 0.
        bbox_sigma = bbox_sigma.reshape(batch_size, n_points, n_steps)

        weights_bbox = self.calc_volume_weights(
            di, ray_vector, bbox_sigma, last_dist=0.)
        bbox = torch.sum(weights_bbox, dim=-1, keepdim=True)
        bbox = bbox.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)
        bbox = bbox.permute(0, 1, 3, 2)
        if res != return_size:
            bbox = F.interpolate(bbox, size=return_size, mode='bilinear')
        return bbox

    def volume_render_image(
        self,
        latent_codes,
        camera_matrices,
        transformations,
        bg_transformations,
        mode='train',
        not_render_background=False,
        only_render_background=False,
    ):

        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        n_points = res * res
        depth_range = self.depth_range
        batch_size = latent_codes[0].shape[0]
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes
        assert(not (not_render_background and only_render_background))

        # Arange Pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        camera_world = origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
            torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        if mode == 'train':
            di = self.add_noise_to_interval(di)

        n_boxes = latent_codes[0].shape[1]
        feat, sigma = [], []
        n_iter = n_boxes if not_render_background else n_boxes + 1
        if only_render_background:
            n_iter = 1
            n_boxes = 0

        for i in range(n_iter):
            if i < n_boxes:  # Object
                p_i, r_i = self.get_evaluation_points(
                    pixels_world, camera_world, di, transformations, i)
                z_shape_i, z_app_i = z_shape_obj[:, i], z_app_obj[:, i]

                feat_i, sigma_i = self.decoder(p_i, r_i, z_shape_i, z_app_i)

                if mode == 'train':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

                # Mask out values outside
                padd = 0.1
                mask_box = torch.all(
                    p_i <= 1. + padd, dim=-1) & torch.all(
                        p_i >= -1. - padd, dim=-1)
                sigma_i[mask_box == 0] = 0.

                # Reshape
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)
            else:  # Background
                p_bg, r_bg = self.get_evaluation_points_bg(
                    pixels_world, camera_world, di, bg_transformations)

                feat_i, sigma_i = self.background_generator(
                    p_bg, r_bg, z_shape_bg, z_app_bg)
                sigma_i = sigma_i.reshape(batch_size, n_points, n_steps)
                feat_i = feat_i.reshape(batch_size, n_points, n_steps, -1)

                if mode == 'train':
                    # As done in NeRF, add noise during training
                    sigma_i += torch.randn_like(sigma_i)

            feat.append(feat_i)
            sigma.append(sigma_i)
        sigma = F.relu(torch.stack(sigma, dim=0))
        feat = torch.stack(feat, dim=0)

        # Composite
        sigma_sum, feat_weighted = self.composite_function(sigma, feat)

        # Get Volume Weights
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)

        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(
            batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map

    def forward(
        self,
        img_rep,
        mode='train',
        not_render_background=False,
        only_render_background=False,
    ):
        latent_codes, camera_matrices, transformations, uvr, rval = \
            self.reform_representation(img_rep)

        batch_size = latent_codes[0].size(0)

        bg_R = self.get_random_bg_rotation(batch_size)
        bg_t = torch.zeros([batch_size, 3], device=self.device)

        # randomly translate bg during training
        if mode == 'train':
            bg_t = self.get_random_bg_transformations(batch_size)

        if self.pos_share:
            bg_t[:, 2] = transformations[1][:, 0, 2]  # obj scene z share

        bg_transformations = (bg_t, bg_R)

        img_rep = self.img_representation(
            latent_codes, uvr, transformations, rval)

        rgb_v = self.volume_render_image(
            latent_codes,
            camera_matrices,
            transformations,
            bg_transformations,
            mode=mode,
            not_render_background=not_render_background,
            only_render_background=only_render_background,
        )

        return rgb_v


############################### Stylegan Generator ###############################

class style_Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = style_Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = style_Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.contiguous().view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, im_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, im_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, im_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class StyleRenderer(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        im_channel=3,
        starting_feat_size=16,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        starting_dim=None,
        mix_prob=0.9,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim
        self.mix_prob = mix_prob

        self.channels = {
            4: 256,
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        if starting_dim is None:
            starting_dim = self.channels[starting_feat_size]

        # self.conv1 = StyledConv(
        #     starting_dim,
        #     self.channels[starting_feat_size],
        #     1,
        #     style_dim,
        #     blur_kernel=blur_kernel
        # )
        self.conv1 = StyledConv(
            starting_dim,
            self.channels[starting_feat_size],
            3,
            style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(
            self.channels[starting_feat_size],
            style_dim,
            im_channel=im_channel,
            upsample=False
        )

        self.log_size = int(math.log(size, 2))
        self.s_log_size = int(math.log(starting_feat_size, 2))
        self.num_layers = (self.log_size - self.s_log_size) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[starting_feat_size]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 9) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(self.s_log_size+1, self.log_size+1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, im_channel=im_channel))

            in_channel = out_channel

        self.n_latent = (self.log_size - self.s_log_size + 1) * 2

    def make_noise(self, device):
        noises = [torch.randn(1, 1, 2 ** self.s_log_size, 2 ** self.s_log_size, device=device)]

        for i in range(self.s_log_size+1, self.log_size+1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def forward(
        self,
        input_feat,
        styles,
        inject_index=None,
        mode='train',
    ):
        if mode == 'train':
            noise = [None] * self.num_layers
        else:
            noise = [
                getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
            ]

        if inject_index is None:
            if len(styles) > 1 and random.random() < self.mix_prob:
                inject_index = random.randint(0, self.n_latent-1)
            else:
                inject_index = self.n_latent

        if inject_index >= self.n_latent:
            latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        elif inject_index == 0:
            latent = styles[1].unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent0 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent1 = styles[1].unsqueeze(1).repeat(1, self.n_latent-inject_index, 1)
            latent = torch.cat((latent0, latent1), dim=1)

        out = self.conv1(input_feat, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        return skip, out


class RefineRdr(nn.Module):
    def __init__(
        self,
        style_dim,
        n_styledconv=2,
        kernal_size=3,
        im_channel=3,
        size=512,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.style_dim = style_dim
        self.size = size
        self.n_styledconv = n_styledconv

        self.channels = {
            4: 256,
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        out_ch = self.channels[size]

        self.convs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.n_styledconv):
            shape = [1, 1, self.size, self.size]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for _ in range(n_styledconv):
            self.convs.append(
                StyledConv(
                    out_ch,
                    out_ch,
                    kernal_size,
                    style_dim,
                    blur_kernel=blur_kernel
                )
            )

        self.to_rgb = ToRGB(
            out_ch,
            style_dim,
            upsample=False,
            im_channel=im_channel
        )

    def make_noise(self, device):
        noises = []
        for _ in range(self.n_styledconv):
            noises.append(torch.randn(1, 1, self.size, self.size, device=device))

        return noises

    def forward(
        self,
        input_feat,
        styles,
        mode='train',
    ):
        if mode == 'train':
            noise = [None] * self.n_styledconv
        else:
            noise = [
                getattr(self.noises, f'noise_{i}') for i in range(self.n_styledconv)
            ]

        latent = styles.unsqueeze(1).repeat(1, self.n_styledconv+1, 1)

        i = 0
        out = input_feat
        for conv in self.convs:
            out = conv(out, latent[:, i], noise=noise[i])
            i += 1

        img = self.to_rgb(out, latent[:, i])
        return img

class GIRAFFEHDGenerator(nn.Module):
    '''
    Args:
        n_mlp (int): number of latent code mapping network layers
        lr_mlp (int): learning rate of mapping network
        channel_multiplier (int): channel multiplier factor
        refine_n_styledconv (int): number of refinement network layers
        refine_kernal_size (int): conv kernal size of refinement layers
        mix_prob (float): stylemixing probability
        grf_use_mlp (bool): whether to use mapping network layer projected latent codes for giraffe
        pos_share (int): whether to use position sharing
        grf_use_z_app (bool): whether to use z_app in feature prediction in giraffe
        fg_gen_mask (bool): whether to generate fg_mask at fg/residual stage
    '''
    def __init__(
        self,
        device,
        z_dim,
        z_dim_bg,
        size,
        resolution_vol,
        feat_dim,
        range_u,
        range_v,
        fov,
        scale_range_max,
        scale_range_min,
        translation_range_max,
        translation_range_min,
        rotation_range,
        bg_translation_range_max,
        bg_translation_range_min,
        bg_rotation_range,
        n_mlp=8,
        lr_mlp=0.01,
        channel_multiplier=2,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        mix_prob=0.9,
        grf_use_mlp=True,
        pos_share=False,
        use_viewdirs=False,
        grf_use_z_app=False,
        fg_gen_mask=False,
        ):
        super().__init__()

        self.size = size
        self.grf_use_mlp = grf_use_mlp
        self.resolution_vol = resolution_vol
        self.fg_gen_mask = fg_gen_mask

        if fg_gen_mask:
            fg_dim = 4
            res_dim = 3
        else:
            fg_dim = 3
            res_dim = 4

        self.vol_generator = GIRAFFEGenerator(
            device=device,
            z_dim=z_dim,
            z_dim_bg=z_dim_bg,
            resolution_vol=resolution_vol,
            rgb_out_dim=feat_dim,
            range_u=range_u,
            range_v=range_v,
            fov=fov,
            scale_range_max=scale_range_max,
            scale_range_min=scale_range_min,
            translation_range_max=translation_range_max,
            translation_range_min=translation_range_min,
            rotation_range=rotation_range,
            bg_translation_range_max=bg_translation_range_max,
            bg_translation_range_min=bg_translation_range_min,
            bg_rotation_range=bg_rotation_range,
            pos_share=pos_share,
            use_viewdirs=use_viewdirs,
            use_z_app=grf_use_z_app
        )

        self.fg_renderer = StyleRenderer(
            size=size,
            style_dim=z_dim,
            starting_feat_size=resolution_vol,
            im_channel=fg_dim,
            channel_multiplier=channel_multiplier,
            mix_prob=mix_prob,
        )

        self.bg_renderer = StyleRenderer(
            size=size,
            style_dim=z_dim_bg,
            starting_feat_size=resolution_vol,
            im_channel=3,
            channel_multiplier=channel_multiplier,
            mix_prob=mix_prob,
        )

        self.refine_renderer = RefineRdr(
            style_dim=z_dim_bg,
            n_styledconv=refine_n_styledconv,
            kernal_size=refine_kernal_size,
            im_channel=res_dim,
            size=size,
        )

        self.styles = nn.ModuleList()
        for _z_dim in [z_dim, z_dim_bg]:
            layers = [PixelNorm()]

            for _ in range(n_mlp):
                layers.append(
                    EqualLinear(
                        _z_dim, _z_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
            self.styles.append(nn.Sequential(*layers))

    def make_noise(self, device):
        fg_noise = self.fg_renderer.make_noise(device)
        bg_noise = self.bg_renderer.make_noise(device)
        fine_noise = self.refine_renderer.make_noise(device)
        noises = [fg_noise, bg_noise, fine_noise]
        return noises

    def get_rand_rep(self, batch_size):
        return self.vol_generator.get_rand_rep(batch_size)

    def get_rep_from_code(self, latent_codes):
        return self.vol_generator.get_rep_from_code(latent_codes)

    def get_2Dbbox(self, img_rep, n_steps=64, return_size=None, render_size=None, padd=0.1):
        if self.grf_use_mlp:
            z_s_fg, z_a_fg, z_s_bg, z_a_bg = img_rep[0:4]
            w_s_fg, w_a_fg, w_s_bg, w_a_bg = [
                self.styles[0](z_s_fg), self.styles[0](z_a_fg), self.styles[1](z_s_bg), self.styles[1](z_a_bg)]
            grf_latents = [w_s_fg, w_a_fg, w_s_bg, w_a_bg]
            img_rep = grf_latents + img_rep[4:]

        if return_size is None:
            return_size = self.size

        if render_size is None:
            render_size = self.resolution_vol

        bbox = self.vol_generator.get_2Dbbox(
            img_rep, return_size=return_size, n_steps=n_steps, render_size=render_size, padd=padd)
        return bbox

    def forward(self, img_rep, inject_index=None, return_ids=[0], mode='train'):
        '''
        return_ids (list): image at specified indices to return, input [] to return all
        0: fnl_img, 1: fg_img, 2: bg_img, 3: _fg_img, 4: fg_residual_img, 5: fg_mk
        '''
        z_s_fg, z_a_fg, z_s_bg, z_a_bg = img_rep[0:4]
        w_s_fg, w_a_fg, w_a_bg = [
            self.styles[0](z_s_fg), self.styles[0](z_a_fg), self.styles[1](z_a_bg)]

        if not self.grf_use_mlp:
            grf_latents = [z_s_fg, z_a_fg, z_s_bg, z_a_bg]
        else:
            w_s_bg = self.styles[1](z_s_bg)
            grf_latents = [w_s_fg, w_a_fg, w_s_bg, w_a_bg]

        _img_rep = grf_latents + img_rep[4:]

        fg_feat = self.vol_generator(not_render_background=True, img_rep=_img_rep, mode=mode)
        bg_feat = self.vol_generator(only_render_background=True, img_rep=_img_rep, mode=mode)

        fg, fg_out = self.fg_renderer(fg_feat, [w_s_fg, w_a_fg], inject_index=inject_index, mode=mode)
        _fg_img = fg[:, 0:3]

        bg_img = torch.tanh(self.bg_renderer(bg_feat, [w_a_bg], mode=mode)[0])

        fg_residual = self.refine_renderer(fg_out, w_a_bg, mode=mode)
        fg_residual_img = fg_residual[:, 0:3]

        if self.fg_gen_mask:
            fg_mk = torch.sigmoid(fg[:, 3:4])
        else:
            fg_mk = torch.sigmoid(fg_residual[:, 3:4])

        fg_img = torch.tanh(_fg_img + fg_residual_img)

        fnl_img = fg_img * fg_mk + bg_img * (1 - fg_mk)

        img_li = [fnl_img, fg_img, bg_img, _fg_img, fg_residual_img, fg_mk]

        if return_ids != []:
            for i in range(len(img_li)):
                if i not in return_ids:
                    img_li[i] = None

        return img_li


############################### Stylegan Discriminator ###############################

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(style_Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, im_channel=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(im_channel, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4],
                        activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out



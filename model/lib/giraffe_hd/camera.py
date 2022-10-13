import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    loc, u, v = sample_on_sphere(range_u, range_v, size=(batch_size))

    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    radius = radius.unsqueeze(-1)
    loc = loc * radius

    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT, (u, v, radius)


def uvr_to_pose(uvr, invert=False):
    u, v, radius = uvr
    batch_size = u.size(0)
    device = u.device
    loc = to_sphere(u, v)
    loc = loc * radius

    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)

    return RT.to(device)


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = torch.arccos(1 - 2 * v)
    cx = torch.sin(phi) * torch.cos(theta)
    cy = torch.sin(phi) * torch.sin(theta)
    cz = torch.cos(phi)
    return torch.cat([cx, cy, cz], dim=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,)):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    u = torch.tensor(u).float().view(-1, 1)
    v = torch.tensor(v).float().view(-1, 1)

    sample = to_sphere(u, v)

    return sample, u, v


def look_at(eye, at=torch.tensor([0, 0, 0]), up=torch.tensor([0, 0, 1]), eps=1e-5):
    device = eye.device

    at = at.float().view(1, 3).to(device)
    up = up.float().view(1, 3).to(device)
    eye = eye.view(-1, 3)
    up = up.repeat(eye.size(0)//up.size(0), 1)
    eps = torch.tensor([eps]).view(1, 1).repeat(up.size(0), 1).to(device)

    z_axis = eye - at
    z_axis = z_axis / torch.max(torch.stack([torch.linalg.norm(z_axis,
                                              dim=1, keepdims=True), eps]))

    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / torch.max(torch.stack([torch.linalg.norm(x_axis,
                                              dim=1, keepdims=True), eps]))

    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.max(torch.stack([torch.linalg.norm(y_axis,
                                              dim=1, keepdims=True), eps]))

    r_mat = torch.cat(
        (x_axis.view(-1, 3, 1), y_axis.view(-1, 3, 1), z_axis.view(
            -1, 3, 1)), dim=2)

    return r_mat


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_dcm()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r

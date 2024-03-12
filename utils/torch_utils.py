# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def unit_vec(vec, dim=-1):
    """
    Returns a normalized unit vector along the provided dimension

    Args:
        vec (tensor): (..., N, ...) arbitrary vector
        dim (int): Dimension to normalize

    Returns:
        tensor: (..., N, ...) vector normalized along @dim
    """
    # type: (Tensor, int) -> Tensor
    return vec / torch.norm(vec, dim=dim, keepdim=True)


@torch.jit.script
def dot(v1, v2, dim=-1, keepdim=False):
    """
    Computes dot product between two vectors along the provided dim, optionally keeping the dimension

    Args:
        v1 (tensor): (..., N, ...) arbitrary vector
        v2 (tensor): (..., N, ...) arbitrary vector
        dim (int): Dimension to sum over for dot product
        keepdim (bool): Whether to keep dimension over which dot product is calculated

    Returns:
        tensor: (..., [1,] ...) dot product of vectors, with optional dimension kept if @keepdim is True
    """
    # type: (Tensor, Tensor, int, bool) -> Tensor
    return torch.sum(v1 * v2, dim=dim, keepdim=keepdim)


@torch.jit.script
def quat_slerp(quat0, quat1, frac, shortestpath=True, eps=1.0e-15):
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) initial quaternion
        quat1 (tensor): (..., 4) tensor where the final dim is (x,y,z,w) final quaternion
        frac (tensor): Values in [0.0, 1.0] representing fraction of interpolation
        shortestpath (bool): If True, will calculate shortest path
        eps (float): Value to check for singularities
    Returns:
        tensor: (..., 4) Interpolated
    """
    # type: (Tensor, Tensor, Tensor, bool, float) -> Tensor
    # reshape quaternion
    quat_shape = quat0.shape
    quat0 = unit_vec(quat0.reshape(-1, 4), dim=-1)
    quat1 = unit_vec(quat1.reshape(-1, 4), dim=-1)

    # Check for endpoint cases
    where_start = frac <= 0.0
    where_end = frac >= 1.0

    d = dot(quat0, quat1, dim=-1, keepdim=True)
    if shortestpath:
        quat1 = torch.where(d < 0.0, -quat1, quat1)
        d = torch.abs(d)
    angle = torch.acos(torch.clip(d, -1.0, 1.0))

    # Check for small quantities (i.e.: q0 = q1)
    where_small_diff = torch.abs(torch.abs(d) - 1.0) < eps
    where_small_angle = abs(angle) < eps

    isin = 1.0 / torch.sin(angle)
    val = quat0 * torch.sin((1.0 - frac) * angle) * isin + \
        quat1 * torch.sin(frac * angle) * isin

    # Filter edge cases
    val = torch.where(
        where_small_diff | where_small_angle | where_start,
        quat0,
        torch.where(
            where_end,
            quat1,
            val,
        ),
    )

    # Reshape and return values
    return val.reshape(list(quat_shape))


@torch.jit.script
def axisangle_between_vec(vec1, vec2, eps=1e-6):
    """
    Calculates the rotation between two 3D vectors from @vec1 to @vec2 in axis angle representation
    See https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/ for formula

    Args:
        vec1 (tensor): (..., 3) where final dim represents (x, y, z) 3D vector
        vec2 (tensor): (..., 3) where final dim represents (x, y, z) 3D vector
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 3) where final dim represents (ax, ay, az) axis-angle rotation from @vec1 to @vec2.
    """
    # type: (Tensor, Tensor, float) -> Tensor
    where_small = torch.abs(torch.sum(vec2 - vec1, dim=-1)) < eps
    # Normalize vectors
    vec1 = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
    vec2 = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
    # Calculate angle and axis
    angle = torch.acos(torch.sum(vec1 * vec2, dim=-1, keepdim=True))
    axis = torch.cross(vec1, vec2)
    axis_norm = torch.norm(axis, dim=-1)
    where_small = where_small | (axis_norm < eps)
    axis[where_small] = 0.0
    axis[~where_small] = axis[~where_small] / axis_norm[~where_small].unsqueeze(-1)
    # Combine and return value
    return angle * axis


@torch.jit.script
def rotate_vec_by_quat(vec, quat):
    """
    This function rotates a 3D vector @vec by the direction and angle represented by @quat.

    See https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    for more information.

    Args:
        vec (tensor): (..., 3) where final dim represents (x, y, z) vectors
        quat (tensor): (..., 4) where final dim represents desired (x, y, z, w) quaternion rotations
    Returns:
        tensor: (..., 3) where final dim is newly rotated (x, y, z) vectors
    """
    # Parse quaternion into vector and scalar components
    q_vec, q_s = quat[..., :3], quat[..., 3].unsqueeze(-1)
    # Calculate new vector and return
    return 2.0 * dot(q_vec, vec, keepdim=True) * q_vec + \
           (q_s * q_s - dot(q_vec, q_vec, keepdim=True)) * vec + \
           2.0 * q_s * torch.cross(q_vec, vec)


@torch.jit.script
def rotate_vec_by_axisangle(vec, aa_vec):
    """
    This function rotates a 3D vector @vec by the direction and angle represented by @aa_vec.

    See https://stackoverflow.com/questions/32485772/how-do-axis-angle-rotation-vectors-work-and-how-do-they-compare-to-rotation-matr
    for more information.

    Args:
        vec (tensor): (..., 3) where final dim represents (x, y, z) vectors
        aa_vec (tensor): (..., 3) where final dim represents desired (ax, ay, az) axis-angle rotations
    Returns:
        tensor: (..., 3) where final dim is newly rotated (x, y, z) vectors
    """
    # Extract angle and unit vector from axis-angle vectors
    angle = torch.norm(aa_vec, dim=-1, keepdim=True)
    aa_v = aa_vec / angle
    # Map all NaNs to 0
    aa_v[aa_v != aa_v] = 0.0

    # # Record angles that are zero so we don't get nan's
    # idx = torch.nonzero(angle.squeeze(dim=-1)).tolist()
    # aa_v = torch.zeros_like(aa_vec)
    # aa_v[idx] = aa_vec[idx] / angle[idx]

    # Rotate the vector using the formula (see link above)
    c, s = torch.cos(angle), torch.sin(angle)
    vec_rot = vec * c + torch.cross(aa_v, vec) * s + \
        aa_v * (torch.sum(aa_v * vec, dim=-1, keepdim=True)) * (1 - c)

    return vec_rot


@torch.jit.script
def quat_conjugate(q):
    input_shape = q.shape
    q = (-q).reshape(-1, input_shape[-1])
    q[:, 3] = -q[:, 3]
    return q.reshape(list(input_shape))


@torch.jit.script
def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (tensor): (..., 3, 3) where final two dims are 2d array representing target orientation matrix
        current (tensor): (..., 3, 3) where final two dims are 2d array representing current orientation matrix
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-2]
    desired = desired.reshape(-1, 3, 3)
    current = current.reshape(-1, 3, 3)

    # grab relevant info
    rc1 = current[:, :, 0]
    rc2 = current[:, :, 1]
    rc3 = current[:, :, 2]
    rd1 = desired[:, :, 0]
    rd2 = desired[:, :, 1]
    rd3 = desired[:, :, 2]

    error = 0.5 * (torch.cross(rc1, rd1, dim=-1) + torch.cross(rc2, rd2, dim=-1) + torch.cross(rc3, rd3, dim=-1))

    # Reshape
    error = error.reshape(list(input_shape) + [3])

    return error


@torch.jit.script
def orientation_error_from_quat(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector, where inputs are quaternions

    Args:
        desired (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
        current (tensor): (..., 4) where final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-1]
    desired = desired.reshape(-1, 4)
    current = current.reshape(-1, 4)

    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return (q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)).reshape(list(input_shape) + [3])


@torch.jit.script
def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.
    Args:
        quaternion (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3, 3) tensor whose final two dimensions are 3x3 rotation matrices
    """
    # convert quat convention
    inds = torch.tensor([3, 0, 1, 2])
    input_shape = quaternion.shape[:-1]
    q = quaternion.reshape(-1, 4)[:, inds]

    # Conduct dot product
    n = torch.bmm(q.unsqueeze(1), q.unsqueeze(-1)).squeeze(-1).squeeze(-1)        # shape (-1)
    idx = torch.nonzero(n).reshape(-1)
    q_ = q.clone()          # Copy so we don't have inplace operations that fail to backprop
    q_[idx, :] = q[idx, :] * torch.sqrt(2.0 / n[idx].unsqueeze(-1))

    # Conduct outer product
    q2 = torch.bmm(q_.unsqueeze(-1), q_.unsqueeze(1)).squeeze(-1).squeeze(-1)        # shape (-1, 4 ,4)

    # Create return array
    ret = torch.eye(3, 3, device=q.device).reshape(1, 3, 3).repeat(torch.prod(torch.tensor(input_shape)), 1, 1)
    ret[idx, :, :] = torch.stack([
        torch.stack([1.0 - q2[idx, 2, 2] - q2[idx, 3, 3], q2[idx, 1, 2] - q2[idx, 3, 0], q2[idx, 1, 3] + q2[idx, 2, 0]], dim=-1),
        torch.stack([q2[idx, 1, 2] + q2[idx, 3, 0], 1.0 - q2[idx, 1, 1] - q2[idx, 3, 3], q2[idx, 2, 3] - q2[idx, 1, 0]], dim=-1),
        torch.stack([q2[idx, 1, 3] - q2[idx, 2, 0], q2[idx, 2, 3] + q2[idx, 1, 0], 1.0 - q2[idx, 1, 1] - q2[idx, 2, 2]], dim=-1),
    ], dim=1)

    # Reshape and return output
    ret = ret.reshape(list(input_shape) + [3, 3])
    return ret


@torch.jit.script
def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    Args:
        quat (tensor): (..., 4) tensor where the final dim is (x,y,z,w) quaternion
    Returns:
        tensor: (..., 3) axis-angle exponential coordinates
    """
    # reshape quaternion
    quat_shape = quat.shape[:-1]      # ignore last dim
    quat = quat.reshape(-1, 4)
    # clip quaternion
    quat[:, 3] = torch.clamp(quat[:, 3], -1., 1.)
    # Calculate denominator
    den = torch.sqrt(1. - quat[:,3] * quat[:,3])
    # Map this into a mask

    # Create return array
    ret = torch.zeros_like(quat)[:, :3]
    idx = torch.nonzero(den).reshape(-1)
    ret[idx, :] = (quat[idx, :3] * 2. * torch.acos(quat[idx, 3]).unsqueeze(-1)) / den[idx].unsqueeze(-1)

    # Reshape and return output
    ret = ret.reshape(list(quat_shape) + [3, ])
    return ret


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps #torch.nonzero(angle).reshape(-1)
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def compute_heading_and_up(
    torso_rotation, inv_start_rot, to_target, vec0, vec1, up_idx
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(
        num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


@torch.jit.script
def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    vel_loc = quat_rotate_inverse(torso_quat, velocity)
    angvel_loc = quat_rotate_inverse(torso_quat, ang_velocity)

    roll, pitch, yaw = get_euler_xyz(torso_quat)

    walk_target_angle = torch.atan2(targets[:, 2] - torso_positions[:, 2],
                                    targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw

    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target
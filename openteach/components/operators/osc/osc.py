from isaacgym import gymapi, gymutil
import numpy as np
#from utils import clamp, AssetDesc
import math
import hydra
from copy import copy
import gym
from gym.spaces import Box
#import torch





import cv2
from plots.data_logging import Log, ListOfLogs, NoLog, SimpleLog 
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import time
#import torch

#from isaacgymenvs.tasks.base.vec_task import VecTask
#from isaacgymenvs.tasks.base.vec_task import VecTask
#gym=gymapi.aquire_gym()

#@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'envs')
from isaacgym import gymapi
import torch
import numpy as np
from oscar.utils.torch_utils import quat_mul, quat2mat, orientation_error, axisangle2quat
from .base_controller import Controller


class OSCController(Controller):
    """
    Operational Space Controller. Leverages impedance-based end effector control.

    This controller expects 6DOF delta commands (dx, dy, dz, dax, day, daz), where the delta orientation
    commands are in axis-angle form, and outputs low-level torque commands.

    Gains may also be considered part of the action space as well. In this case, the action space would be:
        (
            dx, dy, dz, dax, day, daz                       <-- 6DOF delta eef commands
            [, kpx, kpy, kpz, kpax, kpay, kpaz]             <-- kp gains
            [, drx dry, drz, drax, dray, draz]              <-- damping ratio gains
            [, kpnx, kpny, kpnz, kpnax, kpnay, kpnaz]       <-- kp null gains
        )

    Note that in this case, we ASSUME that the inputted gains are normalized to be in the range [-1, 1], and will
    be mapped appropriately to their respective ranges, as defined by XX_limits

    Alternatively, parameters (in this case, kp or damping_ratio) can either be set during initialization or provided
    from an external source; if the latter, the control_dict should include the respective parameter(s) as
    a part of its keys

    Args:
        input_min (int, float, or array): Minimum values below which received commands will be clipped
        input_max (int, float, or array): Maximum values above which received commands will be clipped
        output_min (int, float, or array): Lower end of range that received commands will be mapped to
        output_max (int, float, or array): Upper end of range that received commands will be mapped to
        control_min (int, float, or array): Minimum control values below which outputted controls will be clipped
        control_max (int, float, or array): Maximum control values above which outputted controls will be clipped
        control_noise (float): Amount of noise to apply. Should be in [0, 1)
        control_dim (int): Outputted control dimension -- should be number of joints from base to eef body frame
        device (str): Which device to send all tensors to by default
        kp (None, int, float, or array): Gain values to apply to 6DOF error.
            If None, will be variable (part of action space)
        kp_limits (2-array): (min, max) values of kp
        damping_ratio (None, int, float, or array): Damping ratio to apply to 6DOF error controller gain
            If None, will be variable (part of action space)
        damping_ratio_limits (2-array): (min, max) values of damping ratio
        kp_null (None, int, float, or array): Gain applied when calculating null torques
            If None, will be variable (part of action space)
        kp_null_limits (2-array): (min, max) values of kp_null
        rest_qpos (None, int, float, or array): If not None, sets the joint configuration used for null torques
        decouple_pos_ori (bool): Whether to decouple position and orientation control or not
        normalize_control (bool): Whether or not to normalize outputted controls to (-1, 1) range
    """
    def __init__(
        self,
        input_min,
        input_max,
        output_min,
        output_max,
        control_min,
        control_max,
        control_noise,
        control_dim,
        device,
        kp=150.0,
        kp_limits=(10.0, 300.),
        damping_ratio=1.0,
        damping_ratio_limits=(0.0, 2.0),
        kp_null=10.0,
        kp_null_limits=(0.0, 50.0),
        rest_qpos=None,
        decouple_pos_ori=False,
        normalize_control=True,
        **kwargs,                   # hacky way to sink extraneous args
    ):
        # Run super init first
        super().__init__(
            command_dim=6,
            input_min=input_min,
            input_max=input_max,
            output_min=output_min,
            output_max=output_max,
            control_min=control_min,
            control_max=control_max,
            control_noise=control_noise,
            control_dim=control_dim,
            device=device,
            normalize_control=normalize_control,
        )

        # Store gains
        self.kp = self.nums2tensorarray(nums=kp, dim=6) if kp is not None else None
        self.damping_ratio = damping_ratio
        self.kp_null = self.nums2tensorarray(nums=kp_null, dim=self.control_dim) if kp_null is not None else None
        self.kd_null = 2 * torch.sqrt(self.kp_null) if kp_null is not None else None  # critically damped
        self.kp_limits = np.array(kp_limits)
        self.damping_ratio_limits = np.array(damping_ratio_limits)
        self.kp_null_limits = np.array(kp_null_limits)

        # Store settings for whether we're learning gains or not
        self.variable_kp = self.kp is None
        self.variable_damping_ratio = self.damping_ratio is None
        self.variable_kp_null = self.kp_null is None

        # Modify input / output scaling based on whether we expect gains to be part of the action space
        for variable_gain, gain_limits, dim in zip(
            (self.variable_kp, self.variable_damping_ratio, self.variable_kp_null),
            (self.kp_limits, self.damping_ratio_limits, self.kp_null_limits),
            (6, 6, self.control_dim),
        ):
            if variable_gain:
                # Add this to input / output limits
                self.input_min = torch.cat([self.input_min, self.nums2tensorarray(nums=-1., dim=dim)])
                self.input_max = torch.cat([self.input_max, self.nums2tensorarray(nums=1., dim=dim)])
                self.output_min = torch.cat([self.output_min, self.nums2tensorarray(nums=gain_limits[0], dim=dim)])
                self.output_max = torch.cat([self.output_max, self.nums2tensorarray(nums=gain_limits[1], dim=dim)])
                # Update command dim
                self.command_dim += dim

        # Other values
        self.rest_qpos = self.nums2tensorarray(nums=rest_qpos, dim=self.control_dim) if rest_qpos is not None else None
        self.decouple_pos_ori = decouple_pos_ori

        # Initialize internal vars
        self.n_envs = None
        self.goal_pos = None
        self.goal_ori_mat = None

    def update_goal(self, control_dict, command, env_ids=None, train=False):
        """
        Updates the internal goal (ee pos and ee ori mat) based on the inputted delta command

        Args:
            control_dict (dict): Dictionary of keyword-mapped tensors including relevant control
                information (eef state, q states, etc.)

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body

            command (tensor): 6+DOF EEF command -- first 6 dimensions should be (dx, dy, dz, dax, day, daz), where the
                delta orientation commands are in axis angle form

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances should have gains updated

            train (bool): If True, will assume env_ids is None and will NOT index specific goals so we avoid inplace
                operations and so that we can backprop later
        """
        # Get useful state info
        ee_pos = control_dict["eef_state"][:, :3]
        ee_quat = control_dict["eef_state"][:, 3:7]

        # Scale the commands appropriately
        command = self.scale_command(command)
        dpose = command[:, :6]
        gains = command[:, 6:]

        # Set n_envs, goal_pos, goal_ori, (and maybe gains) if we haven't done so already or if we need to update values
        if self.n_envs is None or command.shape[0] != self.n_envs:
            self.n_envs = command.shape[0]
            self.goal_pos = torch.zeros(self.n_envs, 3, device=self.device)
            self.goal_ori_mat = torch.zeros(self.n_envs, 3, 3, device=self.device)
            self._reset_variable_gains()

        # If we're training, make sure env_ids is None
        if train:
            assert env_ids is None or len(env_ids) == self.n_envs, \
                "When in training mode, env_ids must be None or len of n_envs!"
            # Directly set goals
            self.goal_pos = ee_pos + dpose[:, :3]
            self.goal_ori_mat = quat2mat(quat_mul(axisangle2quat(dpose[:, 3:6]), ee_quat))
            self._update_variable_gains(gains=gains, env_ids=env_ids, train=train)
        else:
            # If env_ids is None, we update all the envs
            if env_ids is None:
                # DON'T use individual indexes since this breaks backpropping
                env_ids = torch.arange(start=0, end=self.n_envs, device=self.device, dtype=torch.long)

            # Update specific goals
            self.goal_pos[env_ids] = ee_pos[env_ids] + dpose[env_ids, :3]
            self.goal_ori_mat[env_ids] = quat2mat(quat_mul(axisangle2quat(dpose[env_ids, 3:6]), ee_quat[env_ids]))
            self._update_variable_gains(gains=gains, env_ids=env_ids, train=train)

    def compute_control(self, control_dict):
        """
        Computes low-level torque controls using internal eef goal pos / ori.

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                    q: shape of (N, N_dof), current joint positions
                    qd: shape of (N, N_dof), current joint velocities
                    mm: shape of (N, N_dof, N_dof), current mass matrix
                    j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                    the relevant elements to be used for the osc computations

        Returns:
            tensor: Processed low-level torque control actions
        """
        # Possibly grab parameters from dict, otherwise, use internal values
        kp = self.nums2tensorarray(nums=control_dict["kp"], dim=6) if \
            "kp" in control_dict else self.kp
        damping_ratio = self.nums2tensorarray(nums=control_dict["damping_ratio"], dim=6) if \
            "damping_ratio" in control_dict else self.damping_ratio
        kd = 2 * torch.sqrt(kp) * damping_ratio

        # Calculate torques
        u = _compute_osc_torques(
            control_dict=control_dict,
            goal_pos=self.goal_pos,
            goal_ori_mat=self.goal_ori_mat,
            kp=kp,
            kd=kd,
            kp_null=self.kp_null,
            kd_null=self.kd_null,
            rest_qpos=self.rest_qpos,
            control_dim=self.control_dim,
            decouple_pos_ori=self.decouple_pos_ori,
            device=self.device,
        )

        # Post-process torques (clipping + normalization)
        u = self.postprocess_control(u.squeeze(-1))

        # Return the control torques
        return u

    def reset(self, control_dict, env_ids=None):
        """
        Reset the internal vars associated with this controller

        Args:
            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                Expected keys:
                    eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                    q: shape of (N, N_dof), current joint positions
                    qd: shape of (N, N_dof), current joint velocities
                    mm: shape of (N, N_dof, N_dof), current mass matrix
                    j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                    the relevant elements to be used for the osc computations

            env_ids (None or tensor): If specified, should be (integer) IDs corresponding to the
                specific env instances of this policy that should be reset
        """
        # Clear n_envs, goal pos, goal ori (and maybe gains) if we're now controlling a new set of envs
        n_cmds = control_dict["eef_state"].shape[0]
        if self.n_envs != n_cmds:
            self.n_envs = None
            self.goal_pos = None
            self.goal_ori_mat = None
            self._clear_variable_gains()
        # Reset corresponding envs to current positions
        self.update_goal(
            control_dict=control_dict,
            command=torch.zeros(n_cmds, 6),
            env_ids=env_ids
        )

    def get_flattened_goals(self):
        """
        Returns the current goal command in a serialized 2D form

        Returns:
            torch.tensor: (N, -1) current goals in this controller
        """
        return torch.cat([self.goal_pos, self.goal_ori_mat.view(-1, 9)], dim=-1)

    def _clear_variable_gains(self):
        """
        Helper function to clear any gains that we're are variable and considered part of actions
        """
        if self.variable_kp:
            self.kp = None
        if self.variable_damping_ratio:
            self.damping_ratio = None
        if self.variable_kp_null:
            self.kp_null = None
            self.kd_null = None

    def _reset_variable_gains(self):
        """
        Helper function to zero-out any gains that we're are variable and considered part of actions
        """
        if self.variable_kp:
            self.kp = torch.zeros(self.n_envs, 6, device=self.device)
        if self.variable_damping_ratio:
            self.damping_ratio = torch.zeros(self.n_envs, 6, device=self.device)
        if self.variable_kp_null:
            self.kp_null = torch.zeros(self.n_envs, self.control_dim, device=self.device)
            self.kd_null = torch.zeros(self.n_envs, self.control_dim, device=self.device)

    def _update_variable_gains(self, gains, env_ids, train=False):
        """
        Helper function to update any gains that we're are variable and considered part of actions

        Args:
            gains (tensor): (n_envs, X) tensor where X dim is parsed based on which gains are being learned
            env_ids (tensor): 1D Integer IDs corresponding to the
                specific env instances that should have gains updated
        """
        idx = 0
        # Take train and testing case separately
        if train:
            # Ignore indexing
            if self.variable_kp:
                self.kp = gains[:, idx:idx+6]
                idx += 6
            if self.variable_damping_ratio:
                self.damping_ratio = gains[:, idx:idx+6]
                idx += 6
            if self.variable_kp_null:
                self.kp_null = gains[:, idx:idx+self.control_dim]
                self.kd_null = 2 * torch.sqrt(self.kp_null) # critically damped
                idx += self.control_dim
        else:
            # Use indexing
            if self.variable_kp:
                self.kp[env_ids] = gains[env_ids, idx:idx+6]
                idx += 6
            if self.variable_damping_ratio:
                self.damping_ratio[env_ids] = gains[env_ids, idx:idx+6]
                idx += 6
            if self.variable_kp_null:
                self.kp_null[env_ids] = gains[env_ids, idx:idx+self.control_dim]
                self.kd_null[env_ids] = 2 * torch.sqrt(self.kp_null) # critically damped
                idx += self.control_dim

    @property
    def goal_dim(self):
        # This is 3 from pos goals + 9 from the ee ori goal
        return 12

    @property
    def control_type(self):
        # This controller outputs torques
        return gymapi.DOF_MODE_EFFORT

    @property
    def differentiable(self):
        # We can backprop through all osc computations
        return True


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def _compute_osc_torques(
    control_dict,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    control_dim,
    decouple_pos_ori,
    device,
):
    # type: (Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, str) -> Tensor
    # Extract relevant values from the control dict
    q = control_dict["q"][:, :control_dim].to(device)
    qd = control_dict["qd"][:, :control_dim].to(device)
    mm = control_dict["mm"][:, :control_dim, :control_dim]       # Keep mm on cpu because running the mm inverse is OOM quicker, see https://github.com/pytorch/pytorch/issues/42265
    j_eef = control_dict["j_eef"][:, :, :control_dim].to(device)
    ee_pos = control_dict["eef_state"][:, :3].to(device)
    ee_quat = control_dict["eef_state"][:, 3:7].to(device)
    ee_vel = control_dict["eef_state"][:, 7:].to(device)
    # Compute the inverse
    mm_inv = torch.inverse(mm.cpu()).to(device)

    # Calculate error
    pos_err = goal_pos - ee_pos
    ori_err = orientation_error(goal_ori_mat, quat2mat(ee_quat))
    err = torch.cat([pos_err, ori_err], dim=1)

    # Determine desired wrench
    err = (kp * err - kd * ee_vel).unsqueeze(-1)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    if decouple_pos_ori:
        m_eef_pos_inv = j_eef[:, :3, :] @ mm_inv @ torch.transpose(j_eef[:, :3, :], 1, 2)
        m_eef_ori_inv = j_eef[:, 3:, :] @ mm_inv @ torch.transpose(j_eef[:, 3:, :], 1, 2)
        m_eef_pos = torch.inverse(m_eef_pos_inv)
        m_eef_ori = torch.inverse(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:, :3, :]
        wrench_ori = m_eef_ori @ err[:, 3:, :]
        wrench = torch.cat([wrench_pos, wrench_ori], dim=1)
    else:
        wrench = m_eef @ err

    # Compute OSC torques
    u = torch.transpose(j_eef, 1, 2) @ wrench

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + np.pi) % (2 * np.pi) - np.pi)
        u_null = mm @ u_null.unsqueeze(-1)
        u += (torch.eye(control_dim).unsqueeze(0).to(device) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null

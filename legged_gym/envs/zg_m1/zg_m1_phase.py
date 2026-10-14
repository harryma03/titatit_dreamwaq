import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.zg_m1.zg_m1 import ZGM1Robot


class ZGM1PhaseRobot(ZGM1Robot):
    """Optional alternating lift for lateral/turn commands; rolling remains unphased."""

    def _create_envs(self):
        super()._create_envs()
        foot_names = ("FAR_FOOT_LINK", "FBL_FOOT_LINK", "RAR_FOOT_LINK", "RBL_FOOT_LINK")
        handles = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                   for name in foot_names]
        if len(self.feet_indices) != 4 or any(handle < 0 for handle in handles):
            raise ValueError("ZG-M1 phase gait requires FAR/FBL/RAR/RBL foot links")
        self.phase_foot_indices = torch.tensor(handles, dtype=torch.long, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_was_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_gain = torch.zeros(self.num_envs, device=self.device)
        self.phase_angle = torch.zeros(self.num_envs, device=self.device)
        self.phase_obs = torch.zeros(self.num_envs, 2, device=self.device)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.phase_steps[env_ids] = 0
        self.phase_gain[env_ids] = self._phase_gain_from_commands()[env_ids]
        self.phase_was_active[env_ids] = self.phase_gain[env_ids] > 0
        self.phase_angle[env_ids] = 0
        self.phase_obs[env_ids, 0] = 0
        self.phase_obs[env_ids, 1] = self.phase_gain[env_ids]

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        count = len(env_ids)
        if count == 0:
            return
        mode = torch.rand(count, device=self.device)
        turn_ids = env_ids[mode < 0.25]
        side_ids = env_ids[(mode >= 0.25) & (mode < 0.5)]
        self.commands[turn_ids, :2] = 0
        turn = torch_rand_float(-1, 1, (len(turn_ids), 1), device=self.device).squeeze(1)
        self.commands[turn_ids, 2] = torch.sign(turn) * (0.4 + 0.6 * torch.abs(turn))
        self.commands[side_ids, 0] = 0
        self.commands[side_ids, 2] = 0
        side = torch_rand_float(-1, 1, (len(side_ids), 1), device=self.device).squeeze(1)
        self.commands[side_ids, 1] = torch.sign(side) * (0.3 + 0.3 * torch.abs(side))

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.phase_gain = self._phase_gain_from_commands()
        active = self.phase_gain > 0
        entered = active & ~self.phase_was_active
        self.phase_steps = torch.where(active, self.phase_steps + (~entered).long(), 0)
        self.phase_was_active = active
        self.phase_angle = self.phase_steps * (2 * math.pi * self.dt / self.cfg.phase_gait.period)
        self.phase_obs = self.phase_gain.unsqueeze(1) * torch.stack(
            (torch.sin(self.phase_angle), torch.cos(self.phase_angle)), dim=1)

    def _phase_gain_from_commands(self):
        cfg = self.cfg.phase_gait
        lateral = torch.clamp((torch.abs(self.commands[:, 1]) - cfg.lateral_start) / cfg.lateral_ramp, 0, 1)
        turning = torch.clamp((torch.abs(self.commands[:, 2]) - cfg.yaw_start) / cfg.yaw_ramp, 0, 1)
        forward = 1 - torch.clamp((torch.abs(self.commands[:, 0]) - cfg.forward_start) / cfg.forward_ramp, 0, 1)
        return torch.maximum(lateral, turning) * forward

    def compute_observations(self):
        heights = torch.clamp(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1) * self.obs_scales.height_measurements
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.phase_obs,
        ), dim=1)
        self.vel_buf = self.base_lin_vel
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, heights, self.obs_buf), dim=1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_phase_clearance(self):
        foot_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.phase_foot_indices, :3]
        points = ((foot_pos[:, :, :2] + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        px = torch.clamp(points[:, :, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clamp(points[:, :, 1], 0, self.height_samples.shape[1] - 2)
        terrain_height = torch.minimum(torch.minimum(self.height_samples[px, py],
                                  self.height_samples[px + 1, py]),
                                  self.height_samples[px, py + 1]) * self.terrain.cfg.vertical_scale
        clearance = foot_pos[:, :, 2] - terrain_height - self.cfg.control.wheel_radius
        swing = self._phase_swing()
        target = self.cfg.phase_gait.swing_clearance * swing
        return self.phase_gain * torch.mean((clearance - target) ** 2, dim=1)

    def _reward_phase_swing_contact(self):
        contact = (self.contact_forces[:, self.phase_foot_indices, 2] > 1.0).float()
        return self.phase_gain * torch.mean(self._phase_swing() * contact, dim=1)

    def _phase_swing(self):
        positive = torch.clamp(torch.sin(self.phase_angle), min=0)
        negative = torch.clamp(-torch.sin(self.phase_angle), min=0)
        return torch.stack((positive, negative, negative, positive), dim=1)

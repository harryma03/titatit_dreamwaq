import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.fdwheel.fdwheel import FDWheelRobot


class FDWheelM20Robot(FDWheelRobot):
    """FDWheel with M20-inspired pure-turn sampling and contact rewards."""

    def _init_buffers(self):
        super()._init_buffers()
        self.heading_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.direct_yaw_commands = torch.zeros(self.num_envs, device=self.device)
        self.rotation_history_size = max(1, round(self.cfg.rewards.rotation_duty_time / self.dt))
        self.rotation_contact_history = torch.zeros(
            self.num_envs, 4, self.rotation_history_size, dtype=torch.bool, device=self.device)
        self.rotation_contact_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation_contact_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rotation_history_index = 0
        self.rotation_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation_ground_height = torch.zeros_like(self.rotation_air_time)
        self.rotation_max_air_height = torch.zeros_like(self.rotation_air_time)
        self.rotation_has_ground = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.rotation_last_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.rotation_air_reward = torch.zeros(self.num_envs, device=self.device)
        self._update_gait_level()
        self.haa_indices = self._joint_indices("_HAA")
        self.hfe_indices = self._joint_indices("_HFE")
        self.kfe_indices = self._joint_indices("_KFE")
        # The existing height observation has one sample directly below the base.
        self.base_height_index = (
            list(self.cfg.terrain.measured_points_x).index(0.0) * len(self.cfg.terrain.measured_points_y)
            + list(self.cfg.terrain.measured_points_y).index(0.0))
        self.mirror_pairs = [
            (self.dof_names.index(left + suffix), self.dof_names.index(right + suffix))
            for left, right in (("LF_", "RH_"), ("LH_", "RF_"))
            for suffix in ("HAA", "HFE", "KFE")
        ]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids):
            self.rotation_contact_history[env_ids] = False
            self.rotation_contact_sum[env_ids] = 0.0
            self.rotation_contact_samples[env_ids] = 0
            self.rotation_air_time[env_ids] = 0.0
            self.rotation_ground_height[env_ids] = 0.0
            self.rotation_max_air_height[env_ids] = 0.0
            self.rotation_has_ground[env_ids] = False
            self.rotation_last_contact[env_ids] = False
            self.rotation_air_reward[env_ids] = 0.0
            self._update_gait_level()
            self.extras["episode"]["gait_level"] = self.gait_level
            self.extras["episode"].update(getattr(self, "terrain_progress", {}))

    def _update_gait_level(self):
        if self.cfg.terrain.curriculum and hasattr(self, "terrain_levels"):
            level = self.terrain_levels.float().mean()
            self.gait_level = torch.where(
                level > 0, torch.exp(level - self.cfg.rewards.gait_level_full_terrain).clamp(max=1.0), 0.0)
        else:
            self.gait_level = torch.ones((), device=self.device)

    def _update_terrain_curriculum(self, env_ids):
        if self.init_done and len(env_ids):
            distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
            up = distance > self.terrain.env_length / 2
            down = (distance < self.commands[env_ids, :2].norm(dim=1) * self.max_episode_length_s * 0.5) & ~up
            self.terrain_progress = {
                "terrain_move_up_rate": up.float().mean(),
                "terrain_move_down_rate": down.float().mean(),
                "terrain_max_level_wrap_rate": (up & (self.terrain_levels[env_ids] == self.max_terrain_level - 1)).float().mean(),
            }
        super()._update_terrain_curriculum(env_ids)

    def check_termination(self):
        super().check_termination()
        # Reward is evaluated before reset, so this penalty is charged only once.
        self.reset_buf |= self._bad_orientation_mask()

    def _bad_orientation_mask(self):
        return self.projected_gravity[:, 2] > self.cfg.rewards.bad_orientation_gravity_z

    def _joint_indices(self, suffix):
        indices = [i for i, name in enumerate(self.dof_names) if name.endswith(suffix)]
        if len(indices) != 4:
            raise ValueError(f"Expected four {suffix} joints, got {indices}")
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _compute_torques(self, actions):
        scaled = actions.clone()
        scaled[:, self.haa_indices] *= self.cfg.control.haa_action_scale / self.cfg.control.action_scale
        return super()._compute_torques(scaled)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        n = len(env_ids)
        if not n:
            return
        cfg = self.cfg.commands
        fractions = (cfg.zero_command_fraction, cfg.pure_yaw_fraction,
                     cfg.pure_lateral_fraction, cfg.pure_forward_fraction)
        if any(value < 0 for value in fractions) or sum(fractions) > 1.0:
            raise ValueError("Command sampling fractions must be nonnegative and sum to <= 1")
        sample = torch.rand(n, device=self.device)
        zero = sample < fractions[0]
        yaw = (sample >= fractions[0]) & (sample < sum(fractions[:2]))
        lateral = (sample >= sum(fractions[:2])) & (sample < sum(fractions[:3]))
        forward = (sample >= sum(fractions[:3])) & (sample < sum(fractions))
        self.commands[env_ids[zero], :3] = 0.0
        self.commands[env_ids[yaw], :2] = 0.0
        self.commands[env_ids[lateral], 0] = 0.0
        self.commands[env_ids[lateral], 2] = 0.0
        self.commands[env_ids[forward], 1:3] = 0.0
        # Sample the full dedicated-axis range; do not turn tiny inputs into a 1 rad/s turn.
        for mask, axis, limits in ((yaw, 2, self.command_ranges["ang_vel_yaw"]),
                                   (lateral, 1, self.command_ranges["lin_vel_y"]),
                                   (forward, 0, self.command_ranges["lin_vel_x"])):
            selected = env_ids[mask]
            self.commands[selected, axis] = torch.empty(len(selected), device=self.device).uniform_(*limits)
        self.heading_envs[env_ids] = ~(zero | yaw | lateral | forward)
        self.direct_yaw_commands[env_ids] = self.commands[env_ids, 2]
        if cfg.heading_command:
            self._update_heading_commands(env_ids)

    def _update_heading_commands(self, env_ids=slice(None)):
        # Isaac Gym quaternions are xyzw. Use root_states so reset poses are already current.
        x, y, z, w = self.root_states[env_ids, 3:7].unbind(dim=-1)
        heading = torch.atan2(2 * (x * y + w * z), 1 - 2 * (y.square() + z.square()))
        error = torch.remainder(self.commands[env_ids, 3] - heading + torch.pi, 2 * torch.pi) - torch.pi
        yaw = (self.cfg.commands.heading_control_stiffness * error).clamp(*self.command_ranges["ang_vel_yaw"])
        self.commands[env_ids, 2] = torch.where(
            self.heading_envs[env_ids], yaw, self.direct_yaw_commands[env_ids])

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.commands.heading_command:
            self._update_heading_commands()
        self._update_rotation_state()

    def _update_rotation_state(self):
        contact = self._wheel_contacts()
        old_contact = self.rotation_contact_history[:, :, self.rotation_history_index].float()
        self.rotation_contact_sum += contact.float() - old_contact
        self.rotation_contact_history[:, :, self.rotation_history_index] = contact
        self.rotation_history_index = (self.rotation_history_index + 1) % self.rotation_history_size
        self.rotation_contact_samples = (self.rotation_contact_samples + 1).clamp(max=self.rotation_history_size)

        wheel_z = self.foot_positions[:, :, 2]
        self.rotation_max_air_height = torch.where(
            contact, self.rotation_max_air_height, torch.maximum(self.rotation_max_air_height, wheel_z))
        cleared = ((self.rotation_max_air_height - self.rotation_ground_height > self.cfg.rewards.rotation_air_clearance)
                   & self.rotation_has_ground)
        touchdown = contact & ~self.rotation_last_contact
        earned = self.rotation_air_time - self.cfg.rewards.rotation_air_time_threshold
        self.rotation_air_reward = (earned * touchdown * cleared).sum(dim=1)
        self.rotation_air_time = torch.where(contact, 0.0, self.rotation_air_time + self.dt)
        self.rotation_ground_height = torch.where(contact, wheel_z, self.rotation_ground_height)
        self.rotation_max_air_height = torch.where(contact, wheel_z, self.rotation_max_air_height)
        self.rotation_has_ground |= contact
        self.rotation_last_contact[:] = contact

    def _pure_turn(self):
        return ((torch.norm(self.commands[:, :2], dim=1) < self.cfg.rewards.rotation_lin_threshold)
                & (self.commands[:, 2].abs() > self.cfg.rewards.rotation_cmd_threshold)).float()

    def _wheel_contacts(self):
        return torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > self.cfg.rewards.rotation_contact_threshold

    def _rotation_gait_gate(self):
        return ((self.commands[:, :2].norm(dim=1) < self.cfg.rewards.rotation_gait_lin_threshold)
                & (self.commands[:, 2].abs() > self.cfg.rewards.rotation_gait_ang_threshold)
                & ~self._bad_orientation_mask())

    def _rotation_diagonals(self):
        contact = self._wheel_contacts()
        # Wheel order is LF, LH, RF, RH: diagonal pairs are (LF,RH) and (LH,RF).
        z_a = self.foot_positions[:, [0, 3], 2].mean(dim=1)
        z_b = self.foot_positions[:, [1, 2], 2].mean(dim=1)
        a = contact[:, 0] & contact[:, 3] & (z_b - z_a > self.cfg.rewards.rotation_lift_height)
        b = contact[:, 1] & contact[:, 2] & (z_a - z_b > self.cfg.rewards.rotation_lift_height)
        return torch.stack((a, b), dim=1)

    def _reward_rotation_air_time(self):
        active = (self.commands[:, 2].abs() > self.cfg.rewards.rotation_cmd_threshold) & ~self._bad_orientation_mask()
        return self.rotation_air_reward * active * self.gait_level

    def _reward_rotation_slip(self):
        # Official M20 penalizes contact-wheel planar motion relative to base translation.
        velocity = self.foot_velocities - self.root_states[:, None, 7:10]
        quat = self.root_states[:, None, 3:7].expand(-1, 4, -1)
        velocity_b = quat_rotate_inverse(quat.reshape(-1, 4), velocity.reshape(-1, 3)).view(self.num_envs, 4, 3)
        reward = (velocity_b[:, :, :2].norm(dim=-1) * self._wheel_contacts()).sum(dim=1)
        return reward * self._pure_turn() * self.gait_level

    def _reward_rotation_gait(self):
        return self._rotation_diagonals().any(dim=1).float() * self._rotation_gait_gate() * self.gait_level

    def _reward_rotation_symmetry(self):
        samples = self.rotation_contact_samples.clamp(min=1)
        duty = self.rotation_contact_sum / samples[:, None]
        a = duty[:, [0, 3]].mean(dim=1)
        b = duty[:, [1, 2]].mean(dim=1)
        error = (a - self.cfg.rewards.rotation_target_duty).square() + (b - self.cfg.rewards.rotation_target_duty).square()
        reward = torch.exp(-error / self.cfg.rewards.rotation_duty_std**2)
        warmup = self.rotation_contact_samples / self.rotation_history_size
        return reward * warmup * self._rotation_gait_gate() * self.gait_level

    def _reward_action_rate(self):
        return super()._reward_action_rate() * self.gait_level

    def _reward_action_smoothness(self):
        reward = super()._reward_action_smoothness()
        return reward * (self.episode_length_buf > 2) * self.gait_level

    def _reward_dof_acc(self):
        acceleration = (self.last_dof_vel[:, self.leg_indices] - self.dof_vel[:, self.leg_indices]) / self.dt
        return acceleration.square().sum(dim=1)

    def _reward_torques(self):
        return self.torques[:, self.leg_indices].square().sum(dim=1) * self.gait_level

    def _reward_wheel_acc(self):
        acceleration = (self.last_dof_vel[:, self.wheel_indices] - self.dof_vel[:, self.wheel_indices]) / self.dt
        return acceleration.square().sum(dim=1)

    def _reward_contact_forces(self):
        force = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1)
        return (force - self.cfg.rewards.max_contact_force).clamp(min=0.0).sum(dim=1) * self.gait_level

    def _joint_position_error(self, indices):
        error = (self.dof_pos[:, indices] - self.default_dof_pos[:, indices]).norm(dim=1)
        moving = (self.commands[:, :3].norm(dim=1) > 0.1) | (self.base_lin_vel[:, :2].norm(dim=1) > 0.5)
        return error * torch.where(moving, 1.0, self.cfg.rewards.joint_stand_still_scale)

    def _reward_hip_default(self):
        return self._joint_position_error(self.haa_indices)

    def _joint_default_penalty(self, indices):
        error = self._joint_position_error(indices)
        pure_side = ((self.commands[:, 1].abs() > 0.1)
                     & (torch.norm(self.commands[:, [0, 2]], dim=1) < 0.1))
        mean_height = self.measured_heights[:, self.base_height_index] - self.env_origins[:, 2]
        terrain_scale = torch.where(mean_height > self.cfg.rewards.high_terrain_threshold,
                                    self.cfg.rewards.high_terrain_penalty_scale, 1.0)
        active = ~(self._pure_turn().bool() | pure_side)
        return error * active * terrain_scale

    def _reward_hfe_default(self):
        return self._joint_default_penalty(self.hfe_indices)

    def _reward_kfe_default(self):
        return self._joint_default_penalty(self.kfe_indices)

    def _reward_joint_mirror(self):
        # Mirrored Litewheel legs have opposite HFE/KFE angle signs.
        positions = self.dof_pos - self.default_dof_pos
        terms = [(positions[:, i].abs() - positions[:, j].abs()).square() for i, j in self.mirror_pairs]
        return torch.stack(terms, dim=1).mean(dim=1)

    def _reward_bad_orientation(self):
        return self._bad_orientation_mask().float()

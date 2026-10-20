import torch

from legged_gym.envs.fdwheel.fdwheel import FDWheelRobot


class FDWheelM20Robot(FDWheelRobot):
    """FDWheel with M20-inspired pure-turn sampling and contact rewards."""

    def _init_buffers(self):
        super()._init_buffers()
        self.rotation_contact_duty = torch.full((self.num_envs, 4), 0.5, device=self.device)
        self.rotation_pair_seen = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.rotation_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotation_air_valid = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.rotation_last_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.rotation_air_reward = torch.zeros(self.num_envs, device=self.device)
        self.haa_indices = self._joint_indices("_HAA")
        self.hfe_indices = self._joint_indices("_HFE")
        self.kfe_indices = self._joint_indices("_KFE")
        self.mirror_pairs = [
            (self.dof_names.index(left + suffix), self.dof_names.index(right + suffix))
            for left, right in (("LF_", "RH_"), ("LH_", "RF_"))
            for suffix in ("HAA", "HFE", "KFE")
        ]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids):
            self.rotation_contact_duty[env_ids] = 0.5
            self.rotation_pair_seen[env_ids] = False
            self.rotation_air_time[env_ids] = 0.0
            self.rotation_air_valid[env_ids] = False
            self.rotation_last_contact[env_ids] = False
            self.rotation_air_reward[env_ids] = 0.0

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
        # Reject near-zero commands within the dedicated motion groups.
        for mask, axis, limits in ((yaw, 2, self.command_ranges["ang_vel_yaw"]),
                                   (lateral, 1, self.command_ranges["lin_vel_y"]),
                                   (forward, 0, self.command_ranges["lin_vel_x"])):
            selected = env_ids[mask & (self.commands[env_ids, axis].abs() < 0.1)]
            if len(selected):
                sign = torch.where(torch.rand(len(selected), device=self.device) < 0.5, -1.0, 1.0)
                magnitude = min(abs(limits[0]), abs(limits[1]))
                self.commands[selected, axis] = sign * max(0.1, magnitude * 0.5)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_rotation_state()

    def _update_rotation_state(self):
        contact = self._wheel_contacts()
        active = self._pure_turn().bool() & ~self._bad_orientation_mask()
        # Commands can change mid-episode. A prior forward swing must not earn turn rewards.
        self.rotation_air_time[~active] = 0.0
        self.rotation_air_valid[~active] = False
        self.rotation_pair_seen[~active] = False
        alpha = min(1.0, self.dt / self.cfg.rewards.rotation_duty_time)
        self.rotation_contact_duty.lerp_(contact.float(), alpha)
        self.rotation_contact_duty[~active] = 0.5
        lifted = self._rotation_lifted(contact)
        self.rotation_pair_seen |= self._rotation_diagonals(contact, lifted) & active[:, None]
        self.rotation_air_valid |= lifted & active[:, None]
        touchdown = contact & ~self.rotation_last_contact
        earned = (self.rotation_air_time - self.cfg.rewards.rotation_air_time_threshold).clamp(
            min=0.0, max=self.cfg.rewards.rotation_air_time_reward_cap)
        self.rotation_air_reward = (earned * touchdown * self.rotation_air_valid).sum(dim=1) * active
        self.rotation_air_time += self.dt * active[:, None]
        self.rotation_air_time[contact] = 0.0
        self.rotation_air_valid[contact] = False
        self.rotation_last_contact[:] = contact

    def _pure_turn(self):
        return ((torch.norm(self.commands[:, :2], dim=1) < self.cfg.rewards.rotation_lin_threshold)
                & (self.commands[:, 2].abs() > self.cfg.rewards.rotation_cmd_threshold)).float()

    def _wheel_contacts(self):
        return torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > self.cfg.rewards.rotation_contact_threshold

    def _rotation_lifted(self, contact):
        # Wheel-center height relative to support wheels is a local clearance proxy.
        wheel_z = self.foot_positions[:, :, 2]
        support_z = torch.where(contact, wheel_z, torch.full_like(wheel_z, float("inf"))).amin(dim=1)
        support_z = torch.where(torch.isfinite(support_z), support_z, wheel_z.amin(dim=1))
        clearance = (wheel_z - support_z[:, None]).clamp(min=0.0)
        return (~contact) & contact.any(dim=1, keepdim=True) & (clearance >= self.cfg.rewards.rotation_lift_height)

    def _rotation_diagonals(self, contact=None, lifted=None):
        if contact is None:
            contact = self._wheel_contacts()
        if lifted is None:
            lifted = self._rotation_lifted(contact)
        # Wheel order is LF, LH, RF, RH: diagonal pairs are (LF,RH) and (LH,RF).
        a = contact[:, 0] & contact[:, 3] & lifted[:, 1] & lifted[:, 2]
        b = contact[:, 1] & contact[:, 2] & lifted[:, 0] & lifted[:, 3]
        return torch.stack((a, b), dim=1)

    def _reward_rotation_air_time(self):
        return self.rotation_air_reward * self._pure_turn()

    def _reward_rotation_slip(self):
        return self._reward_wheel_lateral_slip() * self._pure_turn()

    def _reward_rotation_gait(self):
        return (self._rotation_diagonals().any(dim=1).float()
                * self._pure_turn() * ~self._bad_orientation_mask())

    def _reward_rotation_symmetry(self):
        duty_error = (self.rotation_contact_duty - 0.5).abs().mean(dim=1)
        # Balanced contact duty alone also rewards hopping; require both diagonal patterns.
        alternated = self.rotation_pair_seen.all(dim=1)
        return ((1.0 - 2.0 * duty_error).clamp(min=0.0) * alternated
                * self._reward_rotation_gait())

    def _reward_action_smoothness(self):
        reward = super()._reward_action_smoothness()
        return reward * (self.episode_length_buf > 2)

    def _reward_dof_acc(self):
        acceleration = (self.last_dof_vel[:, self.leg_indices] - self.dof_vel[:, self.leg_indices]) / self.dt
        return acceleration.square().sum(dim=1)

    def _reward_torques(self):
        return self.torques[:, self.leg_indices].square().sum(dim=1)

    def _reward_wheel_acc(self):
        acceleration = (self.last_dof_vel[:, self.wheel_indices] - self.dof_vel[:, self.wheel_indices]) / self.dt
        return acceleration.square().sum(dim=1)

    def _reward_contact_forces(self):
        force = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1)
        return (force - self.cfg.rewards.max_contact_force).clamp(min=0.0).sum(dim=1)

    def _joint_default_penalty(self, indices):
        error = self.dof_pos[:, indices] - self.default_dof_pos[:, indices]
        pure_side = ((self.commands[:, 1].abs() > 0.1)
                     & (torch.norm(self.commands[:, [0, 2]], dim=1) < 0.1))
        mean_height = self.measured_heights.mean(dim=1) - self.env_origins[:, 2]
        terrain_scale = torch.where(mean_height > self.cfg.rewards.high_terrain_threshold,
                                    self.cfg.rewards.high_terrain_penalty_scale, 1.0)
        active = ~(self._pure_turn().bool() | pure_side)
        return error.square().sum(dim=1) * active * terrain_scale

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

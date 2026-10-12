import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.M20.m20 import M20_Robot


class DynawheelRobot(M20_Robot):
    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            action_delayed = self.update_cmd_action_latency_buffer()
            self.torques = self._compute_torques(action_delayed).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.update_obs_latency_buffer()
        self.post_physics_step()

        self.obs_buf = torch.clip(self.obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return self.obs_buf, self.privileged_obs_buf, self.obs_hist_buf, self.vel_buf, self.rew_buf, self.reset_buf, self.extras

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, :3]
        self.foot_clearance = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.max_foot_clearance = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.feet_were_airborne = torch.zeros_like(self.max_foot_clearance, dtype=torch.bool)
        self.spin_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.spin_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.spin_contact_masks = torch.tensor(
            [[True, False, False, True], [False, True, True, False]],
            dtype=torch.bool,
            device=self.device,
        )
        self.last_foot_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.current_foot_contacts = torch.zeros_like(self.last_foot_contacts)
        self.maneuver_was_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_foot_contacts = torch.zeros_like(self.last_foot_contacts)
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.filtered_torques = torch.zeros_like(self.torques)
        self.torque_cutoff_freq = torch.zeros_like(self.torques)
        self.torque_filter_cfg = getattr(self.cfg.control, "torque_filter", None)
        self.torque_filter_enabled = self.torque_filter_cfg is not None and self.torque_filter_cfg.enable
        slew_rates = getattr(self.cfg.commands, "slew_rate", None)
        self.command_slew_rates = (
            torch.tensor(slew_rates, dtype=torch.float, device=self.device)
            if slew_rates is not None else None
        )
        self.emphase_cfg = getattr(self.cfg.commands, "emphase", None)
        self.emphase_enabled = self.emphase_cfg is not None and self.emphase_cfg.enable
        self.emphase_times = torch.zeros(
            self.num_envs, len(self.emphase_cfg.type) if self.emphase_enabled else 0,
            dtype=torch.long, device=self.device
        )
        self.emphase_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._randomize_torque_filter_cutoff(torch.arange(self.num_envs, device=self.device))
        assert len(self.feet_indices) == 4 and torch.all(self.spin_contact_masks.sum(dim=1) == 2)
        assert 0.0 < self.cfg.rewards.feet_clearance_liftoff < self.cfg.rewards.feet_clearance_target

        max_motor_delay = self.cfg.domain_rand.range_obs_motor_latency[1]
        max_imu_delay = self.cfg.domain_rand.range_obs_imu_latency[1]
        self.obs_motor_latency_buffer = torch.zeros(self.num_envs, self.num_actions * 2, max_motor_delay + 1, device=self.device)
        self.obs_imu_latency_buffer = torch.zeros(self.num_envs, 6, max_imu_delay + 1, device=self.device)
        self.obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_obs_latency_buffer(torch.arange(self.num_envs, device=self.device))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.max_foot_clearance[env_ids] = 0.0
        self.feet_were_airborne[env_ids] = False
        self.spin_active[env_ids] = False
        self.spin_start_step[env_ids] = 0
        self.last_foot_contacts[env_ids] = False
        self.current_foot_contacts[env_ids] = False
        self.maneuver_was_active[env_ids] = False
        self.first_foot_contacts[env_ids] = False
        self.feet_air_time[env_ids] = 0.0
        self.feet_contact_time[env_ids] = 0.0
        self.filtered_torques[env_ids] = 0.0
        self._randomize_torque_filter_cutoff(env_ids)
        self._reset_obs_latency_buffer(env_ids)
        if self.emphase_enabled:
            self._generate_emphase_time(env_ids)
            self._update_emphase_commands(env_ids)

    def _randomize_torque_filter_cutoff(self, env_ids):
        if not self.torque_filter_enabled:
            return
        ranges = self.torque_filter_cfg.cutoff_freq_ranges
        for dof_idx, dof_name in enumerate(self.dof_names):
            for motor_name, cutoff_range in ranges.items():
                if motor_name in dof_name:
                    if self.torque_filter_cfg.randomize_cutoff:
                        cutoff = torch_rand_float(
                            cutoff_range[0], cutoff_range[1],
                            (len(env_ids), 1), device=self.device,
                        ).squeeze(1)
                    else:
                        cutoff = torch.full(
                            (len(env_ids),), sum(cutoff_range) / 2,
                            device=self.device,
                        )
                    self.torque_cutoff_freq[env_ids, dof_idx] = cutoff
                    break

    def _resample_commands(self, env_ids):
        if not self.emphase_enabled:
            super()._resample_commands(env_ids)

    def _generate_emphase_time(self, env_ids):
        phase_types = self.emphase_cfg.type
        duration_ranges = self.emphase_cfg.time_range
        durations = torch.cat([
            torch_rand_float(*duration_ranges[name], (len(env_ids), 1), device=self.device)
            for name in phase_types
        ], dim=1)
        cumulative = torch.cumsum(durations[:, :-1], dim=1)
        boundaries = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative), dim=1)
        self.emphase_times[env_ids] = (
            boundaries / durations.sum(dim=1, keepdim=True) * self.max_episode_length
        ).long()
        self.emphase_phase[env_ids] = -1

    def _sample_balanced_signed_command(self, env_ids, min_abs, max_abs):
        count = len(env_ids)
        pair_count = count // 2
        magnitudes = torch_rand_float(
            min_abs, max_abs, (pair_count + count % 2, 1), device=self.device
        ).squeeze(1)
        values = torch.cat((magnitudes, -magnitudes[:pair_count]))
        if count % 2:
            values[pair_count] *= 2 * torch.randint(0, 2, (), device=self.device) - 1
        return values[torch.randperm(count, device=self.device)]

    def _sample_emphase_command(self, phase_name, env_ids):
        ranges = self.command_ranges
        self.commands[env_ids, :] = 0.0
        if phase_name == "lateral":
            self.commands[env_ids, 1] = self._sample_balanced_signed_command(
                env_ids, 0.25, max(abs(v) for v in ranges["lin_vel_y"])
            )
        elif phase_name == "rotate":
            self.commands[env_ids, 2] = self._sample_balanced_signed_command(
                env_ids, 0.25, max(abs(v) for v in ranges["ang_vel_yaw"])
            )
        elif phase_name == "forward_rotate":
            self.commands[env_ids, 0] = torch_rand_float(0.7, 1.3, (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = self._sample_balanced_signed_command(env_ids, 0.25, 0.8)
        elif phase_name == "omnidirectional":
            for index, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
                self.commands[env_ids, index] = torch_rand_float(*ranges[key], (len(env_ids), 1), device=self.device).squeeze(1)
        elif phase_name == "sagittal_forward":
            self.commands[env_ids, 0] = torch_rand_float(0.0, ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        elif phase_name != "stand":
            raise ValueError(f"Unsupported Dynawheel command phase: {phase_name}")

    def _update_emphase_commands(self, env_ids=None):
        if not self.emphase_enabled:
            return
        phase_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        phase = (self.episode_length_buf[phase_ids, None] >= self.emphase_times[phase_ids]).sum(dim=1) - 1
        phase = phase.clamp_(0, len(self.emphase_cfg.type) - 1)
        changed = phase_ids[self.emphase_phase[phase_ids] != phase]
        if len(changed) == 0:
            return
        changed_phase = phase[self.emphase_phase[phase_ids] != phase]
        for phase_idx, phase_name in enumerate(self.emphase_cfg.type):
            selected = changed[changed_phase == phase_idx]
            if len(selected):
                self._sample_emphase_command(phase_name, selected)
        self.emphase_phase[changed] = changed_phase

    def _post_physics_step_callback(self):
        previous_commands = self.commands[:, :3].clone()
        super()._post_physics_step_callback()
        if self.emphase_enabled:
            self._update_emphase_commands()
        if self.command_slew_rates is not None:
            max_delta = self.command_slew_rates * self.dt
            self.commands[:, :3] = previous_commands + torch.clamp(
                self.commands[:, :3] - previous_commands, -max_delta, max_delta
            )

    def post_physics_step(self):
        super().post_physics_step()
        self.feet_air_time *= ~self.current_foot_contacts
        self.feet_contact_time *= self.current_foot_contacts
        self.last_foot_contacts[:] = self.current_foot_contacts

    def _prepare_foot_contact_state(self):
        maneuver_active = self._maneuver_gait_weight() > 0.0
        entered = maneuver_active & ~self.maneuver_was_active
        self.feet_air_time[entered] = 0.0
        self.feet_contact_time[entered] = 0.0
        self.maneuver_was_active[:] = maneuver_active
        contact_z = self.contact_forces[:, self.feet_indices, 2]
        enter = contact_z > self.cfg.rewards.foot_contact_force_threshold
        hold = contact_z > self.cfg.rewards.foot_contact_release_threshold
        self.current_foot_contacts[:] = torch.where(self.last_foot_contacts, hold, enter)
        self.first_foot_contacts[:] = (
            self.current_foot_contacts & ~self.last_foot_contacts
            & (self.feet_air_time >= self.cfg.rewards.min_touchdown_air_time)
        )
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt

    def _foot_clearances(self):
        points = ((self.foot_positions + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1], 0, self.height_samples.shape[1] - 2)
        terrain_height = torch.minimum(torch.minimum(self.height_samples[px, py], self.height_samples[px + 1, py]), self.height_samples[px, py + 1]) * self.terrain.cfg.vertical_scale
        return self.foot_positions[:, :, 2] - terrain_height - self.cfg.rewards.wheel_radius

    def _update_spin_gait(self):
        yaw = torch.abs(self.commands[:, 2])
        linear = torch.norm(self.commands[:, :2], dim=1)
        enter = (yaw > getattr(self.cfg.rewards, "spin_enter_yaw", 0.25)) & (
            linear < getattr(self.cfg.rewards, "spin_enter_linear", 0.10)
        )
        stay = (yaw > getattr(self.cfg.rewards, "spin_exit_yaw", 0.15)) & (
            linear < getattr(self.cfg.rewards, "spin_exit_linear", 0.15)
        )
        active = enter | (self.spin_active & stay)
        just_entered = active & ~self.spin_active
        self.spin_start_step[just_entered] = self.episode_length_buf[just_entered]
        self.spin_active = active
        self.foot_clearance[:] = self._foot_clearances()

    def _reset_obs_latency_buffer(self, env_ids):
        dof_err = self.dof_pos[env_ids] - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0.0
        motor_now = torch.cat((dof_err * self.obs_scales.dof_pos, self.dof_vel[env_ids] * self.obs_scales.dof_vel), dim=1)
        imu_now = torch.cat((self.base_ang_vel[env_ids] * self.obs_scales.ang_vel, self.projected_gravity[env_ids]), dim=1)
        self.obs_motor_latency_buffer[env_ids] = motor_now.unsqueeze(-1)
        self.obs_imu_latency_buffer[env_ids] = imu_now.unsqueeze(-1)

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            self.obs_motor_latency_simstep[env_ids] = torch.randint(self.cfg.domain_rand.range_obs_motor_latency[0], self.cfg.domain_rand.range_obs_motor_latency[1] + 1, (len(env_ids),), device=self.device)
        else:
            self.obs_motor_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_motor_latency[1]
        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.obs_imu_latency_simstep[env_ids] = torch.randint(self.cfg.domain_rand.range_obs_imu_latency[0], self.cfg.domain_rand.range_obs_imu_latency[1] + 1, (len(env_ids),), device=self.device)
        else:
            self.obs_imu_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_imu_latency[1]

    def update_obs_latency_buffer(self):
        if not self.cfg.domain_rand.add_obs_latency:
            return

        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0.0
        motor_now = torch.cat((self.dof_err * self.obs_scales.dof_pos, self.dof_vel * self.obs_scales.dof_vel), dim=1)
        self.obs_motor_latency_buffer[:, :, 1:] = self.obs_motor_latency_buffer[:, :, :-1].clone()
        self.obs_motor_latency_buffer[:, :, 0] = motor_now

        self.gym.refresh_actor_root_state_tensor(self.sim)
        base_quat = self.root_states[:, 3:7]
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        imu_now = torch.cat((base_ang_vel * self.obs_scales.ang_vel, projected_gravity), dim=1)
        self.obs_imu_latency_buffer[:, :, 1:] = self.obs_imu_latency_buffer[:, :, :-1].clone()
        self.obs_imu_latency_buffer[:, :, 0] = imu_now

    def compute_observations(self):
        self.obs_hist_buf = self.obs_hist_buf[:, 57:]
        self.obs_hist_buf = torch.cat((self.obs_hist_buf, self.obs_buf), dim=-1)
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements

        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0.0
        if self.cfg.domain_rand.add_obs_latency:
            env_ids = torch.arange(self.num_envs, device=self.device)
            motor_obs = self.obs_motor_latency_buffer[env_ids, :, self.obs_motor_latency_simstep]
            imu_obs = self.obs_imu_latency_buffer[env_ids, :, self.obs_imu_latency_simstep]
            ang_vel = imu_obs[:, :3]
            gravity = imu_obs[:, 3:6]
            dof_pos = motor_obs[:, :self.num_actions]
            dof_vel = motor_obs[:, self.num_actions:]
        else:
            ang_vel = self.base_ang_vel * self.obs_scales.ang_vel
            gravity = self.projected_gravity
            dof_pos = self.dof_err * self.obs_scales.dof_pos
            dof_vel = self.dof_vel * self.obs_scales.dof_vel

        self.obs_buf = torch.cat((self.commands[:, :3] * self.commands_scale, ang_vel, gravity, dof_pos, dof_vel, self.actions), dim=-1)
        self.vel_buf = self.base_lin_vel
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, heights, self.obs_buf), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            self.joint_friction_coeffs = torch.zeros(self.num_envs, 1, device=self.device)
            self.joint_damping_coeffs = torch.zeros(self.num_envs, 1, device=self.device)
            self.joint_armatures = torch.zeros(self.num_envs, 1, device=self.device)

        if self.cfg.domain_rand.randomize_joint_friction:
            self.joint_friction_coeffs[env_id] = torch_rand_float(*self.cfg.domain_rand.joint_friction_range, (1, 1), device=self.device)
            props["friction"][:] = self.joint_friction_coeffs[env_id, 0].item()
        if self.cfg.domain_rand.randomize_joint_damping:
            self.joint_damping_coeffs[env_id] = torch_rand_float(*self.cfg.domain_rand.joint_damping_range, (1, 1), device=self.device)
            props["damping"][:] = self.joint_damping_coeffs[env_id, 0].item()
        if self.cfg.domain_rand.randomize_joint_armature:
            self.joint_armatures[env_id] = torch_rand_float(*self.cfg.domain_rand.joint_armature_range, (1, 1), device=self.device)
            props["armature"][:] = self.joint_armatures[env_id, 0].item()
        else:
            for dof_idx, dof_name in enumerate(self.dof_names):
                props["armature"][dof_idx] = (
                    self.cfg.asset.wheel_armature
                    if any(wheel_name in dof_name for wheel_name in self.cfg.asset.wheel_name)
                    else self.cfg.asset.joint_armature
                )
        return props

    def _compute_torques(self, actions):
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0.0
        actions_scaled = actions * self.cfg.control.action_scale + self.motor_zero_offsets
        actions_scaled[:, self.wheel_indices] = 0.0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_ref[:, self.wheel_indices] = actions[:, self.wheel_indices] * self.cfg.control.vel_scale
        torques = self.p_gains_multiplier * self.p_gains * (actions_scaled + dof_err) + self.d_gains_multiplier * self.d_gains * (vel_ref - self.dof_vel)
        torques *= self.torques_multiplier
        if self.torque_filter_enabled:
            alpha = 1.0 - torch.exp(-2.0 * torch.pi * self.torque_cutoff_freq * self.cfg.sim.dt)
            self.filtered_torques.mul_(1.0 - alpha).add_(torques * alpha)
            torques = self.filtered_torques
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _push_robots(self):
        self.root_states[:, 7:9] += torch_rand_float(-self.cfg.domain_rand.max_push_vel_xy, self.cfg.domain_rand.max_push_vel_xy, (self.num_envs, 2), device=self.device)
        self.root_states[:, 10:13] += torch_rand_float(-self.cfg.domain_rand.max_push_ang_vel, self.cfg.domain_rand.max_push_ang_vel, (self.num_envs, 3), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _reward_action_smoothness(self):
        delta = self.actions - 2 * self.last_actions + self.last_last_actions
        delta[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(delta), dim=1)

    def _reward_action_rate_dof(self):
        delta = self.actions - self.last_actions
        delta[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(delta), dim=1)

    def _reward_action_rate_wheel(self):
        return torch.sum(torch.square(self.actions[:, self.wheel_indices] - self.last_actions[:, self.wheel_indices]), dim=1)

    def _reward_smoothness_dof(self):
        delta = self.actions - 2 * self.last_actions + self.last_last_actions
        delta[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(delta), dim=1)

    def _reward_smoothness_wheel(self):
        delta = self.actions[:, self.wheel_indices] - 2 * self.last_actions[:, self.wheel_indices] + self.last_last_actions[:, self.wheel_indices]
        return torch.sum(torch.square(delta), dim=1)

    # def _single_diagonal_support_masks(self):
    #     contacts = self.current_foot_contacts
    #     lf_rh = contacts[:, 0] & contacts[:, 3]
    #     lh_rf = contacts[:, 1] & contacts[:, 2]
    #     return lf_rh & ~lh_rf, lh_rf & ~lf_rh
    def _single_diagonal_support_masks(self):
        contacts = self.current_foot_contacts

        diag_lf_rh = (
            contacts[:, 0]
            & contacts[:, 3]
            & ~contacts[:, 1]
            & ~contacts[:, 2]
        )

        diag_lh_rf = (
            contacts[:, 1]
            & contacts[:, 2]
            & ~contacts[:, 0]
            & ~contacts[:, 3]
        )

        return diag_lf_rh, diag_lh_rf
    
    def _diagonal_swing_mask(self):
        lf_rh, lh_rf = self._single_diagonal_support_masks()
        swing = torch.zeros_like(self.current_foot_contacts)
        swing[:, 1] = lf_rh
        swing[:, 2] = lf_rh
        swing[:, 0] = lh_rf
        swing[:, 3] = lh_rf
        return swing

    def _reward_support_trot_not_only_x(self):
        contacts = self.current_foot_contacts
        diag_lf_rh = contacts[:, 0] & contacts[:, 3] & ~contacts[:, 1] & ~contacts[:, 2]
        diag_lh_rf = contacts[:, 1] & contacts[:, 2] & ~contacts[:, 0] & ~contacts[:, 3]
        moving_lateral_or_yaw = torch.norm(self.commands[:, 1:3], dim=1) >= 0.1
        upright = torch.clamp(-self.projected_gravity[:, 2], 0.0, 0.7) / 0.7
        return (diag_lf_rh | diag_lh_rf).float() * moving_lateral_or_yaw.float() * upright

    def _reward_feet_distance_xy_exp(self):
        feet_body = quat_rotate_inverse(
            self.base_quat[:, None, :].expand(-1, len(self.feet_indices), -1).reshape(-1, 4),
            (self.foot_positions - self.root_states[:, None, :3]).reshape(-1, 3),
        ).view(self.num_envs, len(self.feet_indices), 3)
        target_x = torch.tensor(
            [self.cfg.rewards.stance_length_front, -self.cfg.rewards.stance_length_back,
             self.cfg.rewards.stance_length_front, -self.cfg.rewards.stance_length_back],
            device=self.device,
        )
        target_y = torch.tensor(
            [self.cfg.rewards.stance_width, self.cfg.rewards.stance_width,
             -self.cfg.rewards.stance_width, -self.cfg.rewards.stance_width],
            device=self.device,
        )
        error = torch.square(feet_body[:, :, 0] - target_x) + torch.square(feet_body[:, :, 1] - target_y)
        return torch.exp(-torch.sum(error, dim=1) / self.cfg.rewards.feet_distance_xy_sigma)

    def _reward_feet_air_time(self):
        target = self.cfg.rewards.air_time
        sigma = self.cfg.rewards.air_time_sigma
        touchdown_score = torch.exp(-torch.square((self.feet_air_time - target) / sigma))
        moving = (torch.norm(self.commands[:, :3], dim=1) >= 0.1).float()
        return torch.sum(touchdown_score * self.first_foot_contacts.float(), dim=1) * moving

    def _reward_foot_clearance_world(self):
        deficit = torch.clamp(
            (self.cfg.rewards.swing_clearance_target - self.foot_clearance)
            / self.cfg.rewards.swing_clearance_sigma,
            min=0.0,
            max=2.0,
        )
        active = (torch.norm(self.commands[:, 1:3], dim=1) >= 0.1).float()
        return torch.sum(torch.square(deficit) * self._diagonal_swing_mask().float(), dim=1) * active

    def _reward_diagonal_swing_height(self):
        diag_lf_rh, diag_lh_rf = self._single_diagonal_support_masks()
        sigma = self.cfg.rewards.diagonal_height_sigma
        mismatch_lh_rf = torch.clamp(torch.square((self.foot_clearance[:, 1] - self.foot_clearance[:, 2]) / sigma), max=1.0)
        mismatch_lf_rh = torch.clamp(torch.square((self.foot_clearance[:, 0] - self.foot_clearance[:, 3]) / sigma), max=1.0)
        active = (torch.norm(self.commands[:, 1:3], dim=1) >= 0.1).float()
        return (mismatch_lh_rf * diag_lf_rh.float() + mismatch_lf_rh * diag_lh_rf.float()) * active



    def _reward_foot_phase_stall(self):
        max_duration = self.cfg.rewards.max_foot_phase_duration
        sigma = self.cfg.rewards.foot_phase_duration_sigma
        air_excess = torch.clamp((self.feet_air_time - max_duration) / sigma, min=0.0, max=1.0)
        contact_excess = torch.clamp((self.feet_contact_time - max_duration) / sigma, min=0.0, max=1.0)
        return torch.mean(air_excess + contact_excess, dim=1) * self._maneuver_gait_weight()

    def _maneuver_gait_weight(self):
        """Blend stepping into lateral motion / near-stationary turns; pure x stays rolling."""
        cfg = self.cfg.rewards
        lateral = ((self.commands[:, 1].abs() - cfg.trot_lateral_deadband)
                   / cfg.trot_lateral_ramp).clamp(0.0, 1.0)
        yaw = ((self.commands[:, 2].abs() - cfg.trot_yaw_deadband)
               / cfg.trot_yaw_ramp).clamp(0.0, 1.0)
        forward_blend = ((cfg.trot_forward_off - self.commands[:, 0].abs())
                         / (cfg.trot_forward_off - cfg.trot_forward_full)).clamp(0.0, 1.0)
        return torch.maximum(lateral, yaw) * forward_blend

    def _reward_maneuver_trot(self):
        # Either diagonal may support; no unobserved clock or forced phase reset.
        mismatch = (self.current_foot_contacts[:, None, :]
                    != self.spin_contact_masks[None, :, :]).float().mean(dim=2)
        return mismatch.min(dim=1).values * self._maneuver_gait_weight()

    def _reward_feet_impact_vel(self):
        downward_speed = torch.clamp(-self.rigid_body_state[:, self.feet_indices, 9], min=0.0)
        return torch.sum(torch.square(downward_speed) * self.first_foot_contacts.float(), dim=1)

    def _reward_feet_clearance(self):
        contact = self.current_foot_contacts
        command_strength = self._maneuver_gait_weight()
        moving = command_strength > 0.0
        airborne = ~contact & moving.unsqueeze(1)
        self.max_foot_clearance = torch.where(airborne, torch.maximum(self.max_foot_clearance, self.foot_clearance), self.max_foot_clearance)
        self.feet_were_airborne |= airborne

        touchdown = contact & self.feet_were_airborne & moving.unsqueeze(1)
        shortfall = torch.clamp((self.cfg.rewards.feet_clearance_target - self.max_foot_clearance) / self.cfg.rewards.feet_clearance_target, 0.0, 1.0)
        # Ramp the minimum up after liftoff, rather than demand full height instantly.
        # Once the swing peak is sufficient, do not punish the descent before touchdown.
        target = self.cfg.rewards.feet_clearance_target
        ramp = (self.feet_air_time / self.cfg.rewards.trot_clearance_rise_time).clamp(0.0, 1.0)
        swing_shortfall = ((target * ramp - self.max_foot_clearance) / target).clamp(0.0, 1.0)
        penalty = torch.mean(torch.square(swing_shortfall) * airborne
                             + torch.square(shortfall) * touchdown, dim=1) * command_strength

        # A wheel that never leaves the ground cannot enter the airborne branch
        # above. During pure spin, make either diagonal pair a valid swing choice:
        # lifting that pair clears the deficit only while the other pair supports.
        spin_elapsed = (self.episode_length_buf - self.spin_start_step).float() * self.dt
        spin_ramp = (spin_elapsed / self.cfg.rewards.spin_clearance_grace_time).clamp(0.0, 1.0)
        spin_shortfall = ((target * spin_ramp[:, None] - self.max_foot_clearance)
                          / target).clamp(0.0, 1.0).square()
        pair_costs = []
        for pair in range(2):
            swing = self.spin_contact_masks[pair]
            support_error = (~contact[:, ~swing]).float().mean(dim=1)
            pair_costs.append((spin_shortfall[:, swing].mean(dim=1) + support_error).clamp(max=1.0))
        spin_penalty = torch.minimum(pair_costs[0], pair_costs[1])
        penalty = torch.where(self.spin_active, spin_penalty * command_strength, penalty)

        reset = touchdown | ~moving.unsqueeze(1)
        self.max_foot_clearance = torch.where(reset, torch.zeros_like(self.max_foot_clearance), self.max_foot_clearance)
        self.feet_were_airborne &= ~reset
        return penalty

    def _reward_hip_default(self):
        return super()._reward_hip_default()

    def _reward_run_still(self):
        return super()._reward_run_still()

    def _reward_spin_contact_pattern(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        phase_step = self.episode_length_buf - self.spin_start_step
        half_period = getattr(self.cfg.rewards, "spin_half_period", 0.30)
        half_period_steps = max(1, round(half_period / self.dt))
        phase = (phase_step // half_period_steps) % 2
        target_contact = self.spin_contact_masks[phase]
        match = 1.0 - torch.logical_xor(contact, target_contact).float().mean(dim=1)
        return match * self.spin_active

    def _reward_spin_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_vel = self.rigid_body_state[:, self.feet_indices, 7:10]
        base_quat = self.base_quat.unsqueeze(1).expand(-1, len(self.feet_indices), -1)
        foot_vel_body = quat_rotate_inverse(base_quat.reshape(-1, 4), foot_vel.reshape(-1, 3)).view_as(foot_vel)
        return torch.mean(torch.square(foot_vel_body[:, :, 1]) * contact, dim=1) * self.spin_active

    # def _reward_same_side_support(self):
    #     contacts = self.current_foot_contacts

    #     left_pair = (
    #         contacts[:, 0]
    #         & contacts[:, 1]
    #         & ~contacts[:, 2]
    #         & ~contacts[:, 3]
    #     )

    #     right_pair = (
    #         contacts[:, 2]
    #         & contacts[:, 3]
    #         & ~contacts[:, 0]
    #         & ~contacts[:, 1]
    #     )

    #     active = (
    #         torch.norm(self.commands[:, 1:3], dim=1) >= 0.1
    #     ).float()

    #     return (left_pair | right_pair).float() * active

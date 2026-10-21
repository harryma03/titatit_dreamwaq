import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi


class FDWheelRobot(LeggedRobot):
    """Independent FDWheel task using the ZG-M1 training pipeline."""

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")
        self._create_envs()

    def _create_envs(self):
        super()._create_envs()
        wheel_ids = [i for i, name in enumerate(self.dof_names) if self.cfg.asset.wheel_name in name]
        if self.num_dofs != self.num_actions or len(wheel_ids) != 4:
            raise ValueError(f"FDWheel expects 16 DOFs and 4 wheels, got {self.num_dofs} DOFs and {len(wheel_ids)} wheels")
        self.wheel_indices = torch.tensor(wheel_ids, dtype=torch.long, device=self.device)
        leg_ids = [i for i in range(self.num_dofs) if i not in wheel_ids]
        self.leg_indices = torch.tensor(leg_ids, dtype=torch.long, device=self.device)
        foot_names = ("LF_wheel", "LH_wheel", "RF_wheel", "RH_wheel")
        self.feet_indices = torch.tensor([
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            for name in foot_names
        ], dtype=torch.long, device=self.device)
        if torch.any(self.feet_indices < 0):
            raise ValueError(f"FDWheel wheel bodies not found: {foot_names}")


    def _init_custom_buffers__(self):
        super()._init_custom_buffers__()
        self.body_mass_scales = torch.ones(self.num_envs, device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        self.vel_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        min_delay, max_delay = self.cfg.domain_rand.range_cmd_action_latency
        assert 0 <= min_delay <= max_delay, "range_cmd_action_latency must be non-negative and ordered"
        self.cmd_action_latency_buffer = torch.zeros(self.num_envs, self.num_actions, max_delay + 1, device=self.device)
        self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_action_latency(torch.arange(self.num_envs, device=self.device))

    def step(self, actions):
        self.obs_hist_buf = torch.cat((self.obs_hist_buf[:, self.num_obs:], self.obs_buf), dim=-1)
        self.actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            # Delay is measured in physics steps, not policy steps.
            action_delayed = self._update_action_latency_buffer()
            self.torques = self._compute_torques(action_delayed).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.obs_hist_buf, self.vel_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        contact = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        self.reset_buf = torch.any(contact > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        contact_flag = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        self.base_contact_buf = torch.any(contact_flag.unsqueeze(1) < 0.20, dim=1)
        self.reset_buf |= self.base_contact_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.actions[env_ids] = 0.0
        self.obs_hist_buf[env_ids] = 0.0
        self._randomize_dof_props(env_ids, self.cfg)
        self._reset_action_latency(env_ids)

    def _update_action_latency_buffer(self):
        if not self.cfg.domain_rand.add_cmd_action_latency:
            return self.actions
        self.cmd_action_latency_buffer[:, :, 1:] = self.cmd_action_latency_buffer[:, :, :-1].clone()
        self.cmd_action_latency_buffer[:, :, 0] = self.actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.cmd_action_latency_buffer[env_ids, :, self.cmd_action_latency_simstep]

    def _reset_action_latency(self, env_ids):
        self.cmd_action_latency_buffer[env_ids] = 0.0
        delay_range = self.cfg.domain_rand.range_cmd_action_latency
        if self.cfg.domain_rand.randomize_cmd_action_latency:
            self.cmd_action_latency_simstep[env_ids] = torch.randint(delay_range[0], delay_range[1] + 1, (len(env_ids),), device=self.device)
        else:
            self.cmd_action_latency_simstep[env_ids] = delay_range[1]

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and self.common_step_counter % self.cfg.domain_rand.push_interval == 0:
            self._push_robots()

    def _resample_commands(self, env_ids):
        count = len(env_ids)
        for axis, key in enumerate(("lin_vel_x", "lin_vel_y")):
            limits = self.command_ranges[key]
            self.commands[env_ids, axis] = torch_rand_float(
                *limits, (count, 1), device=self.device
            ).squeeze(1)
        if self.cfg.commands.heading_command:
            limits = self.command_ranges["heading"]
            self.commands[env_ids, 3] = torch_rand_float(
                *limits, (count, 1), device=self.device
            ).squeeze(1)
        else:
            limits = self.command_ranges["ang_vel_yaw"]
            self.commands[env_ids, 2] = torch_rand_float(
                *limits, (count, 1), device=self.device
            ).squeeze(1)

        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def compute_observations(self):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1.0, 1.0) * self.obs_scales.height_measurements
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0.0
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            dof_err * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        self.vel_buf = self.base_lin_vel
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, heights, self.obs_buf), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, _cfg=None):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise = self.cfg.noise.noise_scales
        noise_vec[3:6] = noise.ang_vel * self.obs_scales.ang_vel
        noise_vec[6:9] = noise.gravity
        noise_vec[9:25] = noise.dof_pos * self.obs_scales.dof_pos
        noise_vec[25:41] = noise.dof_vel * self.obs_scales.dof_vel
        return noise_vec

    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        if self.cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                self.restitution_coeffs = torch_rand_float(*self.cfg.domain_rand.restitution_range, (self.num_envs, 1), device=self.device)
            for prop in props:
                prop.restitution = self.restitution_coeffs[env_id, 0].item()
        return props

    def _process_rigid_body_props(self, props, env_id):
        if not hasattr(self, "default_body_masses"):
            self.default_body_masses = [prop.mass for prop in props]
            self.default_body_inertias = [
                (
                    (prop.inertia.x.x, prop.inertia.x.y, prop.inertia.x.z),
                    (prop.inertia.y.x, prop.inertia.y.y, prop.inertia.y.z),
                    (prop.inertia.z.x, prop.inertia.z.y, prop.inertia.z.z),
                )
                for prop in props
            ]

        scale = self.body_mass_scales[env_id].item()
        for body_id, prop in enumerate(props):
            prop.mass = self.default_body_masses[body_id] * scale
            values = self.default_body_inertias[body_id]
            inertia = gymapi.Mat33()
            inertia.x = gymapi.Vec3(*(value * scale for value in values[0]))
            inertia.y = gymapi.Vec3(*(value * scale for value in values[1]))
            inertia.z = gymapi.Vec3(*(value * scale for value in values[2]))
            prop.inertia = inertia
        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        # Actor 创建后不再通过 CPU API 重设刚体属性。
        if hasattr(self, "actor_handles"):
            return

        super()._randomize_rigid_body_props(env_ids, cfg)

        if cfg.domain_rand.randomize_link_mass:
            self.body_mass_scales[env_ids] = torch_rand_float(
                *cfg.domain_rand.multiplied_link_mass_range,
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.body_mass_scales[env_ids] = 1.0

    def _randomize_dof_props(self, env_ids, cfg):
        gain = torch_rand_float(
            *cfg.domain_rand.Kp_factor_range, (len(env_ids), 1), device=self.device
        )
        self.Kp_factors[env_ids] = gain
        self.Kd_factors[env_ids] = gain
        self.motor_strengths[env_ids] = 1.0
        self.motor_zero_offsets[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        offsets = torch_rand_float(
            -0.02, 0.02, (len(env_ids), len(self.leg_indices)), device=self.device
        )
        self.dof_pos[env_ids[:, None], self.leg_indices[None, :]] += offsets
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        yaw = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = 0.0
        self.root_states[env_ids, 5] = torch.sin(0.5 * yaw)
        self.root_states[env_ids, 6] = torch.cos(0.5 * yaw)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _compute_torques(self, actions):
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0.0
        position_action = actions * self.cfg.control.action_scale
        position_action[:, self.wheel_indices] = 0.0
        velocity_target = torch.zeros_like(actions)
        velocity_target[:, self.wheel_indices] = (
            actions[:, self.wheel_indices] * self.cfg.control.vel_scale
        )
        self.joint_pos_target[:] = self.default_dof_pos + position_action
        torques = self.Kp_factors * self.p_gains * (position_action + dof_err)
        torques += self.Kd_factors * self.d_gains * (velocity_target - self.dof_vel)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


        
    def _push_robots(self):
        self.root_states[:, 7:9] += torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_xy,
            self.cfg.domain_rand.max_push_vel_xy,
            (self.num_envs, 2),
            device=self.device,
        )
        self.root_states[:, 10:13] += torch_rand_float(
            -self.cfg.domain_rand.max_push_ang_vel,
            self.cfg.domain_rand.max_push_ang_vel,
            (self.num_envs, 3),
            device=self.device,
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    def _reward_action_rate(self):
        delta = self.actions - self.last_actions
        return torch.sum(torch.square(delta), dim=1)


    def _reward_action_smoothness(self):
        delta = self.actions - 2.0 * self.last_actions + self.slast_actions
        return torch.sum(torch.square(delta), dim=1)

    def _reward_wheel_lateral_slip(self):
        """Contact-gated wheel-center side velocity; a horizontal slip proxy."""
        wheels = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        # Litewheel URDF wheel axes are local +/-Y, unchanged by wheel spin.
        local_axis = torch.zeros_like(wheels[:, :, 7:10])
        local_axis[:, :, 1] = 1.0
        axis_world = quat_apply(wheels[:, :, 3:7].reshape(-1, 4), local_axis.reshape(-1, 3))
        axis_xy = axis_world.view(self.num_envs, 4, 3)[:, :, :2]
        axis_xy = axis_xy / torch.norm(axis_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        lateral_velocity = torch.sum(wheels[:, :, 7:9] * axis_xy, dim=-1)
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        lateral_gain = torch.abs(self.commands[:, 1]) / self.cfg.rewards.wheel_lateral_slip_lateral_full
        yaw_gain = torch.abs(self.commands[:, 2]) / self.cfg.rewards.wheel_lateral_slip_yaw_full
        gain = torch.maximum(lateral_gain, yaw_gain).clamp(max=1.0)
        return gain * torch.sum(torch.square(lateral_velocity) * contact, dim=1)


    def _reward_dof_acc(self):
        acceleration = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(acceleration), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        velocity = self.dof_vel.clone()
        velocity[:, self.wheel_indices] = 0.0
        return torch.sum(torch.square(velocity), dim=1)

    def _reward_dof_pos_limits(self):
        error = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        error += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        error[:, self.wheel_indices] = 0.0
        return torch.sum(error, dim=1)

    def _reward_stand_still(self):
        error = self.dof_pos - self.default_dof_pos
        error[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(error), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_hip_default(self):
        hip_ids = [i for i, name in enumerate(self.dof_names) if name.endswith("_HAA")]
        error = (self.dof_pos[:, hip_ids] - self.default_dof_pos[:, hip_ids]).norm(dim=1)
        moving = ((self.commands[:, :3].norm(dim=1) > self.cfg.rewards.hip_command_threshold)
                  | (self.base_lin_vel[:, :2].norm(dim=1) > self.cfg.rewards.hip_velocity_threshold))
        return error * torch.where(moving, 1.0, self.cfg.rewards.hip_stand_still_scale)

    def _reward_run_still(self):
        error = self.dof_pos - self.default_dof_pos
        error[:, self.wheel_indices] = 0.0
        return torch.sum(torch.abs(error), dim=1) * (torch.norm(self.commands[:, :3], dim=1) > 0.1)

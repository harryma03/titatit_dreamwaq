from legged_gym.envs.base.base_config import BaseConfig


class DynawheelCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 57
        num_obs_hist = 5
        num_privileged_obs = 57 + 3 + 187
        num_latent_dims = 16
        num_explicit_dims = 3
        num_history_obs = num_obs_hist * num_observations
        num_actions = 16
        episode_length_s = 20
        env_spacing = 3.0
        send_timeouts = True

    class terrain:
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.1]
        slope_treshold = 0.75

    class commands:
        curriculum = False
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.5]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.62]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "LF_HAA": -0.00,
            "LH_HAA": -0.00,
            "RF_HAA": +0.00,
            "RH_HAA": +0.00,
            "LF_HFE": +0.80,
            "LH_HFE": +0.80,
            "RF_HFE": +0.80,
            "RH_HFE": +0.80,
            "LF_KFE": -1.50,
            "LH_KFE": -1.50,
            "RF_KFE": -1.50,
            "RH_KFE": -1.50,
            "LF_WHEEL_MOTOR": 0.0,
            "LH_WHEEL_MOTOR": 0.0,
            "RF_WHEEL_MOTOR": 0.0,
            "RH_WHEEL_MOTOR": 0.0,
        }

    class control:
        control_type = "P"
        stiffness = {"HAA": 96.0 * 1.5, "HFE": 96.0 * 1.5, "KFE": 96.0 * 1.5, "WHEEL_MOTOR": 0.0}
        damping = {"HAA": 4.8, "HFE": 4.8, "KFE": 4.8, "WHEEL_MOTOR": 0.5}
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/dynawheel/urdf/dynawheel_rl.urdf"
        name = "dynawheel"
        foot_name = "wheel"
        wheel_name = ["WHEEL_MOTOR"]
        penalize_contacts_on = ["thigh", "calf", "Base"]
        terminate_after_contacts_on = ["Base"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        randomize_base_mass = True
        added_base_mass_range = [-1, 5]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.85, 1.15]
        damping_multiplier_range = [0.85, 1.15]
        torque_multiplier_range = [0.85, 1.15]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 3]

    class rewards:
        class scales:
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            torques = -0.000005
            dof_vel = -1e-6
            dof_acc = -1e-7
            collision = -1.0
            action_rate = -0.011
            stand_still = -0.5
            dof_pos_limits = -5.0
            hip_default = -0.5
            run_still = -0.05

        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.62
        max_contact_force = 200.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11.0, 5.0, 3.0]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class DynawheelPPO(BaseConfig):
    seed = 1
    runner_class_name = "DreamWaQRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_obs = 57

    class runner:
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        run_name = ""
        experiment_name = "dynawheel"
        save_interval = 100
        max_iterations = 300000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

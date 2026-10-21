"""M20-inspired Isaac Gym task with the Litewheel robot asset.

Isaac Lab reward terms are adapted to this project's Isaac Gym interface.
Reference: DeepRoboticsLab/rl_training commit 3d05d3b97f7a2390f003547f23a02b3b42eea195.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class FDWheelM20Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 57
        num_obs_hist = 5
        num_history_obs = 285
        num_privileged_obs = 247
        num_latent_dims = 16
        num_explicit_dims = 3
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.1]

    class commands(LeggedRobotCfg.commands):
        heading_command = True
        heading_control_stiffness = 0.5
        # Ordinary motion tracks a heading; the pure-yaw group retains direct yaw velocity.
        zero_command_fraction = 0.2
        pure_yaw_fraction = 0.2
        pure_lateral_fraction = 0.02
        pure_forward_fraction = 0.02

        class ranges:
            lin_vel_x = [-2.0, 2.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-2.0, 2.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            "LF_HAA": 0.0, "LF_HFE": 0.7, "LF_KFE": -1.35, "LF_WHEEL_MOTOR": 0.0,
            "LH_HAA": 0.0, "LH_HFE": -0.7, "LH_KFE": 1.35, "LH_WHEEL_MOTOR": 0.0,
            "RF_HAA": 0.0, "RF_HFE": 0.7, "RF_KFE": -1.35, "RF_WHEEL_MOTOR": 0.0,
            "RH_HAA": 0.0, "RH_HFE": -0.7, "RH_KFE": 1.35, "RH_WHEEL_MOTOR": 0.0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {"HAA": 60.0, "HFE": 120.0, "KFE": 120.0, "WHEEL_MOTOR": 0.0}
        damping = {"HAA": 1.0, "HFE": 1.0, "KFE": 1.0, "WHEEL_MOTOR": 0.5}
        action_scale = 0.25
        haa_action_scale = 0.125
        vel_scale = 5.0
        wheel_radius = 0.0988
        max_wheel_speed = 40.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/litewheel/urdf/litewheel_rl.urdf"
        name = "fdwheel_m20"
        foot_name = "wheel"
        wheel_name = "WHEEL_MOTOR"
        penalize_contacts_on = ["thigh", "calf", "Base"]
        terminate_after_contacts_on = ["Base"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.35, 1.5]
        randomize_restitution = True
        restitution_range = [0.0, 0.7]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
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
        latency_range = [0.0, 0.0]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 3]

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            tracking_lin_vel = 5.0
            tracking_ang_vel = 3.0
            # 机身高度约束
            base_height = -1.5
            
            lin_vel_z = -2.0
            ang_vel_xy = -0.02
            orientation = -50.0
            torques = -2.5e-5
            dof_acc = -2e-7
            wheel_acc = -1e-7
            collision = -1.0
            contact_forces = -1.5e-4
            action_rate = -0.025
            action_smoothness = -0.025
            stand_still = -1.0
            # 保留现有腿部位置惩罚
            hip_default = -3.0
            hfe_default = -1.5
            kfe_default = -0.75
            
            joint_mirror = -0.03
            
            # 转向专项奖励
            rotation_air_time = 0.0
            rotation_slip = -2.0
            rotation_gait = 2.0
            rotation_symmetry = 15.0
            
            bad_orientation = -50.0  # With dt=0.02, a tilted terminal step costs 1.0.

        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.40
        gait_level_full_terrain = 3.0
        rotation_cmd_threshold = 0.1
        rotation_lin_threshold = 0.1
        rotation_gait_lin_threshold = 0.5
        rotation_gait_ang_threshold = 0.05
        rotation_lift_height = 0.05
        rotation_contact_threshold = 1.0
        rotation_air_time_threshold = 0.2
        rotation_air_clearance = 0.0
        rotation_duty_time = 5.0
        rotation_target_duty = 0.5
        rotation_duty_std = 0.2
        bad_orientation_gravity_z = -0.5  # Terminate beyond 60 degrees of tilt.
        joint_stand_still_scale = 5.0
        high_terrain_threshold = 0.06
        high_terrain_penalty_scale = 0.1
        max_contact_force = 150.0

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 100.0


class FDWheelM20PPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.003
        learning_rate = 0.001
        num_obs = 57

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 100
        experiment_name = "fdwheel_m20"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

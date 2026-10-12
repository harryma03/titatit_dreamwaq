from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ZGM1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 57
        num_obs_hist = 5
        num_history_obs = num_obs_hist * num_observations
        num_privileged_obs = 57 + 3 + 187
        num_latent_dims = 16
        num_explicit_dims = 3
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.1]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.2
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.2]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            "FAR_ABAD_JOINT": 0.0,
            "FAR_HIP_JOINT": 0.8,
            "FAR_KNEE_JOINT": -1.5,
            "FAR_FOOT_JOINT": 0.0,
            "FBL_ABAD_JOINT": 0.0,
            "FBL_HIP_JOINT": 0.8,
            "FBL_KNEE_JOINT": -1.5,
            "FBL_FOOT_JOINT": 0.0,
            "RAR_ABAD_JOINT": 0.0,
            "RAR_HIP_JOINT": -0.8,
            "RAR_KNEE_JOINT": 1.5,
            "RAR_FOOT_JOINT": 0.0,
            "RBL_ABAD_JOINT": 0.0,
            "RBL_HIP_JOINT": -0.8,
            "RBL_KNEE_JOINT": 1.5,
            "RBL_FOOT_JOINT": 0.0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {"ABAD_JOINT": 60.0, "HIP_JOINT": 120.0, "KNEE_JOINT": 120.0, "FOOT_JOINT": 0.0}
        damping = {"ABAD_JOINT": 1.0, "HIP_JOINT": 1.0, "KNEE_JOINT": 1.0, "FOOT_JOINT": 0.5}
        action_scale = 0.25
        vel_scale = 5.0
        wheel_radius = 0.0975
        max_wheel_speed = 40.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/zg_m1/urdf/zg_m1_urdf.urdf"
        name = "zg_m1"
        foot_name = "FOOT_LINK"
        wheel_name = "FOOT_JOINT"
        penalize_contacts_on = ["ABAD_LINK", "HIP_LINK", "KNEE_LINK", "BASE_LINK"]
        terminate_after_contacts_on = ["BASE_LINK"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2,1.25]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        randomize_base_mass = True
        added_base_mass_range = [-1,5]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.85, 1.15]  
        damping_multiplier_range = [0.85, 1.15]     
        torque_multiplier_range=[0.85, 1.15]  

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles
        latency_range = [0.0, 0.0]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 3]  # Physics steps: 0-15 ms at sim.dt=0.005.

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.8 # 25/8/23 zsy说不用加
            tracking_lin_vel = 3.0 # 惩罚当前机器人在X、Y方向速度与命令不一致
            tracking_ang_vel = 1.5 # 惩罚当前机器人在角度转向速度与命令不一致
            lin_vel_z = -2.0 # 惩罚机器人在Z轴上的速度 对应现象为机器人上下起伏很大
            ang_vel_xy = -0.05 # 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动
            orientation = -0.5 # 强烈鼓励机器人与初始姿态的基座方向一致
            base_height=-10.0
            torques = -0.000005#
            dof_vel = -1e-6
            dof_acc = -1e-7
            collision = -1.
            # stumble = -0.1
            action_rate = -0.011
            stand_still=-0.5
            dof_pos_limits = -5.0
            hip_default = -0.5
            run_still=-0.05
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.50
        max_contact_force = 150. # forces above this value are penalized
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


class ZGM1PPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 0.001
        num_obs = 57

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        experiment_name = "zg_m1"
        run_name = ""
        save_interval = 100
        max_iterations = 300000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

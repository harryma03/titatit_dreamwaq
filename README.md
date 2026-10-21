# Wheel-Legged Robot Reinforcement Learning Library Dreamwaq

## 🧩 Introduction

This repository provides a **Reinforcement Learning (RL) framework** for wheel-legged robots.
It supports **simulation-based policy training**, **Sim2Sim validation**, and **Sim2Real deployment** on real robots such as **Unitree Go2** and **custom wheel-legged platforms**.

### 🚀 Features

- ✅ Supports vision-proprioception fusion reinforcement learning
- ✅ End-to-end pipeline for training, validation, and deployment
- ✅ Extensible to multiple robot platforms (Go2, m20, etc.)
- ✅ Supports custom reward functions and curriculum learning

---

## 🛠️ 使用指南

### 0. 安装依赖

python环境：3.8

#### Isaacgym 安装

```bash
pip install -e ~/isaacgym/python
python -m pip install -e ~/isaacgym/python
```

#### rsl

```bash
python -m pip install -e ./rsl_rl
```

#### legged_gym

```bash
python -m pip install -e .
```

#### 山猫m20 已经部署实物

![m20](m20_deploy.gif)

## 更新代码

git status
git add .
git commit -m "Update office litewheel config"
git push origin main

## sim2sim更换网络

 cp /home/yons/harryma/titatit_dreamwaq-main/logs/zg_m1/exported/policies/policy_dwaq.onnx /home/yons/harryma/titatit_dreamwaq-main/ddw_legged_rl/src/robot_controller/controller_drl/lib/policy/wheel_run/config_zg_m1/wheel_run.onnx

## 检查sim2sim跟最新训练的网络是否一致

 sha256sum /home/yons/harryma/titatit_dreamwaq-main/logs/zg_m1/exported/policies/policy_dwaq.onnx /home/yons/harryma/titatit_dreamwaq-main/ddw_legged_rl/install/controller_drl/share/policy/zg_m1/wheel_run/wheel_run.onnx

## 开启sim2sim

cd /home/yons/harryma/titatit_dreamwaq-main/ddw_legged_rl && source /opt/ros/humble/setup.bash && source install/setup.bash && env -u __NV_PRIME_RENDER_OFFLOAD -u __NV_PRIME_RENDER_OFFLOAD_PROVIDER -u DRI_PRIME __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 robot=litewheel sn=sim remote_type=joy slam_est_pose=off colony_id=0 ros2 launch robot_plan rl_dynawheel_simulation.launch.py

## Use

### 1. train

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" 
python legged_gym/scripts/train.py --task=dynawheel --headless
```

python legged_gym/scripts/train.py --task=titatit --headless

python legged_gym/scripts/train.py --task=go2_handstand --headless

python legged_gym/scripts/train.py --task=go2_legstand --headless

python legged_gym/scripts/train.py --task=m20 --headless

`python legged_gym/scripts/train.py --task=zg_m1 --headless`

### 2. play policy

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" 
python legged_gym/scripts/play.py --task=zg_m1 --num_envs=50
```

python legged_gym/scripts/play.py --task=go2_handstand --num_envs=50

python legged_gym/scripts/play.py --task=go2_legstand --num_envs=50

python legged_gym/scripts/play.py --task=dynawheel --num_envs=50

### problem

如果你发现自己的urdf训练出来轮子不转，而是抬腿，把urdf中的轮子的continuous改为revolute,上下限改一下-99999 99999，即可训练出轮子转的模型

### 为什么可以训练出轮子转的模型

注意到

action_scale = 0.25

vel_scale = 5.0

故轮子的action更能表现出来故更容易训练出轮子转的模型

# 致谢

https://github.com/XinLang2019/Wheel_Legged_Gym# titatit_dreamwaq

# titatit_dreamwaq

## Dynawheel 步态稳定性优化（2026-09-21）

本次修改参考 `zg/legged_gym/legged_gym/envs/dynawheel_run` 的训练思路，重点改善横移和自旋时的对角交替支撑、拖地和关节抖动。改动位于 `legged_gym/envs/dynawheel/dynawheel.py` 与 `legged_gym/envs/dynawheel/dynawheel_config.py`，不会影响 M20 任务。

### 保持不变的控制参数

以下参数按要求保留：腿部 `Kp=103.9`、`Kd=7.4`，`action_scale=0.25`，轮速 `vel_scale=5.0`，仿真 `dt=0.005 s`、`decimation=4`。因此物理仿真仍为 200 Hz，策略仍为 50 Hz。

### 配置改动与预期收益

- 命令改为直接给定 yaw 角速度，并设置 `slew_rate=[2.5, 2.5, 3.0]`。目标速度逐步变化，减少转向或切换命令时的突然动作。
- 将动作变化惩罚分成腿关节和轮子两组，并采用 `zg` 的权重。横移/旋转时不再额外加倍腿部平滑惩罚，避免惩罚压制必要的摆腿。
- 增加对角支撑、触地后腾空时间、对角摆动高度、摆动净空、步态相位停滞和落脚冲击奖励。摆动净空目标为轮底离地 5 cm；对角支撑、净空、摆动高度和相位奖励只在横移或旋转命令下生效，前后直行不要求额外抬腿。腾空时间奖励用于所有运动命令，落脚冲击惩罚在每次有效落脚时生效。
- 关闭正奖励裁剪，让步态惩罚和速度跟踪奖励都能参与总奖励优化。
- 参考 `zg` 调整摩擦 `[0.2, 1.1]`、恢复系数 `[0.0, 0.2]`、附加机身质量 `[-1, 20] kg` 和机身质心偏移 `±0.10 m`；关闭训练推扰、额外关节摩擦/阻尼随机化与观测延迟，动作延迟设为 1–3 个物理仿真步（约 5–15 ms）。连杆质量仍为 `[0.9, 1.1]` 倍，PD/扭矩随机增益保持原来的 ±5%。
- 添加按关节类型随机化的一阶扭矩低通滤波，并设置腿关节 armature 为 `0.13`、轮子为 `0.008`。预期减少高频扭矩尖峰，同时保留轮子较高的滤波带宽。

这些是训练目标和仿真执行器模型的预期效果；是否改善实际步态仍需训练后用固定命令和同一地形进行 play 对比确认。

### Dynawheel 函数说明

#### 命令、仿真状态和执行器

- `step()`：将策略动作写入动作缓存，在每个 decimation 子步计算扭矩并推进仿真；随后刷新延迟观测和策略观测。
- `_resample_commands()`：沿用原本的随机命令采样，并按比例产生自旋或横移命令。
- `_post_physics_step_callback()`：调用父类更新命令/地形等状态；再按配置限制 x、y、yaw 命令每个策略步的变化量，更新自旋状态、轮底净空和接触计时。
- `post_physics_step()`：父类完成奖励、观测和 reset 后，结束本步接触状态更新；触地腿的腾空计时归零，离地腿的支撑计时归零，并保存本步接触状态供下步比较。
- `_init_buffers()`：创建接触状态、触地事件、腾空/支撑计时、滤波扭矩和每关节滤波截止频率等张量。
- `reset_idx()`：环境重置时清空步态、滤波和接触缓存，避免上个回合的状态泄漏到新回合。
- `_prepare_foot_contact_state()`：用进入/保持两级力阈值识别触地，并在腾空足够时间后标记有效 touchdown；同时更新腾空和支撑时长。
- `_foot_clearances()`：计算轮心相对局部地形的高度，再减轮半径，得到轮底净空。
- `_update_spin_gait()`：根据当前 yaw 和线速度命令判断是否处于自旋状态，并刷新轮底净空。
- `_randomize_torque_filter_cutoff()`：按 HAA、HFE、KFE、WHEEL_MOTOR 的配置频段，为每个环境和关节抽取滤波截止频率。
- `_process_dof_props()`：保留 DOF 限位处理；关闭关节 armature 随机化时，按腿关节/轮子分别写入固定 armature。
- `_compute_torques()`：用保留的 PD/action/轮速 scale 计算扭矩，经过按关节采样的一阶低通滤波后，按 URDF 力矩上限裁剪。
- `_reset_obs_latency_buffer()`、`update_obs_latency_buffer()`、`compute_observations()`：管理延迟观测缓存和拼接策略观测。当前配置下 `update_obs_latency_buffer()` 会直接返回，`compute_observations()` 使用当前传感器状态；reset helper 仍会初始化缓存。
- `_push_robots()`：实现随机推扰；当前 `push_robots=False`，训练过程中不会调用推扰。

#### 奖励函数

- `_single_diagonal_support_masks()`：判断 LF+RH 或 LH+RF 是否构成唯一的支撑对角。
- `_diagonal_swing_mask()`：根据支撑对角标记另一组需要摆动的腿。
- `_reward_support_trot_not_only_x()`：横移/自旋时奖励直立状态下的一组对角支撑，前后直行不触发。
- `_reward_feet_air_time()`：只在有效落脚时奖励腾空时间接近 `0.60 s`，不在每个摆动帧直接给抬腿奖励。
- `_reward_feet_distance_xy_exp()`：奖励四个轮足维持参考的前后位置和左右间距，帮助保持支撑几何。
- `_reward_foot_clearance_world()`：只对由对角支撑确定的摆动腿惩罚净空低于 5 cm 的情况，并且只在横移/旋转时启用。
- `_reward_diagonal_swing_height()`：惩罚同一摆动对角中的两条腿高度不一致，减少一条腿先抬、另一条拖地的情况。
- `_reward_foot_phase_stall()`：惩罚横移/旋转时某条腿长时间持续支撑或持续腾空，促使步态换相。
- `_reward_feet_impact_vel()`：在有效落脚时惩罚向下冲击速度，降低砸地造成的机身振动。
- `_reward_action_rate_dof()` / `_reward_action_rate_wheel()`：分别计算腿关节和轮子相邻策略步之间的动作变化。
- `_reward_smoothness_dof()` / `_reward_smoothness_wheel()`：分别计算腿关节和轮子动作的二阶差分，约束高频抖动。
- `_reward_action_smoothness()`：旧的整体平滑度实现仍保留，但当前配置没有启用对应 reward scale，不参与训练。
- `_reward_feet_clearance()`、`_reward_spin_contact_pattern()`、`_reward_spin_slip()`：旧净空/自旋奖励实现仍保留在文件中，但当前权重为 `0`，不会参与训练。

### 验证状态

已对 Dynawheel Python 文件执行语法编译检查。此修改尚未经过 Isaac Gym 训练或 play 验证；训练日志、对角支撑比例、拖地次数和落脚冲击需要在新 run 中观察。当前 PPO 配置若启用 `resume`，会按 `resume_path` 继续载入已有策略。

"""CPU regression checks for actual M20 reward methods, without Isaac Gym."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch


def rotate(q, v):
    t = 2 * torch.cross(q[:, :3], v, dim=-1)
    return v + q[:, 3:] * t + torch.cross(q[:, :3], t, dim=-1)


def rotate_inverse(q, v):
    conjugate = q.clone()
    conjugate[:, :3] *= -1
    return rotate(conjugate, v)


def load_classes():
    directory = Path(__file__).resolve().parent
    scope = {
        "torch": torch, "inspect": inspect, "quat_apply": rotate, "quat_rotate_inverse": rotate_inverse,
        "wrap_to_pi": lambda a: torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi,
        "torch_rand_float": lambda low, high, shape, device: torch.empty(shape, device=device).uniform_(low, high),
    }
    for path in (directory.parent / "base/base_config.py",
                 directory.parent / "base/legged_robot_config.py",
                 directory / "fdwheel_m20_config.py"):
        classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)

    base = next(node for node in ast.parse((directory.parent / "base/legged_robot.py").read_text()).body
                if isinstance(node, ast.ClassDef))
    base.bases = []
    base.body = [node for node in base.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_update_terrain_curriculum"]
    exec(compile(ast.Module(body=[base], type_ignores=[]), "legged_robot.py", "exec"), scope)

    # Exercise actual sampling and callback code; isolate simulator allocation/reset side effects.
    parent = next(node for node in ast.parse((directory / "fdwheel.py").read_text()).body
                  if isinstance(node, ast.ClassDef))
    parent.body = [node for node in parent.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"check_termination", "_reward_action_smoothness", "_reward_action_rate",
                                     "_resample_commands", "_post_physics_step_callback"}]
    exec(compile(ast.Module(body=[parent], type_ignores=[]), "fdwheel.py", "exec"), scope)
    scope["FDWheelRobot"].reset_idx = lambda self, ids: None
    scope["FDWheelRobot"]._init_buffers = lambda self: None
    scope["FDWheelRobot"]._get_heights = lambda self: self.measured_heights
    path = directory / "fdwheel_m20.py"
    classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)
    return scope["FDWheelM20Robot"], scope["FDWheelM20Cfg"]


def make_robot(n=4):
    Robot, Config = load_classes()
    robot = object.__new__(Robot)
    robot.cfg = Config()
    robot.num_envs = n
    robot.device = "cpu"
    robot.dt = 0.02
    robot.feet_indices = torch.arange(4)
    robot.wheel_indices = torch.tensor([3, 7, 11, 15])
    robot.leg_indices = torch.tensor([i for i in range(16) if i % 4 != 3])
    robot.commands = torch.tensor([[0., 0., 1., 0.]]).repeat(n, 1)
    robot.command_ranges = {key: getattr(robot.cfg.commands.ranges, key)
                            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")}
    robot.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(n, 1)
    robot.contact_forces = torch.zeros(n, 5, 3)
    robot.contact_forces[:, :4, 2] = 20.0
    robot.foot_positions = torch.zeros(n, 4, 3)
    robot.foot_positions[:, :, 2] = 0.1
    robot.episode_length_buf = torch.full((n,), 5)
    robot.max_episode_length = 1000
    robot.termination_contact_indices = torch.tensor([4])
    robot.root_states = torch.zeros(n, 13)
    robot.root_states[:, 2] = 0.55
    robot.root_states[:, 6] = 1.0
    robot.base_quat = robot.root_states[:, 3:7]
    robot.forward_vec = torch.tensor([[1., 0., 0.]]).repeat(n, 1)
    robot.base_lin_vel = torch.zeros(n, 3)
    robot.measured_heights = torch.zeros(n, 187)
    robot.env_origins = torch.zeros(n, 3)
    robot.terrain_levels = torch.full((n,), 3, dtype=torch.long)
    robot.terrain_types = torch.zeros(n, dtype=torch.long)
    robot.max_terrain_level = 10
    robot.terrain_origins = torch.zeros(10, 1, 3)
    robot.terrain = SimpleNamespace(env_length=8.0)
    robot.max_episode_length_s = 20.0
    robot.init_done = True
    robot.common_step_counter = 1
    robot.cfg.domain_rand.push_robots = False
    robot.dof_names = [leg + suffix for leg in ("LF_", "LH_", "RF_", "RH_")
                       for suffix in ("HAA", "HFE", "KFE", "WHEEL_MOTOR")]
    robot.default_dof_pos = torch.zeros(1, 16)
    robot.dof_pos = torch.zeros(n, 16)
    robot.extras = {"episode": {}}
    robot._init_buffers()
    return robot


def set_support(robot, indices):
    robot.contact_forces.zero_()
    robot.contact_forces[:, indices, 2] = 20.0
    robot.foot_positions[:, :, 2] = 0.16
    robot.foot_positions[:, indices, 2] = 0.1


def main():
    torch.manual_seed(13)
    robot = make_robot()
    robot._update_rotation_state()
    assert robot.cfg.rewards.scales.rotation_symmetry == 15.0
    assert torch.all(robot._reward_rotation_symmetry() < 1e-6), "full contact has negligible duty reward"

    for _ in range(130):
        set_support(robot, [])
        robot._update_rotation_state()
        set_support(robot, [0, 1, 2, 3])
        robot._update_rotation_state()
    assert torch.allclose(robot._reward_rotation_symmetry(), torch.ones(4)), (
        "official duty reward alone cannot distinguish synchronous hopping from alternation")
    robot = make_robot()
    robot._update_rotation_state()
    set_support(robot, [0, 3])
    for _ in range(125):
        robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_gait() == 1)
    assert torch.all(robot._reward_rotation_symmetry() < 1e-4), "fixed diagonal has poor duty balance"
    assert torch.all(robot._reward_rotation_symmetry() > 0), "no extra alternation gate"
    set_support(robot, [1, 2])
    robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_symmetry() > 0)
    assert torch.all(robot._reward_rotation_air_time() > 0)
    for _ in range(124):
        robot._update_rotation_state()
    assert torch.allclose(robot._reward_rotation_symmetry(), torch.ones(4))
    set_support(robot, [0, 1, 2, 3])
    robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_symmetry() > 0.99), "balanced history need not hold a lifted pose"
    robot.foot_positions[:, :, 2] = 0.1
    assert not robot._reward_rotation_gait().any(), "contact flicker without clearance must not count"
    robot.commands[:, :3] = torch.tensor([1., 0., 0.])
    robot._update_rotation_state()
    assert not robot._reward_rotation_air_time().any()
    assert not robot._reward_rotation_symmetry().any()
    robot.commands[:, :3] = torch.tensor([0., 0., -1.])
    for _ in range(robot.rotation_history_size):
        robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_symmetry() < 1e-4), "full-contact history has poor duty balance"

    # Restore the short-swing negative term and reject startup/contact-only rewards.
    robot = make_robot()
    robot._update_rotation_state()
    set_support(robot, [0, 3])
    for _ in range(3):
        robot._update_rotation_state()
    set_support(robot, [0, 1, 2, 3])
    robot._update_rotation_state()
    assert torch.allclose(robot._reward_rotation_air_time(), torch.full((4,), -0.28))
    robot._update_rotation_state()
    assert not robot._reward_rotation_air_time().any(), "touchdown reward must occur once"
    robot.rotation_air_time[:] = 10.0
    robot.rotation_max_air_height[:] = 0.16
    robot.rotation_last_contact[:] = False
    robot._update_rotation_state()
    assert torch.allclose(robot._reward_rotation_air_time(), torch.full((4,), 39.2)), (
        "official touchdown reward has no extra positive cap")
    robot.rotation_air_time[:] = 1.0
    robot.reset_idx(torch.tensor([1, 3]))
    assert not robot.rotation_air_time[[1, 3]].any()
    assert robot.rotation_air_time[[0, 2]].all()
    assert not robot.rotation_contact_history[[1, 3]].any()
    assert not robot.rotation_contact_sum[[1, 3]].any()
    robot._update_rotation_state()
    assert not robot._reward_rotation_air_time()[[1, 3]].any(), "reset must not invent an airborne interval"

    for level in (0, 1, 2, 3, 9):
        robot.terrain_levels[:] = level
        robot._update_gait_level()
        expected = min(1.0, torch.exp(torch.tensor(float(level - 3))).item()) if level else 0.0
        assert abs(robot.gait_level.item() - expected) < 1e-6
    robot.terrain_levels[:] = torch.tensor([2, 2, 2, 9])
    robot.root_states[:, 0] = torch.tensor([5., 0., 0., 5.])
    robot.commands[:, :3] = 0.0
    robot.commands[:, 0] = torch.tensor([1., 1., 0., 1.])
    robot._update_terrain_curriculum(torch.arange(4))
    assert robot.terrain_levels[:3].tolist() == [3, 1, 2]
    assert 0 <= robot.terrain_levels[3] < 10
    assert robot.terrain_progress == {"terrain_move_up_rate": 0.5, "terrain_move_down_rate": 0.25,
                                      "terrain_max_level_wrap_rate": 0.25}

    robot = make_robot()
    robot.dof_vel = torch.zeros(4, 16)
    robot.last_dof_vel = torch.zeros(4, 16)
    robot.torques = torch.zeros(4, 16)
    robot.dof_vel[:, robot.wheel_indices] = 10.0
    robot.torques[:, robot.wheel_indices] = 10.0
    assert not robot._reward_dof_acc().any(), "wheels must not be penalized twice"
    assert not robot._reward_torques().any(), "M20 leg torque term must exclude wheels"
    assert torch.all(robot._reward_wheel_acc() > 0)
    robot.dof_vel.zero_()
    robot.dof_vel[:, robot.leg_indices] = 1.0
    assert torch.all(robot._reward_dof_acc() > 0)
    assert not robot._reward_wheel_acc().any()

    robot.actions = torch.ones(4, 16)
    robot.last_actions = torch.zeros(4, 16)
    robot.slast_actions = torch.zeros(4, 16)
    robot.episode_length_buf[:] = torch.tensor([0, 1, 2, 3])
    assert torch.equal(robot._reward_action_smoothness(), torch.tensor([0., 0., 0., 16.]))

    robot.projected_gravity[:, 2] = torch.tensor([-1., -0.6, -0.4, 1.])
    robot.contact_forces[0, 4, 2] = 20.0
    robot.episode_length_buf[1] = 1001
    robot.check_termination()
    assert robot.reset_buf.all(), "retain parent contact/timeout resets and add tilt termination"
    assert torch.equal(robot._reward_bad_orientation(), torch.tensor([0., 0., 1., 1.]))
    penalty = robot._reward_bad_orientation() * robot.cfg.rewards.scales.bad_orientation * robot.dt
    assert penalty.min() == -1.0
    robot.projected_gravity[:, 2] = -1.0
    assert not robot._reward_bad_orientation().any()

    # Contact motion is relative to base translation and measured in the body plane.
    robot = make_robot()
    robot.foot_velocities = torch.zeros(4, 4, 3)
    robot.root_states[:, 7] = 1.0
    robot.foot_velocities[:, :, 0] = 1.0
    assert not robot._reward_rotation_slip().any()
    robot.foot_velocities[:, :, 1] = 0.5
    robot.commands[:, 2] = torch.tensor([1., -1., 1., -1.])
    assert torch.allclose(robot._reward_rotation_slip(), torch.full((4,), 2.0))
    robot.root_states[:, 7:10] = 0.0
    robot.root_states[:, 3:7] = torch.tensor([2**-0.5, 0., 0., 2**-0.5])
    robot.foot_velocities[:] = torch.tensor([0., 0., 1.])
    assert torch.allclose(robot._reward_rotation_slip(), torch.full((4,), 4.0))

    robot = make_robot()
    robot.commands[:, :3] = torch.tensor([1., 0., 0.])
    robot.dof_pos[:, robot.haa_indices] = 1.0
    assert torch.allclose(robot._reward_hip_default(), torch.full((4,), 2.0)), "official posture uses L2 norm"
    robot.commands[:, :3] = 0.0
    assert torch.allclose(robot._reward_hip_default(), torch.full((4,), 10.0))
    robot.dof_pos[:, robot.hfe_indices] = 1.0
    robot.commands[:, :3] = torch.tensor([0., 0., 1.])
    assert not robot._reward_hfe_default().any()
    robot.commands[:, :3] = torch.tensor([1., 0., 0.])
    robot.measured_heights[:] = 1.0
    robot.measured_heights[:, robot.base_height_index] = 0.0
    assert torch.allclose(robot._reward_hfe_default(), torch.full((4,), 2.0))
    robot.measured_heights[:, robot.base_height_index] = 0.1
    assert torch.allclose(robot._reward_hfe_default(), torch.full((4,), 0.2))

    robot = make_robot(4096)
    robot._resample_commands(torch.arange(robot.num_envs))
    assert abs(robot.heading_envs.float().mean().item() - 0.56) < 0.03
    zero = robot.commands[:, :3].norm(dim=1) == 0
    yaw = (robot.commands[:, :2].norm(dim=1) == 0) & ~zero
    assert abs(zero.float().mean().item() - 0.2) < 0.03
    assert abs(yaw.float().mean().item() - 0.2) < 0.03
    assert robot.commands[yaw, 2].min() < -1.9 and robot.commands[yaw, 2].max() > 1.9
    before = robot.commands.clone()
    robot._post_physics_step_callback()
    assert torch.equal(robot.commands[~robot.heading_envs], before[~robot.heading_envs])
    robot.heading_envs[:2] = torch.tensor([True, False])
    robot.commands[:2, 3] = torch.pi / 2
    robot.direct_yaw_commands[1] = -1.2
    robot._update_heading_commands(torch.tensor([0, 1]))
    assert abs(robot.commands[0, 2].item() - torch.pi / 4) < 1e-6
    robot.root_states[0, 3:7] = torch.tensor([0., 0., 2**-0.5, 2**-0.5])
    robot._update_heading_commands(torch.tensor([0, 1]))
    assert abs(robot.commands[0, 2].item()) < 1e-6
    assert abs(robot.commands[1, 2].item() + 1.2) < 1e-6
    robot.cfg.commands.heading_command = False
    robot.commands[:, :3] = torch.tensor([0.3, -0.2, -0.7])
    robot._post_physics_step_callback()
    assert torch.allclose(robot.commands[:, :3], torch.tensor([0.3, -0.2, -0.7]).expand(4096, 3))
    print("PASS: heading/direct-yaw/zero groups, play command preservation, terrain progression and gait scaling,")
    print("      signed uncapped touchdown, five-second duty symmetry, reset isolation, body-frame slip,")
    print("      posture norms, leg/wheel separation, smoothness warmup and tilt termination (CPU tensors only).")


if __name__ == "__main__":
    main()

"""CPU regression checks for actual M20 reward methods, without Isaac Gym."""
import ast
import inspect
from pathlib import Path

import torch


def load_classes():
    directory = Path(__file__).resolve().parent
    scope = {"torch": torch, "inspect": inspect}
    for path in (directory.parent / "base/base_config.py",
                 directory.parent / "base/legged_robot_config.py",
                 directory / "fdwheel_m20_config.py"):
        classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)

    # Retain the real parent's termination/smoothness methods; reset has simulator side effects.
    parent = next(node for node in ast.parse((directory / "fdwheel.py").read_text()).body
                  if isinstance(node, ast.ClassDef))
    parent.bases = []
    parent.body = [node for node in parent.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"check_termination", "_reward_action_smoothness"}]
    exec(compile(ast.Module(body=[parent], type_ignores=[]), "fdwheel.py", "exec"), scope)
    scope["FDWheelRobot"].reset_idx = lambda self, ids: None
    path = directory / "fdwheel_m20.py"
    classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)
    return scope["FDWheelM20Robot"], scope["FDWheelM20Cfg"]


def make_robot():
    Robot, Config = load_classes()
    robot = object.__new__(Robot)
    robot.cfg = Config()
    robot.num_envs = 4
    robot.device = "cpu"
    robot.dt = 0.02
    robot.feet_indices = torch.arange(4)
    robot.wheel_indices = torch.tensor([3, 7, 11, 15])
    robot.leg_indices = torch.tensor([i for i in range(16) if i % 4 != 3])
    robot.commands = torch.tensor([[0., 0., 1., 0.]]).repeat(4, 1)
    robot.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    robot.contact_forces = torch.zeros(4, 5, 3)
    robot.contact_forces[:, :4, 2] = 20.0
    robot.foot_positions = torch.zeros(4, 4, 3)
    robot.foot_positions[:, :, 2] = 0.1
    robot.rotation_contact_duty = torch.full((4, 4), 0.5)
    robot.rotation_pair_seen = torch.zeros(4, 2, dtype=torch.bool)
    robot.rotation_air_time = torch.zeros(4, 4)
    robot.rotation_air_valid = torch.zeros(4, 4, dtype=torch.bool)
    robot.rotation_last_contact = torch.zeros(4, 4, dtype=torch.bool)
    robot.rotation_air_reward = torch.zeros(4)
    robot.episode_length_buf = torch.full((4,), 5)
    robot.max_episode_length = 1000
    robot.termination_contact_indices = torch.tensor([4])
    robot.root_states = torch.zeros(4, 13)
    robot.root_states[:, 2] = 0.55
    robot.measured_heights = torch.zeros(4, 187)
    return robot


def set_support(robot, indices):
    robot.contact_forces.zero_()
    robot.contact_forces[:, indices, 2] = 20.0
    robot.foot_positions[:, :, 2] = 0.16
    robot.foot_positions[:, indices, 2] = 0.1


def main():
    robot = make_robot()
    robot._update_rotation_state()
    assert not robot._reward_rotation_symmetry().any(), "standing must not earn symmetry at reset"

    for _ in range(10):
        set_support(robot, [])
        robot._update_rotation_state()
        set_support(robot, [0, 1, 2, 3])
        robot._update_rotation_state()
    assert not robot._reward_rotation_symmetry().any(), "synchronous hopping is not diagonal alternation"
    assert not robot.rotation_air_valid.any()

    set_support(robot, [0, 3])
    for _ in range(15):
        robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_gait() == 1)
    assert not robot._reward_rotation_symmetry().any(), "one fixed diagonal is not alternation"
    set_support(robot, [1, 2])
    robot._update_rotation_state()
    assert torch.all(robot._reward_rotation_symmetry() > 0)
    assert torch.all(robot._reward_rotation_air_time() > 0)
    robot.foot_positions[:, :, 2] = 0.1
    assert not robot._reward_rotation_gait().any(), "contact flicker without clearance must not count"

    robot.commands[:, :3] = torch.tensor([1., 0., 0.])
    robot._update_rotation_state()
    assert not robot.rotation_pair_seen.any()
    assert not robot.rotation_air_time.any()
    assert not robot._reward_rotation_air_time().any()
    assert not robot._reward_rotation_symmetry().any()

    robot.commands[:, :3] = torch.tensor([0., 0., -1.])
    robot.rotation_air_time[:] = 10.0
    robot.rotation_air_valid[:] = True
    robot.rotation_last_contact[:] = False
    set_support(robot, [0, 1, 2, 3])
    robot._update_rotation_state()
    assert torch.allclose(robot._reward_rotation_air_time(), torch.full((4,), 0.8))
    robot._update_rotation_state()
    assert not robot._reward_rotation_air_time().any(), "touchdown reward must occur once"

    robot.rotation_pair_seen[:] = True
    robot.rotation_air_time[:] = 1.0
    robot.reset_idx(torch.tensor([1, 3]))
    assert not robot.rotation_pair_seen[[1, 3]].any()
    assert robot.rotation_pair_seen[[0, 2]].all()
    assert not robot.rotation_air_time[[1, 3]].any()

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
    print("PASS: tilt termination, no free symmetry/hopping reward, diagonal alternation, bounded touchdown,")
    print("      command/reset isolation, leg/wheel separation, and smoothness warmup (CPU tensors only).")


if __name__ == "__main__":
    main()

from types import SimpleNamespace
import ast
from pathlib import Path

import torch

# Test actual reward methods on CPU without importing Isaac Gym or starting a sim.
env_dir = Path(__file__).resolve().parent
parent_tree = ast.parse((env_dir.parent / "M20/m20.py").read_text())
parent_class = next(n for n in parent_tree.body if isinstance(n, ast.ClassDef))
parent_class.bases = []
parent_class.body = [n for n in parent_class.body if isinstance(n, ast.FunctionDef) and n.name == "_reward_run_still"]
tree = ast.parse((env_dir / "dynawheel.py").read_text())
robot_class = next(n for n in tree.body if isinstance(n, ast.ClassDef))
namespace = {"torch": torch}
exec(compile(ast.Module(body=[parent_class, robot_class], type_ignores=[]), str(env_dir / "dynawheel.py"), "exec"), namespace)
DynawheelRobot = namespace["DynawheelRobot"]
# Existing command scheduling tests also exercise uniform draws.
namespace["torch_rand_float"] = lambda low, high, shape, device: low + (high - low) * torch.rand(shape, device=device)
config_tree = ast.parse((env_dir / "dynawheel_config.py").read_text())
config_class = next(n for n in config_tree.body if isinstance(n, ast.ClassDef))
config_class.bases = []
exec(compile(ast.Module(body=[config_class], type_ignores=[]), "dynawheel_config.py", "exec"), namespace)


def main():
    robot = object.__new__(DynawheelRobot)
    robot.cfg = SimpleNamespace(rewards=namespace["DynawheelCfg"].rewards)
    robot.dt = 0.02
    robot.feet_indices = torch.arange(4)
    robot.dof_pos = torch.ones(4, 16)
    robot.default_dof_pos = torch.zeros(4, 16)
    robot.commands = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    hip_penalty = robot._reward_hip_default()
    assert hip_penalty[0] > 0.0
    assert torch.all(hip_penalty[1:3] > 0.0) and hip_penalty[3] == 0.0
    robot.wheel_indices = torch.tensor([3, 7, 11, 15])
    assert torch.allclose(robot._reward_run_still(), torch.tensor([12.0, 3.6, 3.6, 0.0]))
    assert torch.equal(robot._maneuver_gait_weight(), torch.tensor([0., 1., 1., 0.]))
    robot.commands[0, 0] = -0.5
    robot.commands[0, 2] = 0.5
    assert robot._maneuver_gait_weight()[0] == 0.0  # rolling turn at forward/back speed

    robot.spin_contact_masks = torch.tensor([[True, False, False, True], [False, True, True, False]])
    robot.wheel_indices = torch.tensor([3, 7, 11, 15])
    robot.spin_active = torch.ones(2, dtype=torch.bool)
    robot.spin_start_step = torch.zeros(2, dtype=torch.long)
    robot.episode_length_buf = torch.tensor([0, 15])
    robot.contact_forces = torch.zeros(2, 4, 3)
    robot.contact_forces[:, :, 2] = 2.0 * robot.spin_contact_masks[0]
    assert torch.equal(robot._reward_spin_contact_pattern(), torch.tensor([1.0, 0.0]))

    robot.actions = torch.ones(2, 16)
    robot.actions[:, robot.wheel_indices] = 10.0
    robot.last_actions = torch.zeros(2, 16)
    robot.last_last_actions = torch.zeros(2, 16)
    robot.spin_active[:] = False
    robot.commands = torch.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
    assert torch.equal(robot._reward_action_smoothness(), torch.tensor([12.0, 12.0]))
    robot.commands[1, :3] = 0.0
    robot.spin_active[1] = True
    assert torch.equal(robot._reward_action_smoothness(), torch.tensor([12.0, 12.0]))

    robot.commands = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]])
    robot.episode_length_buf[:] = 15
    robot.spin_start_step.zero_()
    robot.spin_active[:] = torch.tensor([False, True])
    robot.current_foot_contacts = robot.spin_contact_masks.clone()
    robot.feet_air_time = (~robot.current_foot_contacts).float() * 0.2
    robot.feet_contact_time = robot.current_foot_contacts.float() * 0.2
    assert torch.equal(robot._reward_maneuver_trot(), torch.zeros(2))
    assert torch.equal(robot._reward_foot_phase_stall(), torch.zeros(2))
    robot.current_foot_contacts[:] = True
    assert torch.all(robot._reward_maneuver_trot() > 0)  # dragging cannot avoid gait cost
    robot.current_foot_contacts[:] = robot.spin_contact_masks
    robot.foot_clearance = (~robot.current_foot_contacts).float() * 0.01
    robot.max_foot_clearance = torch.zeros(2, 4)
    robot.feet_were_airborne = torch.zeros(2, 4, dtype=torch.bool)
    assert torch.all(robot._reward_feet_clearance() > 0)  # low swing has dense feedback
    robot.foot_clearance = (~robot.current_foot_contacts).float() * 0.15
    assert torch.equal(robot._reward_feet_clearance(), torch.zeros(2))  # no stair height ceiling
    robot.foot_clearance.zero_()
    assert torch.equal(robot._reward_feet_clearance(), torch.zeros(2))  # descent not penalized
    robot.current_foot_contacts[:] = True
    assert torch.equal(robot._reward_feet_clearance(), torch.zeros(2))
    assert not robot.feet_were_airborne.any() and not robot.max_foot_clearance.any()
    assert robot._reward_feet_clearance()[0] == 0.0  # lateral keeps touchdown-only behavior
    assert robot._reward_feet_clearance()[1] > 0.0   # grounded spin cannot avoid clearance
    robot.current_foot_contacts[1] = robot.spin_contact_masks[0]
    robot.max_foot_clearance[1, robot.spin_contact_masks[1]] = 0.10
    assert robot._reward_feet_clearance()[1] == 0.0  # either supported diagonal is valid
    robot.feet_air_time[:] = 1.0
    assert torch.all(robot._reward_foot_phase_stall() > 0)  # cannot hover indefinitely
    robot.commands[:] = torch.tensor([0.5, 0., 0., 0.])
    assert torch.equal(robot._reward_maneuver_trot(), torch.zeros(2))
    assert torch.equal(robot._reward_feet_clearance(), torch.zeros(2))
    assert torch.equal(robot._reward_foot_phase_stall(), torch.zeros(2))
    robot.commands[:] = torch.tensor([-0.5, 0., 0., 0.])
    assert torch.equal(robot._maneuver_gait_weight(), torch.zeros(2))
    robot.commands.zero_()
    assert torch.equal(robot._maneuver_gait_weight(), torch.zeros(2))

    # Long rolling stance must not carry a stale phase penalty into a new maneuver.
    robot.commands[:] = torch.tensor([0., 0.5, 0., 0.])
    robot.maneuver_was_active = torch.zeros(2, dtype=torch.bool)
    robot.last_foot_contacts = torch.ones(2, 4, dtype=torch.bool)
    robot.first_foot_contacts = torch.zeros(2, 4, dtype=torch.bool)
    robot.contact_forces[:, :, 2] = 20.0
    robot.feet_contact_time[:] = 10.0
    robot._prepare_foot_contact_state()
    assert torch.allclose(robot.feet_contact_time, torch.full((2, 4), robot.dt))
    assert torch.equal(robot._reward_foot_phase_stall(), torch.zeros(2))
    # Clearing a maneuver resets swing peaks; they cannot hide a later short swing.
    robot.max_foot_clearance[:] = 0.2
    robot.feet_were_airborne[:] = True
    robot.commands.zero_()
    robot._reward_feet_clearance()
    assert not robot.max_foot_clearance.any() and not robot.feet_were_airborne.any()

    robot.num_envs = 4
    robot.device = "cpu"
    robot.max_episode_length = 1000
    robot.cfg.commands = SimpleNamespace(emphase=SimpleNamespace(
        type=["rotate", "lateral", "stand"],
        time_range={"rotate": [2.0, 2.0], "lateral": [2.0, 2.0], "stand": [2.0, 2.0]},
    ))
    robot.emphase_cfg = robot.cfg.commands.emphase
    robot.emphase_enabled = True
    robot.command_ranges = {
        "lin_vel_x": [-1.2, 1.3],
        "lin_vel_y": [-1.0, 1.0],
        "ang_vel_yaw": [-1.57, 1.57],
    }
    robot.commands = torch.zeros(4, 4)
    robot.episode_length_buf = torch.zeros(4, dtype=torch.long)
    robot.emphase_times = torch.zeros(4, 3, dtype=torch.long)
    robot.emphase_phase = torch.full((4,), -1, dtype=torch.long)
    env_ids = torch.arange(4)
    robot._generate_emphase_time(env_ids)
    assert torch.all(robot.emphase_times[:, 0] == 0)
    assert torch.all(robot.emphase_times[:, 1] > 0)
    assert torch.all(robot.emphase_times[:, 2] > robot.emphase_times[:, 1])
    robot._update_emphase_commands(env_ids)
    assert torch.all(robot.commands[:, 2].abs() >= 0.25)
    robot.episode_length_buf[:] = robot.emphase_times[:, 1]
    robot._update_emphase_commands(env_ids)
    assert torch.all(robot.commands[:, 0] == 0.0) and torch.all(robot.commands[:, 1].abs() >= 0.25)
    robot.episode_length_buf[:] = robot.emphase_times[:, 2]
    robot._update_emphase_commands(env_ids)
    assert torch.all(robot.commands[:, :3] == 0.0)
    print("PASS: maneuver gating, diagonal support, swing clearance, stance, and rolling isolation")


if __name__ == "__main__":
    main()

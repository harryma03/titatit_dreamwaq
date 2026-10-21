"""CPU check of the FDWheel HAA reward without importing Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    path = Path(__file__).with_name("fdwheel.py")
    robot_class = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in robot_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_reward_hip_default")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    config = ast.parse(path.with_name("fdwheel_config.py").read_text())
    cfg_class = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == "FDWheelCfg")
    rewards = next(n for n in cfg_class.body if isinstance(n, ast.ClassDef) and n.name == "rewards")
    params = {n.targets[0].id: ast.literal_eval(n.value) for n in rewards.body
              if isinstance(n, ast.Assign) and n.targets[0].id.startswith("hip_")}
    robot = SimpleNamespace(
        cfg=SimpleNamespace(rewards=SimpleNamespace(**params)),
        dof_names=[leg + suffix for leg in ("LF", "LH", "RF", "RH")
                   for suffix in ("_HAA", "_HFE", "_KFE", "_WHEEL_MOTOR")],
        dof_pos=torch.zeros(4, 16), default_dof_pos=torch.zeros(1, 16),
        commands=torch.zeros(4, 4), base_lin_vel=torch.zeros(4, 3))
    robot.dof_pos[:, [0, 4, 8, 12]] = 0.1
    robot.commands[1, 0] = 1.0
    robot.commands[2, 2] = -1.0
    robot.base_lin_vel[3, 0] = 0.6
    reward = scope["_reward_hip_default"]
    assert torch.allclose(reward(robot), torch.tensor([1., .2, .2, .2]))
    robot.dof_pos[:, [0, 4, 8, 12]] = 0.0
    robot.dof_pos[:, [1, 2, 3]] = 2.0
    assert not reward(robot).any(), "HFE/KFE/wheel positions must not affect HAA reward"
    print("PASS: HAA norm, standing multiplier, translation/yaw/moving-base gates, unrelated joints")


if __name__ == "__main__":
    main()

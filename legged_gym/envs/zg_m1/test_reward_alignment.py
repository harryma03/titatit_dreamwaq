"""Compare actual reward methods with M20 on CPU, without Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def load_rewards(path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {"_reward_action_rate", "_reward_action_smoothness",
               "_reward_dof_acc", "_reward_dof_vel"}
    nodes = [node for node in cls.body
             if isinstance(node, ast.FunctionDef) and node.name in methods]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in methods}


def main():
    directory = Path(__file__).resolve().parent
    actual = load_rewards(directory / "zg_m1.py")
    reference = load_rewards(directory.parent / "M20/m20.py")
    torch.manual_seed(0)
    robot = SimpleNamespace(dt=0.02, wheel_indices=torch.tensor([3, 7, 11, 15]))
    for wheel_only in (False, True):
        for name in ("actions", "last_actions", "slast_actions", "dof_vel", "last_dof_vel"):
            value = torch.randn(8, 16)
            if wheel_only:
                mask = torch.zeros_like(value)
                mask[:, robot.wheel_indices] = 1
                value *= mask
            setattr(robot, name, value)
        robot.last_last_actions = robot.slast_actions
        for name, method in actual.items():
            result = method(robot)
            assert torch.allclose(result, reference[name](robot)), name
            if wheel_only:
                if name == "_reward_dof_vel":
                    assert torch.equal(result, torch.zeros(8))
                else:
                    assert (result > 0).all(), name
    print("PASS: M20 reward equality, including wheel-only action/acceleration changes")


if __name__ == "__main__":
    main()

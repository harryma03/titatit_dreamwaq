"""CPU regression for the real step/delay methods, without importing Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    path = Path(__file__).with_name("zg_m1.py")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"step", "_update_action_latency_buffer", "_reset_action_latency"}]
    scope = {"torch": torch, "gymtorch": SimpleNamespace(unwrap_tensor=lambda x: x)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    robot = scope[cls.name]()
    rand = SimpleNamespace(add_cmd_action_latency=True, randomize_cmd_action_latency=False,
                           range_cmd_action_latency=[0, 3])
    robot.cfg = SimpleNamespace(domain_rand=rand, control=SimpleNamespace(decimation=4),
                                normalization=SimpleNamespace(clip_actions=100., clip_observations=100.))
    robot.num_envs, robot.num_obs, robot.device = 4, 1, "cpu"
    robot.obs_buf = torch.zeros(4, 1)
    robot.obs_hist_buf = torch.zeros(4, 5)
    robot.cmd_action_latency_buffer = torch.zeros(4, 1, 4)
    robot.cmd_action_latency_simstep = torch.arange(4)
    robot.torques = torch.zeros(4, 1)
    robot.privileged_obs_buf = None
    robot.vel_buf = robot.rew_buf = robot.reset_buf = robot.extras = None
    robot.sim = None
    robot.render = robot.post_physics_step = lambda: None
    noop = lambda *args: None
    robot.gym = SimpleNamespace(set_dof_actuation_force_tensor=noop, simulate=noop,
                                fetch_results=noop, refresh_dof_state_tensor=noop)
    applied = []

    def capture(actions):
        applied.append(actions.clone())
        return actions

    robot._compute_torques = capture
    robot.step(torch.ones(4, 1))
    robot.step(torch.full((4, 1), 2.))
    actual = torch.stack(applied).squeeze(-1)
    for delay in range(4):
        expected = torch.tensor([0.] * delay + [1.] * 4 + [2.] * (4 - delay))
        assert torch.equal(actual[:, delay], expected), (delay, actual[:, delay])
    robot._reset_action_latency(torch.arange(4))
    assert not robot.cmd_action_latency_buffer.any()
    assert (robot.cmd_action_latency_simstep == 3).all()
    rand.add_cmd_action_latency = False
    applied.clear()
    robot.step(torch.full((4, 1), 3.))
    assert torch.equal(torch.stack(applied), torch.full((4, 4, 1), 3.))
    print("PASS: delays 0/1/2/3 physics steps, policy transition, reset, disabled delay")


if __name__ == "__main__":
    main()

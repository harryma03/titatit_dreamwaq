import json
import numpy as np
import os
from datetime import datetime
import shlex
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")
for path in (RSL_RL_DIR, ROOT_DIR):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import isaacgym  
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry

import torch
# import debugpy
# print('waiting for debugpy ...')
# debugpy.listen(1990)
# debugpy.wait_for_client()
# print("Debugpy attached, please enter 'F5' for debuging...")


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    with open(os.path.join(ppo_runner.log_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump({"args": vars(args), "env": class_to_dict(env_cfg), "train": class_to_dict(train_cfg)}, file, indent=2, default=str)
    with open(os.path.join(ppo_runner.log_dir, "cmd.txt"), "w", encoding="utf-8") as file:
        file.write(shlex.join([sys.executable, *sys.argv]) + "\n")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)

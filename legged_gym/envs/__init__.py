# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from legged_gym.envs.M20.m20_config import M20_Cfg_Yu,M20_PPO_Yu
from legged_gym.envs.M20.m20 import M20_Robot
from legged_gym.envs.dynawheel.dynawheel_config import DynawheelCfg, DynawheelPPO
from legged_gym.envs.dynawheel.dynawheel import DynawheelRobot
from legged_gym.envs.zg_m1.zg_m1_config import ZGM1Cfg, ZGM1PPO
from legged_gym.envs.zg_m1.zg_m1 import ZGM1Robot
from legged_gym.envs.titatit.titatit_config import Titatit_Cfg, Titatit_PPO
from legged_gym.envs.titatit.titatit import Titatit_Robot

from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand_Config import GO2Cfg_Leggedstand,GO2CfgPPO_Leggedstand
from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand import Go2_legstand

from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand
from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2_legstand", Go2_legstand, GO2Cfg_Leggedstand(), GO2CfgPPO_Leggedstand())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())
task_registry.register( "m20", M20_Robot, M20_Cfg_Yu(), M20_PPO_Yu())
task_registry.register("dynawheel", DynawheelRobot, DynawheelCfg(), DynawheelPPO())
task_registry.register("zg_m1", ZGM1Robot, ZGM1Cfg(), ZGM1PPO())
task_registry.register( "titatit", Titatit_Robot, Titatit_Cfg(), Titatit_PPO())

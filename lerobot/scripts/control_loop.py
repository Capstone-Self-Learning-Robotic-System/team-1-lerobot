import argparse
import logging
import time
import socket
import numpy as np
import json
#import torch
from pathlib import Path
from typing import List

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    delete_current_episode,
    init_dataset,
    save_current_episode,
    add_frame,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int

def busy_wait(dt):   
    current_time = time.time()
    while (time.time() < current_time+dt):
        pass


if __name__ == "__main__":

    fps = 30
    teleop_time_s = 10

    timestamp = 0
    start_episode_t = time.perf_counter()

    # teleoperation loop
    for _ in range(fps*teleop_time_s):
        start_loop_t = time.perf_counter()

        #data = client_socket.recv(24)
        #motor_array = np.frombuffer(data, dtype=np.float32)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        timestamp = time.perf_counter() - start_episode_t
        print(dt_s)

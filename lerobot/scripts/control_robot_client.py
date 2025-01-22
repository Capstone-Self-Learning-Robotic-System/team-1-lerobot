import argparse
import logging
import time
import socket
import numpy as np
import json
import tqdm
import cv2
from pathlib import Path
from typing import List
from datetime import datetime

from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    delete_current_episode,
    init_dataset,
    save_current_episode,
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


########################################################################################
# Util functions
########################################################################################


def busy_wait(dt):   
    current_time = time.time()
    while (time.time() < current_time+dt):
        pass
    

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def remote_teleoperate(
    robot: Robot, 
    fps: int, 
    ip: str, 
    port: int,
    teleop_time_s: float | None
):
    if not robot.is_connected:
        robot.connect()

    # open socket for communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    data = {}
    data['control_mode'] = 'remote_teleoperate'
    data['fps'] = fps
    data['teleop_time_s'] = teleop_time_s
    json_data = json.dumps(data)
    client_socket.send(json_data.encode().ljust(1024))
    
    if teleop_time_s == None:
        teleop_time_s = float("inf")
        log_say(f"Teleoperate for infinite time", True)
    else:
        log_say(f"Teleoperate for {teleop_time_s} seconds", False)
        pbar = tqdm.tqdm(range(teleop_time_s*fps))

    # start timer
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    try:
        # teleoperation loop
        while timestamp < teleop_time_s:
            if teleop_time_s != float("inf"):
                pbar.update(1)
            
            start_loop_t = time.perf_counter()

            motor_array = robot.leader_arms["main"].read("Present_Position")
            client_socket.sendall(motor_array)

            response = client_socket.recv(1024).decode()

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            timestamp = time.perf_counter() - start_episode_t
    
    except KeyboardInterrupt:
        log_say(f"Teleoperation terminated", True)
        robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        robot.disconnect()

        client_socket.close()

    robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.disconnect()
    
    client_socket.close()


@safe_disconnect
def remote_record(
    robot: Robot, 
    fps: int, 
    ip: str, 
    port: int, 
    repo_id: str, 
    model_id: int,
    warmup_time_s=2, 
    episode_time_s=5, 
    num_episodes=3
):

    if not robot.is_connected:
        robot.connect()

    # open socket for communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    data = {}
    data['control_mode'] = 'remote_record'
    data['fps'] = fps
    data['warmup_time_s'] = warmup_time_s
    data['episode_time_s'] = episode_time_s
    data['num_episodes'] = num_episodes
    data['repo_id'] = repo_id
    data['model_id'] = model_id
    json_data = json.dumps(data)
    client_socket.send(json_data.encode().ljust(1024))

    curr_episode = 0

    log_say(f"Warmup record for {warmup_time_s} seconds", False)
    timestamp = 0
    start_episode_t = time.perf_counter()

    pbar = tqdm.tqdm(range(warmup_time_s*fps))

    try:
        # warmup loop
        while timestamp < warmup_time_s:
            pbar.update(1)
            start_loop_t = time.perf_counter()

            motor_array = robot.leader_arms["main"].read("Present_Position")
            client_socket.sendall(motor_array)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            timestamp = time.perf_counter() - start_episode_t

    except KeyboardInterrupt:
        log_say(f"Remote recording terminated", True)
        curr_episode = num_episodes

    while curr_episode < num_episodes:

        log_say(f"Recording episode {curr_episode} for {episode_time_s} seconds", False)

        pbar = tqdm.tqdm(range(episode_time_s*fps))

        timestamp = 0
        start_episode_t = time.perf_counter()

        try:
            # episode loop
            while timestamp < episode_time_s:
                pbar.update(1)
                start_loop_t = time.perf_counter()

                motor_array = robot.leader_arms["main"].read("Present_Position")
                client_socket.sendall(motor_array)

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

                dt_s = time.perf_counter() - start_loop_t
                timestamp = time.perf_counter() - start_episode_t
        
        except KeyboardInterrupt:
            log_say(f"Remote recording terminated", True)
            curr_episode = num_episodes
            break
    
        curr_episode += 1
    
    robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.disconnect()

    client_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch_leader.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    base_parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_teleop = subparsers.add_parser("remote_teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_teleop.add_argument(
        "--teleop-time-s", type=int, default=None, help="Number of seconds of teleoperation (set to None for infinite teleoperation)"
    )
    parser_teleop.add_argument(
        "--ip", type=str, default=None, help="IP address of host remote socket"
    )
    parser_teleop.add_argument(
        "--port", type=int, default=None, help="Port address of host remote socket"
    )


    ############################################################


    parser_record = subparsers.add_parser("remote_record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=2,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=5,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--ip", type=str, default=None, help="IP address of host remote socket"
    )
    parser_record.add_argument(
        "--port", type=int, default=None, help="Port address of host remote socket"
    )
    parser_record.add_argument(
        "--repo-id", type=str, default=str(datetime.now()), help="Dataset identifier",
    )
    parser_record.add_argument(
        "--model-id", type=int, default=None, help="Model identifier",
    )
    parser_record.add_argument(
        "--num-episodes", type=int, default=None, help="Number of episodes recorded",
    )


    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    if control_mode == "remote_teleoperate":
        remote_teleoperate(robot, **kwargs)

    elif control_mode == "remote_record":
        remote_record(robot, **kwargs)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()

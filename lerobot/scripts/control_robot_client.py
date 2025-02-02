import argparse
import logging
import time
import socket
import numpy as np
import json
import tqdm
import cv2
from pynput.keyboard import Key, Listener
import os

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

term_teleop = False
term_session = False

def on_release_ent(key):
    global term_teleop

    if key == Key.enter:
        term_teleop = True

def on_release_esc(key):
    global term_session

    if key == Key.esc:
        term_session = True

listener_esc = Listener(on_release=on_release_esc)
listener_esc.start()

listener_ent = Listener(on_release=on_release_ent)
listener_ent.start()


########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def remote_teleoperate(
    robot: Robot, 
    fps: int
):
    
    global term_session, term_teleop

    if not robot.is_connected:
        robot.connect()

    # open socket for communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("50.39.109.27", 50065))

    data = {}
    data['control_mode'] = 'remote_teleoperate'
    data['fps'] = fps
    json_data = json.dumps(data)
    client_socket.send(json_data.encode().ljust(1024))
    
    log_say(f"Teleoperation Active", True)
    
    teleop = True

    # teleoperation loop
    while teleop and not program_ending:
        
        start_loop_t = time.perf_counter()

        motor_array = robot.leader_arms["main"].read("Present_Position")

        if term_session:
            client_socket.sendall(b"term_teleop", 24)
            term_session = False
            teleop = False
            break
        else:
            client_socket.sendall(motor_array)

        response = client_socket.recv(1024).decode()

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

    robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.disconnect()
    
    client_socket.close()


@safe_disconnect
def remote_record(
    robot: Robot, 
    fps: int, 
    repo_id: str
):

    global term_session, term_teleop

    if not robot.is_connected:
        robot.connect()

    # open socket for communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("50.39.109.27", 50065))

    data = {}
    data['control_mode'] = 'remote_record'
    data['fps'] = fps
    data['repo_id'] = repo_id
    json_data = json.dumps(data)
    client_socket.send(json_data.encode().ljust(1024))

    curr_episode = 0

    recording = True

    while recording and not program_ending:

        log_say(f"Warmup Active", True)

        warmup = True
        
        # warmup loop
        while warmup:
            start_loop_t = time.perf_counter()

            motor_array = robot.leader_arms["main"].read("Present_Position")

            if term_teleop:
                client_socket.sendall(b"term_teleop", 24)
                term_teleop = False
                warmup = False
                continue
            elif term_session:
                client_socket.sendall(b"term_session", 24)
                term_session = False
                record = False
                recording = False
                break
            else:
                client_socket.sendall(motor_array)
                
            response = client_socket.recv(1024).decode()

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        if recording:
            log_say(f"Recording Episode {curr_episode} Active", True)

        episode = True

        # episode loop
        while episode and not program_ending:
            start_loop_t = time.perf_counter()

            motor_array = robot.leader_arms["main"].read("Present_Position")
            
            if term_teleop:
                client_socket.sendall(b"term_teleop", 24)
                term_teleop = False
                episode = False
                continue
            elif term_session:
                client_socket.sendall(b"term_session", 24)
                term_session = False
                episode = False
                recording = False
                break
            else:
                client_socket.sendall(motor_array)
            
            response = client_socket.recv(1024).decode()

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
    
        curr_episode += 1
    
    robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    robot.disconnect()

    client_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for aimport osll the subparsers
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


    ############################################################


    parser_record = subparsers.add_parser("remote_record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--repo-id", type=str, default=str(datetime.now()), help="Dataset identifier",
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

    program_ending = False

    try:

        if control_mode == "remote_teleoperate":
            remote_teleoperate(robot, **kwargs)

        elif control_mode == "remote_record":
            remote_record(robot, **kwargs)

        if robot.is_connected:
            # Disconnect manually to avoid a "Core dump" during process
            # termination due to camera threads not properly exiting.
            robot.disconnect()
    
    except KeyboardInterrupt:
        program_ending = True
        print("Exiting Gracefully")


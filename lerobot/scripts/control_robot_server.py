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
    teleop_time_s: float, 
    client_socket: socket
):
    #if not robot.is_connected:
    #    robot.connect()
    
    # start timer
    timestamp = 0
    start_episode_t = time.perf_counter()

    #log_say(f"Teleoperate for {teleop_time_s} seconds", True)
    
    # teleoperation loop
    while timestamp < teleop_time_s:
        start_loop_t = time.perf_counter()

        data = client_socket.recv(24)
        motor_array = np.frombuffer(data, dtype=np.float32)
        print(motor_array)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        timestamp = time.perf_counter() - start_episode_t
        #log_control_info(robot, dt_s, fps=fps)


@safe_disconnect
def remote_record(
    robot: Robot, 
    fps: int, 
    warmup_time_s: float, 
    episode_time_s: float, 
    num_episodes: int, 
    repo_id: str
):

    if not robot.is_connected:
        robot.connect()

    dataset = init_dataset(
        repo_id=repo_id, 
        fps=fps, 
        root="data", 
        force_override=False, 
        video=True,
        write_images=False,
        num_image_writer_processes=0, 
        num_image_writer_threads=4 * 0
    )

    timestamp = 0
    start_episode_t = time.perf_counter()

    log_say(f"Warmup record for {warmup_time_s} seconds", True)

    # warmup
    for _ in range(warmup_time_s*fps):
        start_loop_t = time.perf_counter()

        #motor_array = robot.leader_arms["main"].read("Present_Position")
        #data = client_socket.recv(48)
        
        #motor_array = np.frombuffer(data, dtype=np.float32)
        #print(motor_array)
        #robot.follower_arms["main"].write("Goal_Position", motor_array)

        dt_s = time.perf_counter() - start_loop_t
        #log_control_info(robot, dt_s, fps=fps)

    while dataset["num_episodes"] < num_episodes:

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index} for {episode_time_s} seconds", True)

        for _ in range(episode_time_s*fps):
            start_loop_t = time.perf_counter()

            motor_array = robot.leader_arms["main"].read("Present_Position")
            #data = client_socket.recv(48)
            
            #motor_array = np.frombuffer(data, dtype=np.float32)
            #robot.follower_arms["main"].write("Goal_Position", motor_array)
            #print(motor_array)

            observation, action = robot.teleop_step(record_data=True)
            add_frame(dataset, observation, action)

            dt_s = time.perf_counter() - start_loop_t
            #log_control_info(robot, dt_s, fps=fps)
        
        save_current_episode(dataset)

    lerobot_dataset = create_lerobot_dataset(dataset, True, False, None, True)


if __name__ == "__main__":

    init_logging()

    robot_path = "lerobot/configs/robot/koch.yaml"

    robot_cfg = init_hydra_config(robot_path)
    robot = make_robot(robot_cfg)

    # open socket for communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("10.0.0.19", 12345))

    while True:
        server_socket.listen(5)
        client_socket, addr = server_socket.accept()

        data = client_socket.recv(1024).decode()
        json_data = json.loads(data)
        control_mode = json_data['control_mode']
        fps = json_data['fps']

        if control_mode == 'remote_teleoperate':
            teleop_time_s = json_data['teleop_time_s']
            remote_teleoperate(robot, fps, teleop_time_s, client_socket)

        elif control_mode == "remote_record":
            warmup_time_s = json_data['warmup_time_s']
            episode_time_s = json_data['episode_time_s']
            num_episodes = json_data['num_episodes']
            repo_id = json_data['repo_id']
            remote_record(robot, fps, warmup_time_s, episode_time_s, num_episodes, repo_id)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()

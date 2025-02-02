import argparse
import logging
import threading
import time
import socket

import cv2
import numpy as np
import json
#import torch
from pathlib import Path
from typing import List
import torch

from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

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
MAX_FPS = 10
def remote_stream(robot: Robot, client: socket, camera_name: str):
    # client.sendall(robot.cameras)
    # try:
    while True and not program_ending:
        image = robot.cameras[camera_name].async_read()

        # Encode to jpeg for smaller transmission
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
        result, enc_img = cv2.imencode('.jpg', image, encode_param)

        # Send to client and wait for response
        client.sendall(np.array(enc_img).tobytes())
        client.send(b'this_is_the_end')

        response = client.recv(1024).decode()
        # print(response)


@safe_disconnect
def remote_teleoperate(
    robot: Robot, 
    fps: int, 
    client_socket: socket
):
    if not robot.is_connected:
        print("Robot was not connected, attempting to reconnect", end="")
        while not robot.is_connected:
            print(".", end="")
            robot.connect()
    
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    
    log_say(f"Teleoperation Active", False)

    teleop = True
    
    # teleoperation loop
    while teleop and not program_ending:
        start_loop_t = time.perf_counter()

        data = client_socket.recv(24)

        if data == b"term_teleop":
            teleop = False
            log_say(f"Teleoperation Terminated", False)
            continue
        
        motor_array = np.frombuffer(data, dtype=np.float32)

        if not np.any(motor_array):
            log_say(f"Teleoperation Terminated", False)
            break

        robot.follower_arms["main"].write("Goal_Position", motor_array)
        client_socket.send("ack".encode())

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        log_control_info(robot, dt_s, fps=fps)
    
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)


@safe_disconnect
def remote_record(
    robot: Robot, 
    fps: int, 
    repo_id: str,
):

    global program_ending

    if not robot.is_connected:
        print("Robot was not connected, attempting to reconnect", end="")
        while not robot.is_connected:
            print(".", end="")
            robot.connect()
    
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)

    dataset = init_dataset(
        repo_id=repo_id, 
        fps=fps, 
        root="data", 
        force_override=False, 
        video=True,
        write_images=True,
        num_image_writer_processes=0, 
        num_image_writer_threads=4
    )

    curr_episode = 0

    while not program_ending:

        log_say(f"Warmup Active", False)

        warmup = True

        # warmup loop
        
        while warmup and not program_ending:
            start_loop_t = time.perf_counter()

            data = client_socket.recv(24)
            
            if data == b"term_teleop":
                warmup = False
                continue
            elif data == b"term_session":
                log_say(f"Remote Recording Rerminated", False)
                program_ending = True
                break

            motor_array = np.frombuffer(data, dtype=np.float32)

            if not np.any(motor_array):
                log_say(f"Remote Recording Rerminated", False)
                break

            robot.follower_arms["main"].write("Goal_Position", motor_array)
            client_socket.send("ack".encode())

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=fps)

        record = True

        log_say(f"Recording episode {curr_episode}", False)

        timestamp = 0
        start_episode_t = time.perf_counter()

        # episode loop
        while record and not program_ending:
            start_loop_t = time.perf_counter()

            data = client_socket.recv(24)

            if data == b"term_teleop":
                record = False
                continue
            elif data == b"term_session":
                log_say(f"Remote Recording Rerminated", False)
                program_ending = True
                break
            
            motor_array = np.frombuffer(data, dtype=np.float32)

            if not np.any(motor_array):
                log_say(f"Remote Recording Rerminated", False)
                break

            robot.follower_arms["main"].write("Goal_Position", motor_array)
            client_socket.send("ack".encode())

            state = []
            state.append(torch.from_numpy(robot.follower_arms["main"].read("Present_Position")))
            state = torch.cat(state)

            action = []
            action.append(torch.from_numpy(motor_array))
            action = torch.cat(action)

            images = {}

            for name in robot.cameras:
                images[name] = robot.cameras[name].async_read()
                images[name] = torch.from_numpy(images[name])

            obs_dict, action_dict = {}, {}
            obs_dict["observation.state"] = state
            action_dict["action"] = action
            for name in robot.cameras:
                obs_dict[f"observation.images.{name}"] = images[name]

            add_frame(dataset, obs_dict, action_dict)

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            timestamp = time.perf_counter() - start_episode_t
            log_control_info(robot, dt_s, fps=fps)
            
        if not program_ending:
            curr_episode += 1
            save_current_episode(dataset)

    lerobot_dataset = create_lerobot_dataset(dataset, True, False, None, True)

    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)

    program_ending = False


def accept_client(robot: Robot, client_socket: socket):
    data = client_socket.recv(1024).decode()
    json_data = json.loads(data)
    control_mode = json_data['control_mode']

    if control_mode == 'remote_teleoperate':
        fps = json_data['fps']
        remote_teleoperate(robot, fps, client_socket)

    elif control_mode == "remote_record":
        fps = json_data['fps']
        repo_id = json_data['repo_id']
        remote_record(robot, fps, repo_id)

    elif control_mode == "remote_stream":
        camera_name = json_data["camera_name"]
        remote_stream(robot, client_socket, camera_name)

    client_socket.close()


if __name__ == "__main__":

    init_logging()

    # Configure robot
    robot_path = "./lerobot/configs/robot/koch_follower.yaml"

    robot_cfg = init_hydra_config(robot_path, None)
    robot = make_robot(robot_cfg)

    # Connect to robot before teleop starts so camera stream can be started
    robot.connect()
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)

    # Open socket for communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("192.168.0.96", 50065))
    server_socket.listen(5)

    program_ending = False
    threads = []

    try:
        while True:
            client_socket, addr = server_socket.accept()

            new_thread = threading.Thread(target=accept_client, args=(robot, client_socket))
            threads.append(new_thread)
            new_thread.start()
    
    except KeyboardInterrupt:
        
        server_socket.close()
        program_ending = True

        print("Waiting for Threads to Stop")
        for thread in threads:
            thread.join()

        robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
        robot.disconnect()
        print("Exiting Gracefully")

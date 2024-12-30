import argparse
import logging
import time
import socket
import numpy as np
import json
from pathlib import Path
from typing import List

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
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int
    

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def remote_teleoperate(
    robot: Robot, 
    fps: int, 
    client_socket: socket,
):
    #if not robot.is_connected:
    #    robot.connect()
    
    # start timer
    timestamp = 0
    start_episode_t = time.perf_counter()
    
    # teleoperation loop
    while True:
        start_loop_t = time.perf_counter()

        #motor_array = robot.leader_arms["main"].read("Present_Position")
        data = client_socket.recv(48)
        if not data:
            client_socket.close()
            break
        
        motor_array = np.frombuffer(data, dtype=np.float32)
        
        #robot.follower_arms["main"].write("Goal_Position", motor_array)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)


@safe_disconnect
def remote_record(
    robot: Robot,
    repo_id : str,
    root: str,
    fps: int | None = None,
    warmup_time_s=2,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    num_image_writer_processes=0,
    num_image_writer_threads_per_camera=4,
    force_override=False,
    display_cameras=True,
    play_sounds=True,
):
    listener = None
    events = None
    policy = None
    device = None
    use_amp = None

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", play_sounds)
    warmup_record(robot, events, enable_teleoperation, warmup_time_s, display_cameras, fps)

    while True:
        if dataset["num_episodes"] >= num_episodes:
            break

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index}", play_sounds)
        record_episode(
            dataset=dataset,
            robot=robot,
            events=events,
            episode_time_s=episode_time_s,
            display_cameras=display_cameras,
            policy=policy,
            device=device,
            use_amp=use_amp,
            fps=fps,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (episode_index < num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", play_sounds)
            reset_environment(robot, events, reset_time_s)

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            delete_current_episode(dataset)
            continue

        # Increment by one dataset["current_episode_index"]
        save_current_episode(dataset)

        if events["stop_recording"]:
            break

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    lerobot_dataset = create_lerobot_dataset(dataset, run_compute_stats, push_to_hub, tags, play_sounds)

    log_say("Exiting", play_sounds)
    return lerobot_dataset


if __name__ == "__main__":

    init_logging()

    robot_path = "lerobot/configs/robot/koch_follower.yaml"

    robot_cfg = init_hydra_config(robot_path)
    robot = make_robot(robot_cfg)

    # open socket for communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 12345))

    while True:
        server_socket.listen(5)
        client_socket, addr = server_socket.accept()

        data = client_socket.recv(1024).decode()
        json_data = json.loads(data)
        control_mode = json_data['control_mode']
        fps = json_data['fps']

        if control_mode == 'remote_teleoperate':
            remote_teleoperate(robot, fps, client_socket)

        elif control_mode == "remote_record":
            remote_record(robot, ...)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()

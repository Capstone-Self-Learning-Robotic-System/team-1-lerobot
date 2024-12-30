import argparse
import logging
import time
import socket
import numpy as np
import json
import tqdm
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
    teleop_time_s: float, 
    ip: str, 
    port: int
):
    #if not robot.is_connected:
    #    robot.connect()

    if teleop_time_s is None:
        teleop_time_s = float("inf")
    
    # start timer
    timestamp = 0
    start_episode_t = time.perf_counter()

    # open socket for communication
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    data = {}
    data['control_mode'] = 'remote_teleoperate'
    data['fps'] = 30
    json_data = json.dumps(data)
    client_socket.send(json_data.encode().ljust(1024))

    pbar = tqdm.tqdm(range(teleop_time_s*fps))
    
    # teleoperation loop
    while timestamp < teleop_time_s:
        pbar.update(1)
        start_loop_t = time.perf_counter()

        #motor_array = robot.leader_arms["main"].read("Present_Position")
        motor_array = np.array([-0.43945312, 133.94531, 179.82422, -18.984375, -1.9335938, 34.541016])
        client_socket.sendall(motor_array)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        #log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
    
    client_socket.close()


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
        "--teleop_time_s", type=int, default=None, help="Number of seconds of teleoperation (set to None for infinite teleoperation)"
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
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser_record.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
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

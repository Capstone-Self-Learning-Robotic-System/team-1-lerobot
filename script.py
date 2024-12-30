from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
import tqdm

leader_port = "/dev/tty.usbmodem58CD1773081"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"main": leader_arm},
    calibration_dir=".cache/calibration/koch",
)

robot.connect()

seconds = 5
frequency = 200
for _ in tqdm.tqdm(range(seconds*frequency)):
    print(robot.leader_arms["main"].read("Present_Position"))
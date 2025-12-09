# joint_sender.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

UR5E_JOINTS = [
    'shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
    'wrist_1_joint','wrist_2_joint','wrist_3_joint'
]

class JointSender(Node):
    def __init__(self, controller='/joint_trajectory_controller'):
        super().__init__('joint_sender')
        self.controller = controller
        self.client = ActionClient(self, FollowJointTrajectory,
                                   f'{controller}/follow_joint_trajectory')

    def send(self, joint_positions_list, dt):
        self.get_logger().info(f"Connecting to {self.controller}/follow_joint_trajectory ...")
        while not self.client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("  Waiting for action server...")
        self.get_logger().info("Action server connected!")

        traj = JointTrajectory()
        traj.joint_names = UR5E_JOINTS

        # t=0 for the very first point
        t = 0.0
        for i, q in enumerate(joint_positions_list):
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            traj.points.append(pt)
            if i == 0:
                t = dt  # first sample at 0, next at dt
            else:
                t += dt

        self.get_logger().info(f"Sending trajectory with {len(traj.points)} points.")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        result_future = self.client._get_result_async(send_future.result())
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.result.error_code != 0:
            self.get_logger().warn(f"Controller error code: {result.result.error_code}")
        else:
            self.get_logger().info("Trajectory executed successfully.")
        return result.result.error_code == 0

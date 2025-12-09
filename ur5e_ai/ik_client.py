# ik_client.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

class IKClient(Node):
    def __init__(self, group_name='ur_manipulator'):
        super().__init__('ik_client')
        self.cli = self.create_client(GetPositionIK, '/compute_ik')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /compute_ik ...')
        self.group_name = group_name

    def solve(self, pose: PoseStamped, seed_state=None):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True

        if seed_state is not None:
            rs = RobotState()
            rs.joint_state = seed_state
            req.ik_request.robot_state = rs

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f"IK result code: {res.error_code.val}")
        self.get_logger().info(f"Returned joints: {res.solution.joint_state.name}")
        return res.solution.joint_state

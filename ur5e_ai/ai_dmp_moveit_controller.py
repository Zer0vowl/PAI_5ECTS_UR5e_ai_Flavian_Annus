import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from ur5e_ai.lemniscate import lemniscate
from ur5e_ai.joint_sender import JointSender

from ur5e_ai.utils import (
    UR_JOINTS,
    START_Q,
    quat_from_rpy,
    limit_step,
    latest_q
)

def build_waypoints(node):
    """
    Build Cartesian waypoints directly from the analytic lemniscate trajectory.
    """
    T, pos, vel, pitch_profile, dt = lemniscate(
        T=10.0,
        dt=0.02,
        a=0.08,
        center=(0.30, 0.0, 0.5),
        close_loop=True,
        rock_z_amp=0.01,
        rock_pitch_amp_deg=8.0,
        rock_phase_mult=1.0,
    )

    waypoints = []
    for i, p in enumerate(pos):
        quat = quat_from_rpy(np.pi, pitch_profile[i], 0.0)
        pose = Pose()
        pose.position.x = float(p[0])
        pose.position.y = float(p[1])
        pose.position.z = float(p[2])
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = quat
        waypoints.append(pose)

    node.get_logger().info(
        f"Built {len(waypoints)} Cartesian waypoints from lemniscate."
    )
    return waypoints

def wait_for_joint_states(node, timeout=10.0):
    got = {"msg": None}

    def cb(msg):
        if msg.name:
            got["msg"] = msg

    sub = node.create_subscription(JointState, "/joint_states", cb, 1)
    t0 = node.get_clock().now().nanoseconds * 1e-9
    while (
        got["msg"] is None
        and (node.get_clock().now().nanoseconds * 1e-9 - t0) < timeout
    ):
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)
    return got["msg"]

def ik_solve(node, ik_client, pose: Pose, seed_state: JointState | None = None):
    """Call MoveIt's /compute_ik for a single pose."""
    if not ik_client.service_is_ready():
        node.get_logger().error("/compute_ik not ready.")
        return None

    ps = PoseStamped()
    ps.header.frame_id = "base_link"
    ps.pose = pose

    from moveit_msgs.srv import GetPositionIK

    req = GetPositionIK.Request()
    req.ik_request.group_name = "ur_manipulator"
    req.ik_request.pose_stamped = ps
    req.ik_request.avoid_collisions = True
    req.ik_request.timeout.sec = 0
    req.ik_request.timeout.nanosec = int(0.1 * 1e9)

    if seed_state is not None:
        rs = RobotState()
        rs.joint_state = seed_state
        req.ik_request.robot_state = rs

    future = ik_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    res = future.result()
    if res is None:
        node.get_logger().warn("IK call returned None")
        return None

    if res.error_code.val != res.error_code.SUCCESS:
        node.get_logger().warn(f"IK failed with code {res.error_code.val}")
        return None

    js = res.solution.joint_state
    joint_map = dict(zip(js.name, js.position))
    try:
        q = [joint_map[j] for j in UR_JOINTS]
    except KeyError:
        node.get_logger().error("IK result missing some UR_JOINTS")
        return None

    return q


def main():
    rclpy.init()
    node = Node("ai_dmp_moveit_controller_node")

    sender = JointSender(controller="/joint_trajectory_controller")

    ik_client = node.create_client(GetPositionIK, "/compute_ik")
    log_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
    log_pub = node.create_publisher(Bool, "/ee_log_enable", log_qos)

    # Move to START_Q

    q_curr = latest_q(node, timeout=10.0)
    if q_curr is None:
        node.get_logger().error("No valid /joint_states received. Aborting.")
        sender.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("Moving to START_Q via joint sender...")
    ramp = limit_step([q_curr, START_Q], max_delta=0.01)
    node.get_logger().info(f"Ramp-to-start has {len(ramp)} points.")
    sender.send(ramp, dt=0.02)
    current_q = START_Q

    # Build waypoints directly from lemniscate
    node.get_logger().info("Generating waypoints")
    waypoints = build_waypoints(node)

    seed_js = wait_for_joint_states(node, timeout=2.0)
    if seed_js is None:
        node.get_logger().error("No /joint_states for IK seed")
        return

    joint_path = []
    valid_count = 0
    for i, pose in enumerate(waypoints):
        q = ik_solve(node, ik_client, pose, seed_state=seed_js)
        if q is None:
            continue

        joint_path.append(q)
        valid_count += 1

        seed_map = dict(zip(seed_js.name, seed_js.position))
        for j_name, q_val in zip(UR_JOINTS, q):
            if j_name in seed_map:
                seed_map[j_name] = q_val
        seed_js.position = [seed_map[n] for n in seed_js.name]

    if not joint_path:
        node.get_logger().error("IK produced 0 valid points")
        return

    node.get_logger().info(
        f"IK produced {valid_count} valid joint points. Smoothing..."
    )

    joint_path_smoothed = limit_step(joint_path, max_delta=0.02)
    node.get_logger().info(
        f"After smoothing: {len(joint_path_smoothed)} joint trajectory points."
    )

    # Enable logging
    log_pub.publish(Bool(data=True))

    # Build a bridge from the live state to the first IK point to avoid path tolerance errors.
    live_q = latest_q(node) or joint_path_smoothed[0]
    bridge = limit_step([live_q, joint_path_smoothed[0]], max_delta=0.01)

    base_path = joint_path_smoothed

    # Loop continuously until Ctrl-C
    try:
        while rclpy.ok():
            # Rebuild bridge from current state each loop to avoid jumps
            live_q = latest_q(node) or base_path[0]
            bridge = limit_step([live_q, base_path[0]], max_delta=0.01)
            full_traj = bridge + base_path[1:] if len(bridge) > 1 else base_path

            finer = limit_step(full_traj, max_delta=0.006)
            node.get_logger().info(
                f"Sending IK joint path with {len(finer)} points via JointSender"
            )
            ok = sender.send(finer, dt=0.04)
            if not ok:
                node.get_logger().error("Controller reported error")
                break
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")

    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

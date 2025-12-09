import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState

from ur5e_ai.lemniscate import lemniscate
from ur5e_ai.dmp import DMP
from ur5e_ai.ik_client import IKClient
from ur5e_ai.joint_sender import JointSender
from ur5e_ai.eval_utils import rms
from ur5e_ai.utils import (
    UR_JOINTS,
    START_Q,
    quat_from_rpy,
    limit_step,
    latest_q,
)

from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Duration as RosDuration


def make_pose(node, position, quat):
    """Build a PoseStamped in the 'base_link' frame from a position and quaternion."""
    pose = PoseStamped()
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.header.frame_id = "base_link"
    pose.pose.position = Point(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
    )
    pose.pose.orientation.x = float(quat[0])
    pose.pose.orientation.y = float(quat[1])
    pose.pose.orientation.z = float(quat[2])
    pose.pose.orientation.w = float(quat[3])
    return pose


def publish_reference_marker(node, Y_ref):
    """
    Publish a latched Marker that visualizes the reference path in RViz.

    The marker is a simple line strip in 'base_link' frame.
    """
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    pub = node.create_publisher(Marker, "/dmp_reference_marker", qos)

    mk = Marker()
    mk.header.frame_id = "base_link"
    mk.header.stamp = node.get_clock().now().to_msg()
    mk.ns = "dmp_ref"
    mk.id = 0
    mk.type = Marker.LINE_STRIP
    mk.action = Marker.ADD
    mk.scale.x = 0.005
    mk.color.r = 0.0
    mk.color.g = 1.0
    mk.color.b = 0.0
    mk.color.a = 1.0

    for p in Y_ref:
        pt = Point()
        pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
        mk.points.append(pt)

    mk.lifetime = RosDuration(sec=0, nanosec=0)

    pub.publish(mk)
    node.get_logger().info(f"Published reference marker with {len(Y_ref)} points.")
    # Keep the publisher around so the marker remains latched.
    return pub


def main():
    rclpy.init()
    node = Node("ai_dmp_controller_node")

    sender = JointSender(controller="/joint_trajectory_controller")
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

    # Generate DMP reference in Cartesian space

    node.get_logger().info("Generating lemniscate path")
    T, Y, dY, pitch_profile, dt = lemniscate(
        T=10.0,
        dt=0.02,
        a=0.08,
        center=(0.30, 0.0, 0.5),
        close_loop=True,
        rock_z_amp=0.01,
        rock_pitch_amp_deg=8.0,
        rock_phase_mult=1.0,
    )

    dmp = DMP(n_dims=3, n_bfs=25, alpha_z=15.0, alpha_s=2.0)
    dmp.fit(T, Y, dY, dt)
    Y_ref = dmp.rollout(T, dt)

    # visual reference
    publish_reference_marker(node, Y_ref)

    # IK along the lemniscate

    ik = IKClient(group_name="ur_manipulator")
    qs = []

    seed_js = JointState()
    seed_js.name = UR_JOINTS
    seed_js.position = list(current_q)
    seed_state = seed_js

    # Move to the first pose of lemniscate so the loop starts smoothly
    first_quat = quat_from_rpy(np.pi, pitch_profile[0], 0.0)
    first_pose = make_pose(node, Y_ref[0], first_quat)
    start_js = ik.solve(first_pose, seed_state=seed_state)
    try:
        start_q = [dict(zip(start_js.name, start_js.position))[j] for j in UR_JOINTS]
    except Exception:
        start_q = None

    if start_q:
        pre_path = limit_step([current_q, start_q], max_delta=0.02)
        node.get_logger().info(
            f"Moving to lemniscate start with {len(pre_path)} steps..."
        )
        sender.send(pre_path, dt=0.02)
        current_q = start_q
        seed_state = start_js
        # start EE logging once we are at the lemniscate start
        log_pub.publish(Bool(data=True))
    else:
        node.get_logger().warn(
            "Could not solve IK for lemniscate start. Using current joints as seed."
        )

    fail_count = 0
    MAX_FAIL = 0  # abort if any point fails
    for i, p in enumerate(Y_ref):
        # Tilt the tool around x (pitch) to get a rocking y-z arc
        quat = quat_from_rpy(np.pi, pitch_profile[i], 0.0)
        pose = make_pose(node, p, quat)
        js = ik.solve(pose, seed_state=seed_state)
        if not js.name:
            fail_count += 1
            continue
        seed_state = js 

        m = dict(zip(js.name, js.position))
        try:
            q = [m[j] for j in UR_JOINTS]
        except KeyError:
            fail_count += 1
            continue
        qs.append(q)

    if fail_count > MAX_FAIL:
        node.get_logger().error(
            f"IK skipped {fail_count} points; aborting for determinism."
        )
        ik.destroy_node()
        sender.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info(f"IK produced {len(qs)} points with {fail_count} failures.")

    qs = limit_step(qs, max_delta=0.01)
    node.get_logger().info(f"After limiting, sending {len(qs)} points per loop.")

    # send trajectory

    log_pub.publish(Bool(data=True))

    dt_send = 0.02
    err = rms(Y_ref, Y)
    node.get_logger().info(f"DMP RMS to demo (cartesian): {err:.4f} m")
    node.get_logger().info("Sending DMP trajectory in a loop. Ctrl-C to stop.")
    try:
        while rclpy.ok():
            live_q = latest_q(node, timeout=0.5) or current_q
            # create a small bridge between current joint state and the first point in qs
            bridge = limit_step([live_q, qs[0]], max_delta=0.01)
            traj = bridge + qs
            ok = sender.send(traj, dt_send)
            if not ok:
                node.get_logger().error("Controller reported an error; stopping.")
                break
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")

    ik.destroy_node()
    sender.destroy_node()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

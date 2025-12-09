# ur5e_ai/utils.py

import time
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# Common robot conventions
# ---------------------------------------------------------------------------

# Common UR5e joint ordering used across controllers
UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Stable, repeatable tool-down seed pose (elbow-forward, wrist straight).
START_Q = [
    0.0,    # shoulder_pan
    -1.57,  # shoulder_lift
    -1.57,  # elbow
    -1.57,  # wrist_1
    0.0,    # wrist_2
    0.0,    # wrist_3
]

# ---------------------------------------------------------------------------
# Math / trajectory helpers
# ---------------------------------------------------------------------------

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Roll–pitch–yaw (rad) -> quaternion (x, y, z, w)."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return float(x), float(y), float(z), float(w)


def limit_step(
    qs: Sequence[Sequence[float]],
    max_delta: float = 0.02
) -> List[List[float]]:
    """Limit per-step joint delta to <= max_delta [rad].

    Takes a list of joint configurations (each an iterable of length 6) and
    inserts intermediate points so that the largest absolute joint
    difference between successive waypoints does not exceed max_delta.
    """
    if len(qs) < 2:
        return [list(q) for q in qs]

    out: List[np.ndarray] = [np.array(qs[0], dtype=float)]
    for i in range(1, len(qs)):
        q_prev = out[-1]
        q_next = np.array(qs[i], dtype=float)
        dq = q_next - q_prev
        max_step = float(np.max(np.abs(dq)))
        if max_step <= max_delta:
            out.append(q_next)
            continue

        n = int(np.ceil(max_step / max_delta))
        n = max(1, n)
        for k in range(1, n + 1):
            out.append(q_prev + dq * (k / n))

    return [q.tolist() for q in out]

# ---------------------------------------------------------------------------
# Joint state helpers
# ---------------------------------------------------------------------------

def wait_for_joint_states(
    node,
    timeout: float = 5.0,
    topic: str = "/joint_states",
) -> Optional[JointState]:
    """Block until a JointState arrives on `topic` or timeout (seconds) elapses."""
    latest = {"msg": None}

    def cb(msg: JointState):
        latest["msg"] = msg

    sub = node.create_subscription(JointState, topic, cb, 10)

    start = time.time()
    try:
        while (
            rclpy.ok()
            and (time.time() - start) < timeout
            and latest["msg"] is None
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)

    return latest["msg"]


def latest_q(
    node,
    timeout: float = 1.0,
    topic: str = "/joint_states",
) -> Optional[List[float]]:
    """Return latest joint vector mapped to UR_JOINTS, or None on timeout/error."""
    js = wait_for_joint_states(node, timeout=timeout, topic=topic)
    if js is None:
        return None

    m = dict(zip(js.name, js.position))
    try:
        return [float(m[j]) for j in UR_JOINTS]
    except KeyError:
        # One or more joints missing from the message
        return None

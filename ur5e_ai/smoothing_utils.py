import numpy as np

def smooth_trajectory(qs, dt, max_vel=0.5):
    """
    Limit joint-to-joint step size so velocity ≤ max_vel [rad/s].
    Returns a new, time-consistent trajectory.
    """
    if len(qs) < 2:
        return qs

    qs = np.array(qs)
    smoothed = [qs[0]]
    t = [0.0]
    for i in range(1, len(qs)):
        dq = qs[i] - smoothed[-1]
        step_time = np.max(np.abs(dq) / max_vel)
        n_steps = max(int(np.ceil(step_time / dt)), 1)
        # Linear interpolation in n_steps sub-segments
        for j in range(1, n_steps + 1):
            smoothed.append(smoothed[-1] + dq / n_steps)
            t.append(t[-1] + dt)
    return np.array(smoothed), np.array(t)

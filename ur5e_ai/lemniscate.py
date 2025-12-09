import numpy as np

def lemniscate(T=10.0, dt=0.02, a=0.15, center=(0.4, 0.0, 0.3), close_loop=False,
               rock_z_amp=0.01, rock_pitch_amp_deg=8.0, rock_phase_mult=2.0):
    """
    Reference for a lemniscate (∞) in xy with optional rocking:
    - rock_z_amp: adds a small z oscillation synced with the figure-8 lobes.
    - rock_pitch_amp_deg: tilts the tool (pitch) to mimic rotary motion about y-z plane.
    Returns: t, pos [N,3], vel [N,3], pitch_profile [N], dt
    """
    # Include the final point at t=T when close_loop=True so start/end coincide.
    t_end = T + (dt if close_loop else 0.0)
    t = np.arange(0, t_end, dt)
    w = 2*np.pi/T

    phase = w * t
    x = center[0] + a * np.sin(phase)
    y = center[1] + a * np.sin(phase) * np.cos(phase)

    z = center[2] + rock_z_amp * np.sin(rock_phase_mult * phase)

    pos = np.stack([x, y, z], axis=1)
    vel = np.gradient(pos, dt, axis=0)

    pitch_profile = np.deg2rad(rock_pitch_amp_deg) * np.sin(rock_phase_mult * phase)

    return t, pos, vel, pitch_profile, dt

import numpy as np

class RhythmicDMP:
    def __init__(self, n_dims=3, n_bfs=30, alpha_z=25.0, beta_z=None, lam=1e-4):
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.alpha_z = alpha_z
        self.beta_z = beta_z if beta_z is not None else alpha_z / 4.0
        self.lam = lam

        # Basis centers evenly over [0, 2π)
        self.centers = np.linspace(0, 2*np.pi, self.n_bfs, endpoint=False)
        # Widths (you can tweak the 1.5 factor if needed)
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5

        self.w = np.zeros((n_dims, n_bfs))
        self.y_center = None   # center of oscillation
        self.omega = None      # angular frequency

    def _psi(self, phi):
        """
        Rhythmic basis functions on the circle.
        phi: scalar or array of shape [N]
        """
        # shape handling: want array of shape [N, n_bfs]
        phi = np.atleast_1d(phi)[:, None]  # [N, 1]
        c = self.centers[None, :]          # [1, n_bfs]
        # classic circular Gaussian: exp(h * (cos(phi - c) - 1))
        return np.exp(self.h * (np.cos(phi - c) - 1.0))

    def fit(self, T, Y, dY, dt):
        """
        Fit rhythmic DMP weights to a single-period demo.
        T, Y, dY as before (Y: [N, n_dims], dY: [N, n_dims]).
        """
        t_vec = np.array(T) - float(T[0])
        T_total = max(t_vec[-1], dt)

        # Angular frequency to cover one full period in T_total
        self.omega = 2.0 * np.pi / T_total
        phi = self.omega * t_vec   # [N], 0→2π over the demo

        # Center of oscillation: mean position of the demo
        self.y_center = Y.mean(axis=0).copy()

        # numerical acceleration
        ddY = np.gradient(dY, dt, axis=0)

        # target forcing term: whatever is left after subtracting spring-damper
        # Note: spring pulls towards y_center, not a "goal" at the end.
        f_target = ddY - (self.alpha_z * (self.beta_z * (self.y_center - Y) - dY))

        # Basis matrix
        Psi = self._psi(phi)              # [N, n_bfs]
        denom = Psi.sum(axis=1, keepdims=True) + 1e-9
        Phi = Psi / denom                 # normalized basis activations

        # Ridge regression per dimension
        A = Phi.T @ Phi + self.lam * np.eye(self.n_bfs)
        for d in range(self.n_dims):
            b = Phi.T @ f_target[:, d]
            self.w[d] = np.linalg.solve(A, b)

    def rollout(self, T, dt, y0=None):
        """
        Roll out one period (or more) using the learned rhythmic DMP.
        T defines how long to run (like before). If T spans multiple periods,
        you’ll get repeated cycles.
        """
        if self.omega is None:
            raise RuntimeError("Call fit() before rollout().")

        # initial phase and state
        phi = 0.0
        if y0 is None:
            y = self.y_center.copy()
        else:
            y = np.array(y0, dtype=float).copy()

        dy = np.zeros_like(y)
        Y_out = []

        for _ in T:
            # basis + forcing
            psi = self._psi(phi)[0]           # [n_bfs]
            f = (psi @ self.w.T) / (psi.sum() + 1e-9)  # [n_dims]

            # rhythmic dynamics: spring to y_center + periodic forcing
            ddy = self.alpha_z * (self.beta_z * (self.y_center - y) - dy) + f
            dy += ddy * dt
            y  += dy * dt
            Y_out.append(y.copy())

            # advance phase (wrap into [0, 2π) for numerical sanity)
            phi += self.omega * dt
            if phi > 2*np.pi:
                phi -= 2*np.pi

        return np.array(Y_out)

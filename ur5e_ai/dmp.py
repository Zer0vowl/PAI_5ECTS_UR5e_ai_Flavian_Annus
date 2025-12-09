import numpy as np

class DMP:
    def __init__(self, n_dims=3, n_bfs=30, alpha_z=25.0, beta_z=None, alpha_s=4.0):
        self.n_dims = n_dims
        self.n_bfs = n_bfs
        self.alpha_z = alpha_z
        self.beta_z = beta_z if beta_z is not None else alpha_z/4.0
        self.alpha_s = alpha_s
        c = np.exp(-self.alpha_s * np.linspace(0,1,self.n_bfs))
        self.centers = c
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / c / self.alpha_s
        self.w = np.zeros((n_dims, n_bfs))
        self.y0 = None
        self.g = None

    def _psi(self, s):
        return np.exp(-self.h * (s - self.centers)**2)

    def fit(self, T, Y, dY, dt):
        t_vec = np.array(T) - float(T[0])
        T_total = max(t_vec[-1], dt)
        s = np.exp(-self.alpha_s * (t_vec / T_total))
        self.y0 = Y[0].copy()
        self.g = Y[-1].copy()
        # numerical acceleration
        ddY = np.gradient(dY, dt, axis=0)
        # full forcing term to match second-order dynamics
        f_target = ddY - (self.alpha_z*(self.beta_z*(self.g - Y) - dY))

        Psi = self._psi(s[:,None])
        denom = Psi.sum(axis=1, keepdims=True) + 1e-9
        Phi = (Psi/denom) * s[:,None]

        lam = 1e-6
        A = Phi.T @ Phi + lam*np.eye(self.n_bfs)
        for d in range(self.n_dims):
            b = Phi.T @ f_target[:,d]
            self.w[d] = np.linalg.solve(A, b)

    def rollout(self, T, dt):
        s = 1.0
        y = self.y0.copy()
        dy = np.zeros_like(y)
        Y = []
        for _ in T:
            # forcing
            psi = self._psi(s)
            f = (psi @ self.w.T) / (psi.sum()+1e-9) * s
            # 2nd-order system
            ddy = self.alpha_z*(self.beta_z*(self.g - y) - dy) + f
            dy += ddy*dt
            y  += dy*dt
            Y.append(y.copy())
            s += (-self.alpha_s*s)*dt
        return np.array(Y)

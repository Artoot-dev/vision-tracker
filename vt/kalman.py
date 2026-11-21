"""Constant-velocity Kalman filter for target state estimation.

dim=3 -> state [x, y, z, vx, vy, vz]^T  (the 6-state model in the spec)
dim=2 -> state [x, y, vx, vy]^T         (used when no depth measurement exists)

Discrete white-acceleration process model, so the process noise scales with dt
and the filter behaves correctly through frame-rate jitter and dropped frames.
"""
from __future__ import annotations

import numpy as np


class KalmanCV:
    def __init__(self, dim=3, process_noise=40.0, meas_noise=3.0,
                 pos=None, vel=None, init_pos_var=400.0, init_vel_var=2000.0):
        self.dim = int(dim)
        self.n = 2 * self.dim
        self.q = float(process_noise)
        self.r = float(meas_noise)
        self.x = np.zeros(self.n, dtype=np.float64)
        if pos is not None:
            self.x[:self.dim] = np.asarray(pos, dtype=np.float64)[:self.dim]
        if vel is not None:
            self.x[self.dim:] = np.asarray(vel, dtype=np.float64)[:self.dim]
        self.P = np.eye(self.n, dtype=np.float64)
        self.P[:self.dim, :self.dim] *= init_pos_var
        self.P[self.dim:, self.dim:] *= init_vel_var
        self._dt = 1.0 / 30.0

    # ---------------------------------------------------------------- predict
    def predict(self, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = self._dt
        # Cap pathological dt (video stalls, timestamp jumps) so Q cannot explode.
        dt = min(dt, 1.0)
        self._dt = dt
        d, n = self.dim, self.n
        F = np.eye(n, dtype=np.float64)
        for i in range(d):
            F[i, d + i] = dt
        Q = np.zeros((n, n), dtype=np.float64)
        for i in range(d):
            Q[i, i] = self.q * dt ** 4 / 4.0
            Q[i, d + i] = Q[d + i, i] = self.q * dt ** 3 / 2.0
            Q[d + i, d + i] = self.q * dt ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.position()

    # ----------------------------------------------------------------- update
    def update(self, z, dims=None):
        """Fuse a measurement. `dims` selects which state components are measured
        (e.g. [0, 1] when only 2D image position is available)."""
        z = np.asarray(z, dtype=np.float64).ravel()
        if dims is None:
            dims = list(range(len(z)))
        dims = list(dims)[:len(z)]
        m = len(dims)
        H = np.zeros((m, self.n), dtype=np.float64)
        for row, i in enumerate(dims):
            H[row, i] = 1.0
        y = z[:m] - H @ self.x
        S = H @ self.P @ H.T + np.eye(m) * self.r ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P
        return self.position()

    # ---------------------------------------------------------------- getters
    def position(self):
        return self.x[:self.dim].copy()

    def velocity(self):
        return self.x[self.dim:].copy()

    def speed_px_s(self):
        return float(np.linalg.norm(self.x[self.dim:self.dim + 2]))

    def mahalanobis(self, z, dims=(0, 1)):
        """Gate distance in position space (sigma units). Used to reject
        detections that are kinematically impossible for this track."""
        z = np.asarray(z, dtype=np.float64).ravel()
        idx = [i for i in dims if i < self.dim]
        if not idx:
            return 0.0
        d = np.array([z[i] for i in range(len(idx))], dtype=np.float64)
        mu = np.array([self.x[i] for i in idx], dtype=np.float64)
        P = self.P[np.ix_(idx, idx)] + np.eye(len(idx)) * self.r ** 2
        diff = d - mu
        try:
            return float(np.sqrt(diff @ np.linalg.solve(P, diff)))
        except np.linalg.LinAlgError:
            return 0.0

    def reset_position(self, pos):
        pos = np.asarray(pos, dtype=np.float64)[:self.dim]
        self.x[:self.dim] = pos
        self.P[:self.dim, :self.dim] = np.eye(self.dim) * 100.0
        return self.position()

    def scale_uncertainty(self, factor):
        self.P[:self.dim, :self.dim] *= float(factor)
        return self.P

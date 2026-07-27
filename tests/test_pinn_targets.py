"""Target consistency: SOH = 100*Q/Q_0, deltas match next_minus_previous."""
from __future__ import annotations

import numpy as np


def test_capacity_soh_consistency() -> None:
    q0 = np.array([1.2, 1.15, 1.1])
    u_hat = np.array([0.95, 0.92, 0.88])
    q_hat = q0 * u_hat
    soh_hat = 100.0 * u_hat
    np.testing.assert_allclose(soh_hat, 100.0 * q_hat / q0)


def test_delta_consistency_next_minus_previous() -> None:
    q0 = np.array([1.2, 1.2])
    q_current = np.array([1.1, 1.05])
    u_hat = np.array([0.95, 0.90])
    q_next = q0 * u_hat
    soh_current = 100.0 * q_current / q0
    soh_next = 100.0 * u_hat
    delta_q = q_next - q_current
    delta_soh = soh_next - soh_current
    np.testing.assert_allclose(delta_q, q_next - q_current)
    np.testing.assert_allclose(delta_soh, soh_next - soh_current)

import numpy as np
from model import score, bce_loss
from data import load_data, load_data_3sensor, estimate_steady_state, estimate_steady_state_3sensor, estimate_c_I_coastdown
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.optimize import minimize
import os


P_SCALE = float(input("P_SCALE (must match generate_data.py, default 1.0): ") or 1.0)
P = 1479 * 1.05 * P_SCALE


def _compute_phi(a_I, c_I, t, p, p_1, t_n):
    """Compute theta(t) - theta_0 for omega_0=0 (the theta_0-independent part)."""
    I_c = 1.0 / c_I
    I_c_2 = I_c ** 2
    phi = np.empty(len(t))
    _t = t[:p_1]
    exp = np.exp(-c_I * _t)
    phi[:p_1] = a_I * p * (I_c * _t + I_c_2 * exp - I_c_2)
    _t = t[p_1:]
    exp = np.exp(-c_I * _t)
    exp1 = np.exp(-c_I * (_t - t_n))
    phi[p_1:] = (a_I * p * (I_c * _t + I_c_2 * exp - I_c_2)
                 - a_I * p * (I_c * (_t - t_n) + I_c_2 * exp1 - I_c_2))
    return phi


def _grid2d_chunk_worker(args):
    """Process-level worker: handles a chunk of omega_ss values with threads for theta0.

    For each omega_ss in the chunk: phi(t) is computed once, then all theta_0
    values are evaluated simultaneously via numpy broadcast (no inner loop).
    ThreadPoolExecutor (no-GIL Python 3.14) runs omega_ss values in parallel,
    sharing sensor_data/t_s in memory with zero copy overhead.
    """
    omega_chunk, c_I, P_val, k, theta0_arr, sensor_data, t_s, P_1, t_n = args
    half_pi = np.pi / 2

    def _eval_one_omega(w):
        a_I_w = w * c_I / P_val
        phi = _compute_phi(a_I_w, c_I, t_s, P_val, P_1, t_n)
        theta = theta0_arr[:, None] + phi[None, :]       # (N_theta, N_t)
        theta_mod = theta % half_pi
        d = np.minimum(theta_mod, half_pi - theta_mod)
        scores = np.mean((d < k).astype(np.float32) - sensor_data[None, :], axis=1) ** 2
        best_idx = np.argmin(scores)
        return float(scores[best_idx]), theta0_arr[best_idx], w

    with ThreadPoolExecutor() as tex:
        results = list(tex.map(_eval_one_omega, omega_chunk))

    return min(results, key=lambda r: r[0])


def main_analytical():
    """
    Analytically determine all parameters except theta_0, then grid-search theta_0.

    Parameter sources:
      c_I      <- log-linear fit to coast-down transition intervals
      omega_ss <- single-sensor (more accurate than 3-sensor above quantization floor)
      a_I      <- derived: a_I = omega_ss * c_I / P
      k        <- single-sensor duty cycle in steady state
      omega_0  <- fixed = 0 (motor at rest before PWM)
      theta_0  <- 3-phase grid search (1D coarse → 2D Nelder-Mead → 2D fine grid)
    """
    t_3, state_data = load_data_3sensor()
    t_s, sensor_data = load_data()
    dt = np.mean(np.diff(t_s))
    P_1 = int(10 / dt)
    t_n = 10.0

    omega_ss_single, k = estimate_steady_state(sensor_data, t_s)
    omega_ss_3sensor   = estimate_steady_state_3sensor(state_data, t_3)
    omega_ss = omega_ss_single
    print(f"omega_ss (single) = {omega_ss_single:.4f} rad/s")
    print(f"omega_ss (3sensor)= {omega_ss_3sensor:.4f} rad/s  (quantization floor = {np.pi/(6*dt):.1f})")

    c_I = estimate_c_I_coastdown(sensor_data, t_s, t_n=t_n)
    a_I = omega_ss * c_I / P

    print(f"c_I      = {c_I:.4f}  (tau = {1/c_I:.3f} s)")
    print(f"a_I      = {a_I:.4f}  (derived)")
    print(f"k        = {k:.4f} rad")
    print(f"omega_0  = 0.0 (fixed)")

    # Phase 1: 1D grid over theta_0 in [0, pi/2)
    theta0_grid = np.linspace(0, np.pi / 2, 200, endpoint=False)
    best_sc, best_theta0 = np.inf, 0.0
    for th0 in theta0_grid:
        sc = score([a_I, c_I, 0.0, th0, k], sensor_data, t_s, P, P_1, t_n)
        if sc < best_sc:
            best_sc, best_theta0 = sc, th0
    print(f"\n[1D grid] theta_0={best_theta0:.4f}  score={best_sc:.6f}")

    # Phase 2: 2D Nelder-Mead over (a_I, theta_0)
    def loss_2d(x):
        return bce_loss([x[0], c_I, 0.0, x[1], k], sensor_data, t_s, P, P_1, t_n)

    res = minimize(loss_2d, x0=[a_I, best_theta0], method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 10000})
    a_I_fine, th0_fine = res.x
    sc_fine = score([a_I_fine, c_I, 0.0, th0_fine, k], sensor_data, t_s, P, P_1, t_n)
    print(f"[2D fine]  a_I={a_I_fine:.4f}  theta_0={th0_fine:.4f}  score={sc_fine:.6f}")

    # Phase 3: coarse-to-fine 2D grid over (omega_ss, theta_0)
    omega_ss_2d = a_I_fine * P / c_I
    n_proc = os.cpu_count()

    def _run_grid(omega_grid, theta0_grid):
        chunks = np.array_split(omega_grid, n_proc)
        chunk_args = [(chunk, c_I, P, k, theta0_grid, sensor_data, t_s, P_1, t_n)
                      for chunk in chunks]
        with ProcessPoolExecutor(max_workers=n_proc, mp_context=get_context('fork')) as pex:
            results = list(pex.map(_grid2d_chunk_worker, chunk_args))
        return min(results, key=lambda r: r[0])

    # Pass A: coarse 100x25, covers full [0, pi/2) phase range
    sc_c, th_c, w_c = _run_grid(
        np.linspace(omega_ss_2d - 4, omega_ss_2d + 4, 100),
        np.linspace(0, np.pi / 2, 25, endpoint=False)
    )
    print(f"\n[grid2D coarse]  omega_ss={w_c:.4f}  theta_0={th_c:.4f}  score={sc_c:.6f}")

    # Pass B: fine step=1e-3, theta_0 within +/-2k of coarse best
    FINE_STEP = 1e-2
    omega_fine  = np.arange(w_c - 0.5, w_c + 0.5, FINE_STEP)
    theta0_fine = np.arange(th_c - 2*k, th_c + 2*k, FINE_STEP)
    print(f"[grid2D fine]    {len(omega_fine)}x{len(theta0_fine)} = {len(omega_fine)*len(theta0_fine):,} evals ...")
    sc_f, th_f, w_f = _run_grid(omega_fine, theta0_fine)
    print(f"[grid2D fine]    omega_ss={w_f:.4f}  theta_0={th_f:.4f}  score={sc_f:.6f}")

    a_I_best = w_f * c_I / P
    best_params = np.array([a_I_best, c_I, 0.0, th_f, k])
    bce_best = bce_loss(list(best_params), sensor_data, t_s, P, P_1, t_n)
    print(f"\nRecovered: a_I={a_I_best:.4f}, c_I={c_I:.4f}, omega_0=0.0, theta_0={th_f:.4f}, k={k:.4f}")
    print(f"Implied omega_ss = {a_I_best*P/c_I:.2f} rad/s  (measured {omega_ss:.2f})")
    print(f"score={sc_f:.6f}  BCE={bce_best:.6f}")
    np.savetxt("parameter.txt", best_params)


if __name__ == "__main__":
    main_analytical()

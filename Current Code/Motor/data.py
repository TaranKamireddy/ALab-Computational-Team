import numpy as np

def load_data():
    data = np.loadtxt(
        "hall_log.txt",  
        dtype=float,
        delimiter=","        
    )
    
    t = data[:, 0]

    t = t / 1e6
    t = t - t[0]
    w = data[:, 1]
    index = t >= 2.26
    t = t[index]
    w = w[index]
    t = t - t[0]   # re-zero to PWM onset (first sample after stripping pre-PWM)
    index = t <= 11 + 3
    return t[index], w[index]

def time_stamp(array, t):
    time = []
    prev = 1 if array[0] > 0.5 else 0
    if prev == 1:
        time.append(t[0])
    for index in range(1, len(array)):
        current = 1 if array[index] > 0.5 else 0
        if prev != current:
            time.append(t[index])
            prev = current
    return np.array(time)

def load_data_3sensor(filename="hall_log_3sensor.txt"):
    data = np.loadtxt(filename, dtype=float, delimiter=",")
    t = data[:, 0] / 1e6
    t = t - t[0]
    w = data[:, 1].astype(int)
    index = t >= 2.26
    t, w = t[index], w[index]
    t = t - t[0]   # re-zero to PWM onset
    index = t <= 11 + 3
    return t[index], w[index]

def estimate_steady_state_3sensor(state_data, t, t_start=3.0, t_end=9.0):
    """
    Estimate omega_ss from 3-sensor state changes.
    Each state change = pi/6 rad mechanical -> omega_ss = pi/(6 * mean_interval).
    """
    mask = (t >= t_start) & (t <= t_end)
    t_ss = t[mask]
    s_ss = state_data[mask]
    change_idx = np.where(np.diff(s_ss) != 0)[0]
    change_times = t_ss[change_idx]
    omega_ss = np.pi / (6 * np.mean(np.diff(change_times)))
    return omega_ss

def estimate_steady_state(sensor_data, t, t_start=3.0, t_end=9.0):
    """
    Estimate omega_ss and k from the steady-state window [t_start, t_end].
    Assumes motor has settled by t_start and PWM is still on at t_end.

    omega_ss: average angle per transition = pi/4 rad  ->  omega_ss = pi / (4 * mean_interval)
    k:        duty cycle = 2k / (pi/2)  ->  k = duty_cycle * pi / 4
    """
    transitions = time_stamp(sensor_data, t)
    ss = transitions[(transitions >= t_start) & (transitions <= t_end)]
    omega_ss = np.pi / (4 * np.mean(np.diff(ss)))

    mask = (t >= t_start) & (t <= t_end)
    k = np.mean(sensor_data[mask]) * np.pi / 4

    return omega_ss, k

def estimate_c_I_coastdown(sensor_data, t, t_n=10.0, omega_ss=None):
    """
    Estimate c_I from the coast-down phase (t > t_n) using a log-linear fit.

    During coast-down: omega(t) = omega_ss * exp(-c_I * (t - t_n))
    Each Hall transition interval: Delta_tau ≈ (pi/4) / omega(t_mid)
    => log(omega) = log(omega_ss) - c_I * (t - t_n)
    => log(Delta_tau) = -log(omega_ss * 4/pi) + c_I * (t - t_n)

    A linear fit of log(Delta_tau) vs (t - t_n) gives slope = c_I.
    """
    transitions = time_stamp(sensor_data, t)
    cd = transitions[transitions > t_n]
    if len(cd) < 4:
        raise ValueError("Too few coast-down transitions to fit c_I.")
    intervals = np.diff(cd)
    t_mid = 0.5 * (cd[:-1] + cd[1:]) - t_n   # time since PWM off
    log_dt = np.log(intervals)
    # linear fit: log_dt = c_I * t_mid + const
    coeffs = np.polyfit(t_mid, log_dt, 1)
    c_I = coeffs[0]
    return c_I

def main():
    return 0
    
if __name__ == "__main__":
    main()
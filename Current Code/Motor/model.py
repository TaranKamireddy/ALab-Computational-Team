import numpy as np
from scipy.special import expit


def bce_loss(intial_guess, sensor_data, t, p, p_1, t_n, sharpness=None):
    a_I, c_I, omega_0, theta_0, k = intial_guess
    I_c = 1 / c_I
    I_c_2 = I_c ** 2
    length = len(sensor_data)
    theta = np.zeros(length)
    _t = t[:p_1]
    exp = np.exp(-c_I * _t)
    theta[:p_1] = theta_0 + omega_0 * I_c * (1 - exp) + a_I * p * (I_c * _t + I_c_2*exp - I_c_2)
    _t = t[p_1:]
    exp = np.exp(-c_I * _t)
    exp_1 = np.exp(-c_I * (_t - t_n))
    theta[p_1:] = theta_0 + omega_0 * I_c * (1 - exp) + a_I * p * (I_c * _t + I_c_2*exp - I_c_2) - a_I * p * (I_c * (_t - t_n) + I_c_2 * exp_1 - I_c_2)

    if sharpness is None:
        sharpness = 1.0 / k

    theta_mod = theta % (np.pi / 2)
    d = np.minimum(theta_mod, np.pi / 2 - theta_mod)
    prob = expit(sharpness * (k - d))
    prob = np.clip(prob, 1e-7, 1 - 1e-7)

    y = sensor_data
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))


def score(intial_guess, sensor_data, t, p, p_1, t_n):
    a_I, c_I, omega_0, theta_0, k = intial_guess
    I_c = 1 / c_I
    I_c_2 = I_c ** 2
    length = len(sensor_data)
    theta = np.zeros(length)
    _t = t[:p_1]
    exp = np.exp(-c_I * _t)
    theta[:p_1] = theta_0 + omega_0 * I_c * (1 - exp) + a_I * p * (I_c * _t + I_c_2*exp - I_c_2)
    _t = t[p_1:]
    exp = np.exp(-c_I * _t)
    exp_1 = np.exp(-c_I * (_t - t_n))
    theta[p_1:] = theta_0 + omega_0 * I_c * (1 - exp) + a_I * p * (I_c * _t + I_c_2*exp - I_c_2) - a_I * p * (I_c * (_t - t_n) + I_c_2 * exp_1 - I_c_2)

    theta_mod = theta % (np.pi / 2)
    d = np.minimum(theta_mod, np.pi / 2 - theta_mod)
    y_pred = (d < k).astype(np.float64)

    return np.mean((y_pred - sensor_data) ** 2)

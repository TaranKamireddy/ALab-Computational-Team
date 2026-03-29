import numpy as np

# True parameters to recover
TRUE_A_I    = 0.755   # alpha/I
TRUE_C_I    = 2.0     # C/I  (tau = 0.5s)
TRUE_OMEGA0 = 0.0     # initial angular velocity (starts from rest)
TRUE_THETA0 = 1.0     # initial angle
TRUE_K      = 0.341   # hall sensor window half-width (gives ~43% duty at steady state)

P_SCALE = 0.46           # set to ~0.46 for low-speed calibration (~272 rad/s)
P       = 1479 * 1.05 * P_SCALE    # PWM amplitude
T_PRE   = 2.26          # pre-PWM rest period (s) — stripped by data.py, aligns t=0 with PWM onset
T_N     = 10.0          # PWM on duration (s) — same as t_n in train.py
DT      = 0.000962       # sampling interval matching Arduino at 115200 baud
T_TOTAL = T_PRE + T_N + 4.0  # pre + PWM on + coast-down; data.py keeps [2.26, 14]

print(f"True params: a_I={TRUE_A_I}, c_I={TRUE_C_I}, omega_0={TRUE_OMEGA0}, "
      f"theta_0={TRUE_THETA0}, k={TRUE_K}")
print(f"omega_ss = {TRUE_A_I*P/TRUE_C_I:.1f} rad/s = "
      f"{TRUE_A_I*P/TRUE_C_I*60/(2*np.pi):.0f} RPM")

N = int(T_TOTAL / DT)
t = np.arange(N) * DT

# --- compute theta(t): motor at rest during pre-PWM, then analytical solution ---
# PWM turns on at physical t=T_PRE; after data.py strip, re-zeroed t=0 = PWM onset
I_c   = 1.0 / TRUE_C_I
I_c_2 = I_c ** 2

pre   = int(T_PRE / DT)   # samples before PWM (at rest, theta=constant)
p_1   = pre + int(T_N / DT)  # sample index where PWM turns off

theta = np.full(N, TRUE_THETA0)  # all pre-PWM samples stay at TRUE_THETA0

# PWM on: t_rel = t - T_PRE (time since PWM onset)
t_rel = t[pre:p_1] - T_PRE
exp   = np.exp(-TRUE_C_I * t_rel)
theta[pre:p_1] = (TRUE_THETA0
                  + TRUE_OMEGA0 * I_c * (1 - exp)
                  + TRUE_A_I * P * (I_c * t_rel + I_c_2 * exp - I_c_2))

# PWM off (coast-down): t_rel continues, subtract PWM-off term
t_rel  = t[p_1:] - T_PRE
exp    = np.exp(-TRUE_C_I * t_rel)
exp1   = np.exp(-TRUE_C_I * (t_rel - T_N))
theta[p_1:] = (TRUE_THETA0
               + TRUE_OMEGA0 * I_c * (1 - exp)
               + TRUE_A_I * P * (I_c * t_rel + I_c_2 * exp - I_c_2)
               - TRUE_A_I * P * (I_c * (t_rel - T_N) + I_c_2 * exp1 - I_c_2))

# --- single Hall sensor (original, pole-tracking) ---
hall = np.zeros(N, dtype=int)
pole = 1
for i in range(N):
    o = theta[i]
    while o >= pole * (np.pi / 2) + TRUE_K:
        pole += 1
    if o >= pole * (np.pi / 2) - TRUE_K:
        hall[i] = 1

# --- 3-sensor Hall state for 4-pole motor (2 pole pairs) ---
# Electrical angle = 2 * mechanical angle
# Each sensor is high for half an electrical revolution (pi rad electrical = pi/2 mech)
# Sensors A, B, C placed at 0, 120, 240 degrees electrical (0, 60, 120 deg mechanical)
# State = (A << 2) | (B << 1) | C  gives values in {1,2,3,4,5,6}, never 0 or 7
theta_elec = 2 * theta
A = ((theta_elec)              % (2*np.pi)) < np.pi
B = ((theta_elec - 2*np.pi/3)  % (2*np.pi)) < np.pi
C = ((theta_elec - 4*np.pi/3)  % (2*np.pi)) < np.pi
hall3 = (A.astype(int) << 2) | (B.astype(int) << 1) | C.astype(int)
# Each state transition = pi/6 rad mechanical (30 deg); 12 transitions per revolution

# --- write single-sensor file ---
us = (t * 1e6).astype(np.int64)
with open("hall_log.txt", "w") as f:
    for i in range(N):
        f.write(f"{us[i]},{hall[i]}\n")

# --- write 3-sensor file ---
with open("hall_log_3sensor.txt", "w") as f:
    for i in range(N):
        f.write(f"{us[i]},{hall3[i]}\n")

# --- verify ---
filt = (t >= 2.26) & (t <= 14.0)
t_f = t[filt]
h_f  = hall[filt]
h3_f = hall3[filt]
print(f"\nGenerated {N} samples -> after data.py filter: {filt.sum()} samples")
print(f"Filtered t: {t_f[0]:.3f}s to {t_f[-1]:.3f}s")
print(f"Single sensor - fraction of 1s: {h_f.mean():.3f}  (expected {4*TRUE_K/np.pi:.3f})")
print(f"3-sensor      - unique states: {np.unique(h3_f)}  (expected 1-6 only)")
transitions_3 = np.sum(np.diff(h3_f) != 0)
print(f"3-sensor      - state changes in filtered data: {transitions_3}"
      f"  (~{transitions_3 / (t_f[-1]-t_f[0]):.0f}/s)")
print(f"P_1 = int({T_N}/dt) = {int(T_N/DT)}")
omega_ss_true = TRUE_A_I * P / TRUE_C_I
quant_floor_3s = np.pi / (6 * DT)
quant_floor_1s = np.pi / (2 * DT * 2)   # single-sensor: interval = pi/4 / omega => need pi/(4*DT)
print(f"\n--- Sampling limits ---")
print(f"omega_ss (true)          = {omega_ss_true:.1f} rad/s")
print(f"3-sensor quantization floor = pi/(6*DT) = {quant_floor_3s:.1f} rad/s")
print(f"  -> 3-sensor accurate?  {'YES' if omega_ss_true < quant_floor_3s else 'NO (floor exceeded)'}")
print(f"Single-sensor interval   = pi/(4*omega_ss) = {np.pi/(4*omega_ss_true)*1e3:.3f} ms  (DT={DT*1e3:.3f} ms)")
print(f"  -> single-sensor accurate? {'YES' if np.pi/(4*omega_ss_true) > DT else 'NO (pole-skipping)'}")
print(f"\nFor reliable 3-sensor omega_ss estimation, need P_SCALE <= {quant_floor_3s / (TRUE_A_I*1479*1.05/TRUE_C_I):.3f}")
print("\nSaved hall_log.txt (single) and hall_log_3sensor.txt (3-sensor)")

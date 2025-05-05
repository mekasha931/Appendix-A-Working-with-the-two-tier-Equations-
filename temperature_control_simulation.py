
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def temperature_model(y, t, k1, k2, T_env, N):
    T = y[:N]
    T_avg = y[N]
    dTdt = -k1 * (T - T_avg)
    dTavg_dt = -k2 * (T_avg - T_env) + 0.1 * np.sum(T - T_avg)
    return np.concatenate([dTdt, [dTavg_dt]])

N = 5
T0 = np.random.uniform(15, 25, size=N)
T_avg0 = np.mean(T0)
y0 = np.concatenate([T0, [T_avg0]])

t = np.linspace(0, 50, 300)
k1, k2 = 0.3, 0.1
T_env = 10

sol = odeint(temperature_model, y0, t, args=(k1, k2, T_env, N))

for i in range(N):
    plt.plot(t, sol[:, i], label=f"T_room_{i+1}")
plt.plot(t, sol[:, N], 'k--', label="Average Temp", linewidth=2)
plt.axhline(T_env, color='r', linestyle=':', label="External Temp")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Smart Home Temperature Control")
plt.legend()
plt.grid(True)
plt.show()

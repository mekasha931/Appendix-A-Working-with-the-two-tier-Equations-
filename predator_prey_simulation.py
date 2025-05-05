
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(y, t, a, b, c, d):
    prey, predator = y
    dprey_dt = a * prey - b * prey * predator
    dpredator_dt = c * prey * predator - d * predator
    return [dprey_dt, dpredator_dt]

y0 = [40, 9]  # initial prey and predator populations
t = np.linspace(0, 15, 500)
params = (0.6, 0.025, 0.01, 0.4)

sol = odeint(lotka_volterra, y0, t, args=params)

plt.plot(t, sol[:, 0], label="Prey")
plt.plot(t, sol[:, 1], label="Predator")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Predator-Prey Dynamics (Lotka-Volterra)")
plt.legend()
plt.grid(True)
plt.show()

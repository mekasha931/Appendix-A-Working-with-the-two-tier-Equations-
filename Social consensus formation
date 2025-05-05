import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def consensus_model(y, t, alpha, beta, N):
    x = y[:N]
    M = y[N]
    dxdt = alpha * (M - x)
    dMdt = beta * (np.mean(x) - M)
    return np.concatenate([dxdt, [dMdt]])

N = 10
x0 = np.random.rand(N)
M0 = np.mean(x0)
y0 = np.concatenate([x0, [M0]])

t = np.linspace(0, 10, 200)
alpha, beta = 1.0, 0.5

sol = odeint(consensus_model, y0, t, args=(alpha, beta, N))

for i in range(N):
    plt.plot(t, sol[:, i], label=f"x_{i+1}")
plt.plot(t, sol[:, N], 'k--', label="Group Consensus M(t)", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Opinion")
plt.title("Consensus Formation in Two-Tier Model")
plt.legend()
plt.grid(True)
plt.show()

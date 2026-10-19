import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Parametry
# --------------------

dt = 1.0

# Macierz przejścia stanu
F = np.array([
    [1, dt],
    [0, 1]
])

# Macierz pomiaru (mierzymy tylko pozycję)
H = np.array([
    [1, 0]
])

# Kowariancja szumu procesu
Q = np.array([
    [0.01, 0],
    [0, 0.01]
])

# Kowariancja szumu pomiaru
R = np.array([[4.0]])

# Początkowy stan [pozycja, prędkość]
x = np.array([
    [0.0],
    [1.0]
])

# Początkowa niepewność
P = np.eye(2) * 10

# --------------------
# Dane testowe
# --------------------

true_positions = np.arange(0, 50)

measurements = true_positions + np.random.normal(0, 2, len(true_positions))

estimated_positions = []

# --------------------
# Filtr Kalmana
# --------------------

for z in measurements:

    # ===== Predykcja =====

    x = F @ x

    P = F @ P @ F.T + Q

    # ===== Korekcja =====

    z = np.array([[z]])

    y = z - H @ x

    S = H @ P @ H.T + R

    K = P @ H.T @ np.linalg.inv(S)

    x = x + K @ y

    P = (np.eye(2) - K @ H) @ P

    estimated_positions.append(x[0, 0])

# --------------------
# Wykres
# --------------------

plt.plot(true_positions, label="Prawdziwa pozycja")
plt.scatter(
    np.arange(len(measurements)),
    measurements,
    color='red',
    s=10,
    label="Pomiar"
)
plt.plot(
    estimated_positions,
    color='green',
    linewidth=2,
    label="Kalman"
)

plt.legend()
plt.grid()
plt.show()
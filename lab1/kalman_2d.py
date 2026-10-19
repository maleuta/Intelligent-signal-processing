import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------
# Parametry
# ---------------------------------------------

N = 200

# ---------------------------------------------
# Kalman Filter
# stan = [x, y, vx, vy]
# ---------------------------------------------

kf = cv2.KalmanFilter(4, 2)

dt = 1.0

kf.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25

kf.errorCovPost = np.eye(4, dtype=np.float32) * 10

kf.statePost = np.array([
    [0],
    [0],
    [0],
    [0]
], dtype=np.float32)

# ---------------------------------------------
# Trajektoria rzeczywista
# ---------------------------------------------

true_x = []
true_y = []

meas_x = []
meas_y = []

pred_x = []
pred_y = []

for t in range(N):

    x = 100 + 50 * np.sin(t / 15)
    y = 2 * t

    true_x.append(x)
    true_y.append(y)

    # zaszumiony pomiar
    mx = x + np.random.normal(0, 5)
    my = y + np.random.normal(0, 5)

    meas_x.append(mx)
    meas_y.append(my)

    # predykcja
    prediction = kf.predict()

    pred_x.append(prediction[0, 0])
    pred_y.append(prediction[1, 0])

    # korekcja
    measurement = np.array([
        [np.float32(mx)],
        [np.float32(my)]
    ])

    kf.correct(measurement)

# ---------------------------------------------
# Wykres X(t)
# ---------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(true_x,
         label="Rzeczywista pozycja X",
         linewidth=3)

plt.scatter(range(N),
            meas_x,
            color="red",
            s=10,
            label="Pomiar X")

plt.plot(pred_x,
         color="green",
         linewidth=2,
         label="Kalman X")

plt.legend()
plt.grid(True)
plt.xlabel("Próbka")
plt.ylabel("Pozycja X")
plt.title("Filtr Kalmana - współrzędna X")

# ---------------------------------------------
# Wykres Y(t)
# ---------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(true_y,
         label="Rzeczywista pozycja Y",
         linewidth=3)

plt.scatter(range(N),
            meas_y,
            color="red",
            s=10,
            label="Pomiar Y")

plt.plot(pred_y,
         color="green",
         linewidth=2,
         label="Kalman Y")

plt.legend()
plt.grid(True)
plt.xlabel("Próbka")
plt.ylabel("Pozycja Y")
plt.title("Filtr Kalmana - współrzędna Y")

# ---------------------------------------------
# Trajektoria 2D
# ---------------------------------------------

plt.figure(figsize=(8, 8))

plt.plot(true_x,
         true_y,
         linewidth=3,
         label="Trajektoria rzeczywista")

plt.scatter(meas_x,
            meas_y,
            s=10,
            color="red",
            label="Pomiary")

plt.plot(pred_x,
         pred_y,
         color="green",
         linewidth=2,
         label="Kalman")

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajektoria 2D")

plt.show()
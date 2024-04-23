import numpy as np
import time
from memory_profiler import profile
import matplotlib.pyplot as plt

A = 0.1 * np.array([[1, 0], [0, 1]])

Q = 0.05 * np.eye(2)
R = 0.04 * np.eye(2)
G = np.eye(2)
P_0 = 0.03 * np.eye(2)
x_0 = np.random.multivariate_normal(np.ones(2), P_0).reshape(-1, 1)



def C_matrix(N):
    c = np.array([1, 0])
    C = np.array([1, 0])
    for i in range(N - 1):
        C = np.vstack((C, c))
    if N == 1:
        C = c
    return C


C = C_matrix(2)


def dynamics(x):
    wt = np.random.multivariate_normal(np.zeros(2), Q).reshape(-1, 1)
    x_next = A @ x + wt
    return x_next


def output(C, x):
    vt = np.random.multivariate_normal(np.zeros(2), R).reshape(-1, 1)
    y = C @ x + vt
    return y


def measurement_update(C, x, sigma, y):
    lhs = C @ sigma @ (C.T) + R
    rhs = y - C @ x
    x_meas = x + sigma @ (C.T) @ np.linalg.solve(lhs, rhs)

    rhs1 = C @ sigma
    sigma_meas = sigma - sigma @ (C.T) @ np.linalg.solve(lhs, rhs1)
    return x_meas, sigma_meas


def time_update(x_meas, sigma_meas):
    x_time = A @ x_meas
    sigma_time = A @ sigma_meas @ (A.T) + G @ Q @ (G.T)
    return x_time, sigma_time


mse_arr = []
n = 50



def memory_test():
    for i in range(100):
        x_true = x_0
        x_pred = np.array([[1], [1]])
        sigma_pred = P_0
        x_true_arr, x_pred_arr, x_meas_arr = [], [], []
        for t in range(n):
            y = output(C, x_true)
            x_meas, sigma_meas = measurement_update(C, x_pred, sigma_pred, y)
            x_pred, sigma_pred = time_update(x_meas, sigma_meas)
            x_true = dynamics(x_true)

            x_true_arr.append(x_true)
            x_pred_arr.append(x_pred)
            x_meas_arr.append(x_meas)

    # plot
        x_plot = np.arange(n)
        y_plot, y_plot2, y_plot3 = [], [], []

        for i in range(len(x_true_arr)):
            y_plot.append(x_true_arr[i][0])
            y_plot2.append(x_meas_arr[i][0])
            y_plot3.append(x_pred_arr[i][0])
        mse = np.mean((np.array(y_plot) - np.array(y_plot2)) ** 2)
        mse_arr.append(mse)
        plt.plot(x_plot, y_plot, label="x_true")
        plt.plot(x_plot[1:], y_plot2[1:], label="x_meas")
        # plt.plot(x_plot, y_plot3, label="x_pred")
        plt.legend()
        plt.ylim(-2, 2)
        plt.title("C=[1,0;1,0], MSE=" + str(mse))
        plt.show()
    print("mse: ", np.mean(mse_arr))



# time
t1 = time.time()
memory_test()
t2 = time.time()
print("time: ", t2 - t1)

# # memory
# @profile
# def memory_test2():
#     x_true = x_0
#     x_pred = np.array([[1], [1]])
#     sigma_pred = P_0
#     y = output(C, x_true)
#     x_meas, sigma_meas = measurement_update(C, x_pred, sigma_pred, y)
#     x_pred, sigma_pred = time_update(x_meas, sigma_meas)
#     x_true = dynamics(x_true)
#
# memory_test2()
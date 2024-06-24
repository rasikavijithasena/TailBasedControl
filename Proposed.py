
import sys
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from control.matlab import ss, c2d, dlqr,ss2tf, tf, step
import control

from scipy.signal import ss2zpk, cont2discrete

from mushroom_rl.algorithms.policy_search import REINFORCE, GPOMDP, eNAC, RWR, PGPE, REPS, ConstrainedREPS, MORE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import StateStdGaussianPolicy
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.distributions import GaussianCholeskyDistribution

from mushroom_rl.utils.parameters import Parameter
from pathlib import Path
from mushroom_rl.core import Serializable

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import matlab
import os
from numpy import array
import random
import cvxpy as cp
# optimum control action
from scipy.linalg import solve_continuous_are

import math
import pickle
import warnings
warnings.filterwarnings("ignore")

import wandb

def find_gain(x, u):
    num_of_samples = len(u)
    x = x.reshape(num_of_samples, 2)
    phi = cp.Variable((1, 2))

    objective = cp.Minimize(cp.sum_squares(u.reshape(num_of_samples, 1) - cp.matmul(phi, x.T)) / num_of_samples)
    problem = cp.Problem(objective)
    problem.solve()
    optimal_phi = phi.value
    return optimal_phi

def cal_power(T,N, NumSys, alpha_u, P_s):
    # average power consumption
    total_power_all_sys = []
    for n in range(N):
        total_power_systemwise = 0
        for i in range(T):
            for j in range(NumSys):
                p = P_s[0, j, n, i]
                alpha_p = alpha_u[0, j, n, i] * p
                total_power_systemwise = total_power_systemwise + alpha_p

        average_power_per_timeslot = total_power_systemwise / T
        total_power_all_sys.append(average_power_per_timeslot)

    #print('Average Transmission Power per timeslot: ', np.mean(total_power_all_sys))
    #print('Total power 200 timeslots: ', np.mean(total_power_all_sys) * T)

    tpower = np.mean(total_power_all_sys)

    return tpower


def write_data(NumSys, R_value, version, avg_power, avg_state_cost, avg_action_cost,var_state, var_control, avg_aoi):
    # write to a file
    path_unique = '_' + str(NumSys) + '_' + str(R_value)
    unique = str(NumSys) + '_' + str(R_value) + '_' + str(version)
    f = open("calculatedInfo/Proposed/stat" + path_unique + ".txt", "a")
    f.write('\n' + 'average power of ' + str(unique) + ' : ' + str(avg_power))
    f.write('\n' + 'average state cost of ' + str(unique) + ' : ' + str(avg_state_cost))
    f.write('\n' + 'average action cost of ' + str(unique) + ' : ' + str(avg_action_cost))
    f.write('\n' + 'variance state cost of ' + str(unique) + ' : ' + str(var_state))
    f.write('\n' + 'variance action cost of ' + str(unique) + ' : ' + str(var_control))
    f.write('\n' + 'average power of ' + str(unique) + ' : ' + str(avg_aoi))
    f.close()

#calculate average aoi
def cal_aoi(T,N, NumSys,beta):
    total_beta_all_sys = []
    for n in range(N):
        for i in range(T):
            for j in range(NumSys):
                p = beta[0, j, n, i]
                total_beta_all_sys.append(p)

    #print('Average beta: ', np.mean(total_beta_all_sys))
    return np.mean(total_beta_all_sys)

#calculate action cost and state cost
def cal_cost(T,N, NumSys,systems_plant, u_cont, R_value):
    # control action power varience
    R = R_value
    control_power_monte_carlo = []
    state_power_monte_carlo = []
    total_power = []

    C = np.array([[1, 0],
                  [0, 1]])

    Q = np.dot(C.T, C)


    for n in range(N):

        for i in range(T):
            for j in range(NumSys):
                beta1 = np.array([0.1, 0.1])
                x = systems_plant[:, j, n, i]

                if (x[0] < 0.0):
                    x_new = -1 * (x[0])
                else:
                    x_new = x[0]

                x_minus_beta = x_new - beta1[0]
                alpha_p = 0.5 * (1 + (x_minus_beta / np.abs(x_minus_beta)))

                alpha = 1
                # calculate position cost
                if (x[0] < 0.0):  # position minus
                    if (x[1] >= 0.0 and x[1] <= beta1[1]):
                        alpha = alpha_p
                    else:
                        alpha = 1

                else:  # position is plus value
                    if (x[1] <= 0.0 and (x[1] >= (-beta1[1]))):
                        alpha = alpha_p
                    else:
                        alpha = 1

                p = u_cont[0, j, n, i] * R * u_cont[0, j, n, i]  # U.T * R*U
                cost = (x.dot(Q).dot(x)) * alpha + p
                s_cost = (x.dot(Q).dot(x)) * alpha

                control_power_monte_carlo.append(p)
                state_power_monte_carlo.append(s_cost)
                total_power.append(cost)

    #print('Average action cost per timeslot: ', np.mean(control_power_monte_carlo))
    #print('Average state per timeslot: ', np.mean(state_power_monte_carlo))

    #print('Variance action cost per timeslot: ', np.var(control_power_monte_carlo))
    #print('Variance state per timeslot: ', np.var(state_power_monte_carlo))

    action_mean = np.mean(control_power_monte_carlo)
    state_mean = np.mean(state_power_monte_carlo)
    control_var = np.var(control_power_monte_carlo)
    state_var = np.var(state_power_monte_carlo)

    return state_mean, action_mean, state_var, control_var

#save data to files
def save_data(systems_plant,u_cont, alpha_u, P_s, beta,version, R, NumSys ):
    unique = str(NumSys)+'_'+str(R)+'_'+str(version)
    with open(r'dump/Proposed/state_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(systems_plant, f)

    with open(r'dump/Proposed/action_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(u_cont, f)

    with open(r'dump/Proposed/schedule_alpha_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(alpha_u, f)

    with open(r'dump/Proposed/power_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(P_s, f)

    with open(r'dump/Proposed/beta_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(beta, f)

def execute(version, arguments):
    model_name = arguments[1]
    agent = Serializable.load(model_name)
    R_value = int(arguments[2])
    NumSys = int(arguments[3])
    iterations = int(arguments[4])
    version = version

    wandb.init(
        # set the wandb project where this run will be logged
        project="GPR_Final",
        # track hyperparameters and run metadata
        config={
            "model": "Proposed",
            "num_sys": NumSys,
            "R": R_value,
            "iterations": iterations,
            "version": version
        }
    )

    gravity = 0.000025
    force = 1.1115
    k = 3
    delta_t = 1
    gk = gravity * k
    gk2 = gravity * k * delta_t

    Ad = np.array([[1 + gk2, 1],
                   [gk2, 1]])

    Bd = np.array([[delta_t],
                   [delta_t]])

    C = np.array([[1, 0],
                  [0, 1]])

    D = np.array([[0],
                  [0]])

    Q = np.dot(C.T, C)

    # Other variable initializations
    N = iterations
    T = 1000  # 500
    NumSys = NumSys
    NumState = 2
    NumCont = 1

    file_name = model_name  # agent name
    agent = Serializable.load(file_name)

    # Initialize arrays
    u_cont = np.zeros((NumCont, NumSys, N, T))
    u_pred = np.zeros((NumCont, NumSys, N, T))
    systems_plant = np.zeros((NumState, NumSys, N, T + 1))
    systems_est = np.zeros((NumState, NumSys, N, T + 1), dtype=complex)
    systems_pred_low = np.zeros((NumState, NumSys, N, T + 1))
    systems_var_low = np.zeros((NumState, NumSys, N, T + 1))
    systems_trace_low = np.zeros((1, NumSys, N, T + 1))
    systems_pred_high = np.zeros((NumState, NumSys, N, T + 1))
    systems_var_high = np.zeros((NumState, NumSys, N, T + 1))
    systems_trace_high = np.zeros((1, NumSys, N, T + 1))
    systems_schedule = np.zeros((NumState, NumSys, N, T + 1), dtype=complex)
    s_new = np.matrix(np.zeros((NumState, T + 1)))
    x_u = np.zeros((T, NumSys, N))
    y_u = np.zeros((NumState, NumSys, N, T))

    # Initialize the systems
    s_new[:, 0] = np.matrix([-1.5, 0.0]).T

    Z = np.matrix(np.eye(2))
    M_T = 2  # Number of antennas Tx and Rx antennas in Uplink
    Hu = np.zeros((M_T, M_T, NumSys, N, T), dtype=complex)  # Empty array for the channel matrix of the sensing link
    noise_T = np.zeros((M_T, 1, NumSys, N, T))
    # Cov_Nu = np.zeros((M_T, M_T, NumSys, N, T))
    P_s = np.zeros((1, NumSys, N, T))  # Wireless transmission power

    Cxy = np.zeros((M_T, M_T, NumSys, N, T), dtype=complex)
    Cyy = np.zeros((M_T, M_T, NumSys, N, T), dtype=complex)
    Est_Error_u = np.zeros((M_T, M_T, NumSys, N, T), dtype=complex)
    Est_u = np.zeros((1, NumSys, N, T), dtype=complex)

    Sigmax = np.matrix(np.eye(M_T))  # State information Covariance matrix
    SNR_th_u = 100  # Signal-to-noise ratio threshold = 20 dB

    Pmax = 1  # Maximum transmission power

    # Optimization parameter
    beta = np.zeros((1, NumSys, N, T))
    alpha_u = np.zeros((1, NumSys, N, T))
    eeta = np.zeros((1, NumSys, N, T))
    # QOS arrays
    P_cont = np.zeros((1, N, T))
    sx = np.zeros((2, NumSys, N, T))

    for n in range(N):
        for sys in range(NumSys):
            systems_plant[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))

    NoTrainSample = 5

    # Virtual Queues Initialization
    Q_B = np.zeros((1, NumSys, N, T + 1))  # beta: AOI
    Q_Pu = np.zeros((1, NumSys, N, T + 1))  # TX Power
    # Q_Pc = np.zeros((1, NumSys, N, T + 1))  # GPR Power
    Q_Cu = np.zeros((1, NumSys, N, T + 1))  # Scheduling variable bound for wireless

    gamm_B = np.zeros((1, NumSys, N, T))  # beta: AOI
    gamm_Pu = np.zeros((1, NumSys, N, T))  # TX power

    for n in range(N):
        for sys in range(NumSys):
            systems_plant[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))
            systems_est[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))
            # print(systems_pred_low[:, sys, n, 0])
            # print(systems_pred_low[:, :, n, 0])
            systems_pred_low[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))
            # print(systems_pred_low[:, sys, n, 0])
            systems_pred_high[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))
            systems_schedule[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))

            for k in range(1, NoTrainSample + 1):
                # A_c = getTransferMatrix(k)
                control_action_cont = agent.draw_action(np.real(systems_schedule[:, sys, n, k - 1]))

                if (np.abs(control_action_cont) > 1000):
                    control_action_cont = 1000 * (control_action_cont / np.abs(control_action_cont))

                u_cont[0, sys, n, k - 1] = control_action_cont * 0.01

                systems_plant[1, sys, n, k] = systems_plant[1, sys, n, k - 1] + (
                        np.real(u_cont[0, sys, n, k - 1]) * delta_t) + gravity * delta_t * math.sin(
                    k * systems_plant[0, sys, n, k - 1]);
                systems_plant[0, sys, n, k] = systems_plant[0, sys, n, k - 1] + systems_plant[1, sys, n, k - 1];
                systems_est[:, sys, n, k] = systems_plant[:, sys, n, k]
                systems_pred_low[:, sys, n, k] = systems_plant[:, sys, n, k]
                systems_pred_high[:, sys, n, k] = systems_plant[:, sys, n, k]
                systems_schedule[:, sys, n, k] = systems_plant[:, sys, n, k]

            x_I = range(NoTrainSample)  # 1x5
            # x_u[:, sys,n] = np.transpose(x_I)  # 5x1
            counter = 0
            for i in x_I:
                x_u[counter][sys][n] = i
                counter = counter + 1

            # print(x_u)

            for l in range(NumState):
                for k in range(NoTrainSample):
                    y_u[l, sys, n, k] = systems_est[l, sys, n, k]

            for k in range(NoTrainSample):
                beta[0, sys, n, k] = 1
                alpha_u[0, sys, n, k] = 1
                eeta[0, sys, n, k] = 1

            Q_B[0, sys, n, NoTrainSample] = T
            Q_Pu[0, sys, n, NoTrainSample] = Pmax
            Q_Cu[0, sys, n, NoTrainSample] = 1

    Val_a = np.zeros((NumSys, N, T))
    I_a = np.zeros((NumSys, N, T))
    a = np.zeros((1, NumSys, N, T))
    m_u_sup = np.zeros((1, N, T))
    sx = np.zeros((4, NumSys, N, T))

    # Wireless Communication Parameters

    SNR_th_u = 100
    Sigmax = np.eye(M_T)
    P_s = np.zeros((1, NumSys, N, T))
    Pmax = 1
    Bmax = T
    Z = np.matrix(np.eye(2))
    window_size = 5

    V = 1000  # Tradeoff control variable
    nu_Bu = 1  # Weight variable of the uplink AoI
    nu_Pu = 1  # Weight variable of the sensor power transmission

    eps = 0.1

    m_u = np.zeros((1, NumSys, N, T))

    KK = np.array([[0.9388, 13.2693]])
    BK = np.dot(Bd, KK)
    A_c = Ad - BK

    for n in range(N):

        for t in range(NoTrainSample, T):

            gain_matrix = np.zeros((NumSys, 2))

            for sys in range(NumSys):
                if t < 21:
                    select_x = systems_schedule[:, sys, n, 0:t - 1]
                    select_u = u_cont[:, sys, n, 0:t - 1]
                else:
                    select_x = systems_schedule[:, sys, n, t - 20:t - 1]
                    select_u = u_cont[:, sys, n, t - 20:t - 1]

                gain_value = find_gain(select_x, select_u[0])
                gain_matrix[sys, :] = gain_value

                BK = np.dot(Bd, gain_value)
                A_c = Ad - BK

            for sys in range(NumSys):
                Hu[:, :, sys, n, t] = (np.random.randn(M_T, M_T) + 1j * np.random.randn(M_T, M_T)) / np.sqrt(2)
                Noise_var_u = 2e-2
                lin = np.linalg.norm(Hu[:, :, sys, n, t], 'fro')
                # P_s[0,sys,n,t] = min(SNR_th_u * Noise_var_u / (lin*lin), Pmax)
                if Q_Pu[0, sys, n, t] >= 0:
                    P_s[0, sys, n, t] = SNR_th_u * Noise_var_u / (lin * lin)
                else:
                    P_s[0, sys, n, t] = Pmax

                noise_T[:, 0, sys, n, t] = np.random.normal(0, (Noise_var_u / 2), M_T).reshape(M_T)
                Cxy[:, :, sys, n, t] = np.sqrt(P_s[0, sys, n, t]) * np.dot(Sigmax, Hu[:, :, sys, n, t].T)
                Cyy[:, :, sys, n, t] = P_s[0, sys, n, t] * np.dot(np.dot(Hu[:, :, sys, n, t], Sigmax),
                                                                  Hu[:, :, sys, n, t].T) + Noise_var_u * np.eye(M_T)
                Est_Error_u[:, :, sys, n, t] = Sigmax - np.dot(np.divide(Cxy[:, :, sys, n, t], Cyy[:, :, sys, n, t]),
                                                               Cxy[:, :, sys, n, t].T)

                gamm_B[0, sys, n, t] = min(max((V * nu_Bu - Q_B[0, sys, n, t]) / Q_B[0, sys, n, t], 1), Bmax)
                gamm_Pu[0, sys, n, t] = min(max((V * nu_Pu - Q_Pu[0, sys, n, t]) / Q_Pu[0, sys, n, t], 0), Pmax)

                s_pred_u = np.zeros((NumState, NumSys, N, T))
                v_pred_u = np.zeros((NumState, NumSys, N, T))
                Pred_u = np.zeros((NumSys, N, T))

                noise_std = 0.75
                kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

                for l in range(NumState):
                    non_zero_x = x_u[:, sys, n][np.where(x_u[:, sys, n] != 0)].astype(int)

                    if len(non_zero_x) > 5:
                        length = len(non_zero_x)
                        non_zero_x = non_zero_x[length - 6: length - 1]

                    selected_y = y_u[l, sys, n, non_zero_x]

                    gaussian_process = GaussianProcessRegressor(
                        kernel=kernel, n_restarts_optimizer=9)
                    gaussian_process.fit(non_zero_x.reshape(-1, 1), selected_y)
                    mean_prediction, std_prediction = gaussian_process.predict(np.array(t).reshape(-1, 1),
                                                                               return_std=True)

                    s_pred_u[l, sys, n, t] = mean_prediction
                    v_pred_u[l, sys, n, t] = std_prediction

                Pred_u[sys, n, t] = np.sum(v_pred_u[:, sys, n, t])
                systems_var_low[:, sys, n, t] = v_pred_u[:, sys, n, t]
                systems_pred_low[:, sys, n, t] = s_pred_u[:, sys, n, t]
                pred_low_temp = systems_pred_low[:, sys, n, t]

                var_low_temp = np.array([[systems_var_low[0, sys, n, t], 0], [0, systems_var_low[1, sys, n, t]]])

                a_1 = np.dot((A_c - eps * np.eye(NumState)), pred_low_temp)
                # a1 = np.dot(np.dot(a_1.T , Z ), a_1)
                a1 = np.linalg.norm(a_1, ord=2) ** 2
                b1 = np.trace(np.dot((np.dot(np.dot(Ad.T, Z), Ad) - eps * Z), var_low_temp))
                j1 = np.trace(np.dot(np.dot(np.dot(BK.T, Z), BK), var_low_temp))
                k1 = np.trace(np.dot(np.dot(np.dot(BK.T, Z), BK), Est_Error_u[:, :, sys, n, t]))

                m_u[0, sys, n, t] = (a1 + b1 + np.trace(Z)) / (j1 - k1)

                a[0, sys, n, t] = Q_Pu[0, sys, n, t] * P_s[0, sys, n, t] - Q_Cu[0, sys, n, t] - Q_B[0, sys, n, t] * beta[0, sys, n, t - 1]

            m_u_sup[0, n, t] = m_u[0, sys, n, t]
            I_a[:, n, t] = np.argsort(a[0, :, n, t])
            Val_a[:, n, t] = sorted(a[0, :, n, t])

            for sys in range(NumSys):
                if ((I_a[0, n, t] == sys) and (Val_a[0, n, t] < 0)):

                    alpha_u[0, sys, n, t] = 1
                    beta[0, sys, n, t] = 1
                    eeta[0, sys, n, t] = 0

                    sx = np.sqrt(P_s[0, sys, n, t]) * np.dot(Hu[:, :, sys, n, t],
                                                             systems_plant[:, sys, n, t]) + noise_T[:, 0, sys, n, t]
                    systems_est[:, sys, n, t] = np.dot(
                        np.dot(Cxy[:, :, sys, n, t], np.linalg.inv(Cyy[:, :, sys, n, t])), sx)
                    systems_schedule[:, sys, n, t] = systems_est[:, sys, n, t]

                    x_u[t, sys, n] = t
                    y_u[:, sys, n, t] = np.real(systems_est[:, sys, n, t])
                else:
                    alpha_u[0, sys, n, t] = 0
                    eeta[0, sys, n, t] = 0
                    beta[0, sys, n, t] = 1 + beta[0, sys, n, t - 1]

                    if t == 0:
                        systems_schedule[:, sys, n, t] = np.zeros(2)
                    else:
                        # systems_schedule[:,sys,n,t] = np.zeros(2)
                        systems_schedule[:, sys, n, t] = systems_pred_low[:, sys, n, t]
                        x_u[t, sys, n] = t
                        y_u[:, sys, n, t] = np.real(systems_schedule[:, sys, n, t])

                control_action_cont = agent.draw_action(np.real(systems_schedule[:, sys, n, t]))

                if (np.abs(control_action_cont) > 1000):
                    control_action_cont = 1000 * (control_action_cont / np.abs(control_action_cont))
                u_cont[0, sys, n, t] = control_action_cont * 0.01

                # systems_plant[:,sys,n,t+1] = np.dot(Ad, systems_plant[:,sys,n,t]) + np.dot(Bd, np.real(u_cont[0,sys,n,t]).reshape(1,))
                systems_plant[1, sys, n, t + 1] = systems_plant[1, sys, n, t] + np.real(
                    u_cont[0, sys, n, t]) * delta_t + gravity * math.sin(k * systems_plant[0, sys, n, t]) * delta_t;
                systems_plant[0, sys, n, t + 1] = systems_plant[0, sys, n, t] + systems_plant[1, sys, n, t + 1];

            for sys in range(NumSys):
                # update the queues
                Q_B[0, sys, n, t + 1] = max(Q_B[0, sys, n, t] - gamm_B[0, sys, n, t], 0) + beta[0, sys, n, t]
                Q_Pu[0, sys, n, t + 1] = max(Q_Pu[0, sys, n, t] - gamm_Pu[0, sys, n, t], 0) + np.dot(
                    alpha_u[0, sys, n, t], P_s[0, sys, n, t])
                Q_Cu[0, sys, n, t + 1] = max(Q_Cu[0, sys, n, t] - alpha_u[0, sys, n, t], 0) + max(
                    min(np.real(m_u_sup[0, n, t]), 1), 0);



    tpower = cal_power(T, N, NumSys, alpha_u, P_s)
    avg_aoi = cal_aoi(T, N, NumSys, beta)
    avg_state, avg_cost, var_state, var_control = cal_cost(T, N, NumSys, systems_plant, u_cont, R_value)
    wandb.log({"avg_aoi": avg_aoi,
               "avg_power": tpower,
               "mean_state_cost": avg_state,
               "mean_control_cost": avg_cost,
               "var_state_cost": var_state,
               "var_control_cost": var_control})

    save_data(systems_plant, u_cont, alpha_u, P_s, beta, version, R_value, NumSys)
    write_data(NumSys, R_value, version, tpower, avg_state, avg_cost, var_state, var_control, avg_aoi)
    wandb.finish()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    iter = int(sys.argv[4])
    for i in range(iter):
        execute(i, sys.argv)

    print('Finish Proposed')


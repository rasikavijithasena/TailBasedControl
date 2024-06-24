
import sys
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from control.matlab import ss, c2d, dlqr,ss2tf, tf, step
import control
import wandb

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

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    f = open("calculatedInfo/RRARIMATail/stat" + path_unique + ".txt", "a")
    f.write('\n' + 'average power of ' + str(unique) + ' : ' + str(avg_power))
    f.write('\n' + 'average state cost of ' + str(unique) + ' : ' + str(avg_state_cost))
    f.write('\n' + 'average action cost of ' + str(unique) + ' : ' + str(avg_action_cost))
    f.write('\n' + 'variance state cost of ' + str(unique) + ' : ' + str(var_state))
    f.write('\n' + 'variance action cost of ' + str(unique) + ' : ' + str(var_control))
    f.write('\n' + 'average power of ' + str(unique) + ' : ' + str(avg_aoi))
    f.close()


def cal_aoi(T,N, NumSys,beta):
    total_beta_all_sys = []
    for n in range(N):
        for i in range(T):
            for j in range(NumSys):
                p = beta[0, j, n, i]
                total_beta_all_sys.append(p)

    #print('Average beta: ', np.mean(total_beta_all_sys))
    return np.mean(total_beta_all_sys)

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

    return state_mean,action_mean,state_var,control_var



#save data to files
def save_data(systems_plant,u_cont, alpha_u, P_s, beta,version, R, NumSys ):
    unique = str(NumSys)+'_'+str(R)+'_'+str(version)
    with open(r'dump/RRARIMATail/state_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(systems_plant, f)

    with open(r'dump/RRARIMATail/action_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(u_cont, f)

    with open(r'dump/RRARIMATail/schedule_alpha_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(alpha_u, f)

    with open(r'dump/RRARIMATail/power_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(P_s, f)

    with open(r'dump/RRARIMATail/beta_data_'+unique+'.pkl', 'wb') as f:
        pickle.dump(beta, f)







def execute(version, arguments):

    model_name = arguments[1]
    agent = Serializable.load(model_name)
    R_value = int(arguments[2])
    NumSys = int(arguments[3])
    iterations = int(arguments[4])


    wandb.init(
        # set the wandb project where this run will be logged
        project="GPR_Final",
        # track hyperparameters and run metadata
        config={
            "model": "RR_ARIMA_Tail",
            "num_sys": NumSys,
            "R": R_value,
            "iterations":iterations,
            "version": version,
        }
    )



    force = 1.1115
    gravity = 0.000025
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
    x_u = np.zeros((T, NumSys))
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

    # QOS arrays
    P_cont = np.zeros((1, N, T))
    sx = np.zeros((2, NumSys, N, T))

    for n in range(N):
        for sys in range(NumSys):
            systems_plant[:, sys, n, 0] = np.transpose(np.array([-1.5, 0.0]))

    # Wireless Communication Parameters

    SNR_th_u = 100
    Sigmax = np.eye(M_T)
    P_s = np.zeros((1, NumSys, N, T))
    Pmax = 1
    window_size = 10

    for n in range(N):
        for t in range(T):
            for sys in range(NumSys):
                Hu[:, :, sys, n, t] = (np.random.randn(M_T, M_T) + 1j * np.random.randn(M_T, M_T)) / np.sqrt(2)
                Noise_var_u = 2e-2
                lin = np.linalg.norm(Hu[:, :, sys, n, t], 'fro')

                P_s[0, sys, n, t] = SNR_th_u * Noise_var_u / (lin * lin)

                noise_T[:, 0, sys, n, t] = np.random.normal(0, (Noise_var_u / 2), M_T).reshape(M_T)
                Cxy[:, :, sys, n, t] = np.sqrt(P_s[0, sys, n, t]) * np.dot(Sigmax, Hu[:, :, sys, n, t].T)
                Cyy[:, :, sys, n, t] = P_s[0, sys, n, t] * np.dot(np.dot(Hu[:, :, sys, n, t], Sigmax),
                                                                  Hu[:, :, sys, n, t].T) + Noise_var_u * np.eye(M_T)

                if t % NumSys == (sys):
                    alpha_u[0, sys, n, t] = 1
                    sx = np.sqrt(P_s[0, sys, n, t]) * np.dot(Hu[:, :, sys, n, t],
                                                             systems_plant[:, sys, n, t]) + noise_T[:, 0, sys, n, t]
                    systems_est[:, sys, n, t] = np.dot(
                        np.dot(Cxy[:, :, sys, n, t], np.linalg.inv(Cyy[:, :, sys, n, t])), sx)
                    systems_schedule[:, sys, n, t] = np.real(systems_est[:, sys, n, t])
                else:
                    alpha_u[0, sys, n, t] = 0

                    if t == 0:
                        # systems_schedule[:,sys,n,t] = np.zeros(2)
                        systems_schedule[:, sys, n, t] = np.transpose(np.array([-1.5, 0.0]))
                    else:
                        # systems_schedule[:,sys,n,t] = np.zeros(2)
                        if (t < 55):
                            for j in range(NumState):
                                model = ARIMA(np.real(systems_schedule[j, sys, n, :]), order=(1, 0, 1))
                                model_fit = model.fit()
                                yhat = model_fit.forecast()[0]
                                systems_schedule[j, sys, n, t] = yhat

                        else:
                            dataset = []
                            for count in range(50):
                                dataset.append(np.real(systems_schedule[j, sys, n, t - count]))

                            for j in range(NumState):
                                model = ARIMA(dataset, order=(1, 0, 1))
                                model_fit = model.fit()
                                yhat = model_fit.forecast()[0]
                                systems_schedule[j, sys, n, t] = yhat

                control_action_cont = agent.draw_action(np.real(systems_schedule[:, sys, n, t]))

                if (np.abs(control_action_cont) > 1000):
                    control_action_cont = 1000 * (control_action_cont / np.abs(control_action_cont))
                u_cont[0, sys, n, t] = control_action_cont * 0.01

                # systems_plant[:,sys,n,t+1] = np.dot(Ad, systems_plant[:,sys,n,t]) + np.dot(Bd, np.real(u_cont[0,sys,n,t]).reshape(1,))
                systems_plant[1, sys, n, t + 1] = systems_plant[1, sys, n, t] + np.real(
                    u_cont[0, sys, n, t]) + gravity * math.sin(k * systems_plant[0, sys, n, t]);
                systems_plant[0, sys, n, t + 1] = systems_plant[0, sys, n, t] + systems_plant[1, sys, n, t + 1];


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


if __name__ == '__main__':

    iter = int(sys.argv[4])
    for i in range(iter):
        execute(i, sys.argv)

    print('Finish RR ARIMA Tail')
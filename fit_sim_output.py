"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
     ** Fit the output correct detection proabilities obtained from simulate_detection_delays and fit them at
     different levels of aggregation.                                                                         **
Change point Detection for M/M/1 waiting times using Non-Overlapping Batch Means and
ladder point process


1. For different rho, time of change and delta rho fit the probability of correct detection vs.  the batch size
"""
import numpy as np
import math
import pandas as pd
from test_code_plot import load_power_df
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm, fisk, lognorm
from scipy import optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, RationalQuadratic, \
    Matern, DotProduct


def log_logistic_pdf(x, alpha, beta):
    """
    x is a scalar
    alpha > 0 scale parameter
    beta > 0 shape parameter
    """
    return (beta / alpha) * math.pow(x / alpha, beta - 1) / ((1 + math.pow((x/alpha), beta)) ** 2)


class ResultParams:
    def __init__(self, params, score, mode, var_mode, bic_score, kernel):
        self.params = params
        self.score = score #coefficient of determination of the prediction
        self.mode = mode
        self.var_mode = var_mode
        self.bic = bic_score
        self.kernel = kernel

class FitClass:
    def __init__(self, data, y=None):
        if y.any():
            self._y = y
            self._x = data
            self._fit_type = "LeastSquare"
        else:
            self._samples = data
            self._fit_type = "MLE"
        self.best_param = self.fit_data()

    def fit_data(self):
        if self._fit_type == "LeastSquare":
            return self.fit_least_square()
        elif self._fit_type == "MLE":
            return self.fit_mle()

    def fit_mle(self):
        pass

    def fit_least_square(self):
        if not self._y.any():
            raise Exception("Expected x and y data, but the user only provided x data.")
        f = lambda params: self.error_function(params)
        grad_f = lambda params: self.grad_function(params)
        initial_param = np.asarray([5, 1])
        # best_param = optimize.minimize(f, [5,1], method="BFGS", jac=grad_f)
        bnds = ((0, 1e6), (0, 1e6))
        best_param = optimize.minimize(f, initial_param, method="L-BFGS-B", jac=grad_f, bounds=bnds)
        print("Found solution: ", best_param)
        return best_param.x

    def error_function(self, params):
        pass

    def grad_function(self, params):
        pass


class LogNormal(FitClass):
    def error_function(self, params):
        mu = params[0]
        sigma = params[1]
        return_score = 0
        for x, y in zip(self._x, self._y):
            density = norm.pdf(np.log(x), loc=mu, scale=sigma)
            return_score += (y - 1.0/x * density) * (y - 1.0/x * density)
        return return_score

    def grad_function(self, params):
        mu, sigma = params
        g = np.zeros_like(params)
        dfdmu = 0
        dfdsigma = 0
        for x, y in zip(self._x, self._y):
            density = norm.pdf(np.log(x), loc=mu, scale=sigma)
            A = (y - 1.0/x * density)
            B = (np.log(x) - mu) / (sigma * sigma)
            dfdmu += B * A
            dfdsigma += A * ((B-1)/(sigma * x)) * density

        g[0] = - 2.0/(sigma * np.sqrt(2 * math.pi)) * dfdmu
        g[1] = - 2.0 * dfdsigma
        return g


class LogLogisticFit(FitClass):
    """
    Create a class for the log logistic distribution
    The base object should take in some data either just samples or (x_i, f(x_i)) pairs
    If samples, perform a mle to return the parameters with the fit functions
    If (x_i, p_i) pairs perform a least square optimization to obtain the parameters with the fit functions
    """

    def grad_function(self, params):
        a, b = params
        g = np.zeros_like(params)
        g[0] = self.log_logistic_error_function_alpha_gradient(a, b)
        g[1] = self.log_logistic_error_function_beta_gradient(a, b)
        return g

    def error_function(self, params):
        alpha = params[0]
        beta = params[1]
        return_score = 0
        for x, y in zip(self._x, self._y):
            return_score += (y - log_logistic_pdf(x, alpha, beta)) * (y - log_logistic_pdf(x, alpha, beta))
        return return_score

    def log_logistic_error_function_alpha_gradient(self, alpha, beta):
        return_value = 0
        for x, y in zip(self._x, self._y):
            denom_base = (1 + math.pow(x/alpha, beta))
            return_value += 2 * (y - log_logistic_pdf(x, alpha, beta)) * (
                    (((beta - 1) / alpha * x - 1) * (beta/(alpha ** 2)) *
                     (x / alpha) ** (beta - 2)) / (denom_base ** 2) + 2 * beta ** 2
                    / alpha * (x/alpha) ** (2 * beta - 1) / denom_base ** 3)
        return -return_value

    def log_logistic_error_function_beta_gradient(self, alpha, beta):
        return_value = 0
        for x, y in zip(self._x, self._y):
            denom_base = (1 + math.pow(x/alpha, beta))
            first_part = (y - log_logistic_pdf(x, alpha, beta))
            return_value += 2 * first_part * (((1 + beta * np.log(x/alpha)) * 1/alpha * math.pow((x/alpha), (beta-1)))
            / (denom_base ** 2) - 2 * (beta/alpha) * math.pow(x/alpha, 2 * beta - 1) * np.log(x/alpha)
                                              / math.pow(denom_base, 3))
        return -return_value


def sample_from_inverse_distribution(x, y_cumsum, nb_points):
    sample_list = []
    for _ in range(nb_points):
        selector_prob = np.random.random()
        for idx, prob_val in enumerate(y_cumsum):
            if prob_val > selector_prob:
                sample_list.append(x[idx])
                break
    return sample_list


def sample_from_discrete_distribution(x_points, y_points, num_points):
    x_vec = list(x_points)
    y_vec = list(y_points)
    # Sort the list
    zipped_lists = zip(x_vec, y_vec)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    x_list, y_list = [list(tuple) for tuple in tuples]
    # Compute cumulative y's and normalize
    y_cumsum = np.cumsum(y_list)
    y_cumsum = y_cumsum / y_cumsum[-1]
    sampled_points = sample_from_inverse_distribution(x_list, y_cumsum, num_points)
    return sampled_points


def plot_lml_landscape(gp):
    # Plot LML landscape
    plt.figure()
    theta0 = np.logspace(-2, 3, 49)
    theta1 = np.logspace(-2, 0, 50)
    Theta0, Theta1 = np.meshgrid(theta0, theta1)
    LML = [[gp.log_marginal_likelihood(np.log([0.36, Theta0[i, j], Theta1[i, j]]))
            for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
    LML = np.array(LML).T

    vmin, vmax = (-LML).min(), (-LML).max()
    vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
    plt.contour(Theta0, Theta1, -LML,
                levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Length-scale")
    plt.ylabel("Noise-level")
    plt.title("Log-marginal-likelihood")
    plt.tight_layout()


class BestParamsLogLogistic:
    def __init__(self, params, coef_det, bic_score, best_kernel, best_gp):
        self.params = params
        self.coef_det = coef_det
        self.bic_coef = bic_score
        self.kernel = best_kernel
        self.gp = best_gp


class SimulationConfigParams:
    def __init__(self, rho, delta_rho, time_of_change):
        self.rho = rho
        self.delta_rho = delta_rho
        self.tc = time_of_change


def optimizer_cg(obj_func, initial_theta, bounds):
    resopt = optimize.minimize(obj_func, initial_theta, bounds=bounds, jac=True, method='CG`')
    return resopt.x, resopt.fun


def get_best_parameters_direct_log_logistic_from_gp_with_mle(covariates, dependent_vars, sim_param, fit_type="fisk"):
    """
        # Learn the parameters of the distribution in three steps
        # 1. Fit a Gaussian Process to the probability of correct detection
        # 2. Sample from the GP to obtain more batch sizes (x) and conditional probabilities (y)
        # 3a. Make sure both are sorted
        # 3b. Compute cumulative distribution by summing up the y's and normalize
        # 3c. sample from the inverted distributions to get multiple batch sizes
        # 4. Fit the batch sizes with MLE
    """
    X = np.atleast_2d(covariates).T
    y = dependent_vars.ravel()
    n = X.shape[0]
    best_bic = np.float("inf")
    best_kernel = None
    best_gp = None

    # Fit multiple kernels, compute the BIC score for each kernel and compare them
    # 1. with RBF + WK
    kernel = WhiteKernel(noise_level=0.5) + C(np.max(y), (1e-4, 1e3)) * RBF(2, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=120, optimizer='fmin_l_bfgs_b')
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    # 2. with RQ + WK
    kernel = WhiteKernel(noise_level=0.5) + C(np.max(y), (1e-6, 1e4)) * RationalQuadratic(alpha=15, length_scale=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=120, optimizer="fmin_l_bfgs_b")
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    # 3. with RQ + WK + RBF
    kernel = WhiteKernel(noise_level=0.1) + C(np.max(y), (1e-6, 1e5)) * RationalQuadratic() \
             + C(np.max(y), (1e-6, 1e5)) * RBF()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=60, optimizer="fmin_l_bfgs_b")
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    # 4. with RBF + WK + RBF
    kernel = WhiteKernel(noise_level=0.1) + C(np.max(y), (1e-6, 1e5)) * RBF(1e-2) \
             + C(np.max(y), (1e-6, 1e5)) * RBF(1000)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=60, optimizer="fmin_l_bfgs_b")
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    # 5. with Matern + WK + RQ
    kernel = WhiteKernel(noise_level=1.0) + C(np.max(y), (1e-6, 1e5)) * Matern(length_scale=5) \
             + C(2) * RationalQuadratic(alpha=15, length_scale=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=60, optimizer="fmin_l_bfgs_b")
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    # 6. with DotProduct + WK + RQ
    kernel = WhiteKernel(noise_level=1.0) + C(np.max(y), (1e-6, 1e5)) * DotProduct() \
             + C(2) * RationalQuadratic(alpha=15, length_scale=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=120, optimizer="fmin_l_bfgs_b")
    # fit to data using MLE of the parameters
    gp.fit(np.log(X), y)
    m = gp.n_features_in_
    log_likelihood_value = gp.log_marginal_likelihood_value_
    bic_score = -2 * log_likelihood_value + m * np.log(n)
    if bic_score < best_bic:
        best_bic = bic_score
        best_kernel = gp.kernel_
        best_gp = gp

    x = np.linspace(min(X), max(X), 20)
    gp_batch_sizes = np.atleast_2d(np.random.uniform(min(x), max(x), size=30)).T
    gp_probabilities, sigma = best_gp.predict(gp_batch_sizes, return_std=True)
    coef_determination = best_gp.score(X, y)
    sampled_batch_sizes = sample_from_discrete_distribution(gp_batch_sizes, gp_probabilities, 120)

    if fit_type == "fisk":
        c, loc, scale = fisk.fit(sampled_batch_sizes, floc=0)
#         print("For a Log-logistic distribution found the parameters: ")
#         print("\t c={0:2.4f}, loc={1:2.4f}, scale={2:2.4f}".format(c, loc, scale))
        params = (c, loc, scale)
    elif fit_type == "log-normal":
        s, loc, scale = lognorm.fit(sampled_batch_sizes, floc=0)
        mu = np.log(scale)
#         print("For a Lognormal distribution found the parameters")
#         print("\t mu={0:2.4f}, loc={1:2.4f}, sigma={2:2.4f}".format(mu, loc, scale))
        params = (s, loc, scale, mu)

    # plot_lml_landscape(gp)
    return BestParamsLogLogistic(params, coef_determination, bic_score, best_kernel, best_gp)


def plot_predicted_gaussian_process(sim_param, X, y, best_gp, x_mode, num_points=20):
    # Plot the Gaussian fit
    plt.figure()
    # plt.plot(dependent_vars, covariates, 'r:', label='Original')
    plt.plot(X.ravel(), y, 'r.', markersize=10, label="Observations")
    x = np.linspace(min(X), max(X), num_points)
    y_pred, sigma = best_gp.predict(np.log(x), return_std=True)
    plt.plot(x, y_pred, 'b-', label="Prediction")
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.vlines(x=x_mode, ymin=0, ymax=max(y_pred), colors='r')
    plt.ylabel("Prob. Correct Detection")
    plt.xlabel("Batch Size")
    plt.savefig("Figures/BatchSizeFit/PassingThroughZero/batch_size_config_rho_({})_delta_rho_({})_tc_{}_gp_fit.png".format(
        sim_param.rho,
        sim_param.delta_rho,
        sim_param.tc))
    plt.legend()
    #plt.show()
    plt.close()


def get_best_parameters_inverted_log_logistic_from_gp_with_mle(covariates, dependent_vars):
    """
        # Learn the parameters of the distribution in three steps
        # 1. Fit a Gaussian Process to the *inverse* of the plot
        # 2. Sample from the inverse with uniform
        # 3. Do a MLE to get the parameters of the distribution
    """
    X = np.atleast_2d(dependent_vars).T
    y = covariates.ravel()
    kernel = C(1.0, (1e-4, 1e3)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(noise_level=0.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=12)

    # fit to data using MLE of the parameters
    gp.fit(X, y)

    # Plot the Gaussian fit
    plt.figure()
    # plt.plot(dependent_vars, covariates, 'r:', label='Original')
    plt.plot(X.ravel(), y, 'r.', markersize=10, label="Observations")
    x = np.linspace(min(X), max(X), 20)
    plt.plot(x, gp.predict(x), 'b-', label="Prediction")
    plt.ylabel("Prob. Correct Detection")
    plt.xlabel("Batch Size")
    plt.legend()
    #plt.show()
    plt.close()

    sampled_prob_values = np.atleast_2d(np.random.uniform(0, 1.0, size=50)).T
    sampled_batch_sizes, sigma = gp.predict(sampled_prob_values, return_std=True)
    c, loc, scale = fisk.fit(sampled_batch_sizes)
    print("For a Log-logistic distribution found the parameters: ")
    print("\t c={0:2.4f}, loc={1:2.4f}, scale={2:2.4f}".format(c, loc, scale))

    plt.figure()
    plt.hist(sampled_batch_sizes, density=True)
    plt.xlabel("Batch Size")
    plt.ylabel("y")
    # plt.show()
    plt.close()
    return c, loc, scale


def get_best_fit_directly_from_samples(x_points, y_points, fit_type="fisk"):
    sampled_batch_sizes = sample_from_discrete_distribution(x_points, y_points, 500)
    if fit_type == "fisk":
        c, loc, scale = fisk.fit(sampled_batch_sizes, floc=0)
        print("For a Log-logistic distribution found the parameters: ")
        print("\t c={0:2.4f}, loc={1:2.4f}, scale={2:2.4f}".format(c, loc, scale))
        params = (c, loc, scale)
    elif fit_type == "log-normal":
        s, loc, scale = lognorm.fit(sampled_batch_sizes, floc=0)
        mu = np.log(scale)
        print("For a Lognormal distribution found the parameters")
        print("\t mu={0:2.4f}, loc={1:2.4f}, sigma={2:2.4f}".format(mu, loc, scale))
        params = (s, loc, scale, mu)

    plt.figure()
    plt.hist(sampled_batch_sizes, density=True )
    plt.xlabel("Batch Size")
    plt.ylabel("y")
    # plt.show()
    return params


def find_best_correct_detection_prob(specific_df, sim_param, fit_type="fisk"):
    rho_selected = specific_df["rho"].iloc[0]
    time_of_change = specific_df["Time of Change"].iloc[0]
    delta_rho_selected = specific_df["delta_rho"].iloc[0]

    # for each batch size, get the mean of the different probability of correct detection

    x = np.array(specific_df["Batch Size"])
    x_unique = np.unique(x)
    y = np.array(specific_df["Conditional_Correct_Detection"])
    reduced_df = specific_df[(specific_df["rho"] == rho_selected) & (specific_df["Time of Change"] == time_of_change)
                             & (specific_df["delta_rho"] == delta_rho_selected)]
    y_unique = []
    for batch_size in reduced_df["Batch Size"].unique():
        y_unique.append(np.max(reduced_df[reduced_df["Batch Size"] == batch_size]["Conditional_Correct_Detection"]))
    eps = 0.001
    x_unique = np.append(x_unique, eps)
    y_unique = np.append(y_unique, eps)

#     # get only one y per x
#     x_vec = list(set(x))
#     y_unique = []
#     for x_elt in x_unique:
#         for idx in range(len(x)):
#             if x[idx] == x_elt:
#                 y_unique.append(y[idx])
#                 break
    x_pred = np.linspace(min(x), max(x))
    list_modes = []
    for _ in range(10):
        best_params = get_best_parameters_direct_log_logistic_from_gp_with_mle(x_unique, np.array(y_unique), sim_param,
                                                                               fit_type
                                                                               )
        params = best_params.params
        coef_det_gp = best_params.coef_det
        bic_score = best_params.bic_coef
        best_kernel = best_params.kernel
    #    params = get_best_fit_directly_from_samples(x_unique, y_unique, fit_type)
        if fit_type == "fisk":
            y_pred = fisk.pdf(x_pred, c=params[0], loc=params[1], scale=params[2])
            c = -params[0]
            alpha = params[2]
            mode = alpha * ((c - 1) / c + 1) ** (1 / c)
        elif fit_type == "log-normal":
            y_pred = lognorm.pdf(x_pred, s=params[0], loc=params[1], scale=params[2])
            sigma = params[0]
            mu = params[3]
            mode = np.exp(mu - sigma * sigma)
        list_modes.append(mode)
    mode = np.mean(list_modes)
    X = np.atleast_2d(x_unique).T
    y = np.array(y_unique).ravel()
    plot_predicted_gaussian_process(sim_param, X, y, best_params.gp, mode, num_points=30)
    print("Mode is {} for a bic score of {}".format(mode, bic_score))
    plt.figure()
    plt.plot(x_unique, y_unique, 'o', label="True")
    plt.plot(x_pred, y_pred, '--', label=fit_type)
    plt.xlabel("Batch Size")
    plt.ylabel("Correct Detection Probability")
    plt.title("Rho={}, Delta_Rho={}, T_change={}".format(rho_selected, delta_rho_selected, time_of_change))
    plt.savefig("Figures/BatchSizeFit/density_rho_{0:2.2f}_deltaRho_{1:2.2f}_tchange_{2:d}_{3}.png"
                .format(rho_selected, delta_rho_selected, int(time_of_change), fit_type))
    plt.legend()
    # plt.show()
    plt.close()
    output_params = ResultParams(params, coef_det_gp, mode, np.var(list_modes), bic_score, best_kernel)
    return output_params


def fit_conditional_correct_detection_at_base_level():
    """
    Fit the conditional probability of correct detection at the 0 level of aggregation
    For different rho, time of change and delta rho fit the probability of correct detection vs.  the batch size

    For each configuration, perform the best fit
    save the coefficients of the fit, the r^2 score and chi-square score.
    That's your training performance
    Validate that against previous data to obtain the validation scores.
    """
    output_file_name  ="best_fit.pkl"
    log_directory = "./Results/GLRT_ROSS/ARL_1/"
    log_file_name = log_directory + "select_detection_delay_test_log_07_10_20.bz2"
    data_df1 = load_power_df(log_file_name)
    results_df = pd.DataFrame(columns=['rho', 'delta_rho', "Time_of_Change",
                                     'c/mu', 'loc', 'scale',
                                     'R^2',
                                     'Batch Size', "Fit Type",
                                       'BIC', 'Kernel'
                                       ])

    # Open second log file
    log_file_name_2 = log_directory + "select_detection_delay_test_log_07_15_20.bz2"
    data_df2 = load_power_df(log_file_name_2)
    frames = [data_df1, data_df2]
    data_df = pd.concat(frames, axis=0, ignore_index=True, verify_integrity=True)
    # data_df = data_df[data_df["Batch Size"] < 100]
    time_of_change_list = list(data_df['Time of Change'].unique())
    rho_list = list(data_df["rho"].unique())
    # rho_list = [0.5]
    fit_type = "log-normal"  # either "fisk" or "log-normal
    for rho_selected in rho_list:
        modified_df = data_df[data_df["rho"] == rho_selected]
        for time_of_change in time_of_change_list:
            specific_df = modified_df[modified_df["Time of Change"] == time_of_change]
            for delta_rho in data_df["delta_rho"].unique():
                specified_df = specific_df[specific_df["delta_rho"] == delta_rho]
                print("For rho={}, delta_rho={} and T_change={}".format(rho_selected, delta_rho, time_of_change))
                simParam = SimulationConfigParams(rho_selected, delta_rho, time_of_change)
                results = find_best_correct_detection_prob(specific_df, simParam, fit_type)
                if fit_type == "log-normal":
                    values_to_add = {'rho': rho_selected, 'delta_rho': delta_rho, "Time_of_Change": time_of_change,
                                     'c/mu': results.params[0], 'loc': results.params[1], 'scale': results.params[2],
                                     'R^2': results.score,
                                     'Batch Size': results.mode, "Var Batch Size": results.var_mode,
                                     "Fit Type": fit_type,
                                     'BIC': results.bic, "Kernel": results.kernel
                                     }
                elif fit_type == "fisk":
                    values_to_add = {'rho': rho_selected, 'delta_rho': delta_rho, "Time_of_Change": time_of_change,
                                     'c/mu': results.params[0], 'loc': results.params[1], 'scale': results.params[2],
                                     'R^2': results.score,
                                     'Batch Size': results.mode, "Var Batch Size": results.var_mode,
                                     "Fit Type": fit_type,
                                     'BIC': results.bic, "Kernel": results.kernel
                                     }
                print(values_to_add)
                results_df = results_df.append(values_to_add, ignore_index=True)
            # break
        # break
    print(results_df.tail())
    results_df.to_pickle(output_file_name)
    fit_type = "fisk"  # either "fisk" or "log-normal
    for rho_selected in rho_list:
        modified_df = data_df[data_df["rho"] == rho_selected]
        for time_of_change in time_of_change_list:
            specific_df = modified_df[modified_df["Time of Change"] == time_of_change]
            for delta_rho in data_df["delta_rho"].unique():
                specified_df = specific_df[specific_df["delta_rho"] == delta_rho]
                print("For rho={}, delta_rho={} and T_change={}".format(rho_selected, delta_rho, time_of_change))
                simParam = SimulationConfigParams(rho_selected, delta_rho, time_of_change)
                results = find_best_correct_detection_prob(specific_df, simParam, fit_type)
                if fit_type == "log-normal":
                    values_to_add = {'rho': rho_selected, 'delta_rho': delta_rho, "Time_of_Change": time_of_change,
                                     'c/mu': results.params[0], 'loc': results.params[1], 'scale': results.params[2],
                                     'R^2': results.score,
                                     'Batch Size': results.mode, "Var Batch Size": results.var_mode,
                                     "Fit Type": fit_type,
                                     'BIC': results.bic, "Kernel": results.kernel
                                     }
                elif fit_type == "fisk":
                    values_to_add = {'rho': rho_selected, 'delta_rho': delta_rho, "Time_of_Change": time_of_change,
                                     'c/mu': results.params[0], 'loc': results.params[1], 'scale': results.params[2],
                                     'R^2': results.score,
                                     'Batch Size': results.mode, "Var Batch Size": results.var_mode,
                                     "Fit Type": fit_type,
                                     'BIC': results.bic, "Kernel": results.kernel
                                     }
                row_to_add = pd.Series(values_to_add)
                print(values_to_add)
                results_df = results_df.append(values_to_add, ignore_index=True)
    results_df.to_csv("best_gp_fit.csv")
    results_df.to_pickle(output_file_name)
    with pd.ExcelWriter("best_gp_fit.xlsx") as writer:
        log_normal_df = results_df[results_df["Fit Type"] == "log-normal"]
        fisk_df = results_df[results_df["Fit Type"] == "fisk"]
        log_normal_df.to_excel(writer, sheet_name="log-normal")
        fisk_df.to_excel(writer, sheet_name="fisk")
    print(results_df.tail())


if __name__ == "__main__":
    fit_conditional_correct_detection_at_base_level()

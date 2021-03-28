"""
  Change Point Detection part of thesis
  Validation of Kemal's results on autocorrelation of waiting times of Non-overlapping Batch Means for  M/M/1 queues

  2020-2021
"""
from math import pi, pow

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import seaborn as sns
from numpy import power
from scipy import optimize  # use for root finding

# import libraries to simulate NOBMs
# from simulate_arl_0 import obtain_batch_mean_wait_time
from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import simulate_deds_return_wait_times, simulate_ladder_point_process


def compute_mm1_mean(arrival_rate, service_rate):
    return (arrival_rate / service_rate) / (service_rate - arrival_rate)


def compute_mm1_var(arrival_rate, service_rate):
    rho = arrival_rate / service_rate
    return rho * (2 - rho) / ((service_rate - arrival_rate) * (service_rate - arrival_rate))


def get_correlation_wait_time(rho, lag):
    """
    Compute the autocorrelation between wait times
    :param rho: the traffic intensity
    :param lag: the lag of the autocorrelation
    :return: autocorrelation between wait times
    """
    a = (4 * rho) / ((1 + rho) * (1 + rho))
    f_fun_t = lambda t: pow(t, 1.5) * pow((a - t), 0.5) * pow(t, lag) / pow((1 - t), 3)
    c = pow(1 - rho, 3) * (1 + rho) / (2 * pi * pow(rho, 3) * (2 - rho))
    integral, error_integral = integrate.quad(f_fun_t, 0, a, epsrel=1e-10)
    return c * integral


def compute_batch_waiting_times_variance(m, arrival_rate, service_rate):
    """
    Compute the variance of the batch mean wait times
    :param m: the number of batches
    :param arrival_rate: lambda
    :param service_rate: mu
    """
    rho = arrival_rate / service_rate
    var_single_observation = compute_mm1_var(arrival_rate, service_rate)
    sum_corr = sum([(1 - i / m) * get_correlation_wait_time(rho, i) for i in range(1, m)])
    var = var_single_observation / m * (1 + 2 * sum_corr)
    return var


def compute_lag_one_waiting_times_autocorrelation(samples, service_rate, arrival_rate, batch_size):
    rho = arrival_rate / service_rate
    mean = compute_mm1_mean(arrival_rate, service_rate)
    var = compute_batch_waiting_times_variance(batch_size, arrival_rate, service_rate)
    last_n_1_elements = np.array(samples[1:]) - mean
    first_n_1_elements = np.array(samples[:-1]) - mean
    return np.dot(first_n_1_elements, last_n_1_elements) / (var * len(last_n_1_elements))


def compute_batch_lag_one_queue_length_autocorrelation(samples, service_rate, arrival_rate, batch_size):
    mean = arrival_rate * arrival_rate / ( service_rate * (service_rate - arrival_rate))
    last_n_1_elements = np.array(samples[1:]) - mean
    first_n_1_elements = np.array(samples[:-1]) - mean
    var = compute_batch_queue_length_variance(batch_size, arrival_rate, service_rate)
    return np.dot(first_n_1_elements, last_n_1_elements) / (var * len(last_n_1_elements))


def compute_batch_queue_length_variance(m, arrival_rate, service_rate):
    """
    Compute the variance of the batch mean wait times
    :param m: the number of batches
    :param arrival_rate: lambda
    :param service_rate: mu
    """
    pre_multiplier = arrival_rate * service_rate / ((arrival_rate - service_rate) * (arrival_rate - service_rate) * m)
    corr_sum = 0
    for i in range(1, m):
        corr_sum += (1 - i / m) * compute_lag_queue_length_autocorrelation(arrival_rate, service_rate, lag=i)
    post_multiplier = 1 + 2 * corr_sum
    return pre_multiplier * post_multiplier


def compute_lag_queue_length_autocorrelation(arrival_rate, service_rate, lag=1):
    return arrival_rate / service_rate + np.exp(- (service_rate - arrival_rate) ** 2 * (lag / arrival_rate))


def compute_a(rho):
    return 4.0 * rho / ((1 + rho) ** 2)


def compute_c(rho):
    return (1 - rho) ** 3 * (1 + rho) / (2 * pi * rho ** 3 * (2 - rho))


def f_function_t(t, rho):
    return power(t, 1.5) * power(compute_a(rho) - t, 0.5) / (pow(1 - t, 3))


def compute_numerator_integral(a_const, m, rho):
    integrated_func = lambda t: t / m * ((1 - t ** m) / (1 - t)) ** 2 * f_function_t(t, rho)
    return integrate.quad(integrated_func, 0, a_const, epsrel=1e-10)


def compute_denominator_integral(a_const, m, rho):
    integrated_func = lambda t: t * (pow(t, m) - m * t + m - 1) / (m * (1 - t) ** 2) * f_function_t(t, rho)
    return integrate.quad(integrated_func, 0, a_const, epsrel=1e-10)


# Compute autocorrelation using Kemal's equation
def compute_nobm_lag_autocorrelation(m, rho):
    c_const = compute_c(rho)
    a_const = compute_a(rho)
    integral_1, error_integral_1 = compute_numerator_integral(a_const, m, rho)
    num = c_const * integral_1
    integral_2, error_integral_2 = compute_denominator_integral(a_const, m, rho)
    den = 1 + 2 * c_const * integral_2
    # print("\t The bounds on the first error is : ", error_integral_1)
    # print("\t The bounds on the second error is : ", error_integral_2)
    return num / den


def estimate_lag_one_autocorrelation_from_sim(m, rho, mu, sim_type="DEDS", use_nobm=True):
    NUM_SIM = 2000
    LEN_SIM = 5000
    NUM_BATCHES = 100
    if use_nobm:
        LEN_SIM = max(LEN_SIM, m * NUM_BATCHES)
    # For NUM_SIM simulations
    start_time = 0
    end_time = LEN_SIM
    time_of_changes = [end_time]
    autocorrelations = []
    my_arrival_rates = [rho * mu, rho * mu]
    my_service_rates = [mu, mu]
    batch_size = m
    warm_up_time = int(0.1 * end_time)
    mean_wait_times = []
    var_wait_times = []
    for _ in range(NUM_SIM):
        # 1. Simulate wait times
        # 2. Create batches of size m and return the mean of the batches
        if sim_type == "DEDS":
            wait_times, wait_times_ts = simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates,
                                                                        time_of_changes, my_service_rates)
        else:
            wait_times, wait_times_ts = simulate_ladder_point_process(start_time, end_time, my_arrival_rates,
                                                                      time_of_changes, my_service_rates)
        if use_nobm:
            nobm_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts, batch_size)

            warm_up_index = int(0.1 * len(nobm_wait_times))
            # print("Number of wait times: {} and warm_up_index: {}".format(len(nobm_wait_times), warm_up_index))
            # 3. Compute autocorrelation
            # corr = sm.tsa.acf(nobm_wait_times[warm_up_index:], nlags=1, fft=True)
            #             corr = pd.Series(nobm_wait_times[warm_up_index:]).autocorr(lag=1)
            corr = compute_lag_one_waiting_times_autocorrelation(nobm_wait_times[warm_up_index:], mu, rho * mu, batch_size)
            mean_wait_times.append(np.mean(nobm_wait_times[warm_up_index:]))
            var_wait_times.append(np.var(nobm_wait_times[warm_up_index:]))
        else:
            start_index = int(0.1 * len(wait_times))
            #             corr = pd.Series(wait_times[start_index:]).autocorr(lag=1)
            corr = compute_lag_one_waiting_times_autocorrelation(wait_times[start_index:], mu, rho * mu, batch_size)
            mean_wait_times.append(np.mean(wait_times[start_index:]))
            var_wait_times.append(np.var(wait_times[start_index:]))
        # 4. add to avg_autocorrelation
        # print(corr)
        autocorrelations.append(corr)
    # divide avg_autocorrelation by NUM_SIM and return the mean
    print("Found mean {0:2.4f} and variance {1:2.4f}".format(np.mean(mean_wait_times), np.mean(var_wait_times)))
    standard_error = np.std(autocorrelations, dtype=np.float64, ddof=1) / np.sqrt(NUM_SIM)
    return np.mean(autocorrelations), np.median(autocorrelations), standard_error


def main():
    rho_vec = [0.25, 0.5, 0.75]
    batch_sizes = np.linspace(1, 50)
    # compute the autocorrelation using Kemal's formula
    linestyle = ["-", "--", "-."]
    for idx, rho in enumerate(rho_vec):
        autocorrelations = []
        for b in batch_sizes:
            print("rho={} and batch_size={}".format(rho, b))
            kemal_autocorr = compute_nobm_lag_autocorrelation(b, rho)
            autocorrelations.append(kemal_autocorr)
            # print("\tKemal obtains an autocorrelation of value: {0:2.5f}".format(kemal_autocorr))
            # my_autocorr = estimate_lag_one_autocorrelation_from_sim(b, rho)
            # print("\tThe Monte Carlo simulation returns {0:2.5f}.".format(my_autocorr))
        plt.plot(batch_sizes, autocorrelations, linestyle[idx], label="{}".format(rho))
    axes = plt.gca()
    axes.set_ylim([0.0, 1.0])
    plt.xlabel("Batch size (m)")
    plt.ylabel("Correlation(m, rho)")
    plt.savefig("Kemal_sim_results_confirmation.png", dpi=800)


def compute_expected_wait_time(rho, mu):
    return (1.0 / mu) / (1.0 - rho)


def compute_var_wait_time(rho, mu):
    lambda_val = rho * mu
    return rho * (2.0 - rho) / ((mu - lambda_val) * (mu - lambda_val))


def main_testing_simulations_off_autocorrelations_lag_1():
    """
    Compare the different simulations based on how close they are to Kemal's derived results.
    Use a lag of 1
    """
    # Compare teh autocorrelation of Kemal to the simulation estimation
    rho = 0.5
    mu = 1.0
    batch_size = 100
    print("\t\t*** Configuration ***\t\t")
    print("\t batch size={},\t rho={}, mu={}".format(batch_size, rho, mu))
    mean_wait_time = compute_expected_wait_time(rho, mu)
    var_wait_time = compute_var_wait_time(rho, mu)
    print("\t  Theoretical values -->\t E[W]={0:2.4f}, \t Var[W]={1:2.4f}".format(mean_wait_time, var_wait_time))
    kemal_autocorr = compute_nobm_lag_autocorrelation(batch_size, rho)
    print("\tKemal obtains an autocorrelation of value: {0:2.5f}".format(kemal_autocorr))
    deds_nobm, deds_nobmedian, deds_nobm_std = estimate_lag_one_autocorrelation_from_sim(batch_size, rho, mu,
                                                                                         "DEDS", True)
    print("\tThe DEDS simulation with batch returns mean: {0:2.5f}, median: {1:2.5f}, std: {2:2.5f}.".format(
        deds_nobm, deds_nobmedian, deds_nobm_std)
    )


def compare_and_plot_kemal_monte_carlo():
    rho_list = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, .8, 0.9]
    batch_sizes = [1, 5, 25, 50, 100, 150, 200]
    mu = 1
    df = pd.DataFrame(columns=rho_list)
    for batch in batch_sizes:
        kemal_auto_correlations = []
        monte_carlo_auto_correlations = []
        monte_carlo_auto_corr_error = []
        values_to_add = {}
        for rho in rho_list:
            kemal_auto_correlations.append(compute_nobm_lag_autocorrelation(batch, rho))
            ladder_nobm, _, autocorr_std = estimate_lag_one_autocorrelation_from_sim(batch, rho, mu, "ladder", True)
            monte_carlo_auto_correlations.append(ladder_nobm)
            monte_carlo_auto_corr_error.append(2 * autocorr_std)
            values_to_add[rho] = ladder_nobm
        row_to_add = pd.Series(values_to_add, name=batch)
        df = df.append(row_to_add)
        print("Batch of size ", batch)
        rms_error = np.sqrt(np.dot(np.array(kemal_auto_correlations) - np.array(monte_carlo_auto_correlations),
                                   np.array(kemal_auto_correlations) - np.array(monte_carlo_auto_correlations)))
        print("Root Mean Square error: {0:2.5f} \n".format(rms_error))
        plt.figure()
        plt.plot(rho_list, kemal_auto_correlations, 'r-', label="Theory")
        #        plt.plot(rho_list, monte_carlo_auto_correlations, 'bo', label="Monte Carlo")
        plt.errorbar(rho_list, monte_carlo_auto_correlations, monte_carlo_auto_corr_error, label="MonteCarlo")
        plt.xlabel(r"$\rho$")
        plt.ylabel("Lag 1 auto-correlation")
        plt.legend()
        plt.savefig("Kemal_autocorelation_validation_batch_{}".format(batch))
        plt.close()
    df.to_csv("EstimatedAutocorrelations.csv")


def find_batch_size(correlation, rho):
    autocorrelation_func = lambda batch: compute_nobm_lag_autocorrelation(batch, rho) - correlation
    try:
        sol = optimize.root_scalar(autocorrelation_func, bracket=[1, 40000])
    except ValueError:
        return None
    print("For a correlation:{0:2.2f} and a rho:{1:2.2f} found a batch of {2:2.4f} after {3:d} iterations".format(
        correlation, rho, sol.root, sol.iterations
    ))
    return sol.root


def extract_correlation_batch_size_rho_relationship(correlation_list, rho_list):
    """
        Extract a dataframe with data on different batch sizes for different correlations and intensity ratios
        :param correlation_list: a list of auto-correlation values (-1, 1)
        :param rho_list: list of traffic loads
        :param rho_list: a list of traffic intensity ratios or traffic loads
        :return: a dataframe
    """
    # let's create the dataframe
    # for each batch and rho compute the kemal autocorrelation, then store that in
    # let's add a new element row by row for each experiment
    df = pd.DataFrame()
    for auto_correlation in correlation_list:
        for rho in rho_list:
            print("For a correlation: {0:2.2f} and a rho: {1:2.2f} ".format(auto_correlation, rho))
            batch = find_batch_size(auto_correlation, rho)
            if batch:
                results_dict = {"batch_size": batch, "rho": rho, "correlation": auto_correlation}
                row_to_add = pd.Series(results_dict)
                df = df.append(row_to_add, ignore_index=True)
    return df


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c


def obtain_exponential_fit(df):
    """
        Ingest the dataframe and obtain an exponential fit
    """
    corr_list = list(df["correlation"].unique())
    output_df = pd.DataFrame()
    for corr in corr_list:
        reduced_df = df[df["correlation"] == corr]
        x_rho = reduced_df["rho"]
        y_batch = reduced_df["batch_size"]
        popt, pcov = optimize.curve_fit(exponential_func, x_rho, y_batch)
        results_dict = {"correlation": corr, "var_x": pcov[0][0], "var_y": pcov[1][1],
                        "a": popt[0], "b": popt[1], "c": popt[2]}
        row_to_add = pd.Series(results_dict)
        output_df = output_df.append(row_to_add, ignore_index=True)
    return output_df


def extract_and_plot_correlation_relationship():
    # instead we find the batch that gives us a desired correlation
    # desired_correlations = np.linspace(0.01, 0.45, num=10)
    desired_correlations = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rho_list = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, .8, 0.9]
    # rho_list = [0.25, 0.5, 0.75]
    info_df = extract_correlation_batch_size_rho_relationship(desired_correlations, rho_list)

    # Generate exponential fit
    output_df = obtain_exponential_fit(info_df)
    output_df.to_csv("KemalCorrelationDF.csv")

    # plot the results using seaborn
    # plot batch size vs. rho for different correlations (color or gradient)
    min_cor_val = info_df["correlation"].min()
    max_cor_val = info_df["correlation"].max()
    f, ax = plt.subplots()
    splot = sns.scatterplot(data=info_df, x="rho", y="batch_size", hue="correlation", hue_norm=(min_cor_val,
                                                                                                max_cor_val))
    splot.set(yscale="log")
    plt.savefig("KemalCorrelationChoice_batchSize_rho.png", dpi=900)
    plt.close(f)

    f, ax = plt.subplots()
    g = sns.relplot(data=info_df, x="rho", y="batch_size", col="correlation", kind="line", col_wrap=3)
    g.set(yscale="log")
    plt.savefig("KemalCorrelationChoiceFacet.png", dpi=900)
    plt.close()

    for _, row in output_df.iterrows():
        f, ax = plt.subplots()
        local_exp_func = lambda x: exponential_func(x, row["a"], row["b"], row["c"])
        predicted_batches = list(map(local_exp_func, rho_list))
        corr = row["correlation"]
        reduced_df = info_df[info_df["correlation"] == corr]
        plt.plot(reduced_df["rho"], reduced_df["batch_size"], 'rx', label="Actual")
        plt.plot(rho_list, predicted_batches, "b-.", label="Exp. Fit")
        plt.legend()
        plt.xlabel(r"$\rho$")
        plt.ylabel("Batch Size")
        plt.title("Correlation of {0:2.3f}".format(corr))
        plt.savefig("KemalCorrelation_rho_{}.png".format(int(corr * 100)))
        plt.close(f)


if __name__ == "__main__":
    # main()
    # main_testing_simulations_off_autocorrelations_lag_1()
    #     extract_and_plot_correlation_relationship()
    compare_and_plot_kemal_monte_carlo()

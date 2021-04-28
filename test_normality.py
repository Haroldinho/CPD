"""
            File to check i.i.d normal assumption of NOBM for waiting times, queue length and waiting times in queue
Check Normality Assumptions of the NOBM
Verify that the autocorrelation gets small as the batch size increase and that the batches are approximately normal for:
    - waiting times
    - waiting times in queue
    - queue lengths

# 3/19/2021 Refactor code from main_tests.py to test_normality.py for added clarity.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels

from Kemal_autocorrelation_validation import compute_lag_one_waiting_times_autocorrelation, \
    compute_batch_lag_one_queue_length_autocorrelation
from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import SingleServerMarkovianQueue, LadderPointQueue, simulate_deds_return_wait_times, \
    simulate_deds_return_age, simulate_deds_return_queue_length


def compute_autocorrelation(x, length=20):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])


def repeat_decorator(*argso, **kwargso):
    num_iter = kwargso['reps']

    def real_decorator(func):
        def wrapper(*args, **kwargs):
            my_mean = []
            for _ in range(num_iter):
                my_mean.append(func(*args, **kwargs))
            return np.mean(my_mean)

        return wrapper

    return real_decorator


def simulate_single_server_queue_and_compute_ljung_box_score(start_time, end_time, arrival_rate, service_rate,
                                                             batch_sizes, num_runs,
                                                             observation_type="wait_times",
                                                             include_autocorrelation_plots=False
                                                             ):
    my_arrival_rates = [arrival_rate, arrival_rate]
    my_service_rates = [service_rate, service_rate]
    time_of_changes = [0]
    autocorrelation_scores_lag_1 = {batch: [] for batch in batch_sizes}
    autocorrelation_scores_lag_5 = {batch: [] for batch in batch_sizes}
#     num_passed_scores_lag1 = {batch: 0 for batch in batch_sizes}
#     num_passed_scores_lag5 = {batch: 0 for batch in batch_sizes}
#     success_rate_lag_1_dictionary = {batch: 0 for batch in batch_sizes}
#     success_rate_lag_5_dictionary = {batch: 0 for batch in batch_sizes}
    mean_lag_1_autocorrelation = {batch: 0 for batch in batch_sizes}
    mean_lag_5_autocorrelation = {batch: 0 for batch in batch_sizes}
    for run in range(num_runs):
        # 1. Simulate waiting times or age of a process
        if observation_type == "wait_times":
            orig_time_samples, orig_time_stamps = simulate_deds_return_wait_times(start_time, end_time,
                                                                                  my_arrival_rates, time_of_changes,
                                                                                  my_service_rates)
        elif observation_type == "age":
            orig_time_samples, orig_time_stamps = simulate_deds_return_age(start_time, end_time,
                                                                           my_arrival_rates, time_of_changes,
                                                                           my_service_rates)
        elif observation_type == "queue":
            orig_time_samples, orig_time_stamps = simulate_deds_return_queue_length(start_time, end_time,
                                                                                    my_arrival_rates, time_of_changes,
                                                                                    my_service_rates)

        if run % 100 == 0:
            print(
                "Simulating {} {} from t={} to t={}.".format(len(orig_time_stamps), observation_type, min(orig_time_stamps),
                                                             max(orig_time_stamps)))
        for batch_size in batch_sizes:
            # 2. Batch them into groups returning the mean of the batches and their centers.
            batch_mean_orig_time_samples, batch_centers = create_nonoverlapping_batch_means(orig_time_samples,
                                                                                            orig_time_stamps,
                                                                                            batch_size=batch_size)
            # Optional: Plot PACF and ACF
            if run == 1:
                if include_autocorrelation_plots:
                    plot_autocorrelation(batch_mean_orig_time_samples, batch_size, observation_type)

            # 3. Perform Ljung-box for independence
#             lbvalue, pvalue = acorr_ljungbox(batch_mean_orig_time_samples, lags=5, return_df=False)
#             print("Ljung-Box test statistic={}, pvalue={}".format(lbvalue, pvalue))
#             if pvalue[0] < 0.05:
#                 # we fail to reject the null at the 0.01 level
#                 # the probability of incorrectly rejecting the null may be only 7% for the POP
#                 # https://blog.minitab.com/en/adventures-in-statistics-2/how-to-correctly-interpret-p-values
#                 num_passed_scores_lag1[batch_size] += 1
#             if pvalue[4] < 0.05:
#                 num_passed_scores_lag5[batch_size] += 1
            # 4. Compute autocorrelation moved by 1
            if observation_type == "wait_times":
                lag_1_autocorrelation = compute_lag_one_waiting_times_autocorrelation(batch_mean_orig_time_samples,
                                                                                      service_rate,
                                                                                      arrival_rate, batch_size)
            elif observation_type == "queue":
                lag_1_autocorrelation = compute_batch_lag_one_queue_length_autocorrelation(batch_mean_orig_time_samples,
                                                                                           service_rate, arrival_rate,
                                                                                           batch_size)
            else:
                lag_1_autocorrelation = compute_autocorrelation(batch_mean_orig_time_samples, 1)
            autocorrelation_scores_lag_1[batch_size].append(lag_1_autocorrelation)
            autocorrelation_scores_lag_5[batch_size].append(compute_autocorrelation(batch_mean_orig_time_samples, 5))
    for my_batch_size in batch_sizes:
        print(f"\tFor a batch of size {my_batch_size}")
#         num_of_failed_rejections = num_passed_scores_lag1[my_batch_size]
#         success_rate_lag_1_dictionary[my_batch_size] = num_of_failed_rejections / float(num_runs)
#         success_rate_lag_5_dictionary[my_batch_size] = num_passed_scores_lag5 / float(num_runs)
        mean_lag_1_autocorrelation[my_batch_size] = np.mean(autocorrelation_scores_lag_1[my_batch_size])
        mean_lag_5_autocorrelation[my_batch_size] = np.mean(autocorrelation_scores_lag_5[my_batch_size])
#         print(f"\tOut of {num_runs} tests, {num_of_failed_rejections} are independent")
        print("\tThe lag-1 autocorrelation is {} and the lag-5 autocorrelation is {}. \n".format(
            mean_lag_1_autocorrelation[my_batch_size], mean_lag_5_autocorrelation[my_batch_size]
        ))
    return mean_lag_1_autocorrelation, mean_lag_5_autocorrelation


def simulate_single_server_queue_and_compute_ad_wilk_score(start_time, end_time, my_arrival_rate, my_service_rate,
                                                           batch_sizes, num_runs,
                                                           observation_type="wait_times",
                                                           ):
    my_arrival_rates = [my_arrival_rate, my_arrival_rate]
    my_service_rates = [my_service_rate, my_service_rate]
    time_of_changes = [0]
    anderson_darling_successes = {batch: 0 for batch in batch_sizes}
    shapiro_wilk_successes = {batch: 0 for batch in batch_sizes}
    for run in range(num_runs):
        print(f"Run {run}/{num_runs}")
        # 1. Simulate waiting times or age of a process
        # discard 10% of the samples
        if observation_type == "wait_times":
            orig_time_samples, orig_time_stamps = simulate_deds_return_wait_times(start_time, end_time,
                                                                                  my_arrival_rates, time_of_changes,
                                                                                  my_service_rates, 0.1)
        elif observation_type == "age":
            orig_time_samples, orig_time_stamps = simulate_deds_return_age(start_time, end_time,
                                                                           my_arrival_rates, time_of_changes,
                                                                           my_service_rates, 0.1)
        elif observation_type == "queue":
            orig_time_samples, orig_time_stamps = simulate_deds_return_queue_length(start_time, end_time,
                                                                                    my_arrival_rates, time_of_changes,
                                                                                    my_service_rates, 0.1)

        print("Simulating {} wait times from t={} to t={}.".format(len(orig_time_stamps), min(orig_time_stamps),
                                                                   max(orig_time_stamps)))
        assert (max(orig_time_stamps) == orig_time_stamps[-1])
        for batch_size in batch_sizes:
            # 2. Batch them into groups returning the mean of the batches and their centers.
            batch_mean_orig_time_samples, batch_centers = create_nonoverlapping_batch_means(orig_time_samples,
                                                                                            orig_time_stamps,
                                                                                            batch_size=batch_size)
            print("Recorded {} batches from t={} to t={}".format(len(batch_mean_orig_time_samples),
                                                                 batch_centers[0], batch_centers[-1]))

            # 3. Test the batch means for normality
            # 3.1     Q-Q plot  Kolmogorov Smirnov
            if run == 1:
                scipy.stats.probplot(batch_mean_orig_time_samples, plot=plt)
                plt.title("Batch {} Q-Q plot [batch size={}] ".format(observation_type, batch_size))
                plt.savefig("./Figures/NormalityTest/QQPlots/QQ_n_{}_{}_{}.png".format(batch_size,
                                                                                       my_arrival_rate,
                                                                                       observation_type),
                            dpi=500)
                plt.close()
            # 3.2      Anderson-Darling Test
            _, did_ad_test_pass = perform_anderson_darling_test(batch_mean_orig_time_samples)
            anderson_darling_successes[batch_size] += int(did_ad_test_pass)
            # 3.2      Anderson-Darling Test
            _, did_wilk_test_pass = perform_shapiro_wilk_test(batch_mean_orig_time_samples)
            shapiro_wilk_successes[batch_size] += int(did_wilk_test_pass)
    anderson_darling_success_rates = {my_batch: success / float(num_runs) for my_batch, success
                                      in anderson_darling_successes.items()}
    shapiro_wilk_success_rates = {my_batch: success / float(num_runs) for my_batch, success
                                  in shapiro_wilk_successes.items()}
    return anderson_darling_success_rates, shapiro_wilk_success_rates


def perform_anderson_darling_test(batch_data):
    ad_stat, ad_crit, ad_sign_level = scipy.stats.anderson(batch_data)
    if ad_stat <= ad_crit[2]:
        print("\n\t For alpha=0.05, stat({}) is lower than critical value({}) so we reject the test (AD)".format(
            ad_stat, ad_crit[2]))
        test_passed = False
    else:
        print("\n\t We failed to reject normality assumption using an Anderson Darling Test")
        test_passed = True
    print("\t At level: {}".format(ad_sign_level[2]))
    return ad_stat, test_passed


def perform_shapiro_wilk_test(batch_data):
    shapiro_stat, shapiro_p_value = scipy.stats.shapiro(batch_data)
    p_value = 0.05
    if shapiro_p_value <= p_value:
        print("\n\t For alpha=0.05, stat({}) is lower than critical value({}) so we reject the test (SW)".format(
            shapiro_stat, shapiro_p_value))
        test_passed = False
    else:
        print("\n\t We failed to reject normality assumption using a Shapiro-Wilk Test")
        test_passed = True
    return shapiro_p_value, test_passed


@repeat_decorator(reps=5)
def simulate_ladder_point_process_and_compute_correlation(start_time, end_time, my_arrival_rates, time_of_changes,
                                                          my_service_rates, batch_size):
    ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                            time_of_changes)
    wait_times = ladder_point_creator.simulate_ladder_point_process()
    recorded_times = ladder_point_creator.report_wait_record_times()

    print("Simulating {} wait times from t={} to t={}.".format(len(recorded_times), min(recorded_times),
                                                               max(recorded_times)))
    assert (max(recorded_times) == recorded_times[-1])
    # 2. Batch them into groups returning the mean of the batches and their centers.
    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, recorded_times,
                                                                             batch_size=batch_size)
    print("Recorded {} batches from t={} to t={}".format(len(batch_mean_wait_times),
                                                         batch_centers[0], batch_centers[-1]))
    correlation = np.correlate(batch_mean_wait_times, batch_mean_wait_times)

    return correlation


def test_for_normality(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates, batch_size):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>  Multiple Repetitions  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>> Anderson-Darling Test <<<<<<<<<<<<<<<<<<<<
    ad_crit = [0.572, 0.651, 0.782, 0.912, 1.084]
    ad_stat = simulate_single_server_queue_and_compute_ad_wilk_score(start_time, end_time, my_arrival_rates,
                                                                     time_of_changes, my_service_rates, batch_size, 20)
    print("After 5 simulations, for a batch size of {} points, mean AD_score={}".format(batch_size, ad_stat))
    if ad_stat > ad_crit[2]:
        print("\n\t For alpha=0.05, stat({}) is greater than critical value({}) so we reject the test".format(
            ad_stat, ad_crit[2]))
    else:
        print("\n\t We failed to reject normality assumption using an Anderson Darling Test")


def plot_autocorrelation(batch_mean_wait_times, batch_size, observation_type="wait_times", simulation_type="DEDS"):
    # PLOT ACF
    statsmodels.graphics.tsaplots.plot_acf(batch_mean_wait_times)
    plt.title('ACF [batch size={}]'.format(batch_size))
    plt.savefig('./Figures/{}_{}_SIM_acf_{}.png'.format(simulation_type, observation_type, batch_size))
    #    plt.show()

    # PLOT PACF
    statsmodels.graphics.tsaplots.plot_pacf(batch_mean_wait_times)
    plt.title('PACF [batch size={}]'.format(batch_size))
    plt.savefig('./Figures/CorrelationTest/{}_{}_SIM/pacf_{}.png'.format(simulation_type, observation_type, batch_size))


def test_wait_times_batch_means():
    """
    Verify that the batch means algorithm produces uncorrelated and approximatively normal batch wait times
    """
    # 1. Generate 100,000 wait times
    start_time = 0
    service_rate = 5
    arrival_rate = 4
    batch_size = 250
    num_departures = 100000
    end_time = 1.5 * num_departures / np.abs(service_rate - arrival_rate)
    my_arrival_rates = [arrival_rate]
    my_service_rates = [service_rate]
    time_of_changes = [0]

    # Single Run code
    simulation_type = "DEDS"
    if simulation_type == "DEDS":
        deds_sim = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                              time_of_changes)
        wait_times = deds_sim.simulate_deds_process(warm_up=30)
        recorded_times = deds_sim.get_recorded_times()
        glrt_threshold_t = 7
    elif simulation_type == "LADDER":
        ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                                my_service_rates,
                                                time_of_changes)
        wait_times = ladder_point_creator.simulate_ladder_point_process()
        recorded_times = ladder_point_creator.report_wait_record_times()

    print("Simulating {} wait times from t={} to t={}.".format(len(recorded_times), min(recorded_times),
                                                               max(recorded_times)))
    assert (max(recorded_times) == recorded_times[-1])
    # 2. Batch them into groups returning the mean of the batches and their centers.
    batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, recorded_times,
                                                                             batch_size=batch_size)
    print("Recorded {} batches from t={} to t={}".format(len(batch_mean_wait_times),
                                                         batch_centers[0], batch_centers[-1]))
    #
    # 3. Test the batch means for normality
    #       Q-Q plot  Kolmogorov Smirnov
    scipy.stats.probplot(batch_mean_wait_times, plot=plt)
    plt.title("Batch Wait Times Q-Q plot [batch size={}] ".format(batch_size))
    plt.savefig("./Figures/Wait_Times_{}_QQ_n_{}.png".format(simulation_type, batch_size), dpi=500)
    plt.show()
    #       Anderson-Darling Test
    ad_stat, ad_crit, ad_sign_level = scipy.stats.anderson(batch_mean_wait_times)
    print("Anderson Darling Test [Batch Size={}]".format(batch_size))
    print("\t Found Statistic: {}".format(ad_stat))
    for i in range(len(ad_crit)):
        print("\t alpha={}, crit_val={}".format(ad_sign_level[i], ad_crit[i]))
    #
    # 4. Test the batch means for independence
    #       lack of correlation np.corrcoef
    #       from statsmodels.graphics.tsaplots import plot_acf
    #       plot_pacf and plot_acf
    r_coef = np.corrcoef(batch_mean_wait_times)
    correlation = np.correlate(batch_mean_wait_times, batch_mean_wait_times)
    print("\nFor [batch-size={}], Obtained correlation coefficients: ".format(batch_size, r_coef))
    print("Obtained convolved correlation: {}".format(correlation))


def test_normality(batch_sizes, observation_type="wait_times"):
    """
    Verify the normality assumption for different arrival rates and service rates
    3 steps
    Q-Q plot
    Anderson Darling test (Specification of Kolmogorov-Smirnov to Gaussian distribution)
    Shapiro-Wilk Test
    """
    start_time = 0
    end_time = 1e5
    arrival_rates = [0.25, 0.5, 0.75, 0.9]
    my_service_rate = 1.0
    normality_success_rates_list = []
    num_runs = 100
    for my_arrival_rate in arrival_rates:
        print("ARRIVAL RATE of {}".format(my_arrival_rate))
        anderson_darling_dic, shapiro_wilk_dic = simulate_single_server_queue_and_compute_ad_wilk_score(
            start_time, end_time, my_arrival_rate, my_service_rate, batch_sizes, num_runs, observation_type)
        crt_arrival_rates = [my_arrival_rate] * len(anderson_darling_dic)
        anderson_darling_results = [val for key, val in anderson_darling_dic.items()]
        shapiro_wilk_results = [val for key, val in shapiro_wilk_dic.items()]
        normality_success_rates_df = pd.DataFrame(
            {
                "Batch Size": batch_sizes,
                "Arrival Rates": crt_arrival_rates,
                "Anderson Darling Rates": anderson_darling_results,
                "Shapiro Wilk Rates": shapiro_wilk_results
            }
        )
        normality_success_rates_list.append(normality_success_rates_df)
    # combine the p values for all the tests
    normality_df = pd.concat(normality_success_rates_list, axis=0)
    # save it to a file
    normality_df.to_csv(f"Results/NormalityResults/normality_hypothesis_test_{observation_type}.csv")


def test_independence(batch_sizes, full_corr_df=[]):
    """
    Ljung-box test for independence
    H0: The data are independently distributed
    (i.e. the correlations in the population from which the sample is taken are 0,
    so that any observed correlations in the data result from randomness of the sampling process).
    Ha: The data are not independently distributed; they exhibit serial correlation.
    source:https://en.wikipedia.org/wiki/Ljung%E2%80%93Box_test
    """
    start_time = 0
    end_time = 1e5
    arrival_rates = [0.25, 0.5, 0.75, 0.9]
    my_service_rate = 1.0
    num_runs = 50
    observation_types = ["wait_times", "age", "queue"]
    for observation_type in observation_types:
        for my_arrival_rate in arrival_rates:
            print(f"For an arrival rate of {my_arrival_rate}")
            mean_lag1_autocorr, mean_lag5_autocorr = simulate_single_server_queue_and_compute_ljung_box_score(
                start_time, end_time, my_arrival_rate, my_service_rate, batch_sizes, num_runs, observation_type, False
            )
            crt_arrival_rates = [my_arrival_rate] * len(batch_sizes)
#             correlation_results_lag_1 = [corr for k, corr in correlation_test_lag1_dic.items()]
#             correlation_results_lag_5 = [corr for k, corr in correlation_test_lag5_dic.items()]
            observation_col = [observation_type] * len(batch_sizes)
            correlation_df = pd.DataFrame({
                "Batch Size": batch_sizes,
                "Observation": observation_col,
                "Arrival Rate": crt_arrival_rates,
#                 "Lag 1 Success Rate": correlation_results_lag_1,
                "Lag 1 Autocorrelation": [autocorr for key, autocorr in mean_lag1_autocorr.items()],
#                 "Lag 5 Success Rate": correlation_results_lag_5,
                "Lag 5 Autocorrelation": [autocorr for k, autocorr in mean_lag5_autocorr.items()]
            })
            full_corr_df.append(correlation_df)

    corr_df = pd.concat(full_corr_df, axis=0)
    corr_df.to_csv("Results/NormalityResults/LjungBoxResults.csv")
    return corr_df


if __name__ == "__main__":
    # Batch sizes to consider
    my_test_batch_sizes = [100, 150, 200, 500, 1000, 2000]
    for observation_type in ["wait_times", "age", "queue"]:
        test_normality(my_test_batch_sizes, observation_type)
    # test_independence(my_test_batch_sizes)

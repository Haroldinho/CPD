"""
            File to check i.i.d normal assumption of NOBM for waiting times, queue length and waiting times in queue

# 3/19/2021 Refactor code from main_tests.py to test_normality.py for added clarity.
TODO: Replace Anderson Darling by shapiro-wilk
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels

from batch_means_methods import create_nonoverlapping_batch_means
from generate_m_m_1_processes import SingleServerMarkovianQueue, LadderPointQueue


def repeat_decorator(*argso, **kwargso):
    num_iter = kwargso['reps']

    def real_decorator(func):
        def wrapper(*args, **kwargs):
            mean = []
            for _ in range(num_iter):
                mean.append(func(*args, **kwargs))
            return np.mean(mean)

        return wrapper

    return real_decorator


@repeat_decorator(reps=5)
def simulate_ladder_point_process_and_compute_ad_score(start_time, end_time, my_arrival_rates, time_of_changes,
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

    # 3. Test the batch means for normality
    #       Q-Q plot  Kolmogorov Smirnov
    scipy.stats.probplot(batch_mean_wait_times, plot=plt)
    plt.title("Batch Wait Times Q-Q plot [batch size={}] ".format(batch_size))
    plt.savefig("./Figures/Wait_Times_QQ_n_{}.png".format(batch_size), dpi=500)
    plt.show()
    #       Anderson-Darling Test
    return perform_anderson_darling_test(batch_mean_wait_times)


def perform_anderson_darling_test(batch_data):
    ad_stat, ad_crit, ad_sign_level = scipy.stats.anderson(batch_data)
    if ad_stat > ad_crit[2]:
        print("\n\t For alpha=0.05, stat({}) is greater than critical value({}) so we reject the test (AD)".format(
            ad_stat, ad_crit[2]))
    else:
        print("\n\t We failed to reject normality assumption using an Anderson Darling Test")
    return ad_stat


def perform_shapiro_wilk_test(batch_data):
    shapiro_stat, shapiro_crit, shapiro_sign_level = scipy.stats.shapiro(batch_data)
    if shapiro_stat > shapiro_crit[2]:
        print("\n\t For alpha=0.05, stat({}) is greater than critical value({}) so we reject the test (SW)".format(
            shapiro_stat, shapiro_crit[2]))
    else:
        print("\n\t We failed to reject normality assumption using a Shapiro-Wilk Test")
    return shapiro_stat


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
    ad_stat = simulate_ladder_point_process_and_compute_ad_score(start_time, end_time, my_arrival_rates,
                                                                 time_of_changes, my_service_rates, batch_size)
    print("After 5 simulations, for a batch size of {} points, mean AD_score={}".format(batch_size, ad_stat))
    if ad_stat > ad_crit[2]:
        print("\n\t For alpha=0.05, stat({}) is greater than critical value({}) so we reject the test".format(
            ad_stat, ad_crit[2]))
    else:
        print("\n\t We failed to reject normality assumption using an Anderson Darling Test")


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

    batch_mean_wait_times = np.array(batch_mean_wait_times)
    # PLOT ACF
    statsmodels.plot_acf(batch_mean_wait_times)
    plt.title('ACF [batch size={}]'.format(batch_size))
    plt.savefig('./Figures/{}_SIM/acf_{}.png'.format(simulation_type, batch_size))
    #    plt.show()

    # PLOT PACF
    statsmodels.plot_pacf(batch_mean_wait_times)
    plt.title('PACF [batch size={}]'.format(batch_size))
    plt.savefig('./Figures/{}_SIM/pacf_{}.png'.format(simulation_type, batch_size))

#    plt.show()

#    lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(batch_mean_wait_times,lags=5)
#    print("Ljung-Box test statistic={}, pvalue={}".format(lbvalue, pvalue))

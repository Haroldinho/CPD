def simulate_joint_change_points(data_df: pd.DataFrame, rho: List[float], delta_rhos: List[float],
                                 arr_rate_0: float, num_runs: int, start_time: float, end_time: float,
                                 my_service_rates: List[float],
                                 batch_size: List[int], power_delay_log: float, cpm_func,
                                 age_type="median"):
    """
    This code is to use the implementation of CPM in R directly from R using rpy2
    :param data_df: dataframe that will contain the performance characteristics of the test
    :param rho: list of intensity ratios
    :param delta_rhos: list of changes in intensity ratio
    :param arr_rate_0: initial arrival rate
    :param num_runs: number of runs
    :param start_time: start time of the sim 0 by default
    :param end_time: end time of the sim
    :param my_service_rates:
    :param my_thresholds:
    :param batch_size:
    :param cpm_func: R wrapper function to cpm
    :param power_delay_log: used to save data in between runs
    :param age_type: whether we use the mean or the median of the processes in the queue
    :return: dataframe of the data_df
    """
    # Look at the change given positive age, queue, wait
    changepoint_age_pos_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_pos_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_pos_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_pos_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_pos = {delta_rho: 0 for delta_rho in delta_rhos}
    changepoint_age_neg_queue_neg_wait_neg = {delta_rho: 0 for delta_rho in delta_rhos}

    tn_tn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}

    fp_tn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}

    fn_tp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}

    fn_fp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_queue_wait = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_age_queue = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_age_wait = {delta_rho: 0 for delta_rho in delta_rhos}

    tp_tp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_tn_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tp_fn_fn = {delta_rho: 0 for delta_rho in delta_rhos}

    fp_tp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_tn_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fp_fn_fn = {delta_rho: 0 for delta_rho in delta_rhos}

    fn_tp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_tn_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    fn_fn_fn = {delta_rho: 0 for delta_rho in delta_rhos}

    tn_tp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fp_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_tn_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_tp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_fp = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_tn = {delta_rho: 0 for delta_rho in delta_rhos}
    tn_fn_fn = {delta_rho: 0 for delta_rho in delta_rhos}
    disregard_frac = 0.05
    effective_sample_time = disregard_frac * end_time
    # Distribution of time till false detection
    # just keep a list of all the detection times when you have a false positive
    for run_idx in range(num_runs):
        print(f"Run {run_idx} of {num_runs}")
        for delta_rho in delta_rhos:
            arr_rate_1 = arr_rate_0 * (1 + delta_rho)
            my_arrival_rates = [arr_rate_0, arr_rate_1]
            if delta_rho == 0.0:
                time_of_change = float('inf')
            else:
                time_of_change = generate_random_change_point_time(end_time, effective_sample_time)
            time_of_changes = [-1, time_of_change]
            queue_lengths, queue_length_times, mean_age_times, recording_times, wait_times, departure_times = \
                simulate_deds_return_age_queue_wait(start_time, end_time, my_arrival_rates,
                                                    time_of_changes, my_service_rates)

            age_of_customers, age_times_ts = disregard_by_length_of_interval(mean_age_times, recording_times, end_time,
                                                                             disregard_frac)
            wait_times, wait_times_ts = disregard_by_length_of_interval(wait_times, departure_times, end_time,
                                                                        disregard_frac)
            queue_lengths, queue_lengths_ts = disregard_by_length_of_interval(queue_lengths, queue_lengths_ts,
                                                                              end_time,
                                                                              disregard_frac)
            batch_mean_ages, batch_centers = create_nonoverlapping_batch_means(age_of_customers, age_times_ts,
                                                                               batch_size=batch_size)
            rbatch_mean_ages = FloatVector(batch_mean_ages)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_ages)
            r_estimated_changepoint_age_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_ages)
            batch_mean_wait_times, batch_centers = create_nonoverlapping_batch_means(wait_times, wait_times_ts,
                                                                                     batch_size=batch_size)
            rbatch_mean_wait_times = FloatVector(batch_mean_wait_times)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_wait_times)
            r_estimated_changepoint_wait_times_index = cpm_func.Detect_r_cpm_GaussianChangePoint(batch_mean_wait_times)
            batch_queue_lengths, batch_centers = create_nonoverlapping_batch_means(queue_lengths,
                                                                                   queue_lengths_ts,
                                                                                   batch_size=batch_size)
            rbatch_mean_queue_lengths = FloatVector(batch_queue_lengths)
            r.assign('remote_batch_mean_wait_times', rbatch_mean_queue_lengths)
            r_estimated_changepoint_queue_index = cpm_func.Detect_r_cpm_GaussianChangePoint(rbatch_mean_queue_lengths)
            estimated_changepoint_age_idx = r_estimated_changepoint_age_index[0]
            estimated_changepoint_queue_idx = r_estimated_changepoint_queue_index[0]
            estimated_changepoint_wait_times_idx = r_estimated_changepoint_wait_times_index[0]
            if np.isinf(time_of_change):  # no changepoint, either we have a FP or a TN
                if estimated_changepoint_age_idx > 0:
                    if estimated_changepoint_queue_idx > 0:
                        fp_fp_age_queue[delta_rho] += 1

                    else:
                        fp_tn_age_queue[delta_rho] += 1
                    if estimated_changepoint_wait_times_idx > 0:
                        fp_fp_age_wait[delta_rho] += 1
                    else:
                        fp_tn_age_wait[delta_rho] += 1

                else:
                    if estimated_changepoint_queue_idx > 0:
                        tn_fp_age_queue[delta_rho] += 1
                    else:
                        tn_tn_age_queue[delta_rho] += 1
                    if estimated_changepoint_wait_times_idx > 0:
                        tn_fp_age_wait[delta_rho] += 1
                    else:
                        tn_tn_age_wait[delta_rho] += 1
                if estimated_changepoint_wait_times_idx > 0:
                    if estimated_changepoint_queue_idx > 0:
                        fp_fp_queue_wait[delta_rho] += 1
                    else:
                        fp_tn_queue_wait[delta_rho] += 1

                else:
                    if estimated_changepoint_queue_idx > 0:
                        tn_fp_queue_wait[delta_rho] += 1
                    else:
                        tn_tn_queue_wait[delta_rho] += 1
            else:
                dd_age = batch_centers[estimated_changepoint_age_idx - 1] if (
                        estimated_changepoint_age_idx > 0) else np.nan
                dd_queue = batch_centers[estimated_changepoint_age_idx - 1] if (
                        estimated_changepoint_queue_idx > 0) else np.nan
                dd_wait = batch_centers[estimated_changepoint_age_idx - 1] if (
                        estimated_changepoint_wait_times_idx > 0) else np.nan
                if not np.isnan(dd_age):
                    if not np.isnan(dd_queue):
                        if dd_age >= 0:
                            if dd_queue >= 0:
                                tp_tp_age_queue[delta_rho] += 1
                            else:
                                tp_fp_age_queue[delta_rho] += 1
                        else:
                            if dd_queue >= 0:
                                fp_tp_age_queue[delta_rho] += 1
                            else:
                                fp_fp_age_queue[delta_rho] += 1
                    else:  # queue is fn
                        if dd_age >= 0:
                            tp_fn_age_queue[delta_rho] += 1
                        else:
                            fp_fn_age_queue[delta_rho] += 1
                else:  # dd_age is fn
                    if not np.isnan(dd_queue):
                        if dd_queue >= 0:
                            fn_tp_age_queue[delta_rho] += 1
                        else:
                            fn_fp_age_queue[delta_rho] += 1
                    else:  # dd_queue is fn
                        fn_fn_age_queue += 1

                if not np.isnan(dd_age):
                    if not np.isnan(dd_wait):
                        if dd_age >= 0:
                            if dd_wait >= 0:
                                tp_tp_age_wait[delta_rho] += 1
                            else:
                                tp_fp_age_wait[delta_rho] += 1
                        else:
                            if dd_wait >= 0:
                                fp_tp_age_wait[delta_rho] += 1
                            else:
                                fp_fp_age_wait[delta_rho] += 1
                    else:  # wait is fn
                        if dd_age >= 0:
                            tp_fn_age_wait[delta_rho] += 1
                        else:
                            fp_fn_age_wait[delta_rho] += 1
                else:  # dd_age is fn
                    if not np.isnan(dd_wait):
                        if dd_wait >= 0:
                            fn_tp_age_wait[delta_rho] += 1
                        else:
                            fn_fp_age_wait[delta_rho] += 1
                    else:  # dd_wait is fn
                        fn_fn_age_wait += 1

                if not np.isnan(dd_queue):
                    if not np.isnan(dd_wait):
                        if dd_queue >= 0:
                            if dd_wait >= 0:
                                tp_tp_queue_wait[delta_rho] += 1
                            else:
                                tp_fp_queue_wait[delta_rho] += 1
                        else:
                            if dd_wait >= 0:
                                fp_tp_queue_wait[delta_rho] += 1
                            else:
                                fp_fp_queue_wait[delta_rho] += 1
                    else:  # wait is fn
                        if dd_queue >= 0:
                            tp_fn_queue_wait[delta_rho] += 1
                        else:
                            fp_fn_queue_wait[delta_rho] += 1
                else:  # dd_queue is fn
                    if not np.isnan(dd_wait):
                        if dd_wait >= 0:
                            fn_tp_queue_wait[delta_rho] += 1
                        else:
                            fn_fp_queue_wait[delta_rho] += 1
                    else:  # dd_wait is fn
                        fn_fn_queue_wait += 1

"""

            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: plotting_utility.py
Description:
    general plots code to support the online parallel detection tests
supports online_parallel_detection_test_simulator.py

Test how early and with what accuracy a change can be detected
Author: Harold Nemo Adodo Nikoue
part of my chapter on parallel partial observability in my thesis
Date: 10/23/2021
"""

from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

from online_parallel_test_config import TrackedStatisticsPerConfig, PolicyName, reorder_roc_lists


def plot_roc_curve(fpr: List[float], tpr: List[float], auc_score: float, fig_name: str) -> None:
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, 'b', label='AUC = {:.2f}'.format(auc_score))
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(fig_name)


class ROCExperimentParam:
    fpr: List[float]
    fpr_no_change: List[float]
    tpr: List[float]
    tpr_no_change: List[float]
    experiment_name: str
    auc_score: float
    auc_score_no_change: float

    def __init__(self, fpr: List[float], fpr_no_change: List[float],
                 tpr: List[float],
                 tpr_no_change: List[float],
                 auc_score: float,
                 auc_score_no_change: float, experiment_name: str):
        self.fpr = fpr
        self.fpr_no_change = fpr_no_change
        self.tpr = tpr
        self.tpr_no_change = tpr_no_change
        self.auc_score = auc_score
        self.auc_score_no_change = auc_score_no_change
        self.experiment_name = experiment_name


def plot_roc_curve_multiple_experiments(roc_params: List[ROCExperimentParam], fig_name: str,
                                        change_param: str = "Change") -> None:
    plt.figure()
    for roc_param in roc_params:
        if change_param == "Change":
            plt.plot(roc_param.fpr, roc_param.tpr, 'b',
                     label='{} AUC = {:.2f}'.format(roc_param.experiment_name, roc_param.auc_score))
        else:
            plt.plot(roc_param.fpr_no_change, roc_param.tpr_no_change, 'b',
                     label='{} AUC = {:.2f}'.format(roc_param.experiment_name, roc_param.auc_score))

        plt.legend(loc="lower right")

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(fig_name)
    plt.close()


def convert_all_experiments_results_to_roc_params(
        dic_of_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_score: float,
        auc_score_no_change: float,
        policy_name: PolicyName
) -> ROCExperimentParam:
    fpr_list = []
    fpr_no_change_list = []
    tpr_list = []
    for rho, inner_dic in dic_of_results.items():
        for delta_rho, tracked_stats in inner_dic.items():
            fpr_list.append(tracked_stats.return_fp_rate())
            fpr_no_change_list.append(tracked_stats.return_fp_rate_no_change())
            tpr_list.append(tracked_stats.return_tp_rate())
    old_tpr_list = tpr_list.copy()
    fpr_list, tpr_list = reorder_roc_lists(fpr_list, tpr_list)
    fpr_no_change_list, tpr_no_change_list = reorder_roc_lists(fpr_no_change_list, old_tpr_list)
    if policy_name == PolicyName.DETECT_ON_ALL_OBSERVATIONS:
        experiment_name = "all observations"
    elif policy_name == PolicyName.DETECT_ON_AGE:
        experiment_name = "Age"
    elif policy_name == PolicyName.DETECT_ON_QUEUE:
        experiment_name = "Queue"
    elif policy_name == PolicyName.DETECT_ON_WAIT:
        experiment_name = "Wait"
    else:
        raise ValueError
    return ROCExperimentParam(fpr_list, fpr_no_change_list, tpr_list, tpr_no_change_list, auc_score,
                              auc_score_no_change, experiment_name)


def generate_analysis_plots(
        dic_of_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_score: float,
        auc_score_no_change: float,
        policy: PolicyName
) -> None:
    # Plot ROC
    roc_params: ROCExperimentParam = convert_all_experiments_results_to_roc_params(dic_of_results, auc_score,
                                                                                   auc_score_no_change, policy)
    if len(roc_params.fpr) > 1:
        plot_roc_curve(roc_params.fpr, roc_params.tpr, roc_params.auc_score,
                       "Figures/ParallelTestFinalResults/all_observations_roc.png")
    if len(roc_params.fpr_no_change) > 1:
        plot_roc_curve(roc_params.fpr_no_change, roc_params.tpr_no_change, roc_params.auc_score_no_change,
                       "Figures/ParallelTestFinalResults/all_observations_roc_no_change.png")
    return None


def generate_multiple_observation_roc_plots(
        dic_of_all_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_all: float,
        auc_no_change_all: float,
        dic_of_age_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_age: float,
        auc_no_change_age: float,
        dic_of_queue_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_queue: float,
        auc_no_change_queue: float,
        dic_of_wait_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        auc_wait: float,
        auc_no_change_wait: float
) -> None:
    roc_params_age = convert_all_experiments_results_to_roc_params(dic_of_age_results, auc_age, auc_no_change_age,
                                                                   PolicyName.DETECT_ON_AGE)
    roc_params_queue = convert_all_experiments_results_to_roc_params(dic_of_queue_results, auc_queue,
                                                                     auc_no_change_queue, PolicyName.DETECT_ON_QUEUE)
    roc_params_wait = convert_all_experiments_results_to_roc_params(dic_of_wait_results, auc_wait,
                                                                    auc_no_change_wait, PolicyName.DETECT_ON_WAIT)
    roc_params_all = convert_all_experiments_results_to_roc_params(dic_of_all_results, auc_all, auc_no_change_all,
                                                                   PolicyName.DETECT_ON_ALL_OBSERVATIONS)
    list_of_roc_params = [roc_params_all, roc_params_age, roc_params_queue, roc_params_wait]
    plot_roc_curve_multiple_experiments(list_of_roc_params, "comparison_roc.png")


#    plot_roc_curve_multiple_experiments(list_of_roc_params,
#                                        "ParallelTestFinalResults/comparison_roc_no_change.png", "no_change")


def return_lift_params(dic_of_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]]) -> \
        Tuple[List[float], List[float]]:
    x_tp_list = []
    y_yrate_list = []
    for _, inner_dic in dic_of_results.items():
        for _, tracked_stats in inner_dic.items():
            x_tp_list.append(tracked_stats.num_correct_detections)
            y_yrate_list.append(tracked_stats.return_y_rate())
    return x_tp_list, y_yrate_list


def return_fa_cdr(dic_of_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]]) -> \
        Tuple[List[float], List[float]]:
    x_fa_list = []
    y_cdr_list = []
    for _, inner_dic in dic_of_results.items():
        for _, tracked_stats in inner_dic.items():
            x_fa_list.append(tracked_stats.return_fp_rate())
            y_cdr_list.append(tracked_stats.return_cdr())
    return x_fa_list, y_cdr_list


def plot_comp_curves(
        vals_all: Tuple[List[float], List[float]],
        vals_age: Tuple[List[float], List[float]],
        vals_queue: Tuple[List[float], List[float]],
        vals_wait: Tuple[List[float], List[float]],
        xlabel: str,
        ylabel: str,
        file_name
) -> None:
    x_all, y_all = order_two_list_by_first_element(vals_all[0], vals_all[1])
    x_age, y_age = order_two_list_by_first_element(vals_age[0], vals_age[1])
    x_queue, y_queue = order_two_list_by_first_element(vals_queue[0], vals_queue[1])
    x_wait, y_wait = order_two_list_by_first_element(vals_wait[0], vals_wait[1])
    plt.figure()
    plt.plot(x_all, y_all, c="k", marker="+", label="All observations")
    plt.plot(x_age, y_age, c="b", marker="d", label="Age")
    plt.plot(x_queue, y_queue, c="g", marker="*", label="Queue length")
    plt.plot(x_wait, y_wait, c="c", marker=".", label="Waiting times")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.close()


def generate_multiple_observations_lift_curves(
        dic_of_all_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_age_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_queue_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_wait_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]]
) -> None:
    lift_tuple_all = return_lift_params(dic_of_all_results)
    lift_tuple_age = return_lift_params(dic_of_age_results)
    lift_tuple_queue = return_lift_params(dic_of_queue_results)
    lift_tuple_wait = return_lift_params(dic_of_wait_results)
    plot_comp_curves(lift_tuple_all, lift_tuple_age, lift_tuple_queue, lift_tuple_wait, "Yrate", "TP",
                     "Figures/ParallelTestFinalResults/LiftCurve.png")


def generate_multiple_observationss_cdr_vs_fa(
        dic_of_all_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_age_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_queue_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]],
        dic_of_wait_results: Dict[float, Dict[float, TrackedStatisticsPerConfig]]
) -> None:
    cdr_tuple_all = return_fa_cdr(dic_of_all_results)
    cdr_tuple_age = return_fa_cdr(dic_of_age_results)
    cdr_tuple_queue = return_fa_cdr(dic_of_queue_results)
    cdr_tuple_wait = return_fa_cdr(dic_of_wait_results)
    plot_comp_curves(cdr_tuple_all, cdr_tuple_age, cdr_tuple_queue, cdr_tuple_wait, "FA", "CDR",
                     "Figures/ParallelTestFinalResults/CDR_vs_FA.png")


def order_two_list_by_first_element(list_1: List[float], list_2: List[float]):
    # transform two lists into list of tuples
    list_of_tuples = list(tuple(zip(list_1, list_2)))
    # sort by the first element
    sorted_list_of_tuples = sorted(list_of_tuples)
    # return the sorted elements
    return map(list, zip(*sorted_list_of_tuples))

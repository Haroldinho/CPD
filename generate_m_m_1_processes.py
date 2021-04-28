"""
            ***** PARTIAL OBSERVABILITY OF QUEUES: CHANGE POINT DETECTION ******
FILE: generate_stochastic_streams.py
Description: Create sequences of random variates from a general M/M/1 queue

I have two implementations:
 - Ladder point process implementation which is the minimum simulation framework for waiting times and idle times
 - DEDS that simulate both arrival and departure events individually
Author: Harold Nemo Adodo Nikoue
part of my partial observability thesis
"""
import random
from collections import deque
from heapq import heappush, heappop
from typing import List

import numpy as np
from dataclasses import dataclass, field

# from scipy.stats import expon
#
# from utilities import my_timer

object_num = 0
random.seed(0)


def get_expected_wait_time(inter_arr_rate, service_rate):
    """
    Get the expected wait time of the nth customer in queue.
    This is the queue time in Gallager
    :param inter_arr_rate: The rate of arrivals of the customers
    :param service_rate: The service rate of the queue
    """
    rho = float(inter_arr_rate) / service_rate
    # return rho / (service_rate - inter_arr_rate)
    return rho


def get_variance_wait_time(inter_arr_rate, service_rate):
    """
    Get the variance of the wait time of the nth customer in queue.
    This was obtained by using the CDF of the wait time derived by Prabhu.
    Taking the derivative of that distribution and using it to compute the second moment and second central moment.
    This doesn't take into account the weight at zero and the fact that the distribution is not purely continuous.
    It is therefore an approximation.
    TODO: USED A MORE CORRECT MODEL IF AVAILABLE
    """
    rho = float(inter_arr_rate) / service_rate
    return (2 - rho) * rho / ((service_rate - inter_arr_rate) * (service_rate - inter_arr_rate))


def get_std_wait_time(inter_arr_rate, service_rate):
    """
    Get the variance of the wait time of the nth customer in queue.
    This was obtained by using the CDF of the wait time derived by Prabhu.
    Taking the derivative of that distribution and using it to compute the second moment and second central moment.
    This doesn't take into account the weight at zero and the fact that the distribution is not purely continuous.
    It is therefore an approximation.
    TODO: USED A MORE CORRECT MODEL IF AVAILABLE
    """
    rho = float(inter_arr_rate) / service_rate
    # return np.sqrt((2 - rho) * rho) / (service_rate - inter_arr_rate)
    return rho + rho * rho


def generate_iid_expo(rate: float):
    return np.random.exponential(1.0 / rate)


def get_rate(rate_list, change_point_times, time):
    if len(rate_list) <= 1:
        return max(rate_list[0], 0.1)
    idx = 0
    while idx < len(change_point_times) and change_point_times[idx] <= time:
        idx += 1
    return max(rate_list[idx - 1], 0.0001)


def simulate_ladder_point_process(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates):
    ladder_point_creator = LadderPointQueue(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                            time_of_changes)
    wait_times = ladder_point_creator.simulate_ladder_points()
    recorded_times = ladder_point_creator.report_wait_record_times()
    return wait_times, recorded_times


def simulate_deds_return_wait_times(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                    discard_fraction=0.0,
                                    warm_up_time=None):
    deds_object = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                             my_service_rates, time_of_changes)
    if warm_up_time:
        wait_times = deds_object.simulate_deds_process(warm_up=warm_up_time)
    else:
        wait_times = deds_object.simulate_deds_process()
    departure_times = deds_object.get_recorded_times()
    num_to_start_keeping = int(discard_fraction * len(wait_times))
    wait_times = wait_times[num_to_start_keeping:]
    departure_times = departure_times[num_to_start_keeping:]
    return wait_times, departure_times


def simulate_deds_return_age(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                             type="median", discard_fraction=0.0, warm_up_time=None):
    deds_object = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                             my_service_rates, time_of_changes)
    if warm_up_time:
        deds_object.simulate_deds_process(warm_up=warm_up_time)
    else:
        deds_object.simulate_deds_process()
    recording_times = deds_object.return_recording_times_age()
    mean_age_times = deds_object.return_mean_process_age()
    num_to_start_keeping = int(discard_fraction * len(mean_age_times))
    mean_age_times = mean_age_times[num_to_start_keeping:]
    recording_times = recording_times[num_to_start_keeping:]
    return mean_age_times, recording_times


def simulate_deds_return_queue_length(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                      discard_fraction=0.1, warm_up_time=None):
    deds_object = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                             my_service_rates, time_of_changes)
    if warm_up_time:
        deds_object.simulate_deds_process(warm_up=warm_up_time)
    else:
        deds_object.simulate_deds_process()
    queue_lengths = deds_object.return_recorded_queue_lengths()
    queue_length_times = deds_object.return_recording_time_queue_length()
    num_to_start_keeping = int(discard_fraction * len(queue_lengths))
    queue_lengths = queue_lengths[num_to_start_keeping:]
    queue_length_times = queue_length_times[num_to_start_keeping:]
    return queue_lengths, queue_length_times


def simulate_deds_return_age_queue_wait(start_time, end_time, my_arrival_rates, time_of_changes, my_service_rates,
                                        discard_fraction=0.1, warm_up_time=None):
    deds_object = SingleServerMarkovianQueue(start_time, end_time, my_arrival_rates, time_of_changes,
                                             my_service_rates, time_of_changes)
    if warm_up_time:
        wait_times = deds_object.simulate_deds_process(warm_up=warm_up_time)
    else:
        wait_times = deds_object.simulate_deds_process()
    queue_lengths = deds_object.return_recorded_queue_lengths()
    queue_length_times = deds_object.return_recording_time_queue_length()
    recording_times = deds_object.return_recording_times_age()
    mean_age_times = deds_object.return_mean_process_age()
    departure_times = deds_object.get_recorded_times()
    num_to_start_keeping = int(discard_fraction * len(wait_times))
    return queue_lengths, queue_length_times, mean_age_times, recording_times, wait_times, departure_times


class LadderPointQueue:
    """
    Class used to compute wait times and idle times between the M/M/1 queue as a Ladder point process
    """

    def __init__(self, start_time, stop_time, arrival_rates, arrival_change_points, service_rates,
                 service_change_points):
        """

        :param start_time: The start of the M/M/1 simulation
        :param stop_time: The end of the simulation
        :param arrival_rates: the m different arrival rates
        :param arrival_change_points: The m different starts to a new arrival rate period
        :param service_rates: the m different arrival rates
        :param service_change_points: The m different starts to a new arrival rate period
        :return: None
        Test: All wait times are positive
             All idle times are positive
             I can extract all idle times and wait times from ladder points
        """
        self._start_time = start_time
        self._stop_time = stop_time
        self._arrival_rates = arrival_rates
        self._arrival_change_points = arrival_change_points
        self._service_rates = service_rates
        self._service_change_points = service_change_points
        self._ladder_times = []
        self._wait_times = []
        self._idle_times = []
        self._event_times = []
        # in a ladder process we are recording wait or idle times at each arrival

    def generate_inter_arrival_time(self, current_time):
        arrival_rate = get_rate(self._arrival_rates, self._arrival_change_points, current_time)
        return generate_iid_expo(arrival_rate)

    def generate_service_time(self, current_time):
        service_rate = get_rate(self._service_rates, self._service_change_points, current_time)
        return generate_iid_expo(service_rate)

    ## Deprecated
    #     # @my_timer
    #     def simulate_ladder_point_process(self):
    #         """
    #         Simulate wait times and idle times for a M/M/1 queue
    #         Return wait times
    #         Test: verify that all wait times are positive and follow a similar distribution
    #
    #         """
    #         current_time = self._start_time
    #         current_wait_time = 0
    #         current_idle_time = 0
    #         count = 0
    #         while current_time < self._stop_time:
    #             inter_arrival_time = self.generate_inter_arrival_time(current_time)
    #             service_time = self.generate_service_time(current_time)
    #             current_time += inter_arrival_time
    #             ladder_step_time = service_time - inter_arrival_time
    #             new_wait_time = max(current_wait_time + ladder_step_time, 0)
    #             new_idle_time = - min(current_idle_time + ladder_step_time, 0)
    #             self._event_times.append(current_time)
    #             self._wait_times.append(new_wait_time)
    #             self._idle_times.append(new_idle_time)
    #             if new_wait_time > 0:
    #                 self._ladder_times.append(new_wait_time)
    #             else:
    #                 self._ladder_times.append(-new_idle_time)
    #         return self._wait_times

    def simulate_ladder_points(self, is_antithetic=False):
        """
        Simulate ladder points much faster
        """
        wait_times = []
        max_arr_rate = np.max(self._arrival_rates)
        max_dep_rate = np.max(self._service_rates)
        # get the max number of elements to simulate
        num_uniforms = 2 * max(max_arr_rate, max_dep_rate) * (self._stop_time - self._start_time)
        uniform_random_variables = np.random.random(int(num_uniforms) + 1)
        if is_antithetic:
            uniform_random_variables = [1 - u for u in uniform_random_variables]
        # t is the start of service time
        # because it's used to determine the rate of arrival or service
        t = self._start_time
        index = 0
        total_inter_arrival_time = 0
        wait_time = 0
        # Generate steady-state waiting-time
        rho = self._arrival_rates[0] / self._service_rates[0]
        mu = self._service_rates[0]
        arr_rate = self._arrival_rates[0]
        u = uniform_random_variables[index]
        if u <= 1 - rho:
            wait_time = 0
        else:
            wait_time = - np.log((1 - u) / rho) / (mu - arr_rate)
        index = index + 1
        while t < self._stop_time:
            self._event_times.append(t)
            arr_rate = get_rate(self._arrival_rates, self._arrival_change_points, t)
            service_rate = get_rate(self._service_rates, self._service_change_points, t)
            inter_arrival_time = 1.0 / arr_rate * np.log(1 / uniform_random_variables[index])
            index += 1
            service_time = 1.0 / service_rate * np.log(1.0 / uniform_random_variables[index])
            index += 1
            total_inter_arrival_time += inter_arrival_time
            wait_time = max(0, wait_time + service_time - inter_arrival_time)
            wait_times.append(wait_time)
            t = max(t + service_time, total_inter_arrival_time)
        self._wait_times = wait_times
        return wait_times

    def simmulate_ladder_points_no_change_point(self):
        """
        Generate random waiting times with no change point using the computationally efficient method in Song 2013
        """
        wait_times = []
        arr_rate = self._arrival_rates[0]
        dep_rate = self._service_rates[0]
        # get the max number of elements to simulate
        num_uniforms = max(arr_rate, dep_rate) * (self._stop_time - self._start_time)
        uniform_random_variables = np.random.random(int(num_uniforms) + 1)
        # t is the start of service time
        # because it's used to determine the rate of arrival or service
        t = self._start_time
        index = 0
        total_inter_arrival_time = 0
        wait_time = 0
        # Generate steady-state waiting-time
        rho = self._arrival_rates[0] / self._service_rates[0]
        mu = self._service_rates[0]
        u = uniform_random_variables[index]
        if u <= 1 - rho:
            wait_time = 0
        else:
            wait_time = - np.log((1 - u) / rho) / (mu - arr_rate)
        index = index + 1
        while index < num_uniforms:
            u = uniform_random_variables[index]
            if u > mu / (arr_rate + mu):
                y = - 1 / mu * np.log((1 - u) * (arr_rate + mu) / arr_rate)
            else:
                y = 1 / arr_rate * np.log(u * (arr_rate + mu))
            wait_time = max(0, wait_time + y)
            wait_time.append(wait_time)
        return wait_times

    # return departure times
    def report_wait_record_times(self):
        return self._event_times

    def get_arrival_times(self):
        return self._event_times

    def get_idle_times(self):
        return self._idle_times

    def get_wait_times(self):
        return self._wait_times

    def get_ladder_times(self):
        return self._ladder_times


@dataclass(order=True)
class Event:
    time: float = field(default=0.0, metadata={'unit': 'seconds'}, compare=True)
    object_id: int = field(default=0, compare=True)
    # can only be 'A': arrival or 'D': departure
    type: str = field(default='A', compare=False)

    def __str__(self):
        print("{type} event occurring at {time} with id {id}.".format(
            type="Arrival" if self.type == 'A' else 'Departure', time=self.time, id=self.object_id))


class MM1MasterClass:
    def __init__(self):
        self._count = 0

    def generate_class_id(self):
        self._count += 1
        return self._count

    def get_num_customers(self):
        return self._count


m_m_1_controller = MM1MasterClass()


class Customer:
    def __init__(self, arr_time):
        self._id = m_m_1_controller.generate_class_id()
        self._arrival_time = arr_time
        self.start_of_service_time = None
        self.departure_time = None
        self.wait_time = None
        self.cycle_time = None

    def get_id(self):
        return self._id

    def get_arrival_time(self):
        return self._arrival_time

    def get_arrival_event(self):
        return Event(self._arrival_time, self._id, 'A')

    def get_departure_event(self, service_time):
        return Event(self.start_of_service_time + service_time, self._id, 'D')

    def set_start_of_service(self, service_start_time):
        self.start_of_service_time = service_start_time
        self.wait_time = self.start_of_service_time - self._arrival_time
        assert (self.wait_time >= 0.0)

    def set_departure_time(self, dep_time):
        self.departure_time = dep_time
        self.cycle_time = dep_time - self._arrival_time
        assert (self.cycle_time >= 0.0)


def get_mean_waiting_time_of_queue(a_queue: List[Customer], current_time: float):
    waiting_times = [current_time - cust.get_arrival_time() for cust in a_queue]
    return np.mean(waiting_times)


def get_median_waiting_time_of_queue(a_queue: List[Customer], current_time: float):
    waiting_times = [current_time - cust.get_arrival_time() for cust in a_queue]
    return np.median(waiting_times)


class SingleServerMarkovianQueue:
    """
    M/M/1 queue
    Why don't I clear departures as I clear arrivals
    """

    def __init__(self, start_time, stop_time, arrival_rates, arrival_change_points, service_rates,
                 service_change_points):
        """

        :param start_time: The start of the M/M/1 simulation
        :param stop_time: The end of the simulation
        :param arrival_rates: the m different arrival rates
        :param arrival_change_points: The m different starts to a new arrival rate period with the first one being zero
        by default, NOTE_TO_SELF: maybe it should be the start_time
        :param service_rates: the m different service rates
        :param service_change_points: The m different starts to a new service rate period
        :return: None
        Test: All wait times are positive
        """
        self._dictionary_customers = {}
        # the idea of a warm up period is not to collect data until the end of the warm-up period.
        # So it should not count for the definition of the time epoch
        # t=0 at the end end of the warm-up period!!!
        # but instead of doing that, I shift all the change point times, start times and end times by the warm-up length
        self._warm_up = 0
        self._start_time = start_time
        self._stop_time = stop_time
        self._arrival_rates = arrival_rates
        self._arrival_change_points = arrival_change_points
        self._service_rates = service_rates
        self._service_change_points = service_change_points
        # will use heapq methods on this list to reproduce the dynamics of a priority queue
        self._priority_event_list = []
        self._is_queue_empty = True
        self._is_server_busy = False
        self._queue = deque()
        self._current_time = 0
        # They should all have the same length
        self._wait_times = []
        self._mean_age = []
        self._arrival_times = []
        self._inter_arrival_times = []
        self._age_recorded_times = []
        self._recorded_queue_lengths = []
        self._time_at_recorded_queue_lengths = []
        self._service_times = []
        self._departure_times = []
        self._cycle_times = []
        self._service_start_times = []

    def update_times_with_warm_up(self, warm_up):
        self._stop_time += warm_up
        # The following statement doesn't do anything.
        #  It just copies the first element of the array and the  remaining of the array before concatenating
        #  to create a new array
        self._arrival_change_points = [self._arrival_change_points[0]] + [t + warm_up
                                                                          for t in self._arrival_change_points[1:]]
        self._service_change_points = [self._service_change_points[0]] + [t + warm_up
                                                                          for t in self._service_change_points[1:]]

    def generate_inter_arrival_time(self, current_time, warm_up_period=None):
        if warm_up_period:
            # why do I add the warm_up_period to the current_time
            arrival_rate = get_rate(self._arrival_rates, self._arrival_change_points, current_time + warm_up_period)
        else:
            arrival_rate = get_rate(self._arrival_rates, self._arrival_change_points, current_time)
        return generate_iid_expo(float(arrival_rate))

    def generate_service_time(self, current_time, warm_up_period=None):
        if warm_up_period:
            service_rate = get_rate(self._service_rates, self._service_change_points, current_time + warm_up_period)
        else:
            service_rate = get_rate(self._service_rates, self._service_change_points, current_time)
        return generate_iid_expo(float(service_rate))

    def generate_arrivals(self):
        time = self._start_time
        while time < self._stop_time:
            inter_arrival_time = self.generate_inter_arrival_time(time)
            time += inter_arrival_time
            new_customer = Customer(time)
            self._dictionary_customers[new_customer.get_id()] = new_customer
            new_arrival_event = new_customer.get_arrival_event()
            self.add_event(new_arrival_event)
            self._arrival_times.append(time)
            self._inter_arrival_times.append(inter_arrival_time)
        # print("\nCreated arrival times: \n\t", self._arrival_times[:10])

    def return_arrival_times(self):
        return self._arrival_times

    def return_inter_arrival_times(self):
        return self._inter_arrival_times

    def return_service_times(self):
        return self._service_times

    # @my_timer
    def simulate_deds_process(self, warm_up=0):
        # Simulate the events of a single-queue queueing process with Markovian arrivals and Markovian departures
        # Integrate a warm-up period before measuring the output of the simulation
        # First generate all possible arrivals in the period
        # To process departures iterate through the priority queue until it's empty
        self._warm_up = warm_up
        self.generate_arrivals()
        time = 0
        # go through the whole list of event
        average_age = 0
        while self._priority_event_list or not self._is_queue_empty:
            # If there is someone in the queue and the server is empty, we take the first customer from the queue
            # and move it to the server to be served.
            if not (self._is_queue_empty or self._is_server_busy):
                first_customer_in_queue = self._queue.pop()
                #                 self._age_recorded_times.append(time)
                #                 if len(self._queue) > 0:
                #                     mean_inq_waiting_time = get_mean_waiting_time_of_queue(self._queue, time)
                #                     median_inq_waiting_time = get_median_waiting_time_of_queue(self._queue, time)
                #                     self._mean_age.append(mean_inq_waiting_time)
                #                     self._median_age.append(median_inq_waiting_time)
                #                 else:
                #                     self._mean_age.append(0)
                #                     self._median_age.append(0)
                first_customer_in_queue.set_start_of_service(time)
                if self._service_start_times:
                    assert (time >= self._service_start_times[-1])
                # This is the time of the start of service indeed
                self._service_start_times.append(time)
                service_time = self.generate_service_time(time)
                self._service_times.append(service_time)
                new_departure_event = first_customer_in_queue.get_departure_event(service_time)
                self.add_event(new_departure_event)
                # Hold the server
                self._is_server_busy = True
            # Get the next event. It could be either an arrival at the queue or a departure from the server
            next_event = self._get_next_event()
            next_event_id = next_event.object_id
            customer = self._dictionary_customers[next_event_id]
            # Verify that the order of events is always respected
            # It's an event driven simulation. Time moves from event to event.
            if time:
                assert (time <= next_event.time)
            # move time to the time of the next event
            old_time = time
            time = next_event.time
            dtime = time - old_time
            old_queue_size = len(self._queue)
            # Either the server is busy or the queue is empty or both
            if next_event.type == 'D':
                self._wait_times.append(customer.wait_time)
                # set the departure time of the event
                customer.set_departure_time(time)
                self._cycle_times.append(customer.cycle_time)
                if self._departure_times:
                    assert (time >= self._departure_times[-1])
                self._departure_times.append(time)
                # release server
                self._is_server_busy = False
            elif next_event.type == 'A':
                # is queue empty
                self._recorded_queue_lengths.append(len(self._queue))
                self._time_at_recorded_queue_lengths.append(time)
                if self._is_queue_empty:
                    if not self._is_server_busy:
                        # Create a departure event
                        customer.set_start_of_service(time)
                        if self._service_start_times:
                            assert (time > self._service_start_times[-1])
                        self._service_start_times.append(time)
                        self._is_server_busy = True
                        service_time = self.generate_service_time(time)
                        self._service_times.append(service_time)
                        new_departure_event = customer.get_departure_event(service_time)
                        self.add_event(new_departure_event)
                    else:
                        # put new arrival in queue
                        self._queue.appendleft(customer)
                        self._is_queue_empty = False
                else:
                    # the server is busy for sure
                    # there is at least one element in the queue
                    # put new customer in the queue
                    self._queue.appendleft(customer)
            else:
                # I have only implemented Arrivals and Departures in the sim so far
                raise Exception('Wrong event type in simulate_queue')
            self._is_queue_empty = (len(self._queue) == 0)
            # get difference in queue size
            dqueue = len(self._queue) - old_queue_size
            average_age = max((old_queue_size * average_age + dtime * dqueue), 0) / len(self._queue) \
                if len(self._queue) > 0 else 0
            self._age_recorded_times.append(time)
            self._mean_age.append(average_age)
        assert (len(self._service_start_times) == len(self._arrival_times))
        # TODO: Check _extract_wait_times
        # self._extract_wait_times()
        # TODO: Check _extract_cycle_times
        # self._extract_cycle_times()
        # TODO: Check account_for_warm_up_period
        #         self.account_for_warm_up_period()
        return self._wait_times

    def account_for_warm_up_period(self):
        # 1. Get the index for the first arrival time greater or equal to the warm-up period
        idx_warm_up = 0
        while idx_warm_up < len(self._arrival_times):
            if self._arrival_times[idx_warm_up] >= self._warm_up:
                break
            idx_warm_up += 1
        assert (idx_warm_up < len(self._arrival_times))
        # 2. update all output vectors
        self._arrival_times = self._arrival_times[idx_warm_up:]
        self._service_start_times = self._service_start_times[idx_warm_up:]
        self._wait_times = self._wait_times[idx_warm_up:]
        self._cycle_times = self._cycle_times[idx_warm_up:]

    def add_event(self, event: Event):
        heappush(self._priority_event_list, event)

    def _get_next_event(self):
        return heappop(self._priority_event_list)

    def _extract_wait_times(self):
        """
        Wait times are just the difference between service start times and arrival times
        Because we have a single server, order is ALWAYS respected.

        Could I extract the wait times from all the customer objects
        """
        wait_times = []
        for idx in range(len(self._service_start_times)):
            wait_time = self._service_start_times[idx] - self._arrival_times[idx]
            assert (wait_time >= 0)
            wait_times.append(wait_time)
        self._wait_times = wait_times
        # print("\n Service Start times: \n\t", self._service_start_times[:10])
        # print("\n Wait Times: \n\t", self._wait_times[:10])

    def _extract_cycle_times(self):
        """
        Cycle times are just the difference between departure times and arrival times
        Because we have a single server, order is ALWAYS respected.
        """
        cycle_times = []
        for idx in range(len(self._departure_times)):
            cycle_times.append(self._departure_times[idx] - self._arrival_times[idx])
        self._cycle_times = cycle_times
        # print("\n Departure times: \n\t", self._departure_times[:10])
        # print("\n Cycle Times: \n\t", self._cycle_times)

    def get_cycle_times(self):
        return self._cycle_times

    def get_recorded_times(self):
        return self._arrival_times

    def return_mean_process_age(self):
        return self._mean_age

    #     def return_median_process_age(self):
    #         return self._median_age

    def return_recorded_queue_lengths(self):
        return self._recorded_queue_lengths

    def return_recording_times_age(self):
        return self._age_recorded_times

    def return_recording_time_queue_length(self):
        return self._time_at_recorded_queue_lengths

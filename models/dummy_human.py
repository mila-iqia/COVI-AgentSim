import sys
import os
sys.path.append(os.getcwd())
import datetime
from utils import float_to_binary, binary_to_float, quantize_risk
from bitarray import bitarray
from collections import namedtuple
import numpy as np

# A utility class for re-inflating human objects with just the stuff we need for message passing / risk prediction
class DummyHuman:
    def __init__(self, name=None, timestamp=None, rng=None):
        self.name = name
        self.M = {}
        self.sent_messages = {}
        self.messages = []
        self.update_messages = []
        self.risk = 0
        self.rng = rng
        self.all_reported_symptoms = [[]]
        self.all_symptoms = []
        self.timestamp = timestamp
        self._uid = None
        self.time_of_recovery = datetime.datetime.max
        self.time_of_infectiousness_start = datetime.datetime.max
        self.time_of_death = datetime.datetime.max
        self.test_time = datetime.datetime.max
        self.time_of_exposure = datetime.datetime.max
        self.symptoms_start = datetime.datetime.max
        self.infectiousness_start = datetime.datetime.max
        self.tested_positive_contact_count = 0
        self.Message = namedtuple('message', 'uid risk day unobs_id')
        self.UpdateMessage = namedtuple('update_message', 'uid new_risk risk day unobs_id')

        num_days = 14
        num_states = 4 # Susceptible, exposed, infected, recovered
        self.state_array = np.zeros((num_days, num_states))


    def cur_message(self, day):
        """creates the current message for this user"""
        message = self.Message(self.uid, quantize_risk(self.risk), day, self.name)
        return message

    def cur_message_risk_update(self, day, old_risk):
        return self.UpdateMessage(self.uid, quantize_risk(self.risk), old_risk, day, self.name)

    def purge_messages(self, todays_date):
        num_purged = 0
        for m in self.messages:
            if todays_date - m.day > 14:
                num_purged += 1
                self.messages.remove(m)
        for m in self.update_messages:
            if todays_date - m.day > 14:
                self.update_messages.remove(m)

    def shuffle_messages(self):
        self.rng.shuffle(self.messages)

    @property
    def uid(self):
        return self._uid

    def update_uid(self):
        try:
            self._uid.pop()
            self._uid.extend([self.rng.choice([True, False])])
        except AttributeError:
            self._uid = bitarray()
            self._uid.extend(self.rng.choice([True, False], 4))  # generate a random 4-bit code

    def reported_symptoms_at_time(self, now):
        sickness_day = (now - self.symptoms_start).days
        all_reported_symptoms_till_day = []
        for day in range(sickness_day-1):
            if not self.all_reported_symptoms:
                return []
            if sickness_day > len(self.all_reported_symptoms):
                return self.all_reported_symptoms
            all_reported_symptoms_till_day.append(self.all_reported_symptoms[day])
        return all_reported_symptoms_till_day

    def symptoms_at_time(self, now):
        sickness_day = (now - self.symptoms_start).days
        all_symptoms_till_day = []
        for day in range(sickness_day-1):
            all_symptoms_till_day.append(self.all_symptoms[day])
        return all_symptoms_till_day

    def get_test_result_array(self, date):
        results = np.zeros(14)
        result_day = (date - self.test_time).days
        if result_day >= 0 and result_day < 14:
            results[result_day] = 1
        return results

    def is_exposed(self, date):
        exposed = False
        exposure_day = (date - self.time_of_exposure).days
        if exposure_day >= 0 and exposure_day < 14:
            exposed = True
        else:
            exposure_day = None
        return exposed, exposure_day

    def is_infectious(self, date):
        is_infectious = False
        infectious_day = (date - self.time_of_infectiousness_start).days
        if infectious_day >= 0 and infectious_day < 14:
            is_infectious = True
        else:
            infectious_day = None
        return is_infectious, infectious_day

    def is_recovered(self, date):
        is_recovered = False
        recovery_day = (date - self.time_of_recovery).days
        if recovery_day >= 0 and recovery_day < 14:
            is_recovered = True
        else:
            recovery_day = None
        return is_recovered, recovery_day

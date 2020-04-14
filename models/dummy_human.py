import sys
import os
sys.path.append(os.getcwd())
import datetime
from utils import float_to_binary, binary_to_float
from bitarray import bitarray
from collections import namedtuple
import numpy as np

# A utility class for re-inflating human objects with just the stuff we need for message passing / risk prediction
class DummyHuman:
    def __init__(self, name=None, timestamp=None, rng=None):
        self.name = name
        self.M = {}
        self.messages = []
        self.risk = 0
        self.rng = rng
        self.all_reported_symptoms = []
        self.timestamp = timestamp
        self._uid = None
        self.is_infectious = False
        self.time_of_recovery = datetime.datetime.max
        self.time_of_death = datetime.datetime.max
        self.test_time = datetime.datetime.max
        self.test_result = None
        self.infectiousness_start = datetime.datetime.max
        self.tested_positive_contact_count = 0

    @property
    def message_risk(self):
        """quantizes the risk in order to be used in a message"""
        if self.risk == 1.0:
            return bitarray('1111')
        return bitarray(float_to_binary(float(self.risk), 0, 4))

    def cur_message(self, time):
        """creates the current message for this user"""
        Message = namedtuple('message', 'uid risk time unobs_id')
        message = Message(self.uid, self.message_risk, time, self.name)
        return message

    def preprocess_messages(self):
        """ Gets my current messages ready for writing to dataset"""
        current_encounter_messages = []
        for m in self.messages:
            m_risk = binary_to_float("".join([str(x) for x in np.array(m[0].tolist()).astype(int)]), 0, 4)
            uid = binary_to_float("".join([str(x) for x in np.array(m[1].tolist()).astype(int)]), 2, 4)

    def purge_messages(self, todays_date):
        for m in self.messages:
            if todays_date - m.time > datetime.timedelta(days=14):
                self.messages.remove(m)

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
        # TODO: this is kind of a lossy way to take into account the information currently available
        try:
            sickness_day = (self.symptoms_start - now).days
            all_reported_symptoms_till_day = []
            for day in range(sickness_day+1):
                all_reported_symptoms_till_day.extend(self.all_reported_symptoms[sickness_day])
            return all_reported_symptoms_till_day
        except Exception as e:
            return []

import sys
import os
sys.path.append(os.getcwd())
from utils import float_to_binary
from bitarray import bitarray
from collections import namedtuple

# A utility class for re-inflating human objects with just the stuff we need for message passing / risk prediction
class DummyHuman:
    def __init__(self, name=None, timestamp=None, rng=None):
        self.name = name
        self.M = {}
        self.messages = []
        self.risk = 0
        self.rng = rng
        self.all_reported_symptoms = []
        self.test_logs = []
        self.timestamp = timestamp
        self._uid = None
        self.is_infectious = False

    @property
    def message_risk(self):
        """quantizes the risk in order to be used in a message"""
        if self.risk == 1.0:
            return bitarray('1111')
        return bitarray(float_to_binary(self.risk, 0, 4))

    def cur_message(self, time):
        """creates the current message for this user"""
        Message = namedtuple('message', 'uid risk time unobs_id')
        message = Message(self.uid, self.message_risk, time, self.name)
        return message

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

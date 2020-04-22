import os
import numpy as np
from config import *
from models.utils import Message, encode_message, decode_message, binary_to_float
import operator
import datetime
from collections import defaultdict
from bitarray import bitarray
from models.dummy_human import Message, UpdateMessage
""" This file contains the core of the side simulation, which is run on the output encounters from the main simulation.
It's primary functionality is to run the message clustering and risk prediction algorithms.
"""
class RiskModelBase:
    @classmethod
    def update_risk_encounter(self, human, message):
        # This function is called for every encounter message
        raise "Unimplemented"

    @classmethod
    def update_risk_risk_update(self, human, update_message):
        # This function is called for every risk update message
        raise "Unimplemented"

    @classmethod
    def update_risk_daily(cls, human, now):
        """ This function calculates a risk score based on the person's symptoms."""
        # if they get tested, it takes TEST_DAYS to get the result, and they are quarantined for QUARANTINE_DAYS.
        # The test_timestamp is set to datetime.min, unless they get a positive test result.
        # Basically, once they know they have a positive test result, they have a risk of 1 until after quarantine days.
        if human.time_of_recovery < now:
            return 0.
        if human.time_of_death < now:
            return 0.
        if human.test_result and human.test_time < now + datetime.timedelta(days=2):
            return 1.

        reported_symptoms = human.reported_symptoms_at_time(now)
        if 'severe' in reported_symptoms:
            return 0.75
        if 'moderate' in reported_symptoms:
            return 0.5
        if 'mild' in reported_symptoms:
            return 0.25
        if len(reported_symptoms) > 3:
            return 0.25
        if len(reported_symptoms) > 1:
            return 0.1
        if len(reported_symptoms) > 0:
            return 0.05
        return 0.0

class RiskModelLenka(RiskModelBase):
    @classmethod
    def update_risk_encounter(cls, human, message):
        # Get the binarized contact risk
        m_risk = binary_to_float("".join([str(x) for x in np.array(message.risk.tolist()).astype(int)]), 0, 4)
        human.update = m_risk * RISK_TRANSMISSION_PROBA


class RiskModelYoshua(RiskModelBase):
    @classmethod
    def update_risk_encounter(cls, human, message):
        """ This function updates an individual's risk based on the receipt of a new message"""

        # Get the binarized contact risk
        m_risk = binary_to_float("".join([str(x) for x in np.array(message.risk.tolist()).astype(int)]), 0, 4)

        update = 0
        if human.risk < m_risk:
            update = (m_risk - m_risk * human.risk) * RISK_TRANSMISSION_PROBA
        print(f"human.risk: {human.risk}, m_risk: {m_risk}, update: {update}")

        human.risk += update



class RiskModelEilif(RiskModelBase):
    @classmethod
    def update_risk_encounter(cls, message):
        """ This function updates an individual's risk based on the receipt of a new message"""
        # Get the binarized contact risk
        m_risk = binary_to_float("".join([str(x) for x in np.array(message.risk.tolist()).astype(int)]), 0, 4)
        msg_enc = encode_message(message)
        if msg_enc not in human.M:
            # update is delta_risk
            update = m_risk * RISK_TRANSMISSION_PROBA
        else:
            previous_risk = human.M[msg_enc]['previous_risk']
            carry_over_transmission_proba = human.M[msg_enc]['carry_over_transmission_proba']
            update = ((m_risk - previous_risk) * RISK_TRANSMISSION_PROBA + previous_risk * carry_over_transmission_proba)

        # Update contact history
        human.M[msg_enc]['previous_risk'] = m_risk
        human.M[msg_enc]['carry_over_transmission_proba'] = RISK_TRANSMISSION_PROBA * (1 - update)


class RiskModelTristan(RiskModelBase):
    risk_map = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/log_risk_mapping.npy")
    risk_map[0] = np.log(0.01)

    @classmethod
    def quantize_risk(cls, risk):
        if risk == 0.:
            return 15
        # returns the quantized log probability (int 0 to 15)
        for idx, log_prob in enumerate(cls.risk_map):
            if risk >= log_prob and risk < cls.risk_map[idx+1]:
                return idx

    @classmethod
    def update_risk_daily(cls, human, now):
        """ This function calculates a risk score based on the person's symptoms."""
        # if they get tested, it takes TEST_DAYS to get the result, and they are quarantined for QUARANTINE_DAYS.
        # The test_timestamp is set to datetime.min, unless they get a positive test result.
        # Basically, once they know they have a positive test result, they have a risk of 1 until after quarantine days.
        if human.time_of_recovery < now:
            return np.log(0.01)
        if human.time_of_death < now:
            return np.log(0.01)
        if human.test_time < now + datetime.timedelta(days=2):
            return np.log(1.)
        return np.log(0.01)

    @classmethod
    def update_risk_encounter(cls, human, message):
        """ This function updates an individual's risk based on the receipt of a new message"""
        # if you already have a positive test result, ya risky.
        if human.risk == np.log(1.):
            human.risk = np.log(1.)
            return

        # if the encounter message indicates they had a positive test result, increment counter
        message = decode_message(message)
        if message.risk == 15:
            human.tested_positive_contact_count += 1

        init_population_level_risk = 0.01
        expo = (1 - RISK_TRANSMISSION_PROBA) ** human.tested_positive_contact_count
        tmp = (1. - init_population_level_risk) * (1. - expo)
        mask = tmp < init_population_level_risk

        if mask:
            human.risk = np.log(init_population_level_risk) + np.log1p(tmp / init_population_level_risk)
        else:
            human.risk = np.log(1. - init_population_level_risk) + np.log1p(-expo) + np.log1p(init_population_level_risk / tmp)


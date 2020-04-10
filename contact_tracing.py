import pickle
import json
from base import Event
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from utils import _encode_message, _decode_message
from bitarray import bitarray
import operator
from collections import defaultdict


if __name__ == "__main__":
    PATH_TO_DATA = "data.pkl"
    PATH_TO_HUMANS = "humans.pkl"

    with open(PATH_TO_DATA, "rb") as f:
        logs = pickle.load(f)
    with open(PATH_TO_HUMANS, "rb") as f:
        humans = pickle.load(f)
    enc_logs = [l for l in logs if l["event_type"] == Event.encounter]

    hd = {}
    for human in humans:
        # # privacy
        human.rng = np.random.RandomState(0)
        human.update_uid()
        hd[human.name] = human


    for log in enc_logs:
        now = log['time']
        unobs = log['payload']['unobserved']
        h1 = unobs['human1']['human_id']
        h2 = unobs['human2']['human_id']
        this_human = hd[h1]
        other_human = hd[h2]
        if this_human.cur_day != now.day:
            this_human.cur_date = now.day
            this_human.update_uid()
            for i in range(len(this_human.pending_messages)):
                this_human.handle_message(this_human.pending_messages.pop())

        this_human.pending_messages.append(other_human.cur_message(now))


    contact_histories = []
    for human in humans:
        contact_histories.append(human.A)
    json.dump(contact_histories, open('contact_histories.json', 'w'))



    import pdb; pdb.set_trace()
    pass
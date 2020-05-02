from collections import defaultdict


class Clusters:
    """ This class manages the storage and clustering of messages and message updates.
     You can think of each message as database record, with the message updates indicating
     an update to a record. We have various clues (uid, risk, number of updates sent for a day) as
     signals about which records to update, as well as whether our clusters are correct."""

    def __init__(self):
        self.num_messages = 0
        self.clusters = defaultdict(list)
        self.clusters_by_day = defaultdict(dict)

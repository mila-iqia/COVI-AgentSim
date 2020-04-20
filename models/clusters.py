import datetime
from collections import defaultdict
from models.utils import Message, decode_message, encode_message


# TODO: include risk level in clustering, currently only uses quantized uid
# TODO: check for mutually exclusive messages in order to break up a group and re-run nearest neighbors
# TODO: storing m_i_enc in dict M is a bug, we're overwriting some messages -- we need to make a unique encoding that uses the timestamp

class Clusters:
    """ This class manages the storage and clustering of messages and message updates.
     You can think of each message as database record, with the message updates indicating
     an update to a record. We have various clues (uid, risk, number of updates sent for a day) as
     signals about which records to update, as well as whether our clusters are correct."""

    def __init__(self):
        self.all_messages = []
        self.clusters = defaultdict(list)
        self.clusters_by_day = defaultdict(dict)

    def add_to_clusters_by_day(self, cluster, day, m_i_enc):
        if self.clusters_by_day[day].get(cluster):
            self.clusters_by_day[day][cluster].append(m_i_enc)
        else:
            self.clusters_by_day[day][cluster] = [m_i_enc]

    def add_message(self, message:Message):
        """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
        m_i_enc = encode_message(message)
        # otherwise score against previous messages
        best_cluster, best_message, best_score = self.score_matches(message)
        if best_score >= 0:
            cluster_id = best_cluster
        elif not self:
            cluster_id = 0
        else:
            cluster_id = self.num_messages + 1
        self.all_messages.append(m_i_enc)
        self.clusters[cluster_id].append(m_i_enc)
        self.add_to_clusters_by_day(cluster_id, message.day, m_i_enc)

    def score_matches(self, m_i):
        best_cluster = 0
        best_message = None
        best_score = -1
        for cluster_id, messages in self:
            for message in messages:
                obs_uid, risk, day, unobs_uid = decode_message(message)
                m = Message(obs_uid, risk, day, unobs_uid)
                if m_i.uid == m.uid and m_i.day == m.day:
                    best_cluster = cluster_id
                    best_message = message
                    best_score = 3
                    break
                elif m_i.uid[:3] == m.uid[1:] and m_i.day - 1 == m.day:
                    best_cluster = cluster_id
                    best_message = message
                    best_score = 2
                elif m_i.uid[:2] == m.uid[2:] and m_i.day - 2 == m.day:
                    best_cluster = cluster_id
                    best_message = message
                    best_score = 1
                elif m_i.uid[:1] == m.uid[3:] and m_i.day - 2 == m.day:
                    best_cluster = cluster_id
                    best_message = message
                    best_score = 0
                else:
                    best_cluster = cluster_id
                    best_message = message
                    best_score = -1

        if best_message:
            best_message = decode_message(best_message)
        return best_cluster, best_message, best_score

    def group_by_received_at(self, update_messages):
        TIME_THRESHOLD = datetime.timedelta(minutes=1)
        grouped_messages = defaultdict(list)
        for m1 in update_messages:
            if len(grouped_messages) == 0:
                grouped_messages[m1.received_at].append(m1)
            else:
                for received_at, m2 in grouped_messages.items():
                    if m1.received_at - received_at < TIME_THRESHOLD or m1.received_at + received_at < TIME_THRESHOLD:
                        grouped_messages[received_at].append(m1)
        return grouped_messages

    def update_record(self, old_cluster_id, new_cluster_id, message, updated_message):
        old_m_enc = encode_message(message)
        new_m_enc = encode_message(updated_message)
        del self.clusters[old_cluster_id][self.clusters[old_cluster_id].index(old_m_enc)]
        del self.all_messages[self.all_messages.index(old_m_enc)]
        del self.clusters_by_day[message.day][old_cluster_id][self.clusters_by_day[message.day][old_cluster_id].index(message)]

        self.clusters[new_cluster_id].append(encode_message(updated_message))
        self.all_messages.append(new_m_enc)
        self.add_to_clusters_by_day(new_cluster_id, updated_message)


    def update_records(self, update_messages):
        grouped_update_messages = self.group_by_received_at(update_messages)

        for received_at, update_messages in grouped_update_messages.items():
            updated_messages = []
            best_clusters = []
            best_messages = []
            for update_message in update_messages:
                best_cluster, best_message, best_score = self.score_matches(update_message)
                best_clusters.append(best_cluster)
                best_messages.append(best_message)
                updated_message = Message(best_message.uid, update_message.new_risk, best_message.day, best_message.unobs_id)
                self.update_record(best_cluster, best_cluster, best_message, updated_message)
            # for i in range(len(updated_messages)):
                # change the cluster
                # if len(self.clusters[best_cluster]) != len(update_messages):
                #     self.update_record(best_clusters[i], len(self), best_messages[i], updated_messages[i])
                # # keep the old cluster and just update the risk
                # else:
                # self.update_record(best_clusters[i], best_clusters[i], best_messages[i], updated_messages[i])
        return self

    def purge(self, current_day):
        for cluster_id, messages in self.clusters_by_day[current_day - 14].items():
            for message in messages:
                del self.clusters[cluster_id][self.clusters[cluster_id].index(message)]
                del self.all_messages[self.all_messages.index(message)]
        to_purge = []
        for cluster_id, messages in self.clusters.items():
            if len(self.clusters[cluster_id]) == 0:
                to_purge.append(cluster_id)
        for cluster_id in to_purge:
            del self.clusters[cluster_id]
        if current_day - 14 >= 0:
            del self.clusters_by_day[current_day - 14]
        self.update_messages = []

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return len(self.clusters.keys())

    @property
    def num_messages(self):
        return len(self.all_messages)

    def __next__(self):
        if self.n <= len(self.clusters) - 1:
            to_ret = (self.n, self.clusters[self.n])
            self.n += 1
            return to_ret
        else:
            raise StopIteration
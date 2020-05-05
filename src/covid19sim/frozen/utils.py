from collections import namedtuple, defaultdict

Message = namedtuple('message', 'uid risk day unobs_id')
UpdateMessage = namedtuple('update_message', 'uid new_risk risk day received_at unobs_id')


def encode_message(message):
    # encode a contact message as a list
    return [*message]


def encode_update_message(message):
    # encode a contact message as a list
    return [*message]


def decode_message(message):
    return Message(*message)


def decode_update_message(update_message):
    return UpdateMessage(*update_message)


def create_new_uid(rng):
    # generate a 4 bit random code
    return rng.randint(0, 16)


def update_uid(uid, rng):
    uid = "{0:b}".format(uid).zfill(4)[1:]
    uid += rng.choice(['1', '0'])
    return int(uid, 2)


def hash_to_cluster(message):
    """ This function grabs the 8-bit code for the message """
    bin_uid = "{0:b}".format(message.uid).zfill(4)
    bin_risk = "{0:b}".format(message.risk).zfill(4)
    binary = "".join([bin_uid, bin_risk])
    cluster_id = int(binary, 2)
    return cluster_id


def hash_to_cluster_day(message):
    """ Get the possible clusters based off UID (and risk) """
    clusters = defaultdict(list)
    bin_uid = "{0:b}".format(message.uid).zfill(4)
    bin_risk = "{0:b}".format(message.risk).zfill(4)

    for days_apart in range(1, 4):
        if days_apart == 1:
            for possibility in ["0", "1"]:
                bin_uid = "{0:b}".format(int(possibility + bin_uid[:3], 2)).zfill(4)
                binary = "".join([bin_uid, bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
        if days_apart == 2:
            for possibility in ["00", "01", "10", "11"]:
                bin_uid = "{0:b}".format(int(possibility + bin_uid[:2], 2)).zfill(4)
                binary = "".join([bin_uid, bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
        if days_apart == 3:
            for possibility in ["000", "001", "011", "010", "100", "101", "110", "111"]:
                bin_uid = "{0:b}".format(int(possibility + bin_uid[:1], 2)).zfill(4)
                binary = "".join([bin_uid, bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
    return clusters

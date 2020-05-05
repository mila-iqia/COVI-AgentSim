import numpy as np
import typing
from collections import namedtuple, defaultdict

import covid19sim.frozen.message_utils as new_utils

Message = namedtuple('message', 'uid risk day unobs_id')
UpdateMessage = namedtuple('update_message', 'uid new_risk risk day received_at unobs_id')


def convert_message_to_new_format(
        message: typing.Union[typing.AnyStr, Message, UpdateMessage],
) -> new_utils.GenericMessageType:
    """Converts a message (string or namedtuple) to its new dataclass format.

    Note that this will leave some unobserved attributes (e.g. real receiver UID) empty.
    """
    if isinstance(message, Message):
        return new_utils.EncounterMessage(
            uid=message.uid,
            risk_level=message.risk,
            encounter_time=message.day,
            _sender_uid=message.unobs_id,
        )
    elif isinstance(message, UpdateMessage):
        return new_utils.UpdateMessage(
            uid=message.uid,
            old_risk_level=message.risk,
            new_risk_level=message.new_risk,
            encounter_time=message.day,
            update_time=message.received_at,
            _sender_uid=message.unobs_id,
        )
    else:
        assert isinstance(message, str) and "_" in message, \
            f"unexpected old message type: {type(message)}"
        attribs = message.split("_")
        assert len(attribs) == 4 or len(attribs) == 6, \
            f"unexpected string attrib count ({len(attribs)}); should be 4 (encounter) or 6 (update)"
        if len(attribs) == 4:
            return new_utils.EncounterMessage(
                uid=np.uint8(int(attribs[0])),
                risk_level=np.uint8(int(attribs[1])),
                encounter_time=np.int64(int(attribs[2])),
                _sender_uid=np.uint8(int(attribs[3])),
            )
        else:
            return new_utils.UpdateMessage(
                uid=np.uint8(int(attribs[0])),
                old_risk_level=np.uint8(int(attribs[2])),
                new_risk_level=np.uint8(int(attribs[1])),
                encounter_time=np.int64(int(attribs[3])),
                update_time=np.int64(int(attribs[4])),
                _sender_uid=np.uint8(int(attribs[5])),
            )


def convert_message_to_old_format(
        message: new_utils.GenericMessageType,
) -> typing.Union[Message, UpdateMessage]:
    """Converts a message (in the new dataclass format) to its old namedtuple format.

    Note that we will not to type conversions or any value adaptation here.
    """
    if isinstance(message, new_utils.EncounterMessage):
        return Message(message.uid, message.risk_level, message.encounter_time, message._sender_uid)
    elif isinstance(message, new_utils.UpdateMessage):
        return UpdateMessage(
            message.uid, message.new_risk_level, message.old_risk_level,
            message.encounter_time, message.update_time, message._sender_uid,
        )
    else:
        raise AssertionError(f"unexpected old message type: {type(message)}")


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

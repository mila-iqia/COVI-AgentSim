import numpy as np
import typing
from collections import namedtuple, defaultdict

import covid19sim.frozen.message_utils as new_utils

Message = namedtuple('message', 'uid risk day unobs_id')
UpdateMessage = namedtuple('update_message', 'uid new_risk risk day received_at unobs_id')


def convert_messages_to_batched_new_format(
        messages: typing.List[typing.Union[typing.AnyStr, Message, UpdateMessage]],
        encoded_exposure_message: typing.Optional[typing.AnyStr] = None,
) -> typing.List[typing.List[new_utils.GenericMessageType]]:
    batched_messages_map = {}
    # we will batch messages based on observable variables only, but return the full messages in lists
    for message in messages:
        message = convert_message_to_new_format(message)
        if isinstance(message, new_utils.EncounterMessage):
            msg_code = (message.uid, message.risk_level, message.encounter_time)
        else:
            msg_code = (message.uid, message.old_risk_level, message.new_risk_level,
                        message.encounter_time, message.update_time,)
        if msg_code not in batched_messages_map:
            batched_messages_map[msg_code] = []
        batched_messages_map[msg_code].append(message)
    batched_messages = list(batched_messages_map.values())
    if encoded_exposure_message:
        # since the original messages do not carry the 'exposition' flag, we have to set it manually:
        exposure_message = convert_message_to_new_format(encoded_exposure_message)
        for batch in batched_messages:
            # hopefully this will only match the proper messages (based on _real_sender_id comparisons)...
            for m in batch:
                if isinstance(m, new_utils.EncounterMessage) and m == exposure_message:
                    m._exposition_event = True
    return batched_messages


def convert_message_to_new_format(
        message: typing.Union[typing.AnyStr, Message, UpdateMessage],
) -> new_utils.GenericMessageType:
    """Converts a message (string or namedtuple) to its new dataclass format.

    Note that this will leave some unobserved attributes (e.g. real receiver UID) empty.
    """
    # @@@@@ _exposition_event
    if isinstance(message, Message):
        return new_utils.EncounterMessage(
            uid=new_utils.UIDType(message.uid),
            risk_level=new_utils.RiskLevelType(message.risk),
            encounter_time=new_utils.TimestampType(message.day),
            _sender_uid=new_utils.RealUserIDType(message.unobs_id.split(":")[-1]),
        )
    elif isinstance(message, UpdateMessage):
        return new_utils.UpdateMessage(
            uid=new_utils.UIDType(message.uid),
            old_risk_level=new_utils.RiskLevelType(message.risk),
            new_risk_level=new_utils.RiskLevelType(message.new_risk),
            encounter_time=new_utils.TimestampType(message.day),
            update_time=new_utils.TimestampType(message.received_at),
            _sender_uid=new_utils.RealUserIDType(message.unobs_id.split(":")[-1]),
        )
    else:
        assert isinstance(message, str) and "_" in message, \
            f"unexpected old message type: {type(message)}"
        attribs = message.split("_")
        assert len(attribs) == 4 or len(attribs) == 6, \
            f"unexpected string attrib count ({len(attribs)}); should be 4 (encounter) or 6 (update)"
        if len(attribs) == 4:
            return new_utils.EncounterMessage(
                uid=new_utils.UIDType(attribs[0]),
                risk_level=new_utils.RiskLevelType(attribs[1]),
                encounter_time=new_utils.TimestampType(attribs[2]),
                _sender_uid=new_utils.RealUserIDType(attribs[3].split(":")[-1]),
            )
        else:
            return new_utils.UpdateMessage(
                uid=new_utils.UIDType(attribs[0]),
                old_risk_level=new_utils.RiskLevelType(attribs[2]),
                new_risk_level=new_utils.RiskLevelType(attribs[1]),
                encounter_time=new_utils.TimestampType(attribs[3]),
                update_time=new_utils.TimestampType(attribs[4]),
                _sender_uid=new_utils.RealUserIDType(attribs[5].split(":")[-1]),
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

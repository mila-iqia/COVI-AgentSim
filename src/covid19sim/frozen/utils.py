import datetime
import time
import typing
from collections import namedtuple, defaultdict

import covid19sim.frozen.message_utils as new_utils

Message = namedtuple('message', 'uid risk day unobs_id')
UpdateMessage = namedtuple('update_message', 'uid new_risk risk day received_at unobs_id')


def convert_messages_to_batched_new_format(
        messages: typing.List[typing.Union[typing.AnyStr, Message, UpdateMessage,
                                           new_utils.EncounterMessage, new_utils.UpdateMessage]],
        exposure_message: typing.Optional[typing.Union[typing.AnyStr, Message,
                                                       new_utils.EncounterMessage]] = None,
) -> typing.List[typing.List[new_utils.GenericMessageType]]:
    # we will batch messages based on observable variables only, but return the full messages in lists
    # note: try to keep batches sequential instead of reshuffling them for backwards compat
    batched_messages = []
    latest_msg_code = None
    latest_msg_batch = []
    for message in messages:
        if not isinstance(message, (new_utils.EncounterMessage, new_utils.UpdateMessage)):
            message = convert_message_to_new_format(message)
        if isinstance(message, new_utils.EncounterMessage):
            msg_code = (message.uid, message.risk_level, message.encounter_time)
        else:
            msg_code = (message.uid, message.old_risk_level, message.new_risk_level,
                        message.encounter_time, message.update_time,)
        if latest_msg_code and msg_code == latest_msg_code:
            latest_msg_batch.append(message)
        else:
            if latest_msg_code:
                batched_messages.append(latest_msg_batch)
            latest_msg_code = msg_code
            latest_msg_batch = [message]
    if latest_msg_code:
        batched_messages.append(latest_msg_batch)
    if exposure_message:
        if not isinstance(exposure_message, new_utils.EncounterMessage):
            exposure_message = convert_message_to_new_format(exposure_message)
        # since the original messages do not carry the 'exposition' flag, we have to set it manually:
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
        assert isinstance(message, list) and (len(message) == 4 or len(message) == 6), \
            f"unexpected old message type: {message}"
        if len(message) == 4:
            return new_utils.EncounterMessage(
                uid=new_utils.UIDType(message[0]),
                risk_level=new_utils.RiskLevelType(message[1]),
                encounter_time=new_utils.TimestampType(message[2]),
                _sender_uid=new_utils.RealUserIDType(message[3].split(":")[-1]),
            )
        else:
            return new_utils.UpdateMessage(
                uid=new_utils.UIDType(message[0]),
                old_risk_level=new_utils.RiskLevelType(message[2]),
                new_risk_level=new_utils.RiskLevelType(message[1]),
                encounter_time=new_utils.TimestampType(message[3]),
                update_time=new_utils.TimestampType(message[4]),
                _sender_uid=new_utils.RealUserIDType(message[5].split(":")[-1]),
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


def convert_json_to_new_format(
        block: typing.Union[typing.List[typing.Dict], typing.Dict],
        extract_self_reported_risks: bool,  # TODO: what should we use by default...?
        timestamp_offset: typing.Optional[datetime.datetime] = None,  # default = offset from epoch
        ticks_per_step: typing.Optional[new_utils.TimeOffsetType] = 24 * 60 * 60,

) -> typing.List[new_utils.GenericMessageType]:
    """Converts a JSON contact block (as defined in the data spec) into a list of
    messages that can be ingested by the clustering algorithm.

    A data block defines a set of update messages received by Alice for encounters
    on a specific (past) day at a specific geospatial location. The attributes in a
    contact block include the accuracy of the geospatial region and the griddate.

    If an update message has the same value for its old and new risk levels, it is
    assumed to be a fresh encounter with a new user at the block's griddate.
    """
    if isinstance(block, list):
        return [m for b in block for m in convert_json_to_new_format(
            block=b, extract_self_reported_risks=extract_self_reported_risks,
            timestamp_offset=timestamp_offset, ticks_per_step=ticks_per_step,
        )]
    assert isinstance(block, dict), f"unexpected contact block type: {type(block)}"
    assert all([isinstance(k, str) for k in block.keys()]), \
        "unexpected block key types; these should all be strings!"
    expected_block_keys = ["accuracy", "griddate", "risks"]
    assert all([k in block for k in expected_block_keys]), \
        f"a mandatory key is missing from the data block; it should have: {expected_block_keys}"
    # TODO: this function needs to contain the logic to convert the day-of-month integer
    #       for the update time to an actual date (currently, we parse it as-is)
    encounter_date = datetime.datetime.strptime(block["griddate"], "%Y-%m-%d")
    encounter_timestamp = new_utils.TimestampType(encounter_date.timestamp())
    if timestamp_offset is not None:
        encounter_timestamp -= timestamp_offset
    if ticks_per_step is not None:
        encounter_timestamp //= ticks_per_step
    encounter_messages, update_messages = [], []
    assert isinstance(block["risks"], dict)
    for sender_uid, messages in block["risks"].items():
        assert isinstance(sender_uid, str) and len(sender_uid) == 2
        sender_uid = new_utils.UIDType(int(sender_uid, 16))
        for message in messages:
            assert isinstance(message, dict) and len(message) == 1
            assert "0" in message and len(message["0"]) == 12
            bitstring = message["0"]
            if extract_self_reported_risks:
                new_risk_level = new_utils.RiskLevelType(int(bitstring[4:6], 16))
                old_risk_level = new_utils.RiskLevelType(int(bitstring[6:8], 16))
            else:  # otherwise, use test result risk levels
                new_risk_level = new_utils.RiskLevelType(int(bitstring[0:2], 16))
                old_risk_level = new_utils.RiskLevelType(int(bitstring[2:4], 16))
            if new_risk_level == old_risk_level:
                # special case used to identify new encounters
                encounter_messages.append(new_utils.EncounterMessage(
                    uid=sender_uid,
                    risk_level=new_risk_level,
                    encounter_time=encounter_timestamp
                ))
                continue
            update_day_of_month = int(bitstring[8:10], 16)
            if update_day_of_month < encounter_date.day:
                # just assume it was done last month, and loop the datetime
                lookback = encounter_date - datetime.timedelta(days=encounter_date.day)
                update_date = datetime.datetime(
                    year=lookback.year, month=lookback.month, day=update_day_of_month)
            else:
                update_date = datetime.datetime(
                    year=encounter_date.year, month=encounter_date.month, day=update_day_of_month)
            update_timestamp = new_utils.TimestampType(update_date.timestamp())
            if timestamp_offset is not None:
                update_timestamp -= timestamp_offset
            if ticks_per_step is not None:
                update_timestamp //= ticks_per_step
            update_messages.append(new_utils.UpdateMessage(
                uid=sender_uid,
                old_risk_level=old_risk_level,
                new_risk_level=new_risk_level,
                encounter_time=encounter_timestamp,
                update_time=update_timestamp,
            ))
    return [*encounter_messages, *update_messages]


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
                binary = "".join(["{0:b}".format(int(possibility + bin_uid[:3], 2)).zfill(4),
                                  bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
        if days_apart == 2:
            for possibility in ["00", "01", "10", "11"]:
                binary = "".join(["{0:b}".format(int(possibility + bin_uid[:2], 2)).zfill(4),
                                  bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
        if days_apart == 3:
            for possibility in ["000", "001", "011", "010", "100", "101", "110", "111"]:
                binary = "".join(["{0:b}".format(int(possibility + bin_uid[:1], 2)).zfill(4),
                                  bin_risk])
                cluster_id = int(binary, 2)
                clusters[days_apart].append(cluster_id)
    return clusters

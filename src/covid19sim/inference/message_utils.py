import collections
import dataclasses
import datetime
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.interventions.tracing import BaseMethod

TimestampType = datetime.datetime
TimeOffsetType = datetime.timedelta
TimestampDefault = datetime.datetime.utcfromtimestamp(0)
RealUserIDType = typing.Union[int, str]

UIDType = int  # should be at least 16 bytes for truly unique keys?
message_uid_bit_count = 128  # to be adjusted with the actual real bit count of the mailbox keys
message_uid_mask = UIDType((1 << message_uid_bit_count) - 1)

RiskLevelType = np.uint8
risk_level_bit_count = 4
risk_level_mask = RiskLevelType((1 << risk_level_bit_count) - 1)

GenericMessageType = typing.Union["EncounterMessage", "UpdateMessage"]


def create_new_uid(rng=None) -> UIDType:
    """Returns a randomly initialized uid."""
    if rng is None:
        rng = np.random
    if message_uid_bit_count <= 32:
        return UIDType(rng.randint(0, 1 << message_uid_bit_count))
    else:
        assert (message_uid_bit_count % 32) == 0, "missing implementation"
        uid, bits_left = 0, message_uid_bit_count
        while bits_left > 0:
            uid += rng.randint(0, 1 << 32) << (bits_left - 32)
            bits_left -= 32
        return uid


@dataclasses.dataclass
class EncounterMessage:
    """Contains all the observed+unobserved data related to a user encounter message."""

    #####################
    # observed variables

    uid: UIDType
    """Unique Identifier (UID) of the encountered user."""

    risk_level: RiskLevelType
    """Quantified risk level of the encountered user."""

    encounter_time: TimestampType
    """Discretized encounter timestamp."""

    #############################################
    # unobserved variables (for debugging only!)

    _sender_uid: typing.Optional[RealUserIDType] = None
    """Real Unique Identifier (UID) of the encountered user."""

    _receiver_uid: typing.Optional[RealUserIDType] = None
    """Real Unique Identifier (UID) of the user receiving the message."""

    _real_encounter_time: typing.Optional[TimestampType] = None
    """Real encounter timestamp."""

    _exposition_event: typing.Optional[bool] = None
    """Flags whether this encounter corresponds to an exposition event for the receiver."""

    _applied_update_count: typing.Optional[int] = None
    """List of update messages which have been applied to this encounter."""


@dataclasses.dataclass
class UpdateMessage:
    """Contains all the observed+unobserved data related to a user update message."""

    #####################
    # observed variables

    uid: UIDType
    """Unique Identifier (UID) of the updater at the time of the encounter."""

    old_risk_level: RiskLevelType
    """Previous quantified risk level of the updater."""

    new_risk_level: RiskLevelType
    """New quantified risk level of the updater."""

    encounter_time: TimestampType
    """Discretized encounter timestamp."""

    update_time: TimestampType
    """Update generation timestamp."""  # TODO: this might be a 1-31 rotating day id?

    #############################################
    # unobserved variables (for debugging only!)

    _sender_uid: typing.Optional[RealUserIDType] = None
    """Real Unique Identifier (UID) of the updater."""

    _receiver_uid: typing.Optional[RealUserIDType] = None
    """Real Unique Identifier (UID) of the user receiving the message."""

    _real_encounter_time: typing.Optional[TimestampType] = None
    """Real encounter timestamp."""

    _real_update_time: typing.Optional[TimestampType] = None
    """Real update generation timestamp."""

    _exposition_event: typing.Optional[bool] = None
    """Flags whether the original encounter corresponds to an exposition event for the receiver."""

    _update_reason: typing.Optional[str] = None
    """Reason why this update message was sent (for debugging)."""


def generate_encounter_message(
        sender: "Human",
        receiver: "Human",
        env_timestamp: TimestampType,
        use_gaen_key: bool = False,
) -> EncounterMessage:
    """Generates an encounter message to pass from a sender to a receiver.

    TODO: determine whether we should actually use a real 15-min lifespan key for exchanges?
    """
    return EncounterMessage(
        uid=create_new_uid() if use_gaen_key else None,
        risk_level=None,  # dirty hack, but it will do for now (we will have to check for None)
        encounter_time=datetime.datetime.combine(env_timestamp.date(), datetime.datetime.min.time()),
        _sender_uid=sender.name,
        _receiver_uid=receiver.name,
        _real_encounter_time=env_timestamp,
        _exposition_event=None,  # we don't decide this here, it will be done in the caller
    )


def exchange_encounter_messages(
        h1: "Human",
        h2: "Human",
        env_timestamp: datetime.datetime,
        initial_timestamp: datetime.datetime,
        use_gaen_key: bool = False,
) -> typing.Tuple[EncounterMessage, EncounterMessage]:
    """Creates & exchanges encounter messages between two humans.

    This function is written as part of the GAEN refactoring. This means that the only data that
    is exchanged directly between the users is a unique "mailbox key" using which they can later
    exchange update messages. As such, this function creates encounter messages that are kept
    in the user's own contact book as a way to keep track of who they contacted & need to update.

    In reality, the mailbox key would be updated every 15 minutes by each user, but here, for
    simplicity, we just create a new key for every encounter inside this function.

    Returns both newly created encounter messages, which contain the mailbox keys as uids. The
    contact books of both users will be updated by this function.
    """
    h1_msg = generate_encounter_message(h1, h2, env_timestamp, use_gaen_key=use_gaen_key)
    h2_msg = generate_encounter_message(h2, h1, env_timestamp, use_gaen_key=use_gaen_key)
    # the encounter messages above are essentially reminders that we need to update that contact
    curr_day_idx = (env_timestamp - initial_timestamp).days
    assert 0 <= curr_day_idx
    if curr_day_idx not in h1.contact_book.encounters_by_day:
        assert curr_day_idx not in h1.contact_book.mailbox_keys_by_day
        h1.contact_book.encounters_by_day[curr_day_idx] = []
        h1.contact_book.mailbox_keys_by_day[curr_day_idx] = []
    h1.contact_book.encounters_by_day[curr_day_idx].append(h1_msg)
    h1.contact_book.mailbox_keys_by_day[curr_day_idx].append(h2_msg.uid)  # message uid == mailbox key
    if curr_day_idx not in h2.contact_book.encounters_by_day:
        assert curr_day_idx not in h2.contact_book.mailbox_keys_by_day
        h2.contact_book.encounters_by_day[curr_day_idx] = []
        h2.contact_book.mailbox_keys_by_day[curr_day_idx] = []
    h2.contact_book.encounters_by_day[curr_day_idx].append(h2_msg)
    h2.contact_book.mailbox_keys_by_day[curr_day_idx].append(h1_msg.uid)  # message uid == mailbox key
    return h1_msg, h2_msg


def create_update_message(
        encounter_message: EncounterMessage,
        new_risk_level: RiskLevelType,
        current_time: TimestampType,
        update_reason: typing.Optional[str] = None,
) -> UpdateMessage:
    """Creates and returns an update message for a given encounter.

    Args:
        encounter_message: the encounter message for which to create an update.
        new_risk_level: the new risk level of the sender.
        current_time: the current time of the simulation.
        update_reason: the (optional) reason why this update is being generated.

    Returns:
        The update message object to send.
    """
    assert new_risk_level <= message_uid_mask
    assert current_time >= encounter_message.encounter_time
    return UpdateMessage(
        uid=encounter_message.uid,
        old_risk_level=encounter_message.risk_level,
        new_risk_level=new_risk_level,
        encounter_time=encounter_message.encounter_time,
        update_time=datetime.datetime.combine(current_time.date(), datetime.datetime.min.time()),
        _sender_uid=encounter_message._sender_uid,
        _receiver_uid=encounter_message._receiver_uid,
        _real_encounter_time=encounter_message._real_encounter_time,
        _real_update_time=current_time,
        _exposition_event=encounter_message._exposition_event,
        _update_reason=update_reason,
    )


def create_encounter_from_update_message(
        update_message: UpdateMessage,
) -> EncounterMessage:
    """Creates and returns an encounter message for a given update message.

    It may happen later that an update message is received for which we cannot find a corresponding
    encounter. This will allow us to create a dummy encounter instead.

    Args:
        update_message: the update message.

    Returns:
        The compatible encounter message object, already updated to the latest risk level.
    """
    return EncounterMessage(
        uid=update_message.uid,
        risk_level=update_message.new_risk_level,
        encounter_time=update_message.encounter_time,
        _sender_uid=update_message._sender_uid,
        _receiver_uid=update_message._receiver_uid,
        _real_encounter_time=update_message._real_encounter_time,
        _exposition_event=None,  # cannot properly deduct if the original encounter was an exposition
    )


def create_updated_encounter_with_message(
        encounter_message: EncounterMessage,
        update_message: UpdateMessage,
        blind_update: bool = False,
) -> EncounterMessage:
    """Creates and returns a new encounter message based on the update of the provided one.

    This function will throw if any "observed" parameters of the messages are mismatched, and will
    silently set unobserved values to `None` if those are mismatched.

    Args:
        encounter_message: the encounter message object to create an updated copy from.
        update_message: the update message that will provide the new risk level.
        blind_update: defines whether to check for a UID match or not.

    Returns:
        A newly instantiated encounter message with the updated attributes.
    """
    if not blind_update:
        assert encounter_message.uid == update_message.uid
    assert encounter_message.risk_level == update_message.old_risk_level
    assert encounter_message.encounter_time == update_message.encounter_time
    old_update_count = encounter_message._applied_update_count \
        if encounter_message._applied_update_count else 0
    return EncounterMessage(
        uid=update_message.uid,
        risk_level=update_message.new_risk_level,
        encounter_time=update_message.encounter_time,
        _sender_uid=update_message._sender_uid if
            update_message._sender_uid == encounter_message._sender_uid else None,
        _receiver_uid=update_message._receiver_uid if
            update_message._receiver_uid == encounter_message._receiver_uid else None,
        _real_encounter_time=update_message._real_encounter_time if
            update_message._real_encounter_time == encounter_message._real_encounter_time else None,
        _exposition_event=encounter_message._exposition_event,
        _applied_update_count=old_update_count + 1,
    )


def combine_update_messages(
        oldest_update_message: UpdateMessage,
        newest_update_message: UpdateMessage,
        blind_update: bool = False,
) -> UpdateMessage:
    """Creates and returns a new update message based on the combination of two updates.

    This function will throw if any "observed" parameters of the messages are mismatched, and will
    silently set unobserved values to `None` if those are mismatched.

    Args:
        oldest_update_message: the older of the two update message objects to combine.
        newest_update_message: the newer of the two update message objects to combine.
        blind_update: defines whether to check for a UID match or not.

    Returns:
        A newly instantiated update message with the updated attributes.
    """
    if not blind_update:
        assert oldest_update_message.uid == newest_update_message.uid
    assert oldest_update_message.new_risk_level == newest_update_message.old_risk_level
    assert oldest_update_message.encounter_time == newest_update_message.encounter_time
    assert oldest_update_message.update_time <= newest_update_message.update_time
    return UpdateMessage(
        uid=newest_update_message.uid,
        old_risk_level=oldest_update_message.old_risk_level,
        new_risk_level=newest_update_message.new_risk_level,
        encounter_time=oldest_update_message.encounter_time,
        update_time=newest_update_message.update_time,
        _sender_uid=newest_update_message._sender_uid if
        oldest_update_message._sender_uid == newest_update_message._sender_uid else None,
        _receiver_uid=newest_update_message._receiver_uid if
        oldest_update_message._receiver_uid == newest_update_message._receiver_uid else None,
        _real_encounter_time=newest_update_message._real_encounter_time if
        oldest_update_message._real_encounter_time == newest_update_message._real_encounter_time else None,
        _real_update_time=newest_update_message._real_update_time if
        oldest_update_message._real_update_time == newest_update_message._real_update_time else None,
        _exposition_event=newest_update_message._exposition_event if
        oldest_update_message._exposition_event == newest_update_message._exposition_event else None,
        _update_reason=newest_update_message._update_reason,
    )


class ContactBook:
    """
    Contact book used to store all past encounters & provide a simple interface to query information
    for tracing.

    Each human owns a contact book. This contact book can be used (for simulation tracing only!) to
    gather statistics on the Nth-order contacts of its owner. By default, it will simply provide a
    way to know who to inform when update messages must be generated.
    """

    def __init__(
            self,
            tracing_n_days_history: int,
    ):
        """
        Initializes the contact book.

        Args:
            tracing_n_days_history: length of the contact history to keep in this object.
        """
        self.tracing_n_days_history = tracing_n_days_history
        # the encounters we keep here are the messages we sent, not the ones we received
        self.encounters_by_day: typing.Dict[int, typing.List[EncounterMessage]] = {}
        # the mailbox keys are used to fetch update messages and provide them to the clustering algo
        self.mailbox_keys_by_day: typing.Dict[int, typing.List[UIDType]] = {}
        self.latest_update_time = datetime.datetime.min
        self._is_being_traced = False  # used for internal tracing only

    def get_contacts(
            self,
            humans_map: typing.Dict[str, "Human"],
            only_with_initial_update: bool = False,
            make_sure_15min_minimum_between_contacts: bool = False,
    ) -> typing.List["Human"]:
        """Returns a list of all humans that the contact book owner encountered."""
        # note1: this is based on unobserved variables, so in reality, we can't use it on the app
        # note2: the 'only_with_initial_update' allows us to only fetch contacts that *should* have
        #        been confirmed by an initial update received at the contact book owner's timeslot;
        #        this is an approximation of what would really happen however, since we would need
        #        to check whether we have actually received an update from that contact
        output = {}
        real_human_encounter_times = collections.defaultdict(set)
        for day, msgs in self.encounters_by_day.items():
            for msg in msgs:
                if make_sure_15min_minimum_between_contacts:
                    real_human_encounter_times[msg._receiver_uid].add(msg._real_encounter_time)
                if msg._receiver_uid not in output and \
                        (not only_with_initial_update or msg.risk_level is not None):
                    output[msg._receiver_uid] = humans_map[msg._receiver_uid]
        if make_sure_15min_minimum_between_contacts:
            for human_name, encounter_times in real_human_encounter_times.items():
                encounter_times = sorted(list(encounter_times))
                for encounter_time_idx in range(1, len(encounter_times)):
                    assert (encounter_times[encounter_time_idx] -
                            encounter_times[encounter_time_idx - 1]) >= datetime.timedelta(minutes=15)
        return list(output.values())

    def get_positive_contacts_counts(
            self,
            humans_map: typing.Dict[str, "Human"],
            max_order: int = 1,
            curr_order: int = 0,
            count_map: typing.Optional[typing.Dict[int, int]] = None,
            make_sure_15min_minimum_between_contacts: bool = False,
    ) -> typing.Dict[int, int]:  # returns order-to-count mapping
        """Traces and returns the number of Nth-order contacts that have been tested positive."""
        if curr_order == 0:
            for human in humans_map.values():
                human.contact_book._is_being_traced = False
            self._is_being_traced = True
            count_map = collections.defaultdict(int)
        elif self._is_being_traced:
            return count_map
        for contact in self.get_contacts(
                humans_map,
                only_with_initial_update=True,
                # note: the check below is really just for an assert, it doesn't change behavior
                make_sure_15min_minimum_between_contacts=make_sure_15min_minimum_between_contacts,
        ):
            assert contact.has_app, "how can we be tracing this person without an app?"
            if not contact.contact_book._is_being_traced and \
                    contact.reported_test_result == "positive":
                count_map[curr_order + 1] += 1
                contact.contact_book._is_being_traced = True
            if max_order > curr_order + 1:
                contact.contact_book.get_positive_contacts_counts(
                    humans_map=humans_map,
                    max_order=max_order,
                    curr_order=curr_order + 1,
                    count_map=count_map,
                    make_sure_15min_minimum_between_contacts=make_sure_15min_minimum_between_contacts,
                )
        return count_map


    def get_risk_level_change_score(
            self,
            prev_risk_history_map: typing.Dict[int, float],
            curr_risk_history_map: typing.Dict[int, float],
            proba_to_risk_level_map: typing.Callable,
    ):
        """Returns the 'risk level change' score used for GAEN message impact estimation.

        Args:
            prev_risk_history_map: the previous risk history map of the human who owns this contact book.
            curr_risk_history_map: the current risk history map of the human who owns this contact book.
            proba_to_risk_level_map: the risk-probability-to-risk-level mapping function.

        Returns:
            The risk level change score (a numeric value).
        """
        change = 0
        for day_idx in set(prev_risk_history_map.keys()) & set(curr_risk_history_map.keys()):
            old_risk_level = min(proba_to_risk_level_map(prev_risk_history_map[day_idx]), 15)
            curr_risk_level = min(proba_to_risk_level_map(curr_risk_history_map[day_idx]), 15)
            n_encs_for_day = len(self.encounters_by_day.get(day_idx, []))
            change += abs(curr_risk_level - old_risk_level) * n_encs_for_day
        return change  # Danger potential PII => fewer bits

    def cleanup_contacts(
            self,
            init_timestamp: TimestampType,
            current_timestamp: TimestampType,
    ):
        """Removes all sent/received encounter messages older than TRACING_N_DAYS_HISTORY."""
        current_day_idx = (current_timestamp - init_timestamp).days
        self.encounters_by_day = {
            day: msgs for day, msgs in self.encounters_by_day.items()
            if day >= current_day_idx - self.tracing_n_days_history
        }
        self.mailbox_keys_by_day = {
            day: keys for day, keys in self.mailbox_keys_by_day.items()
            if day >= current_day_idx - self.tracing_n_days_history
        }

    def generate_initial_updates(
            self,
            current_day_idx: int,
            current_timestamp: datetime.datetime,
            risk_history_map: typing.Dict[int, float],
            proba_to_risk_level_map: typing.Callable,
            intervention: typing.Optional["BaseMethod"],
    ):
        """
        Generates and returns the update messages needed to communicate the initial risk
        level of a recent encounter.

        If the tracing method is not yet defined, does nothing.

        Args:
            current_day_idx: the current day index inside the simulation.
            current_timestamp: the current timestamp of the simulation.
            risk_history_map: the risk history map of the human who owns this contact book.
            proba_to_risk_level_map: the risk-probability-to-risk-level mapping function.
            intervention: intervention object that holds the settings used for tracing.

        Returns:
            A list of update messages to send out to contacts (if any).
        """
        update_messages = []
        if intervention is None:
            return update_messages  # no need to generate update messages until tracing is enabled
        assert current_day_idx >= 0
        for encounter_day_idx, encounter_messages in self.encounters_by_day.items():
            for encounter_message in encounter_messages:
                if encounter_message.risk_level is None:  # we never sent the first update w/ the risk level
                    assert 0 <= encounter_day_idx <= current_day_idx, \
                        "can't have encounters before init or after today...?"
                    assert encounter_day_idx in risk_history_map, \
                        "how could we have an encounter without a risk at that point? use default?"
                    encounter_message.risk_level = \
                        min(proba_to_risk_level_map(risk_history_map[encounter_day_idx]), 15)
                    update_messages.append(
                        create_update_message(
                            encounter_message=encounter_message,
                            new_risk_level=RiskLevelType(encounter_message.risk_level),
                            current_time=current_timestamp,
                            update_reason="contact",
                        )
                    )
        return update_messages

    def generate_updates(
            self,
            current_day_idx: int,
            current_timestamp: datetime.datetime,
            prev_risk_history_map: typing.Dict[int, float],
            curr_risk_history_map: typing.Dict[int, float],
            proba_to_risk_level_map: typing.Callable,
            update_reason: str,
            intervention: typing.Optional["BaseMethod"],
    ):
        """
        Will check the human's previously registered risk level over the past 'TRACING_N_DAYS_HISTORY'
        and compare it with the latest values, sending update messages if necessary.

        If the tracing method is not yet defined, does nothing.

        Args:
            current_day_idx: the current day index inside the simulation.
            current_timestamp: the current timestamp of the simulation.
            prev_risk_history_map: the previous risk history map of the human who owns this contact book.
            curr_risk_history_map: the current risk history map of the human who owns this contact book.
            proba_to_risk_level_map: the risk-probability-to-risk-level mapping function.
            update_reason: defines the root cause of the updates (as a string).
            intervention: intervention object that holds the settings used for tracing.

        Returns:
            A list of update messages to send out to contacts (if any).
        """
        update_messages = []
        if intervention is None:
            return update_messages  # no need to generate update messages until tracing is enabled
        assert current_day_idx >= 0
        for encounter_day_idx, encounter_messages in self.encounters_by_day.items():
            assert current_day_idx - encounter_day_idx <= self.tracing_n_days_history, \
                "contact book should have been cleaned up before calling update method...?"
            if encounter_day_idx not in prev_risk_history_map.keys():
                prev_risk_history_map[encounter_day_idx] = curr_risk_history_map[encounter_day_idx]
                continue
            old_risk_level = min(proba_to_risk_level_map(prev_risk_history_map[encounter_day_idx]), 15)
            new_risk_level = min(proba_to_risk_level_map(curr_risk_history_map[encounter_day_idx]), 15)
            if old_risk_level != new_risk_level:
                for encounter_idx, encounter_message in enumerate(encounter_messages):
                    assert encounter_message.risk_level is not None, \
                        "should have already initialized all encounters before updating...?"
                    assert encounter_message.risk_level == old_risk_level or \
                        encounter_message.risk_level == new_risk_level, \
                        "encounter message risk mismatch (should have old level if already initialized " \
                        "or new level if it was initialized just now, but nothing else)"
                    if encounter_message.risk_level != new_risk_level:
                        update_messages.append(
                            create_update_message(
                                encounter_message=encounter_message,
                                new_risk_level=RiskLevelType(new_risk_level),
                                current_time=current_timestamp,
                                update_reason=update_reason,
                            )
                        )
                        encounter_messages[encounter_idx] = create_updated_encounter_with_message(
                            encounter_message, update_messages[-1],  # to keep track of applied updates internally...
                        )
        return update_messages


def batch_messages(
        messages: typing.List[GenericMessageType],
) -> typing.List[typing.Dict[TimestampType, typing.List[GenericMessageType]]]:
    """Creates batches of messages based on their observable risk level attributes for faster clustering."""
    batched_encounter_messages = collections.defaultdict(list)
    batched_update_messages = collections.defaultdict(list)
    for message in messages:
        if isinstance(message, EncounterMessage):
            msg_code = (message.risk_level, message.encounter_time)
            batched_encounter_messages[msg_code].append(message)
        else:
            msg_code = (message.old_risk_level, message.new_risk_level,
                        message.encounter_time, message.update_time)
            batched_update_messages[msg_code].append(message)
    output = []
    for msgs in batched_encounter_messages.values():
        batched_messages_by_timestamps = {}
        for msg in msgs:
            if msg.encounter_time not in batched_messages_by_timestamps:
                batched_messages_by_timestamps[msg.encounter_time] = []
            batched_messages_by_timestamps[msg.encounter_time].append(msg)
        output.append(batched_messages_by_timestamps)
    for msgs in batched_update_messages.values():
        batched_messages_by_timestamps = {}
        for msg in msgs:
            if msg.encounter_time not in batched_messages_by_timestamps:
                batched_messages_by_timestamps[msg.encounter_time] = []
            batched_messages_by_timestamps[msg.encounter_time].append(msg)
        output.append(batched_messages_by_timestamps)
    return output


def convert_json_to_messages(
        block: typing.Union[typing.List[typing.Dict], typing.Dict],
        extract_self_reported_risks: bool,  # TODO: what should we use by default...?
        timestamp_offset: typing.Optional[datetime.datetime] = None,  # default = offset from epoch

) -> typing.List[GenericMessageType]:
    """Converts a JSON contact block (as defined in the data spec) into a list of
    messages that can be ingested by the clustering algorithm.

    A data block defines a set of update messages received by Alice for encounters
    on a specific (past) day at a specific geospatial location. The attributes in a
    contact block include the accuracy of the geospatial region and the griddate.

    If an update message has the same value for its old and new risk levels, it is
    assumed to be a fresh encounter with a new user at the block's griddate.
    """
    # FIXME: outdated, will likely blow up if mixing timestamps/ints for time type
    if isinstance(block, list):
        return [m for b in block for m in convert_json_to_messages(
            block=b, extract_self_reported_risks=extract_self_reported_risks,
            timestamp_offset=timestamp_offset
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
    if TimestampType == datetime.datetime:
        encounter_timestamp = encounter_date
    else:
        assert TimestampType in (int, float)
        encounter_timestamp = TimestampType(encounter_date.timestamp())
    if timestamp_offset is not None:
        encounter_timestamp -= timestamp_offset
    encounter_messages, update_messages = [], []
    assert isinstance(block["risks"], dict)
    for sender_uid, messages in block["risks"].items():
        assert isinstance(sender_uid, str) and len(sender_uid) == 2
        sender_uid = UIDType(int(sender_uid, 16))
        for message in messages:
            assert isinstance(message, dict) and len(message) == 1
            assert "0" in message and len(message["0"]) == 12
            bitstring = message["0"]
            if extract_self_reported_risks:
                new_risk_level = RiskLevelType(int(bitstring[4:6], 16))
                old_risk_level = RiskLevelType(int(bitstring[6:8], 16))
            else:  # otherwise, use test result risk levels
                new_risk_level = RiskLevelType(int(bitstring[0:2], 16))
                old_risk_level = RiskLevelType(int(bitstring[2:4], 16))
            if new_risk_level == old_risk_level:
                # special case used to identify new encounters
                encounter_messages.append(EncounterMessage(
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
            if TimestampType == datetime.datetime:
                update_timestamp = update_date
            else:
                assert TimestampType in (int, float)
                update_timestamp = TimestampType(update_date.timestamp())
            if timestamp_offset is not None:
                update_timestamp -= timestamp_offset
            update_messages.append(UpdateMessage(
                uid=sender_uid,
                old_risk_level=old_risk_level,
                new_risk_level=new_risk_level,
                encounter_time=encounter_timestamp,
                update_time=update_timestamp,
            ))
    return [*encounter_messages, *update_messages]

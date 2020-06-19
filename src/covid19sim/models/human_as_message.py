import datetime
import typing

import collections
import dataclasses
import numpy as np
from covid19sim.frozen.helper import conditions_to_np, encode_age, encode_sex, \
    symptoms_to_np
from covid19sim.frozen.message_utils import UpdateMessage
if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.city import PersonalMailboxType

def make_human_as_message(
        human: "Human",
        personal_mailbox: "PersonalMailboxType",
        conf: typing.Dict,
):
    """
    Creates a human dataclass from a super-ultra-way-too-heavy human object.

    The returned dataclass can more properly be serialized and sent to a remote server for
    clustering/inference. All update messages aimed at the given human found in the global
    mailbox will be popped and added to the dataclass.

    Args:
        human: the human object to convert.
        personal_mailbox: the personal mailbox dictionary to fetch update messages from.
        conf: YAML configuration dictionary with all relevant settings for the simulation.
    Returns:
        The nice-and-slim human dataclass object.
    """
    preexisting_conditions = conditions_to_np(human.preexisting_conditions)
    obs_preexisting_conditions = conditions_to_np(human.obs_preexisting_conditions)
    rolling_all_symptoms = symptoms_to_np(human.rolling_all_symptoms, conf)
    rolling_all_reported_symptoms = symptoms_to_np(human.rolling_all_reported_symptoms, conf)

    # TODO: we could index the global mailbox by day, it might be faster that way
    target_mailbox_keys = [key for keys in human.contact_book.mailbox_keys_by_day.values() for key in keys]
    update_messages = []
    for key in target_mailbox_keys:
        if key in personal_mailbox:
            assert isinstance(personal_mailbox[key], list)
            update_messages.extend(personal_mailbox.pop(key))

    return HumanAsMessage(
        name=human.name,
        age=encode_age(human.age),
        sex=encode_sex(human.sex),
        obs_age=encode_age(human.obs_age),
        obs_sex=encode_sex(human.obs_sex),
        preexisting_conditions=preexisting_conditions,
        obs_preexisting_conditions=obs_preexisting_conditions,

        infectiousnesses=human.infectiousnesses,
        infection_timestamp=human.infection_timestamp,
        recovered_timestamp=human.recovered_timestamp,
        test_results=human.get_test_results_array(human.env.timestamp),
        rolling_all_symptoms=rolling_all_symptoms,
        rolling_all_reported_symptoms=rolling_all_reported_symptoms,
        incubation_days=human.incubation_days,
        recovery_days=human.recovery_days,

        update_messages=update_messages,
        carefulness=human.carefulness,
        has_app=human.has_app
    )


@dataclasses.dataclass
class HumanAsMessage:
    # Static fields
    name: str
    age: int
    sex: str
    obs_age: int
    obs_sex: str
    preexisting_conditions: np.array
    obs_preexisting_conditions: np.array

    # Medical fields
    infectiousnesses: typing.Iterable
    # TODO: Should be reformatted to int timestamp
    infection_timestamp: datetime.datetime
    # TODO: Should be reformatted to int timestamp
    recovered_timestamp: datetime.datetime
    # TODO: Should be reformatted to deque of (int, int timestamp)
    test_results: collections.deque
    rolling_all_symptoms: np.array
    rolling_all_reported_symptoms: np.array
    incubation_days: int  # NOTE: FOR NOW, USED FOR TESTING/DEBUGGING ONLY
    recovery_days: int  # NOTE: FOR NOW, USED FOR TESTING/DEBUGGING ONLY

    # Risk-level-related fields
    update_messages: typing.List[UpdateMessage]
    carefulness: float
    has_app: bool

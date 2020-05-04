import dataclasses
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu


@dataclasses.dataclass
class Visit:
    visitor_real_uid: np.uint64
    visited_real_uid: np.uint64
    exposition: bool
    timestamp: np.uint64
    visitor_uid: typing.Optional[mu.UIDType] = None
    visited_uid: typing.Optional[mu.UIDType] = None
    visitor: typing.Optional["FakeHuman"] = None
    visited: typing.Optional["FakeHuman"] = None


@dataclasses.dataclass
class FakeHuman:

    def __init__(
            self,
            real_uid: int,
            exposition_timestamp: int,
            visits_to_adopt: typing.List[Visit],
            force_init_uid: typing.Optional[np.uint8] = None,
            force_init_risk: typing.Optional[np.uint8] = None,
            allow_spurious_exposition: bool = False,
    ):
        self.real_uid = np.uint64(real_uid)
        self.made_visit_to = [v for v in visits_to_adopt if v.visitor_real_uid == real_uid]
        self.got_visit_from = [v for v in visits_to_adopt if v.visited_real_uid == real_uid]
        self.visits = [v for v in visits_to_adopt
                       if v.visitor_real_uid == real_uid or v.visited_real_uid == real_uid]
        if force_init_uid is None:
            force_init_uid = mu.create_new_uid()
        if force_init_risk is None:
            force_init_risk = np.uint8(not exposition_timestamp)
        self.rolling_uids = np.asarray([force_init_uid])
        self.rolling_exposed = np.asarray([exposition_timestamp == 0])
        self.rolling_risk = np.asarray([force_init_risk])
        self.rolling_visits = [[v for v in self.visits if v.timestamp == 0]]
        max_timestamp = max([v.timestamp for v in self.visits]) if self.visits else 1
        for timestamp in range(1, int(max_timestamp) + 1):
            self.rolling_uids = np.append(self.rolling_uids, mu.update_uid(self.rolling_uids[-1]))
            self.rolling_exposed = np.append(self.rolling_exposed, timestamp >= exposition_timestamp)
            # we gradually increase risk level from time of exposition time (just faking things)
            self.rolling_risk = np.append(self.rolling_risk,
                                          self.rolling_risk[-1] + 1 if timestamp >= exposition_timestamp
                                          else force_init_risk)
            self.rolling_visits.append([v for v in self.visits if v.timestamp == timestamp])
        for v in self.visits:
            # note: under the current logic, only the visitor can infect the visited
            assert v.visitor_real_uid != v.visited_real_uid
            if v.visitor_real_uid == real_uid:
                v.visitor_uid = self.rolling_uids[v.timestamp]
                v.visitor = self
            if v.visited_real_uid == real_uid:
                v.visited_uid = self.rolling_uids[v.timestamp]
                v.visited = self
            if not allow_spurious_exposition:
                # make sure the input exposition matches with the visit logic
                assert not v.exposition or self.rolling_exposed[v.timestamp]
            elif v.exposition and v.visitor_real_uid == real_uid and not self.rolling_exposed[v.timestamp]:
                # allow visits to set the exposition timestamp for this human
                for timestamp in range(int(v.timestamp), int(max_timestamp) + 1):
                    self.rolling_exposed[timestamp] = True
                    self.rolling_risk[timestamp] = self.rolling_risk[max(timestamp - 1, 0)] + 1


def generate_sent_messages(
        humans: typing.List[FakeHuman],
        minimum_risk_level_for_updates: int = 5,
        maximum_risk_level_for_saturaton: int = 10,
):
    """Returns a user-to-sent-messages mapping for clustering logic testing."""
    output = {}
    for human_idx, human in enumerate(humans):
        assert human.real_uid == human_idx
        sent_encounter_messages, sent_update_messages, sent_messages = {}, {}, {}
        for timestamp, visits in enumerate(human.rolling_visits):
            sent_encounter_messages[timestamp] = []
            sent_update_messages[timestamp] = []
            sent_messages[timestamp] = []
            # always do encounters first, send updates at the end of the step
            for visit in visits:
                opposite_human = visit.visited if visit in human.made_visit_to else visit.visitor
                opposite_uid = visit.visited_uid if visit in human.made_visit_to else visit.visitor_uid
                assert opposite_human.rolling_uids[timestamp] == opposite_uid
                risk_level = human.rolling_risk[timestamp]
                if risk_level <= minimum_risk_level_for_updates:
                    risk_level = 0  # fake risk suppression while exposition is unknown
                is_visited = visit.visited_real_uid == human.real_uid
                encounter_message = mu.EncounterMessage(
                    uid=human.rolling_uids[timestamp],
                    risk_level=risk_level,
                    encounter_time=np.uint64(timestamp),
                    _sender_uid=human.real_uid,
                    _receiver_uid=opposite_human.real_uid,
                    _real_encounter_time=np.uint64(timestamp),
                    _exposition_event=visit.exposition and is_visited,
                )
                sent_encounter_messages[timestamp].append(encounter_message)
                sent_messages[timestamp].append(encounter_message)
            if human.rolling_risk[timestamp] == maximum_risk_level_for_saturaton:
                # broadcast update messages systematically once risk threshold is reached
                for prev_timestamp in range(timestamp + 1):
                    # cutoff back updates using the arbitrary null risk inflexion point
                    if human.rolling_risk[prev_timestamp] > 0:
                        for message in sent_encounter_messages[prev_timestamp]:
                            # assume message risk has already been updated; create manually
                            update_message = mu.UpdateMessage(
                                uid=message.uid,
                                old_risk_level=human.rolling_risk[prev_timestamp],
                                new_risk_level=np.uint8(maximum_risk_level_for_saturaton),
                                encounter_time=message.encounter_time,
                                update_time=np.uint64(timestamp),
                                _sender_uid=message._sender_uid,
                                _receiver_uid=message._receiver_uid,
                                _real_encounter_time=message._real_encounter_time,
                                _real_update_time=np.uint64(timestamp),
                                _update_reason="positive_test",
                            )
                            sent_update_messages[timestamp].append(update_message)
                            sent_messages[timestamp].append(update_message)
            elif human.rolling_risk[timestamp] == minimum_risk_level_for_updates:
                # broadcast update messages systematically once risk threshold is reached
                for prev_timestamp in range(timestamp + 1):
                    # cutoff back updates using the arbitrary null risk inflexion point
                    if human.rolling_risk[prev_timestamp] > 0:
                        # remember: previously, if risk was below minimum, it was flattened to zero
                        for message in sent_encounter_messages[prev_timestamp]:
                            update_message = mu.create_update_message(
                                encounter_message=message,
                                new_risk_level=human.rolling_risk[prev_timestamp],
                                current_time=np.uint64(timestamp),
                                update_reason="symptoms",
                            )
                            sent_update_messages[timestamp].append(update_message)
                            sent_messages[timestamp].append(update_message)
        output[human_idx] = {
            "sent_encounter_messages": sent_encounter_messages,
            "sent_update_messages": sent_update_messages,
            "sent_messages": sent_messages,  # this one just combines the other two (in order)
        }
    return output


def generate_received_messages(
        humans: typing.List[FakeHuman],
        minimum_risk_level_for_updates: int = 5,
        maximum_risk_level_for_saturaton: int = 10,
):
    """Returns a user-to-received-messages mapping for clustering logic testing."""
    sent_messages = generate_sent_messages(
        humans=humans,
        minimum_risk_level_for_updates=minimum_risk_level_for_updates,
        maximum_risk_level_for_saturaton=maximum_risk_level_for_saturaton,
    )
    max_timestamp = max([len(h.rolling_uids) for h in humans])
    # we will simply reverse the mapping logic from 'generate_sent_messages'...
    received_messages = {
        human_idx: {
            "received_encounter_messages": {t: [] for t in range(max_timestamp)},
            "received_update_messages": {t: [] for t in range(max_timestamp)},
            "received_messages": {t: [] for t in range(max_timestamp)},
        } for human_idx in range(len(sent_messages))
    }
    # fetch all encounter messages first, then update messages
    for human_idx, sent_map in sent_messages.items():
        for timestamp, msgs in sent_map["sent_encounter_messages"].items():
            for msg in msgs:
                assert msg._receiver_uid != human_idx  # should never self-send
                enc_recv_map = received_messages[msg._receiver_uid]["received_encounter_messages"]
                if timestamp not in enc_recv_map:
                    enc_recv_map[timestamp] = []
                enc_recv_map[timestamp].append(msg)
    for human_idx, sent_map in sent_messages.items():
        for timestamp, msgs in sent_map["sent_update_messages"].items():
            for msg in msgs:
                assert msg._receiver_uid != human_idx  # should never self-send
                updt_recv_map = received_messages[msg._receiver_uid]["received_update_messages"]
                if timestamp not in updt_recv_map:
                    updt_recv_map[timestamp] = []
                updt_recv_map[timestamp].append(msg)
    # finally, fill up the generic arrays by combining encounters+updates
    for human_idx, recv_map in received_messages.items():
        max_timestamp = max(len(recv_map["received_encounter_messages"]),
                            len(recv_map["received_update_messages"]))
        for timestamp in range(max_timestamp):
            recv_map["received_messages"][timestamp] = []
            if timestamp in recv_map["received_encounter_messages"]:
                recv_map["received_messages"][timestamp].extend(
                    recv_map["received_encounter_messages"][timestamp])
            if timestamp in recv_map["received_update_messages"]:
                recv_map["received_messages"][timestamp].extend(
                    recv_map["received_update_messages"][timestamp])
    return received_messages

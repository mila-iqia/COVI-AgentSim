import dataclasses
import numpy as np
import typing
import datetime

import covid19sim.inference.message_utils as mu


class RiskLevelBoundsCheckedType(mu.RiskLevelType):
    """
    Wrapper class aliasing RiskLevelType to bound check passed data to
    RiskLevelType
    """
    lo_bound = 0
    hi_bound = mu.risk_level_mask

    def __new__(cls, *args, **kargs):
        self = super(RiskLevelBoundsCheckedType, cls).__new__( \
                                                        cls, *args, **kargs)
        value = self.item()
        if not cls.lo_bound <= value <= cls.hi_bound:
            raise ValueError(
                "{} {} is out of its expected boundaries [{},{}]".format(
                    cls.__name__,
                    value,
                    cls.lo_bound,
                    cls.hi_bound
                )
            )
        return self


@dataclasses.dataclass
class ObservedRisk:
    """
    A mapping from encountered tick to the observed data at that tick. It
    includes a mandatory field risk_level and an optional list of update
    signatures, i.e., update_risk_levels, update_ticks, and update_reasons
    for update messages to be related to the former encountered message.
    """
    start_time = mu.TimestampDefault
    time_offset = datetime.timedelta(days=1)

    @staticmethod
    def toff(t: typing.Union[int, np.int64]):
        """
        translates discrete time according to daily time offsets
        """
        return ObservedRisk.start_time + t * ObservedRisk.time_offset

    @dataclasses.dataclass
    class _UpdateMessage:
        """
        container for update messages of observed risks
        """
        update_tick: typing.Union[int, np.int64]
        update_risk_level: RiskLevelBoundsCheckedType
        update_reason: typing.Optional[str] = None

    encounter_tick: typing.Union[int, np.int64]
    encounter_risk_level: RiskLevelBoundsCheckedType
    update_signatures: typing.List[_UpdateMessage] = \
        dataclasses.field(default_factory=list, init=False)

    def __post_init__(self):
        """
        post initialization of encounter_risk_level to make sure
        the its bounds are checked
        """
        self.encounter_risk_level = RiskLevelBoundsCheckedType(
                                            self.encounter_risk_level)

    def update(self,
               update_tick: typing.Union[int, np.int64],
               update_risk_level: typing.Union[int, RiskLevelBoundsCheckedType],
               update_reason: typing.Optional[str] = None,
               ):
        """
        adds an update message following an encounter message and the updates
        that followed it
        """
        # validate the passed update tick does not belong to a time older than
        # the last update
        if len(self.update_signatures) > 0:
            if update_tick < self.update_signatures[-1].update_tick:
                raise ValueError(
                    'passed update tick {} < last update tick {}'.format(
                        update_tick,
                        self.update_signatures[-1].update_tick
                    )
                )
        else:
            if update_tick < self.encounter_tick:
                raise ValueError(
                    'passed update tick {} < encounter tick {}'.format(
                        update_tick,
                        self.encounter_tick
                    )
                )
        # validate the bounds of the passed update_risk_level
        update_risk_level = RiskLevelBoundsCheckedType(update_risk_level)

        self.update_signatures.append(
            ObservedRisk._UpdateMessage(
                update_tick=update_tick,
                update_risk_level=update_risk_level,
                update_reason=update_reason
            ))
        return self

    def to_user_messages(self,
                         uid: mu.UIDType,
                         real_uid_counter: mu.RealUserIDType,
                         exposure_tick: typing.Union[int, np.int64] = np.inf,
                         ) -> typing.List[mu.GenericMessageType]:
        """
        converting observed risk levels for a contacted user
        into a list of encounter and update messages
        """
        encounter_time = ObservedRisk.toff(self.encounter_tick)

        encounter_message = mu.EncounterMessage(
            uid=uid,
            risk_level=self.encounter_risk_level,
            encounter_time=encounter_time,
            _sender_uid=real_uid_counter,
            _receiver_uid=None,
            _real_encounter_time=encounter_time,
            _exposition_event=(
                self.encounter_tick >= exposure_tick),
            _applied_updates=[]
        )

        messages: typing.List[mu.GenericMessageType] = [encounter_message]
        last_risk = encounter_message.risk_level
        for update_signature in self.update_signatures:
            update_time = ObservedRisk.toff(update_signature.update_tick)
            update_message = mu.UpdateMessage(
                uid=uid,
                old_risk_level=last_risk,
                new_risk_level=update_signature.update_risk_level,
                encounter_time=encounter_time,
                update_time=update_time,
                _sender_uid=encounter_message._sender_uid,
                _receiver_uid=encounter_message._receiver_uid,
                _real_encounter_time=encounter_time,
                _real_update_time=update_time,
                _exposition_event=encounter_message._exposition_event,
                _update_reason=update_signature.update_reason,
            )

            messages.append(update_message)
            encounter_message._applied_updates.append(update_message)
            last_risk = update_signature.update_risk_level
        return messages


class MessageContextManager():
    """
    A context manager to generate batch of encounter and update
    messages that interface.
    Each public method call counts as one distinct user sending
    messages to a protagonist receiving these messages.
    """

    def __init__(self, max_tick):
        self.max_tick = max_tick
        self.max_history_offset = ObservedRisk.time_offset * max_tick
        self.real_uid_counter: mu.RealUserIDType = 0
        self._contact_messages = []

    def insert_messages(
        self,
        observed_risks: typing.Union[ObservedRisk, typing.List[ObservedRisk]],
        tick_to_uid_map: typing.Dict[int, mu.UIDType] = {},
        exposure_tick: typing.Union[int, np.int64] = np.inf,
    ) -> typing.List[mu.GenericMessageType]:
        """
        inserting a batch of observed_risks to the message context
        :param tick_to_uid_map: the uids to force ticks to adopt,
        random otherwise
        :param exposure_tick: the tick on which all the following messages
        (inclusive) are set in _exposition_event flag in messages' debugging
        attributes this doesn't ensure the risk levels that are passed
        via observed_risks reflect this
        """
        if isinstance(observed_risks, ObservedRisk):
            observed_risks = [observed_risks]
        for observed_risk in observed_risks:
            encounter_tick = observed_risk.encounter_tick
            # uid validation
            uid = tick_to_uid_map.get(encounter_tick)
            if uid is None:
                uid = mu.create_new_uid()
                tick_to_uid_map[encounter_tick] = uid
            self._contact_messages.extend(
                observed_risk.to_user_messages(
                    uid,
                    self.real_uid_counter,
                    exposure_tick=exposure_tick
                )
            )
        self.real_uid_counter += 1

    @staticmethod
    def _chronological_key(message: mu.GenericMessageType):
        """
        Returns a message key that is used to generate chronological sorting
        of messages (from old to new)
        """
        if isinstance(message, mu.UpdateMessage):
            return (message.update_time, 1)
        else: # mu.EncounterMessage
            return (message.encounter_time, 0)

    @property
    def contact_messages(self, sort=True):
        """
        Returns a batch of contact messages sorted in chronological order
        by default. If sort is otherwise set to False, then returns unsorted
        contact messages.
        All generated messages in this manager are to be accessed using this
        property
        """
        if sort:
            self._contact_messages.sort(key=self._chronological_key)
        return self._contact_messages

    def insert_random_messages(
            self,
            n_encounter: int,
            n_update: int,
            exposure_tick: typing.Union[int, np.int64] = np.inf,
            min_risk_level: RiskLevelBoundsCheckedType = 0,
            max_risk_level: RiskLevelBoundsCheckedType = mu.risk_level_mask
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns a set of random encounter/update messages from a single user
        :param exposure_tick: the tick on which all the following messages
        (inclusive) are set in _exposition_event flag in messages' debugging
        attributes this doesn't ensure the risk levels that are passed
        via observed_risks reflect this
        """
        # argument validation
        if n_encounter < 1:
            raise ValueError("n_encounter must be at least 1, passed: " +
                             str(n_encounter))
        if not min_risk_level <= max_risk_level <= mu.risk_level_mask:
            raise ValueError("min_risk_level {} should be less than or equal \
                to max_risk_level {} which in turn should be less than or \
                    equal to risk_level_mask {}".format(
                min_risk_level,
                max_risk_level,
                mu.risk_level_mask
            ))
        n_message = n_encounter + n_update
        risk_levels = np.random.randint(
            min_risk_level, max_risk_level + 1,
            size=n_message).astype(RiskLevelBoundsCheckedType)
        message_ticks = np.random.choice(
            self.max_tick, size=n_message).astype(int)
        partitions = np.random.choice(np.arange(1, n_message),
                                      size=n_encounter-1, replace=False)
        partitions.sort()
        risk_levels = np.split(risk_levels, partitions)
        message_ticks = np.split(message_ticks, partitions)
        observed_risks = [None] * n_encounter
        for i in range(n_encounter):
            message_ticks[i].sort()
            n_encounter_update = message_ticks[i].shape[0] - 1

            encounter_risk = risk_levels[i][0]
            encounter_tick = message_ticks[i][0]

            update_risks = risk_levels[i][1:]
            update_ticks = message_ticks[i][1:]

            observed_risks[i] = ObservedRisk(
                encounter_tick=encounter_tick,
                encounter_risk_level=encounter_risk
            )
            for j in range(n_encounter_update):
                update_tick = update_ticks[j]
                update_risk_level = update_risks[j]
                observed_risks[i].update(
                    update_tick=update_tick,
                    update_risk_level=update_risk_level
                )
        self.insert_messages(observed_risks, exposure_tick=exposure_tick)

    def _generate_linear_saturation_observed_risks(
        self,
        negative_saturation_tick: typing.Union[int, np.int64],
        init_risk_level: RiskLevelBoundsCheckedType,
        final_risk_level: RiskLevelBoundsCheckedType,
        rate: int = 1
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns the observed risks belonging to a linear saturation curve (_/¯)
        :param negative_saturation_tick: the last tick before observing an
        increase in the risk levels
        :param init_risk_level: the initial risk level
        :param final_risk_level: the final risk level if reachable given
        negative_saturation_tick and rate
        :param rate: the rate of increase of ticks over a unit of
        increase of init_risk_level to final_risk_level
        """
        if not negative_saturation_tick:
            negative_saturation_tick = self.max_tick + 1
        # setting the last tick of the increasing pattern in the risk levels
        positive_saturation_tick = negative_saturation_tick + \
            (final_risk_level - init_risk_level) * rate

        if positive_saturation_tick <= self.max_tick:
            observed_risks = [None] * self.max_tick
            for t in range(self.max_tick):
                if t <= negative_saturation_tick:
                    observed_risks[t] = ObservedRisk(
                        encounter_tick=t,
                        encounter_risk_level=init_risk_level,
                    )
                elif t >= positive_saturation_tick:
                    observed_risks[t] = ObservedRisk(
                        encounter_tick=t,
                        encounter_risk_level=final_risk_level,
                    )
                else:
                    observed_risks[t] = ObservedRisk(
                        encounter_tick=t,
                        encounter_risk_level= \
                            (t - negative_saturation_tick) / rate,
                    ).update(
                        update_tick=positive_saturation_tick,
                        update_risk_level=final_risk_level
                    )
        else:
            observed_risks = [
                ObservedRisk(
                    encounter_tick=t,
                    encounter_risk_level=init_risk_level
                )
                for t in range(self.max_tick)
            ]
        return observed_risks

    def insert_linear_saturation_risk_messages(
            self,
            n_encounter: int,
            exposure_tick: typing.Union[int, np.int64] = np.inf,
            init_risk_level: RiskLevelBoundsCheckedType = 0,
            final_risk_level: RiskLevelBoundsCheckedType = mu.risk_level_mask,
    ) -> typing.List[mu.GenericMessageType]:
        """
        Returns a set of random encounter/update messages from a single user
        with observed risks belonging to a linear saturation curve (_/¯)
        :param exposure_tick: the tick on which the risk level start rising
        this tick also marks the debugging flag of _exposition_event in all
        messages happening on and following ticks (inclusive)
        :param init_risk_level: the initial risk level
        :param final_risk_level: the final risk level if reachable given
        negative_saturation_tick and rate
        """
        # argument validation
        if not init_risk_level <= final_risk_level <= mu.risk_level_mask:
            raise ValueError("init_risk_level {} should be less than or equal \
                to final_risk_level {} which in turn should be less than or \
                    equal to risk_level_mask {}".format(
                init_risk_level,
                final_risk_level,
                mu.risk_level_mask
            ))
        # create at least one encounter
        linear_saturation_risks = \
            self._generate_linear_saturation_observed_risks(
                negative_saturation_tick=exposure_tick,
                init_risk_level=init_risk_level,
                final_risk_level=final_risk_level,
            )

        idxs = np.random.choice(
            self.max_tick, size=n_encounter)
        sample_risks = [linear_saturation_risks[idx] for idx in idxs]
        self.insert_messages(sample_risks, exposure_tick=exposure_tick)

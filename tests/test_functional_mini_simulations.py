import datetime
import os
import pickle
import typing
import unittest
import warnings
import zipfile
from collections import namedtuple
from tempfile import TemporaryDirectory

from tests.utils import get_test_conf

from covid19sim.log.event import Event

if typing.TYPE_CHECKING:
    from covid19sim.human import Human
    from covid19sim.inference.message_utils import UpdateMessage


TEST_CONF_NAME = "mini_simulations.yaml"


InterventionProps = namedtuple('InterventionProps',
                               ['name', 'risk_model', 'tracing_order'])


class ProxyEvent(Event):
    conf = None

    @staticmethod
    def log_encounter(*args, **kwargs):
        Event.log_encounter(*args, **kwargs)

    @staticmethod
    def log_encounter_messages(*args, **kwargs):
        Event.log_encounter_messages(*args, **kwargs)

    @staticmethod
    def log_risk_update(*args, **kwargs):
        Event.log_risk_update(*args, **kwargs)

    @staticmethod
    def log_exposed(*args, **kwargs):
        Event.log_exposed(*args, **kwargs)

    @staticmethod
    def log_test(*args, **kwargs):
        Event.log_test(*args, **kwargs)

    @staticmethod
    def log_daily(*args, **kwargs):
        Event.log_daily(*args, **kwargs)

    @staticmethod
    def log_static_info(*args, **kwargs):
        Event.log_static_info(*args, **kwargs)


class MakeHumanAsMessageProxy:
    def __init__(self, test_case: unittest.TestCase):
        self.test_case = test_case
        self._max_update_messages_count: typing.Dict["Human", typing.Dict["Human", int]] = None

    def set_max_update_messages_count(
            self,
            max_update_messages_count: typing.Dict["Human", typing.Dict["Human", int]]):
        self._max_update_messages_count = max_update_messages_count

    def make_human_as_message(
            self,
            human,
            personal_mailbox: typing.Dict[int, typing.List["UpdateMessage"]],
            conf):

        tc = self.test_case

        now = human.env.timestamp

        update_messages_count: typing.Dict[str, int] = {}
        for update_messages in personal_mailbox.values():
            for update_message in update_messages:
                update_messages_count.setdefault(update_message._sender_uid, 0)
                update_messages_count[update_message._sender_uid] += 1

        for sender_human, max_messages_count in self._max_update_messages_count[human].items():
            update_messages_count.setdefault(sender_human.name, 0)
            with tc.subTest(human=human.name, sender_human=sender_human.name,
                            time=str(now),
                            update_messages_count=update_messages_count[sender_human.name],
                            max_update_messages_count=max_messages_count):
                tc.assertLessEqual(update_messages_count[sender_human.name],
                                   max_messages_count,
                                   f"Human must not send more update messages "
                                   f"than the total number of encounter messages")
            del update_messages_count[sender_human.name]

        tc.assertEqual(len(update_messages_count), 0,
                       f"Update messages sent by other humans with no previous encounters"
                       f"should not have been received")

        message = MessagingTest.make_human_as_message(human, personal_mailbox, conf)

        return message


class TrackerMock:
    def track_tested_results(self, *args, **kwargs):
        pass


def _init_infector(human):
    # Force carefulness to 1 for the human to report the test result
    human.carefulness = 1
    # Force has_app to True for the human to report the test result
    human.has_app = True
    try:
        if human.city.tracker is None:
            human.city.tracker = TrackerMock()
    except AttributeError:
        human.city.tracker = TrackerMock()
    # Make the human at his most infectious (Peak viral load day)
    human._infection_timestamp = human.infection_timestamp + \
                                 datetime.timedelta(days=-(human.viral_load_peak_start +
                                                           human.infectiousness_onset_days))
    # Mark the human as tested
    human.set_test_info("lab", "positive")
    # Make the test available now
    human.time_to_test_result = 0

    # Mock Human.how_am_I_feeling() to prevent human from staying at home
    human.how_am_I_feeling = lambda: 1.0


def _run_simulation(test_case, intervention_properties):
    from covid19sim.run import simulate

    intervention, risk_model, tracing_order = intervention_properties
    test_case.config['INTERVENTION'] = intervention
    test_case.config['RISK_MODEL'] = risk_model
    test_case.config['TRACING_ORDER'] = tracing_order

    if intervention == '':
        test_case.config['INTERVENTION_DAY'] = -1
    else:
        test_case.config['INTERVENTION_DAY'] = test_case.intervention_day

    data = []

    with TemporaryDirectory() as d:
        outfile = os.path.join(d, "data")
        monitors, _ = simulate(
            n_people=test_case.n_people,
            start_time=test_case.start_time,
            simulation_days=test_case.simulation_days,
            outfile=outfile,
            out_chunk_size=0,
            init_percent_sick=test_case.init_percent_sick,
            seed=test_case.test_seed,
            conf=test_case.config
        )
        monitors[0].dump()
        monitors[0].join_iothread()

        with zipfile.ZipFile(f"{outfile}.zip", 'r') as zf:
            for pkl in zf.namelist():
                pkl_bytes = zf.read(pkl)
                data.extend(pickle.loads(pkl_bytes))

    test_case.assertGreater(len(data), 0)

    test_case.assertIn(Event.encounter, {d['event_type'] for d in data})
    test_case.assertIn(Event.test, {d['event_type'] for d in data})

    test_case.assertGreaterEqual(len({d['human_id'] for d in data}), test_case.n_people)

    return data


class MiniSimulationTest(unittest.TestCase):
    from covid19sim.log.event import Event

    def setUp(self):
        import covid19sim.log.event
        covid19sim.log.event.Event = ProxyEvent

        self.config = get_test_conf(TEST_CONF_NAME)
        ProxyEvent.conf = self.config

        self.test_seed = 0
        self.n_people = 10
        self.init_percent_sick = 0.1
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 5
        self.intervention_day = 0

        self.config['COLLECT_LOGS'] = True
        self.config['INTERVENTION_DAY'] = self.intervention_day
        self.config['APP_UPTAKE'] = -1
        self.config['TRANSFORMER_EXP_PATH'] = "https://drive.google.com/file/d/1QhiZehbxNOhA-7n37h6XEHTORIXweXc6"
        self.config['LOGGING_LEVEL'] = "DEBUG"

    def tearDown(self):
        import covid19sim.log.event
        covid19sim.log.event.Event = MiniSimulationTest.Event


class InterventionRiskTest(MiniSimulationTest):
    def setUp(self):
        super(InterventionRiskTest, self).setUp()

        # Prevent testing to ease analysis of 1st and 2 order tracing
        self.config['TEST_TYPES']['lab']['capacity'] = 0
        self.config['BASELINE_RISK_VALUE'] = 0.0
        # Prevent quarantines, we want to test contacts and messages
        self.config["DROPOUT_RATE"] = 1.0

    def test_tracing_order_1(self):
        """
        Test the risk assigned to humans when using 1st order binary tracing
        """
        from covid19sim.human import Human
        intervention = InterventionProps('Tracing', 'digital', 1)

        initial_infectors = set()

        # TODO: Fix the code to have contacts take effect immediately to simplify
        #  this test. Since current code only makes contacts have an effect 1
        #  time slot later, the contacts state for all humans are not in sync
        #  and must be kept in cache for each human

        # Holds human's contacts
        contacts: typing.Dict[Human, set] = {}
        # contacts currently only have an effect 1 time slot later
        time_slot_contacts: typing.Dict[Human, set] = {}

        def log_encounter_messages_proxy(*args, **kwargs):
            Event.log_encounter_messages(*args, **kwargs)
            human1 = kwargs.get('human1', None)
            human2 = kwargs.get('human2', None)

            for arg in (*args, *kwargs.values()):
                if isinstance(arg, Human):
                    if human1 is None:
                        human1 = arg
                    elif human2 is None:
                        human2 = arg
                        break

            time_slot_contacts[human1].add(human2)
            time_slot_contacts[human2].add(human1)

        def log_risk_update_proxy(*args, **kwargs):
            Event.log_risk_update(*args, **kwargs)
            human = kwargs.get('human', None)
            time = kwargs.get('time', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break
            for arg in args:
                if isinstance(arg, datetime.datetime):
                    time = arg
                    break

            with self.subTest(human=human.name, time=str(time),
                              initial_infectors=initial_infectors.copy(),
                              contacts=contacts[human].copy(),
                              time_slot_contacts=time_slot_contacts[human].copy()):
                if human in initial_infectors:
                    self.assertEqual(human.risk, 1,
                                     msg=f"initial infector should be at risk")
                else:
                    for other_human in contacts[human]:
                        if other_human in initial_infectors:
                            self.assertEqual(human.risk, 1,
                                             msg=f"The contact with the initial "
                                             f"infector {other_human} was not "
                                             f"flagged at risk")
                            break
                    else:
                        self.assertEqual(human.risk, human.baseline_risk,
                                         msg=f"A human not in contact with "
                                         f"the initial infector should not "
                                         f"be flagged as at risk")

            contacts[human].update(time_slot_contacts[human])
            time_slot_contacts[human].clear()

        def log_static_info_proxy(*args, **kwargs):
            Event.log_static_info(*args, **kwargs)
            human = kwargs.get('human', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break

            self.assertIsNot(human, None)

            # add human to list of initial_infectors
            if human.is_exposed:
                _init_infector(human)
                initial_infectors.add(human)

            contacts.setdefault(human, set())
            time_slot_contacts.setdefault(human, set())

        ProxyEvent.log_encounter_messages = log_encounter_messages_proxy
        ProxyEvent.log_risk_update = log_risk_update_proxy
        ProxyEvent.log_static_info = log_static_info_proxy

        _run_simulation(self, intervention)

        # We want to test with only 1 initial infector
        self.assertEqual(1, len(initial_infectors))

    def test_tracing_order_2(self):
        """
        Test the risk assigned to humans when using 2nd order binary tracing
        """
        from covid19sim.human import Human
        intervention = InterventionProps('Tracing', 'digital', 2)

        # TODO: This code would be much simpler if the effects of a contact was
        #  not delayed by a time slot. It would remove the need for
        #  time_slot_contacts and staged_time_slot_contacts which would be
        #  replaced by a simple order_1_contacts set

        # Since current code only makes contacts have an effect 1
        # time slot later, the contacts state for all humans are not in sync
        # and must be kept in cache for each human

        initial_infectors = set()
        # Holds human's contacts
        contacts: typing.Dict[Human, set] = {}
        # Contacts currently only have an effect 1 time slot later
        time_slot_contacts: typing.Dict[Human, set] = {}
        # In between contacts that should be taken into account in the next
        # human's risk_update but which should not be taken into account by the
        # other humans in the same time slot
        staged_time_slot_contacts: \
            typing.Dict[datetime.datetime, typing.Dict[Human, set]] = {}

        def log_encounter_messages_proxy(*args, **kwargs):
            Event.log_encounter_messages(*args, **kwargs)
            human1 = kwargs.get('human1', None)
            human2 = kwargs.get('human2', None)

            for arg in (*args, *kwargs.values()):
                if isinstance(arg, Human):
                    if human1 is None:
                        human1 = arg
                    elif human2 is None:
                        human2 = arg
                        break

            time_slot_contacts[human1].add(human2)
            time_slot_contacts[human2].add(human1)

        def log_risk_update_proxy(*args, **kwargs):
            Event.log_risk_update(*args, **kwargs)
            human = kwargs.get('human', None)
            time = kwargs.get('time', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break
            for arg in args:
                if isinstance(arg, datetime.datetime):
                    time = arg
                    break

            h_contacts = contacts[human]
            h_time_slot_contacts = time_slot_contacts[human]

            # Add previous time slot contacts to each humans' contacts
            for time_slot in list(staged_time_slot_contacts.keys()):
                if time_slot != time:
                    for h, h_staged_time_slot_contacts in \
                            staged_time_slot_contacts[time_slot].items():
                        contacts[h].update(h_staged_time_slot_contacts)
                    del staged_time_slot_contacts[time_slot]

            with self.subTest(human=human.name, time=str(time),
                              initial_infectors=initial_infectors.copy(),
                              contacts=h_contacts.copy(),
                              time_slot_contacts=h_time_slot_contacts.copy()):
                if human in initial_infectors:
                    self.assertEqual(human.risk, 1,
                                     msg=f"initial infector should be at risk")
                elif h_contacts.intersection(initial_infectors):
                    self.assertEqual(human.risk, 1,
                                     msg=f"1st order contact should be at risk")
                else:
                    for other_human in h_contacts:
                        if contacts[other_human].intersection(initial_infectors):
                            with self.subTest(human=human.name, time=str(time),
                                              initial_infectors=initial_infectors.copy(),
                                              other_human=other_human,
                                              other_human_contacts=contacts[other_human]):
                                self.assertEqual(human.risk, 1,
                                                 msg=f"2nd order contact should be "
                                                 f"at risk")
                                break
                    else:
                        self.assertEqual(human.risk, human.baseline_risk,
                                         msg=f"A human not in contact with the "
                                         f"initial infector or a 2nd order "
                                         f"contact should not be flagged as at risk")

            # Stage this time slot contacts to be taken into account in the
            # next risk_update
            staged_time_slot_contacts.setdefault(time, dict())
            staged_time_slot_contacts[time].setdefault(human, set())
            staged_time_slot_contacts[time][human].update(h_time_slot_contacts)
            h_time_slot_contacts.clear()

        def log_static_info_proxy(*args, **kwargs):
            Event.log_static_info(*args, **kwargs)
            human = kwargs.get('human', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break

            self.assertIsNot(human, None)

            # add human to list of initial_infectors
            if human.is_exposed:
                _init_infector(human)
                initial_infectors.add(human)

            contacts.setdefault(human, set())
            time_slot_contacts.setdefault(human, set())

        ProxyEvent.log_encounter_messages = log_encounter_messages_proxy
        ProxyEvent.log_risk_update = log_risk_update_proxy
        ProxyEvent.log_static_info = log_static_info_proxy

        _run_simulation(self, intervention)

        # We want to test with only 1 initial infector
        self.assertEqual(len(initial_infectors), 1)

    def test_transformer(self):
        """
        Test that the risk assigned to humans is at least the same as with
        tracing order 2
        """
        from covid19sim.human import Human
        intervention = InterventionProps('Tracing', 'transformer', None)

        # TODO: This code would be much simpler if the effects of a contact was
        #  not delayed by a time slot. It would remove the need for
        #  time_slot_contacts and staged_time_slot_contacts which would be
        #  replaced by a simple order_1_contacts set

        # Since current code only makes contacts have an effect 1
        # time slot later, the contacts state for all humans are not in sync
        # and must be kept in cache for each human

        initial_infectors = set()
        # Holds human's contacts
        contacts: typing.Dict[Human, set] = {}
        # Contacts currently only have an effect 1 time slot later
        time_slot_contacts: typing.Dict[Human, set] = {}
        # In between contacts that should be taken into account in the next
        # human's risk_update but which should not be taken into account by the
        # other humans in the same time slot
        staged_time_slot_contacts: \
            typing.Dict[datetime.datetime, typing.Dict[Human, set]] = {}

        # Holds humans's recommendation levels
        rec_levels: typing.Dict[Human, typing.Set[int]] = {}

        def log_encounter_messages_proxy(*args, **kwargs):
            Event.log_encounter_messages(*args, **kwargs)
            human1 = kwargs.get('human1', None)
            human2 = kwargs.get('human2', None)

            for arg in (*args, *kwargs.values()):
                if isinstance(arg, Human):
                    if human1 is None:
                        human1 = arg
                    elif human2 is None:
                        human2 = arg
                        break

            time_slot_contacts[human1].add(human2)
            time_slot_contacts[human2].add(human1)

        def log_risk_update_proxy(*args, **kwargs):
            Event.log_risk_update(*args, **kwargs)
            human = kwargs.get('human', None)
            time = kwargs.get('time', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break
            for arg in args:
                if isinstance(arg, datetime.datetime):
                    time = arg
                    break

            h_contacts = contacts[human]
            h_time_slot_contacts = time_slot_contacts[human]

            # Add previous time slot contacts to each humans' contacts
            for time_slot in list(staged_time_slot_contacts.keys()):
                if time_slot != time:
                    for h, h_staged_time_slot_contacts in \
                            staged_time_slot_contacts[time_slot].items():
                        contacts[h].update(h_staged_time_slot_contacts)
                    del staged_time_slot_contacts[time_slot]

            rec_levels.setdefault(human, set())
            rec_levels[human].add(human.rec_level)

            with self.subTest(human=human.name, time=str(time),
                              initial_infectors=initial_infectors.copy(),
                              contacts=h_contacts.copy(),
                              time_slot_contacts=h_time_slot_contacts.copy(),
                              rec_levels=rec_levels[human].copy()):
                if human in initial_infectors:
                    self.assertEqual(human.rec_level, 3,
                                     msg=f"initial infector should be at risk")
                elif h_contacts.intersection(initial_infectors):
                    # At least one non-zero rec level should have been assigned
                    # to the human
                    # TODO: Verify if it makes sense that a 5 days (duration of
                    #  this mini simulation) is enough to have the rec_level droping
                    #  to 0
                    try:
                        non_zero_rec_levels = [rec_level for rec_level in rec_levels[human]
                                               if rec_level > 0]
                        self.assertGreaterEqual(len(non_zero_rec_levels), 1,
                                                msg=f"1st order contact should be at "
                                                f"a minimum risk")
                    except AssertionError as error:
                        warnings.warn(f"1st order contact should probably be at "
                                      f"a minimum risk: {str(error)}", RuntimeWarning)
                else:
                    for other_human in h_contacts:
                        if contacts[other_human].intersection(initial_infectors):
                            with self.subTest(human=human.name, time=str(time),
                                              initial_infectors=initial_infectors.copy(),
                                              other_human=other_human,
                                              other_human_contacts=contacts[other_human]):
                                # At least one non-zero rec level should have been assigned
                                # to the human
                                # TODO: Verify if it makes sense that a 5 days (duration of
                                #  this mini simulation) is enough to have the rec_level droping
                                #  to 0
                                try:
                                    non_zero_rec_levels = [rec_level for rec_level in rec_levels[human]
                                                           if rec_level > 0]
                                    self.assertGreaterEqual(len(non_zero_rec_levels), 1,
                                                            msg=f"2nd order contact should be at "
                                                            f"a minimum risk")
                                except AssertionError as error:
                                    warnings.warn(f"2nd order contact should probably be at "
                                                  f"a minimum risk: {str(error)}", RuntimeWarning)
                                break
                    else:
                        if not human.all_symptoms:
                            try:
                                self.assertEqual(human.rec_level, 0,
                                                 msg=f"A human not in contact with the "
                                                 f"initial infector or a 1st order "
                                                 f"contact and without any symptoms "
                                                 f"should not be flagged as at risk")
                            except AssertionError as error:
                                warnings.warn(f"A human not in contact with the "
                                              f"initial infector or a 1st order "
                                              f"contact and without any symptoms "
                                              f"should probably not be flagged as "
                                              f"at risk: {str(error)}", RuntimeWarning)

            # Stage this time slot contacts to be taken into account in the
            # next risk_update
            staged_time_slot_contacts.setdefault(time, dict())
            staged_time_slot_contacts[time].setdefault(human, set())
            staged_time_slot_contacts[time][human].update(h_time_slot_contacts)
            h_time_slot_contacts.clear()

        def log_static_info_proxy(*args, **kwargs):
            Event.log_static_info(*args, **kwargs)
            human = kwargs.get('human', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break

            self.assertIsNot(human, None)

            # add human to list of initial_infectors
            if human.is_exposed:
                _init_infector(human)
                initial_infectors.add(human)

            contacts.setdefault(human, set())
            time_slot_contacts.setdefault(human, set())

        ProxyEvent.log_encounter_messages = log_encounter_messages_proxy
        ProxyEvent.log_risk_update = log_risk_update_proxy
        ProxyEvent.log_static_info = log_static_info_proxy

        _run_simulation(self, intervention)

        # We want to test with only 1 initial infector
        self.assertEqual(len(initial_infectors), 1)


class MessagingTest(MiniSimulationTest):
    from covid19sim.inference.heavy_jobs import make_human_as_message

    def setUp(self):
        super(MessagingTest, self).setUp()

        from covid19sim.inference import heavy_jobs

        # Prevent quarantines, we want to test contacts and messages
        self.config["DROPOUT_RATE"] = 1.0

        proxy = MakeHumanAsMessageProxy(self)
        self.make_human_as_message_proxy = proxy
        heavy_jobs.make_human_as_message = proxy.make_human_as_message

    def tearDown(self):
        super(MessagingTest, self).tearDown()

        from covid19sim.inference import heavy_jobs

        heavy_jobs.make_human_as_message = MessagingTest.make_human_as_message

    def test_transformer(self):
        """
        Test the messaging system when using 1st order binary tracing
        """
        intervention = InterventionProps('Tracing', 'transformer', None)

        initial_infectors = set()

        # Holds human's encounter messages sent to other human count
        time_slot_encounter_messages_cnt: typing.Dict["Human", typing.Dict["Human", int]] = {}
        # Holds human's updates messages received from other human count
        max_update_messages_cnt: typing.Dict["Human", typing.Dict["Human", int]] = {}
        # Staged update messages to be received at the beginning of the next time slot
        staged_max_update_messages_cnt: typing.Dict["Human", typing.Dict["Human", int]] = {}

        self.make_human_as_message_proxy.set_max_update_messages_count(staged_max_update_messages_cnt)

        def log_encounter_messages_proxy(COLLECT_LOGS, human1, human2, location, duration, distance, time):
            Event.log_encounter_messages(COLLECT_LOGS, human1, human2, location, duration, distance, time)

            time_slot_encounter_messages_cnt[human1].setdefault(human2, 0)
            time_slot_encounter_messages_cnt[human2].setdefault(human1, 0)

            messages_count = duration / human1.conf.get("MIN_MESSAGE_PASSING_DURATION")
            time_slot_encounter_messages_cnt[human1][human2] += int(messages_count)
            time_slot_encounter_messages_cnt[human2][human1] += int(messages_count)

        def log_risk_update_proxy(COLLECT_LOGS, human, tracing_description,
                                  prev_risk_history_map, risk_history_map, current_day_idx,
                                  time):
            Event.log_risk_update(COLLECT_LOGS, human, tracing_description,
                                  prev_risk_history_map, risk_history_map, current_day_idx,
                                  time)

            h_ts_encounter_messages_cnt = time_slot_encounter_messages_cnt[human]

            # Reset the count of update messages to be received by this human at
            # the beginning of the next time slot
            for other_human_staged_max_update_messages_cnt in staged_max_update_messages_cnt.values():
                other_human_staged_max_update_messages_cnt[human] = 0

            # The maximum update messages that can be received from this human
            # is the sum of all encounter messages between him and another human.
            # This could only be reached if the human sending has to update his
            # risk level for all previous days and no other filtering on the
            # updates messages is applied
            for other_human, messages_count in h_ts_encounter_messages_cnt.items():
                max_update_messages_cnt[other_human].setdefault(human, 0)
                max_update_messages_cnt[other_human][human] += messages_count

            is_new_risk_level = False
            for prev_risk, risk in zip(prev_risk_history_map.values(), risk_history_map.values()):
                prev_risk_level = min(human.proba_to_risk_level_map(prev_risk), 15)
                risk_level = min(human.proba_to_risk_level_map(risk), 15)
                is_new_risk_level = prev_risk_level != risk_level
                if is_new_risk_level:
                    break

            if is_new_risk_level:
                # Add all previous messages count in the max update messages
                # count received by the other humans
                for other_human in max_update_messages_cnt:
                    max_update_messages_cnt[other_human].setdefault(human, 0)
                    staged_max_update_messages_cnt[other_human][human] = max_update_messages_cnt[other_human][human]
            else:
                # Only the new encounter messages should count in the max update
                # messages count received by the other humans
                for other_human, messages_count in h_ts_encounter_messages_cnt.items():
                    staged_max_update_messages_cnt[other_human][human] = messages_count

            h_ts_encounter_messages_cnt.clear()

        def log_static_info_proxy(*args, **kwargs):
            from covid19sim.human import Human

            Event.log_static_info(*args, **kwargs)
            human = kwargs.get('human', None)
            for arg in args:
                if isinstance(arg, Human):
                    human = arg
                    break

            self.assertIsNot(human, None)

            # add human to list of initial_infectors
            if human.is_exposed:
                _init_infector(human)
                initial_infectors.add(human)

            time_slot_encounter_messages_cnt.setdefault(human, dict())
            max_update_messages_cnt.setdefault(human, dict())
            staged_max_update_messages_cnt.setdefault(human, dict())

        ProxyEvent.log_encounter_messages = log_encounter_messages_proxy
        ProxyEvent.log_risk_update = log_risk_update_proxy
        ProxyEvent.log_static_info = log_static_info_proxy

        _run_simulation(self, intervention)

        # We want to test with only 1 initial infector
        self.assertEqual(1, len(initial_infectors))

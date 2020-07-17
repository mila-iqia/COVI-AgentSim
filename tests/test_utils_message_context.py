import unittest
import covid19sim.inference.message_utils as mu
from tests.utils import MessageContextManager, ObservedRisk

class TestMessageContextManager(unittest.TestCase):

    def setUp(self):
        self.message_context = MessageContextManager(max_tick=30)
        self.maxDiff = None

    def test_insert_messages_smoke_test(self):
        # test arguments
        encounter_tick = 0
        tick_to_uid_map = {encounter_tick: 10}
        encounter_risk_level = 14
        update_tick = 30
        update_risk_level = 3
        # record an encounter observation and append an update on it
        o = ObservedRisk(
            encounter_tick=encounter_tick,
            encounter_risk_level=encounter_risk_level
        ).update(update_tick=update_tick, update_risk_level=update_risk_level)
        # compile the previous object into message history
        self.message_context.insert_messages(
            observed_risks=o, tick_to_uid_map=tick_to_uid_map
        )
        encounter_time = ObservedRisk.start_time
        update_time = ObservedRisk.toff(update_tick)
        # define expected messages
        expected_update_message = mu.UpdateMessage(
            uid=tick_to_uid_map[encounter_tick],
            encounter_time=encounter_time,
            update_time=update_time,
            old_risk_level=encounter_risk_level,
            new_risk_level=update_risk_level,
            _real_encounter_time=encounter_time,
            _real_update_time=update_time,
            _sender_uid=0,
            _exposition_event=False,
        )
        expected_encounter_message = mu.EncounterMessage(
            uid=tick_to_uid_map[encounter_tick],
            encounter_time=encounter_time,
            risk_level=encounter_risk_level,
            _real_encounter_time=encounter_time,
            _sender_uid=0,
            _exposition_event=False,
            _applied_updates=[expected_update_message]
        )
        self.assertEqual(self.message_context.contact_messages,
                         [expected_encounter_message, expected_update_message])

    def test_insert_random_messages_smoke_test(self):
        # test arguments
        exposure_tick = 7
        n_encounter = 10
        n_update = 10
        # create random messages
        self.message_context.insert_random_messages(
            exposure_tick=exposure_tick,
            n_encounter=n_encounter,
            n_update=n_update
        )
        curated_messages = self.message_context.contact_messages
        # test if the number of messages match up the arguments
        returned_n_encounter = sum([
            isinstance(message, mu.EncounterMessage)
            for message in curated_messages
        ])
        self.assertEqual(returned_n_encounter, n_encounter)
        self.assertEqual(len(curated_messages), n_encounter + n_update)

    def test_insert_linear_saturation_risk_messages_smoke_test(self):
        # test arguments
        exposure_tick = 10
        n_encounter = 2
        init_risk_level = 5
        final_risk_level = 12
        # create linear saturation messages
        self.message_context.insert_linear_saturation_risk_messages(
            exposure_tick=exposure_tick,
            n_encounter=n_encounter,
            init_risk_level=init_risk_level,
            final_risk_level=final_risk_level
        )
        # test if the update message risks are increasing
        for message in self.message_context.contact_messages:
            if isinstance(message, mu.UpdateMessage):
                self.assertTrue(message.old_risk_level <=
                                message.new_risk_level)

    def tearDown(self):
        pass

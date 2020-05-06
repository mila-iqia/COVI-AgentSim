import datetime
import numpy as np
import unittest

import covid19sim.frozen.message_utils as mu
from tests.utils import FakeHuman, generate_received_messages, generate_sent_messages, Visit

never = 9999  # dirty macro to indicate a human will never get infected


class CommonTests(unittest.TestCase):
    # check all utility functions & data classes...

    def test_create_and_update_uid(self):
        self.assertEqual(mu.message_uid_bit_count, 4)  # current test impl hardcodes stuff
        n_trials = 1000
        for _ in range(n_trials):
            uid = mu.create_new_uid()
            assert 0 <= uid <= 15
            new_uid = mu.update_uid(uid)
            assert 0 <= new_uid <= 15
            self.assertEqual(((uid * 2) & 15), new_uid - (new_uid % 2))

    def test_create_update_message(self):
        n_trials = 1000
        for _ in range(n_trials):
            fake_encounter_time = datetime.datetime.now() + \
                                  datetime.timedelta(seconds=np.random.randint(1000))
            encounter_msg = mu.EncounterMessage(
                uid=mu.create_new_uid(),
                risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(fake_encounter_time.timestamp()),
            )
            fake_update_time = fake_encounter_time + \
                               datetime.timedelta(seconds=np.random.randint(1, 10))
            update_msg = mu.create_update_message(
                encounter_message=encounter_msg,
                new_risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                current_time=int(fake_update_time.timestamp()),
            )
            self.assertEqual(update_msg.uid, encounter_msg.uid)
            self.assertEqual(update_msg.old_risk_level, encounter_msg.risk_level)
            self.assertEqual(update_msg.encounter_time, encounter_msg.encounter_time)
            self.assertGreater(update_msg.update_time, update_msg.encounter_time)
            dummy_recreated_encounter_msg = mu.create_encounter_from_update_message(update_msg)
            self.assertEqual(dummy_recreated_encounter_msg.uid, encounter_msg.uid)
            self.assertEqual(dummy_recreated_encounter_msg.risk_level, update_msg.new_risk_level)
            self.assertEqual(dummy_recreated_encounter_msg.encounter_time, encounter_msg.encounter_time)
            self.assertEqual(dummy_recreated_encounter_msg._sender_uid, encounter_msg._sender_uid)
            self.assertEqual(dummy_recreated_encounter_msg._receiver_uid, encounter_msg._receiver_uid)
            self.assertEqual(dummy_recreated_encounter_msg._real_encounter_time, encounter_msg._real_encounter_time)
            self.assertIsNone(dummy_recreated_encounter_msg._exposition_event)

    def test_create_updated_encounter_with_message(self):
        n_trials = 1000
        for _ in range(n_trials):
            fake_encounter_time = datetime.datetime.now() + \
                                  datetime.timedelta(seconds=np.random.randint(1000))
            uid = mu.create_new_uid()
            init_risk = np.uint8(np.random.randint(mu.risk_level_mask + 1))
            encounter_msg = mu.EncounterMessage(
                uid=uid,
                risk_level=init_risk,
                encounter_time=int(fake_encounter_time.timestamp()),
            )
            fake_update_time = fake_encounter_time + \
                               datetime.timedelta(seconds=np.random.randint(1000))
            update_msg = mu.UpdateMessage(
                uid=uid,
                old_risk_level=init_risk,
                new_risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(fake_encounter_time.timestamp()),
                update_time=int(fake_update_time.timestamp()),
            )
            new_encounter_msg = mu.create_updated_encounter_with_message(encounter_msg, update_msg)
            self.assertEqual(encounter_msg.uid, new_encounter_msg.uid)
            self.assertEqual(update_msg.new_risk_level, new_encounter_msg.risk_level)
            self.assertEqual(encounter_msg.encounter_time, new_encounter_msg.encounter_time)

    def test_find_encounter_match_random(self):
        n_trials = 1000
        for _ in range(n_trials):
            old_encounter_time = datetime.datetime.now() + \
                                 datetime.timedelta(seconds=np.random.randint(100))
            old_encounter_msg = mu.EncounterMessage(
                uid=mu.create_new_uid(),
                risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(old_encounter_time.timestamp()),
            )
            new_encounter_time = old_encounter_time + \
                                 datetime.timedelta(hours=np.random.randint(1, 150))
            new_encounter_msg = mu.EncounterMessage(
                uid=mu.create_new_uid(),
                risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(new_encounter_time.timestamp()),
            )
            score = mu.find_encounter_match_score(
                old_encounter_msg,
                new_encounter_msg,
            )
            self.assertGreaterEqual(score, -1)
            self.assertLessEqual(score, mu.message_uid_bit_count)

    def test_find_encounter_match_positive_or_null(self):
        n_trials = 1000
        for _ in range(n_trials):
            old_encounter_time = datetime.datetime.now() + \
                                 datetime.timedelta(seconds=np.random.randint(100))
            old_encounter_msg = mu.EncounterMessage(
                uid=mu.create_new_uid(),
                risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(old_encounter_time.timestamp()),
            )
            new_uid = old_encounter_msg.uid
            for day_offset in range(0, mu.message_uid_bit_count * 2):
                new_encounter_time = old_encounter_time + \
                                     datetime.timedelta(days=day_offset) + \
                                     datetime.timedelta(hours=np.random.randint(1, 23))
                new_encounter_msg = mu.EncounterMessage(
                    uid=new_uid,
                    risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                    encounter_time=int(new_encounter_time.timestamp()),
                )
                score = mu.find_encounter_match_score(
                    old_encounter_msg,
                    new_encounter_msg,
                )
                if day_offset < mu.message_uid_bit_count:
                    self.assertEqual(score, mu.message_uid_bit_count - day_offset)
                else:
                    self.assertEqual(score, 0)
                new_uid = mu.update_uid(new_uid)

    def test_find_encounter_match_negative(self):
        np.random.seed(0)
        n_trials = 1000
        for trial_idx in range(n_trials):
            old_encounter_time = datetime.datetime.now() + \
                                 datetime.timedelta(seconds=np.random.randint(100))
            old_encounter_msg = mu.EncounterMessage(
                uid=mu.create_new_uid(),
                risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                encounter_time=int(old_encounter_time.timestamp()),
            )
            legit_new_uid = old_encounter_msg.uid
            for day_offset in range(0, mu.message_uid_bit_count):
                new_encounter_time = old_encounter_time + \
                                     datetime.timedelta(days=day_offset) + \
                                     datetime.timedelta(hours=np.random.randint(1, 23))
                bflip = np.random.choice(list(range(mu.message_uid_bit_count - day_offset)))
                new_uid = legit_new_uid ^ np.uint8(1 << (mu.message_uid_bit_count - bflip - 1))
                new_encounter_msg = mu.EncounterMessage(
                    uid=new_uid,
                    risk_level=np.uint8(np.random.randint(mu.risk_level_mask + 1)),
                    encounter_time=int(new_encounter_time.timestamp()),
                )
                score = mu.find_encounter_match_score(
                    old_encounter_msg,
                    new_encounter_msg,
                )
                self.assertEqual(score, -1)
                legit_new_uid = mu.update_uid(legit_new_uid)


class MessageTests(unittest.TestCase):
    # this is essentially pre-testing the logic for clustering, which is the main course

    def test_simple_negative_encounter(self):
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=3),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
        ]
        messages = generate_sent_messages(humans)
        self.assertIn(0, messages)
        self.assertIn(1, messages)
        self.assertFalse(sum([len(msgs) for msgs in messages[0]["sent_update_messages"].values()]))
        self.assertFalse(sum([len(msgs) for msgs in messages[1]["sent_update_messages"].values()]))
        human0_encounters = [(t, m) for t, msgs in messages[0]["sent_encounter_messages"].items() for m in msgs]
        human1_encounters = [(t, m) for t, msgs in messages[1]["sent_encounter_messages"].items() for m in msgs]
        self.assertEqual(len(human0_encounters), 1)
        self.assertEqual(len(human1_encounters), 1)
        self.assertEqual(human0_encounters[0][0], human1_encounters[0][0])
        self.assertEqual(human0_encounters[0][0], 3)
        human0_encounter = human0_encounters[0][1]
        human1_encounter = human1_encounters[0][1]
        self.assertEqual(human0_encounter._sender_uid, 0)
        self.assertEqual(human0_encounter._receiver_uid, 1)
        self.assertEqual(human1_encounter._sender_uid, 1)
        self.assertEqual(human1_encounter._receiver_uid, 0)
        self.assertEqual(human0_encounter._real_encounter_time, 3)
        self.assertEqual(human1_encounter._real_encounter_time, 3)
        self.assertEqual(human0_encounter._real_encounter_time, human0_encounter.encounter_time)
        self.assertEqual(human1_encounter._real_encounter_time, human1_encounter.encounter_time)
        self.assertEqual(human0_encounter.risk_level, 0)
        self.assertEqual(human1_encounter.risk_level, 0)
        self.assertEqual(human0_encounter.uid, humans[0].rolling_uids[3])
        self.assertEqual(human1_encounter.uid, humans[1].rolling_uids[3])
        self.assertFalse(human0_encounter._exposition_event)
        self.assertFalse(human1_encounter._exposition_event)

    def test_simple_positive_encounter(self):
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=True, timestamp=3),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=2, visits_to_adopt=visits),
            FakeHuman(real_uid=1, exposition_timestamp=3, visits_to_adopt=visits),
        ]
        self.assertTrue(all([
            not humans[0].rolling_exposed[0],
            not humans[0].rolling_exposed[1],
            humans[0].rolling_exposed[2],
            not humans[1].rolling_exposed[0],
            not humans[1].rolling_exposed[1],
            not humans[1].rolling_exposed[2],
            humans[1].rolling_exposed[3],
        ]))
        messages = generate_sent_messages(humans, minimum_risk_level_for_updates=5)
        self.assertIn(0, messages)
        self.assertIn(1, messages)
        # should not have updates since we don't reach day 5 (first update day)
        self.assertFalse(sum([len(msgs) for msgs in messages[0]["sent_update_messages"].values()]))
        self.assertFalse(sum([len(msgs) for msgs in messages[1]["sent_update_messages"].values()]))
        human0_encounters = [(t, m) for t, msgs in messages[0]["sent_encounter_messages"].items() for m in msgs]
        human1_encounters = [(t, m) for t, msgs in messages[1]["sent_encounter_messages"].items() for m in msgs]
        self.assertEqual(len(human0_encounters), 1)
        self.assertEqual(len(human1_encounters), 1)
        self.assertEqual(human0_encounters[0][1].risk_level, 0)
        self.assertEqual(human1_encounters[0][1].risk_level, 0)
        self.assertEqual(human0_encounters[0][1].uid, humans[0].rolling_uids[3])
        self.assertEqual(human1_encounters[0][1].uid, humans[1].rolling_uids[3])
        self.assertTrue(human0_encounters[0][1]._exposition_event)
        self.assertFalse(human1_encounters[0][1]._exposition_event)

    def test_simple_update(self):
        visits = [
            Visit(visitor_real_uid=0, visited_real_uid=1, exposition=True, timestamp=4),
            # 2nd visit used to make sure we populate the message tree beyond the 5-day mark
            Visit(visitor_real_uid=0, visited_real_uid=1, exposition=False, timestamp=6),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=1, visits_to_adopt=visits),  # starts off sick
            FakeHuman(real_uid=1, exposition_timestamp=4, visits_to_adopt=visits),  # gets sick on day 4
        ]
        messages = generate_sent_messages(humans, minimum_risk_level_for_updates=5)  # will send updates on day 5
        human0_encounters = [(t, m) for t, msgs in messages[0]["sent_encounter_messages"].items() for m in msgs]
        human1_encounters = [(t, m) for t, msgs in messages[1]["sent_encounter_messages"].items() for m in msgs]
        human0_updates = [(t, m) for t, msgs in messages[0]["sent_update_messages"].items() for m in msgs]
        human1_updates = [(t, m) for t, msgs in messages[1]["sent_update_messages"].items() for m in msgs]
        self.assertTrue(all([
            len(human0_encounters) == 2,
            len(human1_encounters) == 2,
            len(human0_updates) == 1,
            len(human1_updates) == 0,
        ]))
        self.assertTrue(all([
            messages[0]["sent_encounter_messages"][4][0].risk_level == 0,
            messages[0]["sent_update_messages"][5][0]._update_reason == "symptoms",
        ]))
        human0_update = human0_updates[0][1]
        self.assertEqual(human0_update.uid, humans[0].rolling_uids[4])
        self.assertEqual(human0_update.old_risk_level, 0)
        self.assertEqual(human0_update.new_risk_level, 4)
        self.assertEqual(human0_update.encounter_time, 4)
        self.assertEqual(human0_update.update_time, 5)
        self.assertEqual(human0_update._sender_uid, 0)
        self.assertEqual(human0_update._receiver_uid, 1)
        self.assertEqual(human0_update._real_encounter_time, 4)
        self.assertEqual(human0_update._real_update_time, 5)
        self.assertEqual(human0_update._update_reason, "symptoms")

    def test_chain_updates(self):
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=0, visited_real_uid=1, exposition=False, timestamp=2),
            Visit(visitor_real_uid=0, visited_real_uid=1, exposition=False, timestamp=2),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=True, timestamp=4),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=7),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=8),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=12),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=4, visits_to_adopt=visits),  # sick on day 4 (transmission)
            FakeHuman(real_uid=1, exposition_timestamp=1, visits_to_adopt=visits),  # sick on day 1 (random)
        ]
        messages = generate_sent_messages(
            humans,
            minimum_risk_level_for_updates=5,
            maximum_risk_level_for_saturaton=10,
        )
        human0_updates = [(t, m) for t, msgs in messages[0]["sent_update_messages"].items() for m in msgs]
        human1_updates = [(t, m) for t, msgs in messages[1]["sent_update_messages"].items() for m in msgs]
        human0_encounters = [(t, m) for t, msgs in messages[0]["sent_encounter_messages"].items() for m in msgs]
        human1_encounters = [(t, m) for t, msgs in messages[1]["sent_encounter_messages"].items() for m in msgs]
        self.assertTrue(all([
            len(human0_encounters) == 7,  # both humans share 7 visits in total
            len(human1_encounters) == 7,
            len(human1_updates) == 8,  # will update day 2+4 encounters on day 6 (level5 broadcast)
                                       # ... and will update 2+4+7+8 on day 11 (level10 broadcast)
            len(human0_updates) == 3,  # will update day 4+7+8 encounters on day 8 (level5 broadcast)
        ]))
        self.assertTrue(all([m._receiver_uid == 1 for _, m in human0_encounters]))
        self.assertTrue(all([m._receiver_uid == 0 for _, m in human1_encounters]))
        self.assertTrue(all([  # track update behavior for day 4 encounter
            messages[1]["sent_encounter_messages"][4][0].risk_level == 0,
            messages[1]["sent_update_messages"][5][2]._update_reason == "symptoms",
            messages[1]["sent_update_messages"][5][2].old_risk_level == 0,
            messages[1]["sent_update_messages"][5][2].new_risk_level == 4,
            messages[1]["sent_update_messages"][10][2]._update_reason == "positive_test",
            messages[1]["sent_update_messages"][10][2].old_risk_level == 4,
            messages[1]["sent_update_messages"][10][2].new_risk_level == 10,
        ]))

    def test_multiple_encounters(self):
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=3),
            Visit(visitor_real_uid=0, visited_real_uid=2, exposition=False, timestamp=3),
            Visit(visitor_real_uid=2, visited_real_uid=1, exposition=False, timestamp=3),
            Visit(visitor_real_uid=3, visited_real_uid=0, exposition=False, timestamp=3),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=1, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=2, exposition_timestamp=never, visits_to_adopt=visits),
            FakeHuman(real_uid=3, exposition_timestamp=never, visits_to_adopt=visits),
        ]
        messages = generate_sent_messages(humans)
        self.assertTrue(all(idx in messages for idx in range(4)))
        self.assertFalse(sum([sum([len(msgs) for msgs in messages[idx]["sent_update_messages"].values()]) for idx in range(4)]))
        flat_encounters = [[(t, m) for t, msgs in messages[idx]["sent_encounter_messages"].items() for m in msgs] for idx in range(4)]
        self.assertEqual(len(flat_encounters[0]), 3)
        self.assertEqual(len(flat_encounters[1]), 2)
        self.assertEqual(len(flat_encounters[2]), 2)
        self.assertEqual(len(flat_encounters[3]), 1)

    def test_gen_received_messages(self):
        visits = [
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=0),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=True, timestamp=4),
            Visit(visitor_real_uid=0, visited_real_uid=2, exposition=False, timestamp=6),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=7),
            Visit(visitor_real_uid=1, visited_real_uid=0, exposition=False, timestamp=8),
            Visit(visitor_real_uid=0, visited_real_uid=2, exposition=False, timestamp=9),
        ]
        humans = [
            FakeHuman(real_uid=0, exposition_timestamp=4, visits_to_adopt=visits),  # sick on day 4 (transmission)
            FakeHuman(real_uid=1, exposition_timestamp=1, visits_to_adopt=visits),  # sick on day 1 (random)
            FakeHuman(real_uid=2, exposition_timestamp=never, visits_to_adopt=visits),  # never sick
        ]
        sent_messages = generate_sent_messages(
            humans=humans, minimum_risk_level_for_updates=5, maximum_risk_level_for_saturaton=10,
        )
        self.assertEqual(sum([len(msgs) for msgs in sent_messages[0]["sent_encounter_messages"].values()]), 6)
        self.assertEqual(sum([len(msgs) for msgs in sent_messages[0]["sent_update_messages"].values()]), 4)
        self.assertEqual(len(sent_messages[0]["sent_update_messages"][8]), 4)
        received_messages = generate_received_messages(
            humans=humans, minimum_risk_level_for_updates=5, maximum_risk_level_for_saturaton=10,
        )
        self.assertEqual(sum([len(msgs) for msgs in received_messages[0]["received_encounter_messages"].values()]), 6)
        self.assertEqual(sum([len(msgs) for msgs in received_messages[0]["received_update_messages"].values()]), 1)
        self.assertEqual(sum([len(msgs) for msgs in received_messages[1]["received_update_messages"].values()]), 3)
        self.assertEqual(sum([len(msgs) for msgs in received_messages[2]["received_update_messages"].values()]), 1)
        self.assertEqual(len(received_messages[2]["received_update_messages"][8]), 1)
        update_from_0_to_2_on_day_8 = received_messages[2]["received_update_messages"][8][0]
        self.assertEqual(update_from_0_to_2_on_day_8.update_time, 8)
        self.assertEqual(update_from_0_to_2_on_day_8.encounter_time, 6)
        self.assertEqual(update_from_0_to_2_on_day_8.uid, humans[0].rolling_uids[6])
        self.assertEqual(update_from_0_to_2_on_day_8.old_risk_level, 0)
        self.assertEqual(update_from_0_to_2_on_day_8.new_risk_level, 3)


if __name__ == "__main__":
    unittest.main()

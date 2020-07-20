"""
Class and functions to plan human's schedule.
"""
import datetime
import math
import numpy as np
from copy import deepcopy
from collections import defaultdict, deque

from covid19sim.utils.utils import _random_choice, filter_queue_max, filter_open, compute_distance, _normalize_scores
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE
ACTIVITIES = ["work", "socialize", "exercise", "grocery"]

class Activity(object):
    def __init__(self, start_in_seconds, end_in_seconds, duration, name, location):
        self.start_in_seconds = start_in_seconds # (float) number of seconds starting from midnight
        self.end_in_seconds = end_in_seconds # (float) number of seconds starting from midnight
        self.duration = duration # (float) in seconds
        self.name = name # (str) type of activity
        self.location = location

        # absolute times (starting from simulation start time)
        self.start_time = None # (datetime.datetime) object to be initialized in _patch_schedule
        self.end_time = None # (datetime.datetime) object to be initialized in _patch_schedule
        self.rsvp = set()
        self.parent_activity_pointer = None # stores pointer to the parent Activity for kids with supervision
        self.sender = None # pointer to human who sent this activity

    @property
    def date(self):
        return self.start_time.date()

    def __repr__(self):
        if self.start_time:
            return f"<{self.name} on {self.start_time.date()} from {self.start_time.time()} to {self.end_time.time()} at {self.location}>"
        return f"<TBD: {self.name} at {self.location} for {self.duration} seconds>"

    def clone(self, append_name="clone"):
        name = self.name
        if append_name:
            name = f"{append_name}-{self.name}"

        x = Activity(self.start_in_seconds, self.end_in_seconds,
                            self.duration, name, self.location)
        x.start_time = self.start_time
        x.end_time = self.end_time
        x.parent_activity_pointer = self.parent_activity_pointer
        return x

    def align(self, new_activity, cut_left=True, append_name=""):
        """
        Cuts short the duration of `self` to match the starting time of `activity`
        if - = activity (`x`), . = new_activity, then
        (cut_left = True) ...--.-- ==> .....---
        (cut_left = False) --.--... ==> --....


        Args:
            activity (Activity): activity to align with
            cut_left (bool): True if end_time of `self` is to be matched with `activity`. False for start_times
            append_name (str): name to prepend to the clone

        Returns:
            (Activity): new activity object with aligned time with `activity`
        """
        x = self.clone(append_name)
        if cut_left:
            # assert self.start_time  < new_activity.end_time, "not a proper alignment"
            x.start_time = new_activity.end_time
            x.start_in_seconds = x.start_time.hour * SECONDS_PER_HOUR + x.start_time.minute * SECONDS_PER_MINUTE + x.start_time.second
        else:
            # assert self.end_time  > new_activity.start_time, "not a proper alignment"
            x.end_time = new_activity.start_time
            x.end_in_seconds = x.end_time.hour * SECONDS_PER_HOUR + x.end_time.minute * SECONDS_PER_MINUTE + x.end_time.second

        x.duration = (x.end_time - x.start_time).total_seconds()

        assert x.duration >= 0, "negative duration encountered"
        return x

    def refresh_location(self):
        assert self.parent_activity_pointer is not None,  "refresh shouldn't be called without supervision or invitation"
        self.location = self.parent_activity_pointer.location

    def set_sender(self, human):
        self.sender = human

    def set_location_tracker(self, activity):
        """
        Sets the `parent_activity_pointer` to activity.location for future reference.

        Args:
            activity (Activity): activity to follow
        """
        self.parent_activity_pointer = activity

class MobilityPlanner(object):
    """
    Scheduler planning object that prepares `human`s schedule from the time of waking up to sleeping on the same day.

    Args:
        human (covid19sim.human.Human): `human` for whom this schedule needs to be planned
        env (simpy.Environment): simpy environment that schedules these `activities`
        conf (dict): yaml configuration of the experiment
    """
    def __init__(self, human, env, conf):
        self.human = human
        self.env = env
        self.conf = conf
        self.rng = human.rng

        self.invitation = {
            "accepted": {},
            "sent": set(),
            "received": set()
        }
        self.schedule_for_day = []
        self.current_activity = None
        self.follows_adult_schedule, self.adult_to_follow = False, []
        self.schedule_prepared = set()

    def __repr__(self):
        return f"<MobilityPlanner for {self.human}>"

    def initialize(self):
        """
        Initializes current activity to be sleeping until AVG_SLEEPING_MINUTES.
        Prepares a tentative schedule for the entire simulation so that only the location needs to be determined.
        Following cases are considered -
            1. `human` is a kid that can't be without parent supervision
            2. `human` is a kid that can go to school, but needs parent supervision at other times
            3. `human` who is free to do anything.
        """
        # start human from the activity of sleeping. (assuming everyone sleeps for same amount of time)
        AVERAGE_TIME_SLEEPING = self.conf['AVERAGE_TIME_SLEEPING']
        duration = AVERAGE_TIME_SLEEPING * SECONDS_PER_HOUR
        self.current_activity = Activity(0, duration, duration, "sleep", self.human.household)
        self.current_activity.start_time = self.env.timestamp
        self.current_activity.end_time = self.env.timestamp + datetime.timedelta(seconds=duration)
        self.schedule_for_day = deque([self.current_activity])

        # presample activities for the entire simulation
        n_days = self.conf['simulation_days']
        todays_weekday = self.env.timestamp.weekday()

        ## work
        if self.human.does_not_work:
            does_work = np.zeros(n_days)
        else:
            does_work = 1.0 * np.array([(todays_weekday + i) % 7 in self.human.working_days for i in range(n_days)])
            n_working_days = (does_work > 0).sum()
            does_work[does_work > 0] = [_sample_activity_duration("work", self.conf, self.rng) for _ in range(n_working_days)]

        MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
        if self.human.age <= MAX_AGE_CHILDREN_WITHOUT_SUPERVISION:
            self.follows_adult_schedule = True
            adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
            assert len(adults_in_house) > 0, "No adult found"

            adult_to_follow = self.rng.choice(adults_in_house , size=n_days, replace=True)
            self.adult_to_follow = deque(adult_to_follow.tolist())
        else:
            ## other activities
            does_grocery = _presample_activity("grocery", self.conf, self.rng, n_days)
            does_exercise = _presample_activity("exercise", self.conf, self.rng, n_days)
            does_socialize = _presample_activity("socialize", self.conf, self.rng, n_days)

            # schedule them all while satisfying sleep constraints
            # Note: we sample locations on the day of activity
            last_activity = self.current_activity
            full_schedule = []
            for i in range(n_days):
                assert last_activity.name == "sleep", f"found {last_activity} and not sleep"

                # Note: order of appending is important to _patch_schedule
                to_schedule = []
                to_schedule.append(Activity(None, None, does_work[i].item(), "work", self.human.workplace))
                to_schedule.append(Activity(None, None, does_socialize[i].item(), "socialize", None))
                to_schedule.append(Activity(None, None, does_grocery[i].item(), "grocery", None))
                to_schedule.append(Activity(None, None, does_exercise[i].item(), "exercise", None))

                # add idle and sleep acivities too
                schedule = _patch_schedule(self.human, last_activity, to_schedule, self.conf)
                last_activity = schedule[-1]
                full_schedule.append(schedule)

            self.full_schedule = deque(full_schedule)

    def get_schedule(self, force_today=False):
        """
        Moves the schedule pointer to the schedule for the current simulation day.

        Args:
            force_today (bool): return schedule for today even if it hasn't been prepared yet

        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        today = self.env.timestamp.date()

        if force_today:
            assert not self.follows_adult_schedule, "kids do not have preplanned schedule"
            # schedule hasn't been updated yet so we peek a day ahead
            if today not in self.schedule_prepared:
                return self.full_schedule[0]
            else:
                return self.schedule_for_day

        if len(self.schedule_for_day) == 0 and today not in self.schedule_prepared:
            self.schedule_for_day = self._prepare_schedule()
            self.schedule_prepared.add(today)

        return self.schedule_for_day

    def get_next_activity(self):
        """
        Clears `schedule_for_day` by popping the last element and storing it in `current_activity`.
        Also calls `prepare_schedule` when there are no more activities.

        Returns:
            (Activity): activity that human does next
        """
        schedule = self.get_schedule()
        self.current_activity = schedule.popleft()
        if self.current_activity.location is None:
            if any(x in self.current_activity.name for x in ["supervised", "invitation"]):
                self.current_activity.refresh_location()
            else:
                self.current_activity.location = _select_location(self.human, self.current_activity.name, self.human.city, self.rng, self.conf)
        return self.current_activity

    def invite(self, activity, connections):
        """
        Sends `activity` to connections.

        Args:
            activity (Activity):  activity object defining the activity.
            connections (list): list of humans that are to be sent a request
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."
        group = set()
        for human in connections:
            if human == self.human:
                continue
            if human.mobility_planner.receive(activity, self.human):
                group.add(human)

        # print(self.human, "invited", len(group), "others")
        activity.rsvp = group

    def receive(self, activity, human):
        """
        Receives the invite from `human`.

        Args:
            activity (Activity): attributes of activity
            human (covid19sim.human.Human): `human` who sent the invitation

        Returns:
            (bool): True if `self` adds `activity` to its schedule. False o.w.
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."
        today = self.env.timestamp.date()

        if (self.follows_adult_schedule
            or today in self.invitation["accepted"]
            or today in self.invitation["sent"]
            or today in self.invitation["received"]):
            return False

        self.invitation["received"].add(today)

        P_INVITATION_ACCEPTANCE = self.conf['P_INVITATION_ACCEPTANCE']
        if self.rng.random() > P_INVITATION_ACCEPTANCE:
            return False

        # invitations are sent on the day of the event
        # only accept this activity if it fits in the schedule of the day on which it is sent
        # and leave the current schedule unchanged
        new_schedule = self.get_schedule(force_today=True)
        remaining_schedule = [self.current_activity] + list(self.schedule_for_day)

        # /!\ by accepting the invite, `self` doesn't invite others to its social
        # thus, if there is a non-overlapping social on the schedule, `self` will go alone.
        new_activity = activity.clone(append_name="invitation")
        new_activity.set_location_tracker(activity)
        new_activity.set_sender(human)

        new_schedule, valid = _modify_schedule(remaining_schedule, new_activity, new_schedule)
        if valid:
            self.invitation["accepted"][today] = new_activity
            if today not in self.schedule_prepared:
                self.full_schedule[0] = new_schedule
            else:
                self.schedule_for_day = new_schedule
            return True

        return False

    def _prepare_schedule(self):
        """
        Prepares schedule for the next day. Retruns presampled schedule if its an adult.

        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        # if it's a kid that needs supervision, follow athe next-day schedule of a random adult in the household
        if self.follows_adult_schedule:
            adult = self.adult_to_follow.popleft()
            adult_schedule = adult.mobility_planner.get_schedule(force_today=True)
            work = 0
            if not self.human.does_not_work:
                work = _sample_activity_duration("work", self.conf, self.rng)

            remaining_schedule = [self.current_activity] + list(self.schedule_for_day)
            schedule = _patch_kid_schedule(self.human, adult_schedule, work, remaining_schedule, self.conf)
        else:
            schedule = self.full_schedule.popleft()

        return schedule

    def send_social_invites(self):
        """
        Sends invitation for "socialize" activity to `self.human.known_connections`.
        To be called once per day at midnight. By calling it at midnight helps in modifying schedules of those who accept the invitation.
        """
        today = self.env.timestamp.date()

        # invite others
        if (not self.follows_adult_schedule
            and today not in self.invitation["sent"]
            and today not in self.invitation["received"]
            and today not in self.invitation["accepted"]):

            schedule = self.get_schedule(force_today=True)
            socials = [x for x in schedule if x.name == "socialize"]
            assert len(socials) <=1, "more than one socials on one day"
            if socials:
                self.invitation["sent"].add(today)
                self.invite(socials[0], self.human.known_connections)


def _modify_schedule(remaining_schedule, new_activity, new_schedule):
    """
    Finds space for `new_activity` while keeping `remaining_schedule` unchanged

    Args:
        remaining_schedule (list): list of `Activity`s to precede the new schedule
        new_activity (Activity): activity which needs to be fit in `new_schedule`
        new_schedule (list): list of activities that follow `remaining_schedule`

    Returns:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
    """
    valid = True
    assert len(remaining_schedule) > 0, "Empty remaining_schedule. Human should be doing something all the time."
    assert remaining_schedule[-1].name == "sleep", "sleep not found as the last activity"
    assert remaining_schedule[-1].end_time == new_schedule[0].start_time, "two schedules are not aligned"

    last_activity = remaining_schedule[-1]
    # if new_activity completely overlaps with last_activity, do not accept
    if new_activity.end_time <= last_activity.end_time:
        valid = False

    # if new_activity starts before the last_activity, do not accept
    if new_activity.start_time < last_activity.end_time:
        valid = False

    # if new_activity coincides with work in new_schedule, do not accept
    work_activity = [(idx,x) for idx,x in enumerate(new_schedule) if x.name == "work"]
    work_activity_idx = -1
    if work_activity:
        work_activity_idx = work_activity[0][0]
        if work_activity[0][1].start_time < new_activity.start_time < work_activity[0][1].end_time:
            valid = False

    if not valid:
        return deque([]), False

    partial_schedule = []
    other_activities = [x for idx,x in enumerate(new_schedule) if idx >  work_activity_idx]

    # fit thie new_activity into the schedule
    # - = activity, . = new_activity
    for activity in other_activities:
        cut_right, cut_left = False, False

        if activity.start_time <= new_activity.start_time:

            if activity.end_time <= new_activity.start_time:
                partial_schedule.append(activity)
                continue

            # --.--... ==> --.... (cut right)
            cut_right=True
            if activity.end_time > new_activity.end_time:
                # ...--.-- ==> .....--- (cut left also)
                cut_left = True

        if activity.start_time >= new_activity.start_time:
            if activity.end_time <= new_activity.end_time:
                # discard but if both ends are equal add new_activity before discarding or there will be a gap
                if new_activity not in partial_schedule:
                    partial_schedule.append(new_activity)
                continue

            if new_activity.end_time <= activity.start_time:
                partial_schedule.append(activity)
                continue

            # ...--.-- ==> .....--- (cut left only)
            cut_left = True

        if cut_right:
            partial_schedule.append(activity.align(new_activity, cut_left=False, append_name=""))

        if new_activity not in partial_schedule:
            partial_schedule.append(new_activity)

        if cut_left:
            partial_schedule.append(activity.align(new_activity, cut_left=True, append_name=""))

    full_schedule = [x for idx, x in enumerate(new_schedule) if idx <= work_activity_idx]
    full_schedule += partial_schedule

    assert remaining_schedule[-1].end_time == full_schedule[0].start_time, "times do not align"

    # uncomment for rigorous checks
    for a1, a2 in zip(full_schedule, full_schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"

    return deque(full_schedule), True

def _patch_kid_schedule(human, adult_schedule, work, remaining_schedule, conf):
    """
    Makes a schedule that copies `Activity`s in `adult_schedule` to the new schedule aligned with current_activity.
    Patches between adult_schedule and current_activity by adding "idle" activity.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled
        adult_schedule (list): list of `Activity`s which have been presampled for adult for the next day
        work (float): duration of work that needs to be scheduled before anything
        remaining_schedule (list): list of `Activity`s remaining to be executed before following `adult_schedule`
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s that are in continuation with `remaining_schedule`. It doesn't include `remaining_schedule`
    """
    assert len(remaining_schedule) > 0, "Empty remaining_schedule. Human should be doing something all the time."
    assert remaining_schedule[-1].name == "sleep", "sleep not found as the last activity"

    wake_up_time_in_seconds = remaining_schedule[-1].end_in_seconds # relative to midnight
    schedule, awake_duration = [], 0

    current_activity = remaining_schedule[-1]
    # add work if required just after the last activity on the remaining_schedule
    if work > 0:
        new_activity = Activity(None, None, work, "work", human.workplace)
        new_activity = _fill_time_constraints(human, new_activity, current_activity, wake_up_time_in_seconds, conf)
        schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, current_activity, awake_duration)

    assert adult_schedule[-1].name == "sleep", "adult_schedule doesn't have sleep as its last element"

    # 1. discard activities which are completely a subset of schedule upto now
    # 2. align the activity which has a partial overlap with current_activity
    # 3. add rest of them as it is
    for activity in adult_schedule:
        # deques do not allow slicing.
        if activity.name == "sleep":
            continue

        # 1.
        if activity.end_time <= current_activity.end_time:
            continue

        # 2.
        elif activity.start_time < current_activity.end_time < activity.end_time:
            new_activity = activity.align(current_activity, cut_left=True, append_name="supervised")

        # 3.
        else:
            new_activity = activity.clone(append_name="supervised")

        new_activity.set_location_tracker(activity)
        schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, current_activity, awake_duration)

    # finally, close the schedule by adding sleep
    schedule, current_activity, awake_duration = _add_sleep_to_schedule(human, schedule, wake_up_time_in_seconds, current_activity, human.rng, conf, awake_duration)

    full_schedule = remaining_schedule + schedule
    # full_schedule.append(deque(schedule))
    for a1, a2 in zip(full_schedule, full_schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"

    return deque(schedule)

def _patch_schedule(human, last_activity, activities, conf):
    """
    Makes a continuous schedule out of the list of `activities` in continuation to `last_activity`.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled
        last_activity (Activity): last activity that `human` was doing
        activities (list): list of `Activity`s to add to the schedule
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
    """
    wake_up_time_in_seconds = last_activity.end_in_seconds # relative to midnight
    assert 0 <= wake_up_time_in_seconds <= SECONDS_PER_DAY, f"out-of-bounds wake_up_time_in_seconds {wake_up_time_in_seconds}"

    current_activity = last_activity
    schedule, awake_duration = [], 0
    for activity in activities:
        if activity.duration == 0:
            continue

        # set time in accordance to activity before this one and location constraints
        activity = _fill_time_constraints(human, activity, current_activity, wake_up_time_in_seconds, conf)

        # add idle activity if required and add it to the schedule
        schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, activity, current_activity, awake_duration)

    # finally, close the schedule by adding sleep
    schedule, current_activity, awake_duration = _add_sleep_to_schedule(human, schedule, wake_up_time_in_seconds, current_activity, human.rng, conf, awake_duration)

    for a1, a2 in zip(schedule, schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"

    return deque(schedule)

def _fill_time_constraints(human, activity, current_activity, wake_up_time_in_seconds, conf):
    """
    Checks and corrects the time constraints of `activity` with respect to location where it is taking place.

    Args:
        human (covid19sim.human.Human): human for which activity needs to be checked
        activity (Activity): new activity that needs to be scheduled ahead of `current_activity`
        current_activity (Activity): last activity that `human` was doing
        wake_up_time_in_seconds (float): seconds since midnight when `human` woke up
        conf (dict): yaml configuration of the experiment

    Returns:
        (Activity): activity with verified start and end time in seconds
    """
    # opening and closing time for the location of this activity
    opening_time, closing_time = _get_open_close_times(activity, conf)

    # set starting time
    if activity.name == "work":
        activity.start_in_seconds = human.work_start_time
    else:
        # (TODO: add-randomness) sample randomly from now until location is open - duration
        activity.start_in_seconds = current_activity.end_in_seconds

    ## if wake up late, then start the activity then
    activity.start_in_seconds = max(activity.start_in_seconds, wake_up_time_in_seconds)
    activity.start_in_seconds = max(activity.start_in_seconds, opening_time)
    activity.start_in_seconds = min(closing_time, activity.start_in_seconds)

    # set ending time
    ## if it is an all time open location, end_in_seconds can exceed closing time to next day
    if closing_time == SECONDS_PER_DAY:
        activity.end_in_seconds = activity.start_in_seconds + activity.duration
    else:
        activity.end_in_seconds = min(closing_time, activity.start_in_seconds + activity.duration)

    # overwrite duration to follow open and closing constraints of a location
    activity.duration = activity.end_in_seconds - activity.start_in_seconds

    assert activity.duration >= 0, f"negative duration {activity.duration} encountered"
    return activity

def _add_to_the_schedule(human, schedule, activity, last_activity, awake_duration):
    """
    Adds `activity` to the `schedule`. Also adds "idle" `Activity` if there is a time gap between `activity` and `last_activity`.

    Args:
        human (covid19sim.human.Human): human for which `activity` needs to be added to the schedule
        schedule (list): list of `Activity`s
        activity (Activity): new `activity` that needs to be added to the `schedule`
        last_activity (Activity): last activity that `human` was doing
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        awake_duration (float): total amount of time in seconds that `human` has been awake after adding the new `activity`.

    """
    assert activity.start_in_seconds >= 0, "negative start encountered"

    # if there is time between now and next activity add idle activity at household
    idle_time = activity.start_in_seconds - last_activity.end_in_seconds
    # if idle_time <= 0:
    #     warnings.warn(f"Negative idle time {idle_time} encountered.skipping the activity")
    #     return schedule, last_activity, awake_duration

    if idle_time > 0:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, activity, last_activity, awake_duration)
        idle_time = 0

    assert idle_time == 0, "non-zero idle time encountered"
    activity.start_time = last_activity.end_time
    activity.end_time = activity.start_time + datetime.timedelta(seconds=activity.duration) # only place where datetime operation is required
    schedule.append(activity)

    # adjust end_in_seconds if its more than 24 * 60 * 60 = 86400 seconds so that next schedule can be prepared starting from here.
    if activity.end_in_seconds > 86400:
        activity.end_in_seconds -= 86400

    return schedule, activity, awake_duration + activity.duration

def _add_sleep_to_schedule(human, schedule, wake_up_time_in_seconds, last_activity, rng, conf, awake_duration):
    """
    Adds sleep `Activity` to the schedule.

    Args:
        human (covid19sim.human.Human): human for which sleep schedule needs to be added.
        schedule (list): list of `Activity`s
        wake_up_time_in_seconds (float): seconds since midnight when `human` wake up
        last_activity (Activity): last activity that `human` was doing
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        total_duration (float): total amount of time in seconds that `human` had spent across all the activities in the schedule.
    """
    # draw time for which `self` remains awake and sleeps
    max_awake_duration = _sample_activity_duration("awake", conf, rng)
    sleep_duration = _sample_activity_duration("sleep", conf, rng)

    # Note: # if  start_sleep_time_in_seconds > last_activity.end_in_seconds we cut short the last_activity
    start_sleep_time_in_seconds = wake_up_time_in_seconds + max(awake_duration, max_awake_duration)
    activity = Activity(start_sleep_time_in_seconds, start_sleep_time_in_seconds + sleep_duration, sleep_duration, "sleep", human.household)

    # add idle time if its before bed time
    return _add_to_the_schedule(human, schedule, activity, last_activity, awake_duration)

def _add_idle_activity(human, schedule, next_activity, last_activity, awake_duration):
    """
    Adds an idle activity at household.

    Args:
        human (covid19sim.human.Human): human for which "idle" activity needs to be added to the schedule
        schedule (list): list of `Activity`s
        next_activity (Activity): new `activity` that needs to be added to the `schedule`
        last_activity (Activity): last activity that `human` was doing
        awake_duration (float): total amount of time in seconds that `human` had been awake

    Returns:
        schedule (list): list of `Activity`s with the last `Activity` as sleep
        last_activity (Activity): sleep as the last activity
        awake_duration (float): total amount of time in seconds that `human` has been awake after adding the new `activity`.
    """
    idle_time_in_seconds = next_activity.start_in_seconds - last_activity.end_in_seconds

    assert idle_time_in_seconds > 0, f"non-positive duration: {idle_time_in_seconds} encountered for idle time"

    idle_location = human.household
    idle_activity = Activity(last_activity.end_in_seconds, next_activity.start_in_seconds, idle_time_in_seconds, "idle", idle_location)
    return _add_to_the_schedule(human, schedule, idle_activity, last_activity, awake_duration)

def _get_open_close_times(activity, conf):
    """
    Fetches opening and closing time of the location_type where `activity` will take place.

    Args:
        activity (Activity): activity for which opening and closing times of location are requested
        conf (dict): yaml configuration of the experiment

    Returns:
        opening_time (float): opening time in seconds since midnight of `activity.location`
        closing_time (float): closing time in seconds since midnight of `activity.location`
    """
    if activity.name == "work":
        assert activity.location is not None, "workplace is None"
        return activity.location.opening_time, activity.location.closing_time
    elif activity.name == "grocery":
        location_type = "STORE"
    elif activity.name == "socialize":
        location_type = "MISC"
    elif activity.name == "exercise":
        location_type = "PARK"
    else:
        raise ValueError

    # /!\ same calculation is in covid19sim.locations.location.Location.__init__()
    OPEN_CLOSE_TIMES = conf[f'{location_type}_OPEN_CLOSE_HOUR_MINUTE']
    opening_time = OPEN_CLOSE_TIMES[0][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[0][1] * SECONDS_PER_MINUTE
    closing_time = OPEN_CLOSE_TIMES[1][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[1][1] * SECONDS_PER_MINUTE

    return opening_time, closing_time

def _sample_days_to_next_activity(P_ACTIVITY_DAYS, rng):
    """
    Samples days after which next activity can be scheduled.

    Args:
        P_ACTIVITY_DAYS (list): each element is a list - [d, p], where
                d is number of days after which this activity can be scheduled
                p is the probability of sampling d days for this activity.
                Note: p is normalized before being used.
        rng (np.random.RandomState): Random number generator

    Returns:
        (float): Number of days after which next activity can be scheduled
    """
    p = np.array([x[1] for x in P_ACTIVITY_DAYS])
    sampled_day = _random_choice(P_ACTIVITY_DAYS, size=1, P=p/p.sum(), rng=rng)[0]
    return sampled_day[0]

def _presample_activity(type_of_activity, conf, rng, n_days):
    """
    Presamples activity for `n_days`.

    Args:
        P_ACTIVITY_DAYS (list): each element is a list - [d, p], where
                d is number of days after which this activity can be scheduled
                p is the probability of sampling d days for this activity.
                Note: p is normalized before being used.
        rng (np.random.RandomState): Random number generator
        n_days (int): number of days to sample for

    Returns:
        (np.array): An array of size `n_days` containing float, where x implies do that activity for x seconds
    """
    if type_of_activity == "grocery":
        P_ACTIVITY_DAYS = conf['P_GROCERY_SHOPPING_DAYS']
    elif type_of_activity == "socialize":
        P_ACTIVITY_DAYS = conf['P_SOCIALIZE_DAYS']
    elif type_of_activity == "exercise":
        P_ACTIVITY_DAYS = conf['P_EXERCISE_DAYS']
    else:
        raise ValueError

    total_days_sampled = 0
    does_activity = np.zeros(n_days)
    duration = np.zeros(n_days)
    while total_days_sampled <= n_days:
        days = _sample_days_to_next_activity(P_ACTIVITY_DAYS, rng)
        total_days_sampled += days
        if total_days_sampled >= n_days:
            break
        does_activity[days] = _sample_activity_duration(type_of_activity, conf, rng)


    return does_activity

def _sample_activity_duration(activity, conf, rng):
    """
    Samples duration for `activity` according to predefined distribution, parameters of which are defined in the configuration file.
    TODO - Make it age dependent.

    Args:
        activity (str): type of activity
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator

    Returns:
        (float): duration for which to conduct activity (seconds)
    """
    SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR']
    SECONDS_CONVERSION_FACTOR = SECONDS_PER_HOUR
    if activity == "work":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_WORK"]

    elif activity == "grocery":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_GROCERY"]

    elif activity == "exercise":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_EXERCISING']

    elif activity == "socialize":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_SOCIALIZING']

    elif activity == "sleep":
        AVERAGE_TIME = conf['AVERAGE_TIME_SLEEPING']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']

    elif activity == "awake":
        AVERAGE_TIME = conf['AVERAGE_TIME_AWAKE']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']

    else:
        raise ValueError

    # round off to prevent microseconds in timestamps
    return math.floor(rng.gamma(AVERAGE_TIME/SCALE_FACTOR, SCALE_FACTOR) * SECONDS_CONVERSION_FACTOR)

def _select_location(human, activity, city, rng, conf):
    """
    Preferential exploration treatment to visit places in the city.

    Reference -
    Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015)
    Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166

    Args:
        activity (str): type of activity to sample from
        city (covid19sim.locations.city): `City` object in which `self` resides
        additional_visits (int): number of additional visits for `activity`. Used to decide location after some number of these visits.

    Raises:
        ValueError: when location_type is not one of "park", "stores", "hospital", "hospital-icu", "miscs"

    Returns:
        (covid19sim.locations.location.Location): a `Location` object
    """
    if activity == "exercise":
        S = human.visits.n_parks
        pool_pref = human.parks_preferences
        locs = filter_open(city.parks)
        visited_locs = human.visits.parks

    elif activity == "grocery":
        S = human.visits.n_stores
        pool_pref = human.stores_preferences
        # Only consider locations open for business and not too long queues
        locs = filter_queue_max(filter_open(city.stores), conf.get("MAX_STORE_QUEUE_LENGTH"))
        visited_locs = human.visits.stores

    elif activity == "hospital":
        for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(human.location, x)):
            if len(hospital.humans) < hospital.capacity:
                return hospital
        return None

    elif activity == "hospital-icu":
        for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(human.location, x)):
            if len(hospital.icu.humans) < hospital.icu.capacity:
                return hospital.icu
        return None

    elif activity == "socialize":
        # Note 1: a candidate location is human's household
        # Note 2: if human works at one of miscs, we still consider that as a candidate location
        P_HOUSE_OVER_MISC_FOR_SOCIALS = conf['P_HOUSE_OVER_MISC_FOR_SOCIALS']
        if rng.random() < P_HOUSE_OVER_MISC_FOR_SOCIALS:
            return human.household

        S = human.visits.n_miscs
        candidate_locs = city.miscs
        pool_pref = [(compute_distance(human.location, m) + 1e-1) ** -1 for m in candidate_locs]

        # Only consider locations open for business and not too long queues
        locs = filter_queue_max(filter_open(candidate_locs), conf.get("MAX_MISC_QUEUE_LENGTH"))
        visited_locs = human.visits.miscs

    elif activity == "work":
        return human.workplace

    else:
        raise ValueError(f'Unknown activity:{activity}')

    if S == 0:
        p_exp = 1.0
    else:
        p_exp = human.rho * S ** (-human.gamma)

    if rng.random() < p_exp and S != len(locs):
        # explore
        cands = [i for i in locs if i not in visited_locs]
        cands = [(loc, pool_pref[i]) for i, loc in enumerate(cands)]
    else:
        # exploit, but can only return to locs that are open
        cands = [
            (i, count)
            for i, count in visited_locs.items()
            if i.is_open_for_business
            and len(i.queue) <= conf.get("MAX_STORE_QUEUE_LENGTH")
        ]

    if len(cands) == 0:
        return None

    cands, scores = zip(*cands)
    loc = rng.choice(cands, p=_normalize_scores(scores))
    visited_locs[loc] += 1
    return loc

def _patch_kid_schedule_presmapled(human, adult_schedule, work, current_activity, conf):
    """
    Makes a schedule that copies `Activity`s in `adult_schedule`.
    Patches between two schedule by adding "idle" activities.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled
        adult_schedule (list): list of `Activity`s which have been presampled for adult
        work (np.array): each element is the duration in seconds for which `human` works on that day
        current_activity (Activity): last activity that `human` was doing
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of deques where each deque is a priority queue of `Activity`
    """
    full_schedule = []
    for day, day_schedule in enumerate(adult_schedule):
        schedule, awake_duration = [], 0
        wake_up_time_in_seconds = current_activity.end_in_seconds # relative to midnight

        if work[day] > 0:
            new_activity = Activity(None, None, work[day].item(), "work", human.workplace)
            new_activity = _fill_time_constraints(human, new_activity, current_activity, wake_up_time_in_seconds, conf)
            schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, current_activity, awake_duration)

        assert day_schedule[-1].name == "sleep", "day_schedule doesn't have sleep as its last element"
        for activity in day_schedule:
            # deques do not allow slicing.
            if activity.name == "sleep":
                continue
            name = f"supervised-{activity.name}"
            if current_activity.end_in_seconds >= activity.end_in_seconds:
                continue

            elif activity.end_in_seconds > current_activity.end_in_seconds > activity.start_in_seconds:
                duration = activity.end_in_seconds - current_activity.end_in_seconds
                new_activity = Activity(current_activity.end_in_seconds, activity.end_in_seconds, duration, name, activity.location)

            else:
                new_activity = Activity(activity.start_in_seconds, activity.end_in_seconds, activity.duration, name, activity.location)

            new_activity.parent_activity_pointer = activity
            schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, current_activity, awake_duration)

        # finally, close the schedule by adding sleep
        schedule, current_activity, awake_duration = _add_sleep_to_schedule(human, schedule, wake_up_time_in_seconds, current_activity, human.rng, conf, awake_duration)

        full_schedule.append(deque(schedule))

    return deque(full_schedule)

# Human schedule has 6 parts to it -
## work @ working hours and days (workplaces, stores, miscs, hospital, schools, SCR)
#>> WORK OPENING HOURS AND DAYS
## shopping when sustenance runs out (stores)
#>> AVERAGE SUSTENANCE PERIOD
## misc when entertainment runs out (miscs)
#>> AVERAGE ENTERTAINMENT PERIOD
## hospital when health runs out (hospital)
#>>
## sleep when energy runs out (household or senior_residences)
#>> AVERAGE ENERGY
## exercise @ exercise hours (parks)
#>> PROPORTION REGULAR EXERCISE

# Sensible checks --> (requires interaction between two MobilityPlanners)
## Can't leave home if there is a kid below SUPERVISON_AGE
## Can't go out to a restaurant if the other human is not coming out

# Location centeric mobility patterns

# Mobility pattern of human is motivated from the demands of the locations.
# Going to a grocery shop
# Households running out of food stock generates demand for residents to go for shopping.
# A following procedure can be used to simulate the behavior -
#   residents are notified about the urge to stock up
#   In the mobility planner of human, at the next time slot if the shop is open they go out
#   Who goes? Everyone? or just one person?
#       (a) houses like 'couple', 'single_person', etc. a group of person can go.
#           Constraints that need to be satisfied - kids not left alone
# For "other" houses, humans have their own sustenance period that is a part of their mobility planner.

# Going to a workplace - hospitals, misc, stores, workplaces, schools, SCR
# miscs - workplaces are busy on weekends to enable leisure time of people
# stores - they open regular hours on weekdays
# workplaces - they open regular hours on weekdays

# to go to a workplace, human should yield until the next time period.
#   For example, human enters house from work. we check when will the next activity take place, and we yield until then.
#   This is an example of an oracle that can give a peek into the future.
#   Human's workplace opening hours are predfined so we can have a random noise to when he will reach there.

# This planner will also help in modeling the sleep schedule as well as the time when the phone is on or off.
# fatigue level which makes human go to sleep. When human wakes up, fatigue level is Y hours. AVERAGE_TIME_HUMAN_IS_AWAKE (break it down by age)
# After this much time, he should be home to get to bed for AVERAGE_SLEEP_TIME. (break it down by age)
# this defines the time for which human sleeps and he should be at home after that.

# stores as a workplace
# these have a slightly different operation as compared to restaurants
# workers reach there on time and people only come here when these are open.

# miscs as a workplace
# same operation.



## requirements for the MobilityPlanner
# Make a schedule for the day
# Peek into the future so that yielding is prespecified.
# Define the sleep schedule at which there are no known interactions sampled.



# What happens if it is sick?

# Desire centric mobility pattern
# Human's entertainment level is refreshed for X minutes every time they go out to MISC.
# After X minutes, human gets an urge to go out with one of the known connections.
# this requires two mobility planners to interact and make a decision.
# If known connection can go out, they meet otherwise human doesn't go out.


# (B) Workplaces open at certain times. This notifies human to be at work.
#
# this creates a flag in human that is checked in the mobility planner

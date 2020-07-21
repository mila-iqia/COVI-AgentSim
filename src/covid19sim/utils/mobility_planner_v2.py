"""
Class and functions to plan human's schedule.
There are three types of functions -
1. _patch_schedule - Takes in tentative activities to schedule and make a continuous schedule out of them
2. _patch_kid_schedule - takes a current_activity and future schedule to follow  and makes a continuous schedule out of them
3. _modify_schedule - takes a current a schedule and a new activity that needs to be added and makes adjustment to it accordingly
"""
import datetime
import math
import numpy as np
from copy import deepcopy
from collections import defaultdict, deque

from covid19sim.utils.utils import _random_choice, filter_queue_max, filter_open, compute_distance, _normalize_scores, _get_seconds_since_midnight
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE
ACTIVITIES = ["work", "socialize", "exercise", "grocery"]

class Activity(object):
    def __init__(self, start_time, duration, name, location, tentative_date=None, append_name=""):
        self.start_time = start_time # (datetime.datetime) object to be initialized in _patch_schedule
        self.duration = duration # (float) in seconds
        self.name = name # (str) type of activity
        self.location = location
        self.tentative_date = tentative_date
        self.append_name=append_name

        self.rsvp = set()
        self.parent_activity_pointer = None # stores pointer to the parent Activity for kids with supervision
        self.sender = None # pointer to human who sent this activity

    @property
    def end_time(self):
        assert self.start_time is not None, "start time has not been initialized"
        return self.start_time + datetime.timedelta(seconds=self.duration) # (datetime.datetime) object to be initialized in _patch_schedule

    @property
    def date(self):
        return self.start_time.date()

    def __repr__(self):
        name = f"{self.append_name}-{self.name}" if self.append_name else self.name
        if self.start_time:
            return f"<{name} on {self.start_time.date()} from {self.start_time.time()} to {self.end_time.time()} at {self.location}>"
        return f"<TBD: {name} at {self.location} for {self.duration} seconds>"

    def clone(self, append_name="clone"):
        x = Activity(self.start_time, self.duration, self.name, self.location, append_name=append_name)
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
            x.start_time = new_activity.end_time
            x.duration = (x.end_time - x.start_time).total_seconds()
        else:
            # keep the start_time unchanged
            x.duration = (new_activity.start_time - x.start_time).total_seconds()

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

    def adjust_time(self, seconds, start=True):
        """
        Changes start time and duration by `seconds`.

        Args:
            seconds (float): amount of seconds to apply to start or end
            start (bool): whether to apply this adjustment at the start or towards the end
        """
        if start:
            self.start_time += datetime.timedelta(seconds=seconds)
            self.duration -= seconds
        else:
            self.duration += seconds

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
        self.current_activity = Activity(self.env.timestamp, duration, "sleep", self.human.household)
        self.schedule_for_day = deque([self.current_activity])

        # presample activities for the entire simulation
        # simulation is run until these many days pass. We want to sample for all of these days. Add 1 to include the activities on the last day.
        n_days = self.conf['simulation_days'] + 1
        todays_weekday = self.env.timestamp.weekday()

        MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
        if self.human.age <= MAX_AGE_CHILDREN_WITHOUT_SUPERVISION:
            self.follows_adult_schedule = True
            adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
            assert len(adults_in_house) > 0, "No adult found"

            adult_to_follow = self.rng.choice(adults_in_house , size=n_days, replace=True)
            self.adult_to_follow = deque(adult_to_follow.tolist())
        else:
            ## work
            if self.human.does_not_work:
                does_work = np.zeros(n_days)
            else:
                does_work = 1.0 * np.array([(todays_weekday + i) % 7 in self.human.working_days for i in range(n_days)])
                n_working_days = (does_work > 0).sum()
                does_work[does_work > 0] = [_sample_activity_duration("work", self.conf, self.rng) for _ in range(n_working_days)]

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
                # Note: duration of activities is equally important. A variance factor of 10 in the distribution
                # might result in duration spanning two or more days which will violate the assumptions in this planner.
                to_schedule = []
                tentative_date = (self.env.timestamp + datetime.timedelta(days=i)).date()
                to_schedule.append(Activity(None, does_work[i].item(), "work", self.human.workplace, tentative_date))
                to_schedule.append(Activity(None, does_socialize[i].item(), "socialize", None, tentative_date))
                to_schedule.append(Activity(None, does_grocery[i].item(), "grocery", None, tentative_date))
                to_schedule.append(Activity(None, does_exercise[i].item(), "exercise", None, tentative_date))

                # adds idle and sleep acivities too
                schedule = _patch_schedule(self.human, last_activity, to_schedule, self.conf)
                last_activity = schedule[-1]
                full_schedule.append(schedule)

            assert all(schedule[-1].name == "sleep" for schedule in full_schedule), "sleep not found as last element in a schedule"
            assert len(full_schedule) == n_days, "not enough schedule prepared"
            assert full_schedule[-1][-1].end_time >= self.env.timestamp + datetime.timedelta(seconds=n_days * SECONDS_PER_DAY), "not enough schedule length sampled"

            self.full_schedule = deque(full_schedule)

    def get_schedule(self, force_end_time=None):
        """
        Moves the schedule pointer to the schedule for the current simulation day.

        Args:
            force_range (start_time, end_time): returns schedule that spans over all the acitivities across start_time and end_time
            force_end_time (datetime.datetime): return all the activities until first "sleep" which have end_time greater than force_end_time
        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        today = self.env.timestamp.date()

        if force_end_time is not None:
            assert not self.follows_adult_schedule, "kids do not have preplanned schedule"
            try:
                return [self.current_activity]  + list(self.schedule_for_day) + list(self.full_schedule[0])
            except:
                breakpoint()
            # schedule hasn't been updated yet so we peek a day ahead
            # if len(self.schedule_for_day) == 0 and self.current_activity.end_time > force_end_time:
            #     return [self.current_activity]
            #
            # if today not in self.schedule_prepared:
            #     return self.full_schedule[0]
            # else:
            #     return [self.current_activity] + list(self.schedule_for_day)

        if len(self.schedule_for_day) == 0:
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

        # self.current_activity = _do_mobility_reduction_checks(self, self.current_activity)
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
        assert len(self.schedule_for_day) == 0, "_prepare_schedule should only be called when there are no more activities in schedule_for_day"
        assert self.current_activity.name == "sleep", "_prepare_schedule should only be called if current_activity is 'sleep' "
        # if it's a kid that needs supervision, follow athe next schedule (until "sleep") of a random adult in the household
        if self.follows_adult_schedule:
            adult = self.adult_to_follow.popleft()
            adult_schedule = adult.mobility_planner.get_schedule(force_end_time = self.current_activity.end_time)

            work_activity = None
            if not self.human.does_not_work and self.env.timestamp.weekday() in self.human.working_days:
                work = _sample_activity_duration("work", self.conf, self.rng)
                work_activity = Activity(None, work, "work", self.human.workplace, self.env.timestamp.date())

            schedule = _patch_kid_schedule(self.human, adult_schedule, work_activity, self.current_activity, self.conf)
        else:
            schedule = self.full_schedule.popleft()

        return schedule

    def send_social_invites(self):
        """
        Sends invitation for "socialize" activity to `self.human.known_connections`.
        To be called once per day at midnight. By calling it at midnight helps in modifying schedules of those who accept the invitation.
        """
        return None
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
        if (work_activity[0][1].start_time <= new_activity.start_time
            and new_activity.end_time < work_activity[0][1].end_time):
            valid = False

    if not valid:
        return deque([]), False

    partial_schedule = []
    other_activities = [x for idx,x in enumerate(new_schedule) if idx >=  work_activity_idx]

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

    full_schedule = [x for idx, x in enumerate(new_schedule) if idx < work_activity_idx]
    full_schedule += partial_schedule

    assert remaining_schedule[-1].end_time == full_schedule[0].start_time, "times do not align"

    # uncomment for rigorous checks
    for a1, a2 in zip(full_schedule, full_schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"
        assert a1.duration >= 0, "negative duration encountered"

    return deque(full_schedule), True

def _patch_kid_schedule(human, adult_schedule, work_activity, current_activity, conf):
    """
    Makes a schedule that copies `Activity`s in `adult_schedule` to the new schedule aligned with `current_activity` (expects "sleep")
    Adds work activity if a non-zero `work` is provided.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled
        adult_schedule (list): list of `Activity`s which have been presampled for adult for the next day
        current_activity (Activity): activity that `human` is currently doing (expects "sleep")
        work_activity (float): duration of work that needs to be scheduled before anything
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s that are in continuation with `remaining_schedule`. It doesn't include `remaining_schedule`.
    """
    assert current_activity.name == "sleep", "sleep not found as the last activity"
    assert adult_schedule[-1].name == "sleep", "adult_schedule doesn't have sleep as its last element"

    last_sleep_activity = current_activity
    last_activity = current_activity
    assert all(activity.end_time > last_activity.start_time for activity in adult_schedule), "adult activities which spans kid's current activity are expected"
    assert any(activity.start_time > last_activity.end_time for activity in adult_schedule), "at least one adult activity that ends after kid's current activity is expected"

    max_awake_duration = _sample_activity_duration("awake", conf, human.rng)
    schedule, awake_duration = [], 0
    # add work just after the current_activity on the remaining_schedule
    if work_activity is not None:
        work_activity.start_time = _get_datetime_for_seconds_since_midnight(human.work_start_time, work_activity.tentative_date)
        # adjust activities if there is a conflict while keeping the last_activity unchanged because it's should be the current_activity of the kid
        if work_activity.start_time < last_activity.end_time:
            work_activity, last_activity = _make_hard_changes_to_activity_for_scheduling(work_activity, last_activity, keep_last_unchanged=True)

        schedule, last_activity, awake_duration = _add_to_the_schedule(human, schedule, work_activity, last_activity, awake_duration)

    candidate_activities = [activity for activity in adult_schedule if activity.end_time > last_activity.end_time]
    # 1. discard activities which are completely a subset of schedule upto now
    # 2. align the activity which has a partial overlap with current_activity
    # 3. add rest of them as it is
    for activity in candidate_activities:

        # 1.
        if activity.end_time <= last_activity.end_time:
            continue

        # 2.
        elif activity.start_time < last_activity.end_time < activity.end_time:
            new_activity = activity.align(last_activity, cut_left=True, append_name="supervised")

        # 3.
        else:
            new_activity = activity.clone(append_name="supervised")

        new_activity.set_location_tracker(activity)
        schedule, last_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, last_activity, awake_duration)

        if awake_duration > max_awake_duration:
            break

    # finally, close the schedule by adding sleep
    schedule, last_activity, awake_duration = _add_sleep_to_schedule(human, schedule, last_sleep_activity, last_activity, human.rng, conf, awake_duration, max_awake_duration=max_awake_duration)

    full_schedule = [current_activity] + schedule
    for a1, a2 in zip(full_schedule, full_schedule[1:]):
        assert a1.end_time == a2.start_time, "times do not align"

    return deque(schedule)

def _patch_schedule(human, last_activity, activities, conf):
    """
    Makes a continuous schedule out of the list of `activities` in continuation to `last_activity` (expects "sleep") from previous schedule.

    Args:
        human (covid19sim.human.Human): human for which `activities`  needs to be scheduled. expects only duration for them and no start_time
        last_activity (Activity): last activity (expects sleep) that `human` was doing
        activities (list): list of `Activity`s to add to the schedule
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
    """
    assert last_activity.name == "sleep", "sleep not found as the last activity"

    current_activity = last_activity
    schedule, awake_duration = [], 0
    for activity in activities:
        if activity.duration == 0:
            continue

        if activity.name == "work":
            activity.start_time = _get_datetime_for_seconds_since_midnight(human.work_start_time, activity.tentative_date)
            # adjust activities if there is a conflict
            if activity.start_time < current_activity.end_time:
                activity, current_activity = _make_hard_changes_to_activity_for_scheduling(activity, current_activity)
        else:
            # (TODO: add-randomness) sample randomly from now until location is open - duration
            activity.start_time = current_activity.end_time

        # add idle activity if required and add it to the schedule
        schedule, current_activity, awake_duration = _add_to_the_schedule(human, schedule, activity, current_activity, awake_duration)

    # finally, close the schedule by adding sleep
    schedule, current_activity, awake_duration = _add_sleep_to_schedule(human, schedule, last_activity, current_activity, human.rng, conf, awake_duration)

    return deque(schedule)

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
    assert activity.start_time is not None, "only fully defined activities are expected"
    assert activity.start_time >= last_activity.end_time, "function assumes no confilict with the last activity"

    # ** A ** # set up the activity so that it is in accordance to the previous activity and the location's opening and closing constraints

    # opening and closing time for the location of this activity
    opening_time, closing_time = _get_open_close_times(activity.name, human.conf, activity.location)

    ## check the constraints with respect to a location
    seconds_since_midnight = _get_seconds_since_midnight(activity.start_time)
    if seconds_since_midnight > closing_time:
        return schedule, last_activity, awake_duration

    if seconds_since_midnight < opening_time:
        activity.start_time = _get_datetime_for_seconds_since_midnight(opening_time, activity.tentative_date)
        if activity.start_time < last_activity.end_time:
            return schedule, last_activity, awake_duration

    # if it is not an all time open location, end_in_seconds can not exceed closing time
    if closing_time != SECONDS_PER_DAY:
        activity.duration = min(closing_time - seconds_since_midnight, activity.duration)

    if activity.duration == 0:
        # print(human, f"has {activity} of 0 duration")
        pass
    assert activity.duration >= 0, f"negative duration {activity.duration} encountered"

    # ** B ** # Add an idle activity if there is a time gap between this activity and the last activity
    idle_time = (activity.start_time - last_activity.end_time).total_seconds()

    assert idle_time >= 0, f"negative idle_time {idle_time} encountered"

    if idle_time > 0:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, activity, last_activity, awake_duration)

    assert last_activity.end_time == activity.start_time, f"times do not align for {last_activity} and {activity}"

    schedule.append(activity)
    return schedule, activity, awake_duration + activity.duration

def _add_sleep_to_schedule(human, schedule, last_sleep_activity, last_activity, rng, conf, awake_duration, max_awake_duration=0):
    """
    Adds sleep `Activity` to the schedule. We constrain everyone to have an awake duration during which they
    hop from one network to the other, and a sleep during which they are constrained to be at their respective household networks.

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
    if max_awake_duration <= 0:
        max_awake_duration = _sample_activity_duration("awake", conf, rng)
    sleep_duration = _sample_activity_duration("sleep", conf, rng)

    start_time = last_sleep_activity.end_time + datetime.timedelta(seconds=max_awake_duration)
    sleep_activity = Activity(start_time, sleep_duration, "sleep", human.household)

    if sleep_activity.start_time >= last_activity.end_time:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, sleep_activity, last_activity, awake_duration)
        return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

    sleep_activity, last_activity = _make_hard_changes_to_activity_for_scheduling(sleep_activity, last_activity)
    return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

def _make_hard_changes_to_activity_for_scheduling(next_activity, last_activity, keep_last_unchanged=False):
    """
    Makes changes to either of the activities if next_activity starts before last_activity.

    Args:

    Returns:

    """
    def _assert_positive_duration(next_activity, last_activity):
        assert next_activity.duration >= 0 and last_activity.duration >= 0, "negative duration encountered"
        return next_activity, last_activity

    # print(f"making hard changes between next- {next_activity} and last - {last_activity}")
    # 1. if last activity can be safely cut short, do that and leave next_activity unchanged
    if not keep_last_unchanged and last_activity.start_time <= next_activity.start_time:
        last_activity.duration = (next_activity.start_time - last_activity.start_time).total_seconds()
        return _assert_positive_duration(next_activity, last_activity)

    # 2. else cut short next_activity
    # short version -
    # next_activity.start_time = last_activity.end_time
    # next_activity.duration = min(0, (next_activity.end_time - next_activity.start_time).total_seconds())
    # return next_activity, last_activity

    # more explicit
    # 2a. do next_activity late
    if next_activity.end_time >= last_activity.end_time:
        next_activity.start_time = last_activity.end_time
        next_activity.duration = (next_activity.end_time - next_activity.start_time).total_seconds()
        return _assert_positive_duration(next_activity, last_activity)

    # 2b. next_activity was supposed to end before the last activity, hence don't do next_activity
    next_activity.start_time = last_activity.end_time
    next_activity.duration = 0
    return _assert_positive_duration(next_activity, last_activity)

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
    duration = (next_activity.start_time -  last_activity.end_time).total_seconds()
    if duration == 0:
        return schedule, last_activity, awake_duration

    assert duration > 0, "negative duration for idle activity is not allowed"
    idle_activity = Activity(last_activity.end_time, duration, "idle", human.household)

    idle_time = (idle_activity.end_time - next_activity.start_time).total_seconds()
    assert idle_time == 0,  "non-zero idle time after adding idle_activity"

    schedule.append(idle_activity)
    return schedule, idle_activity, awake_duration + duration

def _get_open_close_times(activity_name, conf, location=None):
    """
    Fetches opening and closing time for a `location` (if given) or a typical location where `activity_name` can take place.

    Args:
        name (str): type of activity
        location (covid19sim.locations.Location): location for which opening closing times are requested.
        conf (dict): yaml configuration of the experiment

    Returns:
        opening_time (float): opening time in seconds since midnight of `activity.location`
        closing_time (float): closing time in seconds since midnight of `activity.location`
    """
    if location is not None:
        return location.opening_time, location.closing_time

    elif activity_name == "grocery":
        location_type = "STORE"
    elif activity_name == "socialize":
        location_type = "MISC"
    elif activity_name == "exercise":
        location_type = "PARK"
    else:
        raise ValueError(f"Unknown activity_name:{activity_name}")

    # # /!\ same calculation is in covid19sim.locations.location.Location.__init__()
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
    SECONDS_CONVERSION_FACTOR = SECONDS_PER_HOUR

    if activity == "work":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_WORK"]
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_FOR_WORK']
        MAX_TIME = conf['MAX_TIME_WORK']

    elif activity == "grocery":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_GROCERY"]
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES']
        MAX_TIME = conf["MAX_TIME_SHORT_ACTVITIES"]

    elif activity == "exercise":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_EXERCISING']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES']
        MAX_TIME = conf["MAX_TIME_SHORT_ACTVITIES"]

    elif activity == "socialize":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_SOCIALIZING']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_FOR_SHORT_ACTIVITIES']
        MAX_TIME = conf["MAX_TIME_SHORT_ACTVITIES"]

    elif activity == "sleep":
        AVERAGE_TIME = conf['AVERAGE_TIME_SLEEPING']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']
        MAX_TIME = conf['MAX_TIME_SLEEP']

    elif activity == "awake":
        AVERAGE_TIME = conf['AVERAGE_TIME_AWAKE']
        SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR_SLEEP_AWAKE']
        MAX_TIME = conf['MAX_TIME_AWAKE']

    else:
        raise ValueError

    # round off to prevent microseconds in timestamps
    duration = math.floor(rng.gamma(AVERAGE_TIME/SCALE_FACTOR, SCALE_FACTOR) * SECONDS_CONVERSION_FACTOR)
    return min(duration, MAX_TIME * SECONDS_PER_HOUR)

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

def _get_datetime_for_seconds_since_midnight(seconds_since_midnight, date):
    """
    Adds `seconds_since_midnight` to the `date` object.

    Args:
        seconds_since_midnight (float): seconds to add to the `date`
        date (datetime.date): date on which new datetime object needs to be initialized

    Returns:
        (datetime.datetime): datetime obtained after adding seconds_since_midnight to date
    """
    return datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(seconds=seconds_since_midnight)


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

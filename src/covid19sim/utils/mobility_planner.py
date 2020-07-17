"""
Class and functions to plan human's schedule.
"""
import datetime
from collections import defaultdict, deque
import numpy as np

from covid19sim.utils.utils import _random_choice
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

        self.social_group = set() # (set) used only when name = "socialize" to coordinate between people.

    def __repr__(self):
        if self.start_time:
            return f"<{self.name} on {self.start_time.date()} from {self.start_time.time()} to {self.end_time.time()} at {self.location}>"
        return f"<TBD: {self.name} at {self.location} for {self.duration} seconds>"


class MobilityPlanner(object):
    """
    Scheduel planning object that prepares `human`s schedule from the time of waking up to sleeping on the same day.

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

        self.restrictions = {"socialize":None}
        self.next_date = {activity: _sample_next_date(activity, conf, self.human, self.rng, self.env.timestamp.date()) for activity in ACTIVITIES}
        self.schedule_for_day = []
        self.current_activity = None
        self.follows_adult_schedule, self.adults_in_house = False, []
        self.schedule_prepared = {}

    def initialize(self):
        """
        Initializes current activity to be sleeping until AVG_SLEEPING_MINUTES.
        """
        MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
        if self.human.age <= MAX_AGE_CHILDREN_WITHOUT_SUPERVISION:
            self.follows_adult_schedule = True
            self.adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
            assert len(self.adults_in_house) > 0, "No adult found"

        # start human from the activity of sleeping. (assuming everyone sleeps for same amount of time)
        AVG_SLEEPING_MINUTES = self.conf['AVG_SLEEPING_MINUTES']
        duration = AVG_SLEEPING_MINUTES * SECONDS_PER_MINUTE
        self.current_activity = Activity(0, duration, duration, "sleep", self.human.household)
        self.current_activity.start_time = self.env.timestamp
        self.current_activity.end_time = self.env.timestamp + datetime.timedelta(seconds=duration)
        self.schedule_for_day = deque([self.current_activity])

    def get_schedule(self):
        """
        Prepares the schedule if not prepared yet.

        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        today = self.env.timestamp.date()
        if len(self.schedule_for_day) == 0 and not self.schedule_prepared.get(today, False):
            self.schedule_for_day = self.prepare_schedule()
            self.schedule_prepared[today] = True

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
        return self.current_activity

    def add_restrictions(self, activity):
        """
        Applies restrictions to mobility by accepting / rejecting invitations from others.

        Args:
            activity (Activity): activty and its attributes
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."

        P_INVITATION_ACCEPTANCE = self.conf['P_INVITATION_ACCEPTANCE']

        if (activity.start_time > self.human.work_end_time
            and activity.end_time < self.human.work_start_time):
            if self.restrictions["socialize"] is None and self.rng.uniform() < P_INVITATION_ACCEPTANCE:
                self.next_time["socialize"] = activity.start_time.date()
                self.next_location["socialize"] = activity.location
                self.restrictions["socialize"] = activity
                return True
        return False

    def do_activity_on_date(self, activity, date):
        """
        Decides if the activity is scheduled on `date`.

        Args:
            activity (str): type of activity
            date (datetime.date): date on which it needs to be checked

        Returns:
            (bool): True if it can be scheduled.
        """
        if activity == "work":
            return not self.human.does_not_work and date.weekday() in self.human.working_days

        return self.next_date[activity] is not None and self.next_date[activity] >= date

    def _notify_others_if_required(self, activity):
        """
        Notifies other `human`s in known connections to coordinate for group activities.

        Args:
            activity (Activtiy): type of activity
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."
        group = set()
        for human in self.human.known_connections:
            if human.mobility_planner.add_restrictions(activity):
                group.add(human)

        activity.social_group = group

    def prepare_schedule(self):
        """
        Prepares schedule for `human`. Following cases are considered -
            1. `human` is a kid that can't be without parent supervision
            2. `human` is a kid that can go to school, but needs parent supervision at other times
            3. `human` that is free to do anything.

        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        ALL_ACTIVITIES = ACTIVITIES
        schedule = {activity:None for activity in ACTIVITIES}
        adult = None
        today = self.env.timestamp.date()

        # parent supervision
        # if the kid goes to school then add "work" activity for the required time and add an adult's activity for the rest of the time
        # otherwise just use adult's schedule all the time.
        if self.follows_adult_schedule:
            adult = self.rng.choice(self.adults_in_house, size=1).item()
            if self.human.does_not_work or not self.do_activity_on_date("work", today):
                schedule = adult.mobility_planner.get_schedule()
                return _patch_supervised_schedule(self.human, schedule, self.current_activity, self.conf)
            else:
                ALL_ACTIVITIES = ["work"]

        # without supervision
        for activity in ALL_ACTIVITIES:
            if self.do_activity_on_date(activity, today):
                location = self.human._select_location(activity, self.human.city)
                duration = _sample_activity_duration(activity, self.conf, self.rng)
                schedule[activity] = Activity(None, None, duration, activity, location)
                self.next_date[activity] = _sample_next_date(activity, self.conf, self.human, self.rng, self.env.timestamp.date())

        return _patch_schedule(self.human, schedule, self.current_activity, self.rng, self.conf, self.env.timestamp, adult_to_follow=adult)

def _patch_supervised_schedule(human, adult_schedule, last_activity, conf):
    """
    Makes a schedule that copies `Activity`s in `adult_schedule`.

    Args:
        human (covid19sim.human.Human): human for which sleep schedule needs to be added.
        adult_schedule (list): list of `Activity`s
        last_activity (Activity): last activity that `human` was doing
        conf (dict): yaml configuration of the experiment

    Returns:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
    """
    assert len(adult_schedule) > 0, "empty schedule to follow"
    assert last_activity.name == "sleep", "last activity was not sleep"

    wake_up_time_in_seconds = last_activity.end_in_seconds
    kid_schedule, awake_duration = [], 0
    for activity in adult_schedule:
        if activity.name == "sleep":
            continue
        name = f"supervised-{activity.name}"
        kid_activity = Activity(activity.start_in_seconds, None, activity.duration, name, activity.location)
        kid_schedule, last_activity, awake_duration = _add_to_the_schedule(human, kid_schedule, kid_activity, last_activity, awake_duration)

    kid_schedule, last_activity, awake_duration = _add_sleep_to_schedule(human, kid_schedule, wake_up_time_in_seconds, last_activity, human.rng, conf, awake_duration)

    return deque(kid_schedule)


def _patch_schedule(human, schedule, last_activity, rng, conf, today, adult_to_follow):
    """
    Makes a continuous schedule out of the list of activities.

    Args:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
    """
    # start from the last activity
    assert last_activity.name == "sleep", "last activity is not sleep"
    current_activity = last_activity
    wake_up_time_in_seconds = last_activity.end_in_seconds

    full_schedule = []
    awake_duration = 0
    PRIORITIZED_ACTIVITIES = ["work", "socialize", "grocery", "exercise"]
    for type_of_activity in PRIORITIZED_ACTIVITIES:
        activity = schedule[type_of_activity]

        if activity is None:
            continue

        # set the starting time in seconds (relative to midnight)
        if type_of_activity == "work":
            activity.start_in_seconds = human.work_start_time
        else:
            activity.start_in_seconds = current_activity.end_in_seconds

        activity.start_in_seconds = max(activity.start_in_seconds, wake_up_time_in_seconds)

        full_schedule, current_activity, awake_duration = _add_to_the_schedule(human, full_schedule, activity, current_activity, awake_duration)

    if adult_to_follow is not None:
        assert sum(schedule[activity] is None for activity in ["socialize", "grocery", "exercise"]) == 3, "kids are not to make their decision for non-work activities"
        assert schedule["work"] is not None, "no work assigned to kid but patch_schedule called"
        assert len(full_schedule) <= 2, "more than 2 activities when adult is to be followed"
        adult_schedule = adult_to_follow.mobility_planner.get_schedule()
        work_end_in_seconds = full_schedule[-1].end_in_seconds
        for activity in adult_schedule:
            # if school ends before parent's work, send the kid to the office
            # else kid will go to the same location as parent

            if (activity.name != "sleep"
                and activity.start_in_seconds < work_end_in_seconds < activity.end_in_seconds):
                start_time_in_seconds = work_end_in_seconds
                duration = activity.end_in_seconds - work_end_in_seconds

            elif (activity.name != "sleep"
                  and activity.start_in_seconds >= work_end_in_seconds):
                start_time_in_seconds = activity.start_in_seconds
                duration = activity.duration
            else:
                continue

            kid_activity = Activity(start_time_in_seconds, None, duration, f"supervised-{activity.name}", activity.location)
            full_schedule, current_activity, awake_duration = _add_to_the_schedule(human, full_schedule, kid_activity, current_activity, awake_duration)

    full_schedule, current_activity, awake_duration = _add_sleep_to_schedule(human, full_schedule, wake_up_time_in_seconds, current_activity, human.rng, conf, awake_duration)

    assert full_schedule[-1].end_in_seconds < 86400, "unbounded end_in_seconds encountered"

    # uncomment for more rigorous checks on schedule
    # for a1, a2 in zip(full_schedule, full_schedule[1:]):
    #     assert a1.end_time == a2.start_time, "missing time in between two activities"

    return deque(full_schedule)

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
    AVG_AWAKE_MINUTES = conf['AVG_AWAKE_MINUTES']
    SCALE_AWAKE_MINUTES = conf['SCALE_AWAKE_MINUTES']

    AVG_SLEEPING_MINUTES = conf['AVG_SLEEPING_MINUTES']
    SCALE_SLEEPING_MINUTES = conf['SCALE_SLEEPING_MINUTES']

    # draw time for which `self` remains awake
    max_awake_duration  = rng.gamma(AVG_AWAKE_MINUTES / SCALE_AWAKE_MINUTES, SCALE_AWAKE_MINUTES) * SECONDS_PER_MINUTE

    # add idle time if its before bed time
    start_sleep_time_in_seconds = wake_up_time_in_seconds + max(awake_duration, max_awake_duration)
    sleep_duration = rng.gamma(AVG_SLEEPING_MINUTES / SCALE_SLEEPING_MINUTES, SCALE_SLEEPING_MINUTES) * SECONDS_PER_MINUTE
    activity = Activity(start_sleep_time_in_seconds, None, sleep_duration, "sleep", human.household)
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
    activity.start_in_seconds = max(activity.start_in_seconds, activity.location.opening_time)

    # if it is an all time open location, end_in_seconds can exceed closing time to the next day
    if activity.location.closing_time == SECONDS_PER_DAY:
        activity.end_in_seconds = activity.start_in_seconds + activity.duration
    else:
        activity.end_in_seconds = min(activity.location.closing_time, activity.start_in_seconds + activity.duration)

    # overwrite duration to follow the open and closing constraints of a location
    activity.duration = activity.end_in_seconds - activity.start_in_seconds
    # it can be zero when the location closing time is same as start_in_seconds.
    assert activity.duration >= 0, f"negative duration {activity.duration} encountered"

    # if there is time between now and next activity add idle activity at household
    idle_time = activity.start_in_seconds - last_activity.end_in_seconds
    if idle_time > 0:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, activity, last_activity, awake_duration)
        idle_time = 0

    assert idle_time == 0, "non-zero idle time encountered."

    activity.start_time = last_activity.end_time
    activity.end_time = activity.start_time + datetime.timedelta(seconds=activity.duration) # only place where datetime operation is required
    schedule.append(activity)

    # adjust end_in_seconds if its more than 24 * 60 * 60 = 86400 seconds so that next schedule can be prepared starting from here.
    if activity.end_in_seconds > 86400:
        activity.end_in_seconds -= 86400

    return schedule, activity, awake_duration + activity.duration

def _sample_days_to_next_activity(P_ACTIVITY_DAYS, rng):
    """
    Samples days after which next activity can be scheduled.

    Args:
        P_ACTIVITY_DAYS (list): each element is a list - [d, p], where
                d is number of days after which this activity can be scheduled
                p is the probability of sampling d days for this activity.
                Note: p is normalized before being used.

    Returns:
        (float): Number of days after which next activity can be scheduled
    """
    p = np.array([x[1] for x in P_ACTIVITY_DAYS])
    sampled_day = _random_choice(P_ACTIVITY_DAYS, size=1, P=p/p.sum(), rng=rng)[0]
    return sampled_day[0]

def _sample_next_date(activity, conf, human, rng, today):
    """
    Samples number of days after which `activity` will take place again.

    Args:
        activity (str): type of activity
        conf (dict): yaml configuration of the experiment
        human (covid19sim.human.Human): human object for which next activity needs to be sampled
        rng (np.random.RandomState): Random number generator
        today (datetime.date): today's date

    Returns:
        days (datetime.date): date on which it takes place next
    """
    if activity == "work":
        return None

    if activity == "grocery":
        P_GROCERY_SHOPPING_DAYS = conf['P_GROCERY_SHOPPING_DAYS']
        return today + datetime.timedelta(days=_sample_days_to_next_activity(P_GROCERY_SHOPPING_DAYS, rng=rng))

    if activity == "socialize":
        P_SOCIALIZE_DAYS = conf['P_SOCIALIZE_DAYS']
        return today + datetime.timedelta(days=_sample_days_to_next_activity(P_SOCIALIZE_DAYS, rng=rng))

    if activity == "exercise":
        P_EXERCISE_DAYS = conf['P_EXERCISE_DAYS']
        return today + datetime.timedelta(days=_sample_days_to_next_activity(P_EXERCISE_DAYS, rng=rng))

    raise ValueError

def _sample_activity_duration(activity, conf, rng):
    """
    Samples duration for `activity` according to predefined distribution, parameters of which are defined in the configuration file.

    Args:
        activity (str): type of activity
        conf (dict): yaml configuration of the experiment
        rng (np.random.RandomState): Random number generator

    Returns:
        (float): duration for which to conduct activity (seconds)
    """
    SCALE_FACTOR = conf['TIME_SPENT_SCALE_FACTOR']
    if activity == "work":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_WORK"]

    if activity == "grocery":
        AVERAGE_TIME = conf["AVERAGE_TIME_SPENT_GROCERY"]

    if activity == "exercise":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_EXERCISING']

    if activity == "socialize":
        AVERAGE_TIME = conf['AVERAGE_TIME_SPENT_SOCIALIZING']

    return rng.gamma(AVERAGE_TIME/SCALE_FACTOR, SCALE_FACTOR) * SECONDS_PER_HOUR

from covid19sim.utils.utils import filter_queue_max, filter_open, compute_distance, _normalize_scores
def _select_location(human, activity, city, conf):
    """
    Preferential exploration treatment to visit places in the city.

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
        S = human.visits.n_miscs
        pool_pref = [(compute_distance(human.location, m) + 1e-1) ** -1 for m in city.miscs if
                     m != human.location]
        # Only consider locations open for business and not too long queues
        locs = filter_queue_max(filter_open(city.miscs), conf.get("MAX_MISC_QUEUE_LENGTH"))
        visited_locs = human.visits.miscs

    elif activity == "work":
        return human.workplace

    else:
        raise ValueError(f'Unknown location_type:{location_type}')

    if S == 0:
        p_exp = 1.0
    else:
        p_exp = self.rho * S ** (-human.gamma)

    if self.rng.random() < p_exp and S != len(locs):
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

    if cands:
        cands, scores = zip(*cands)
        loc = human.rng.choice(cands, p=_normalize_scores(scores))
        visited_locs[loc] += 1
        return loc
    else:
        return None

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

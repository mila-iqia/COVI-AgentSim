"""
Class and functions to plan human's schedule.
There are three types of functions -
1. _patch_schedule - Takes in tentative activities to schedule and make a continuous schedule out of them
2. _patch_kid_schedule - takes a current_activity and future schedule to follow  and makes a continuous schedule out of them
3. _modify_schedule - takes a current a schedule and a new activity that needs to be added and makes adjustment to it accordingly
"""
import datetime
import math
import warnings
import numpy as np
from copy import deepcopy
from collections import defaultdict, deque

from covid19sim.utils.utils import _random_choice, filter_queue_max, filter_open, compute_distance, _normalize_scores, _get_seconds_since_midnight, log
from covid19sim.utils.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE
ACTIVITIES = ["work", "socialize", "exercise", "grocery"]

class Activity(object):
    def __init__(self, start_time, duration, name, location, owner, tentative_date=None, prepend_name="", append_name=""):
        self.start_time = start_time # (datetime.datetime) object to be initialized in _patch_schedule
        self.duration = duration # (float) in seconds
        self.name = name # (str) type of activity
        self.location = location# (Location) where this activity will take place
        self.tentative_date = tentative_date
        self.prepend_name = prepend_name # (str) used to denote activity modifier
        self.append_name = append_name # (str) used to denote activity modifier
        self.is_cancelled = False # (bool) to dynamically change the status of this activity
        self.owner = owner # (Location) location where owner should goto in case of cancellation

        # to organize socials, these attributes are used
        self.rsvp = set() # keeps record of who is coming, if someone gets sick, they can remvoe themselves from this list
        self.parent_activity_pointer = None # stores pointer to the parent Activity for kids with supervision and those who are invited to know the location

        self.human_dies = False # to identify if this activity marks the end of human

    @property
    def end_time(self):
        assert self.start_time is not None, "start time has not been initialized"
        return self.start_time + datetime.timedelta(seconds=self.duration) # (datetime.datetime) object to be initialized in _patch_schedule

    def __repr__(self):
        name = f"{self.prepend_name}-{self.name}" if self.prepend_name else self.name
        name = f"{name}-{self.append_name}" if self.append_name else name
        if self.start_time:
            return f"<{name} on {self.start_time.date()} from {self.start_time.time()} to {self.end_time.time()} at {self.location}>"
        return f"<TBD: {name} at {self.location} for {self.duration} seconds>"

    def clone(self, prepend_name="", append_name="", new_owner=None):
        if new_owner is None:
            new_owner = self.owner

        prepend_name = f"{self.prepend_name}-{prepend_name}" if self.prepend_name else prepend_name
        # append_name = f"{self.append_name}-{append_name}" if self.append_name else append_name
        x = Activity(self.start_time, self.duration, self.name, self.location, new_owner, prepend_name=prepend_name, append_name=self.append_name)
        x._add_to_append_name(append_name)
        x.parent_activity_pointer = self.parent_activity_pointer
        return x

    def align(self, new_activity, cut_left=True, prepend_name="", append_name="", new_owner=None):
        """
        Cuts short the duration of `self` to match the starting time of `activity`
        if - = activity (`x`), . = new_activity, then
        (cut_left = True) ...--.-- ==> .....---
        (cut_left = False) --.--... ==> --....


        Args:
            activity (Activity): activity to align with
            cut_left (bool): True if end_time of `self` is to be matched with `activity`. False for start_times
            prepend_name (str): name to prepend to the clone

        Returns:
            (Activity): new activity object with aligned time with `activity`
        """
        x = self.clone(prepend_name, append_name, new_owner)
        if cut_left:
            duration_decrease = (new_activity.end_time - x.start_time).total_seconds()
            x.start_time = new_activity.end_time
            x.duration = x.duration - duration_decrease
        else:
            # keep the start_time unchanged
            x.duration = (new_activity.start_time - x.start_time).total_seconds()

        assert x.duration >= 0, "negative duration encountered"
        return x

    def refresh_location(self):
        """
        Checks for the parent activity's location.
        NOTE: It needs to be called just before `yield`ing for this activity even though the `self.location` might not be none.
        This is being done in `MobilityPlanner._modify_activity_location_if_needed`
        """
        assert any(x in self.prepend_name for x in ["invitation", "supervised"]),  f"{self.prepend_name} not recognized. refresh shouldn't be called without supervision or invitation.\nActivity: {self}\nParent:{self.parent_activity_pointer}\nOwner:{self.owner}\nParent owner:{self.parent_activity_pointer.owner}"
        assert self.parent_activity_pointer is not None,  f"refresh shouldn't be called without supervision or invitation. Activity: {self}"

        if self.parent_activity_pointer.is_cancelled:
            self.location = self.owner.household
        else:
            self.location = self.parent_activity_pointer.location

    def set_location_tracker(self, activity):
        """
        Sets the `parent_activity_pointer` to activity.location for future lookup for location.
        Useful for activities like `supervised` and `invitation` where location is not decided until the time of activity.

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

    def _add_to_append_name(self, name):
        self.append_name += f"{name}"

    def cancel_and_go_to_location(self, reason, location=None):
        """
        Sets the flag `is_cancelled` to True and sets the location of this activity to `owner.household`

        Args:
            reason (str): reason to append to the the name of the activity
        """
        self.is_cancelled = True
        self._add_to_append_name(f"-cancel-{reason}")
        self.location = location if location is not None else self.owner.household # @1

        # (A) Socials
        # (A.1) if the plan was to go to someone else's invitation, then remove `self` from rsvp list
        # location of this activity has already been refreshed to self houeshold @1
        if (
            self.name == "socialize"
            and self.prepend_name == "invitation"
            and not "-cancel-for-kid-" not in self.parent_activity_pointer.append_name
        ):
            assert self.owner in self.parent_activity_pointer.rsvp, f"{self.owner} didnt' accept the {self.parent_activity_pointer}"
            self.parent_activity_pointer.rsvp.remove(self.owner)
            self.parent_activity_pointer = None

        # (A.2) if the plan was to invite others, it will be taken care by is_cancelled flag because that is checked at the time of deciding the location
        # pass

        # (B) Supervised
        # if adult is supervising a kid, that kid will follow the location of an adult's schedule
        # thus, if adult goes to home, kid will go to home as well

        # (C) Being supervised
        # Inverted supervision - change the parent_activity_pointer of adult to `self`.
        # this way, kids already following the adult will tell adult to follow them
        # below will not be helpful if adult's events are processed before kid's events, which is checked via adult.mobility_planner.inverted_supervision
        if self.prepend_name == "supervised":
            self.parent_activity_pointer.set_location_tracker(self)
            self.parent_activity_pointer._add_to_append_name( f"-cancel-for-kid-{reason}")
            self.parent_activity_pointer = None

            # Note: above will change the parent_activity_pointer of invitation-socialize activities
            # Thus, (A.1) will result in an error


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
            "accepted": set(),
            "sent": set(),
            "received": set()
        }
        self.schedule_for_day = []
        self.full_schedule = []
        self.current_activity = None
        self.follows_adult_schedule, self.adult_to_follow = False, []
        self.schedule_prepared = set()
        # since we pop the elements from full_schedule, we keep count of days passed
        self.schedule_day = -1 # denotes the number of schedules that full_schedule has already popped

        # inverted supervision - when kid needs to stay home and adult needs to be informed about this
        self.inverted_supervision = set()
        self.adult_to_follow_today = None

        # flags affected by health induced mobility reduction
        self.human_to_rest_at_home = False # (bool) used as a flag to restrict human to home

        # hospitalization parameters
        P_HOSPITALIZED_GIVEN_SYMPTOMS = self.conf['P_HOSPITALIZED_GIVEN_SYMPTOMS']
        self.human_will_be_hospitalized = self.rng.random() < P_HOSPITALIZED_GIVEN_SYMPTOMS[self.human.age_bin_width_10.index][2]

        P_CRITICAL_GIVEN_HOSPITALIZED = self.conf['P_CRITICAL_GIVEN_HOSPITALIZED']
        self.human_will_be_critical_if_hospitalized = self.rng.random() < P_CRITICAL_GIVEN_HOSPITALIZED[self.human.age_bin_width_10.index][2]

        P_FATALITY_GIVEN_CRITICAL = self.conf['P_FATALITY_GIVEN_CRITICAL']
        self.human_will_die_if_critical = self.rng.random() < P_FATALITY_GIVEN_CRITICAL[self.human.age_bin_width_10.index][2]

        self.hospitalization_timestamp = None
        self.critical_condition_timestamp = None
        self.death_timestamp = None
        self.location_of_hospitalization = None # stores `Location` where self.human is hospitalized or is in ICU, when hospitalized.
        self.human_dies_in_next_activity = False

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
        self.current_activity = Activity(self.env.timestamp, duration, "sleep", self.human.household, self.human)
        self.schedule_for_day = deque([self.current_activity])

        # presample activities for the entire simulation
        # simulation is run until these many days pass. We want to sample for all of these days. Add 1 to include the activities on the last day.
        # Add an additional 1 to be on teh safe side and sample activities for an extra day.
        n_days = self.conf['simulation_days'] + 1
        todays_weekday = self.env.timestamp.weekday()

        MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
        if self.human.age <= MAX_AGE_CHILDREN_WITHOUT_SUPERVISION:
            self.follows_adult_schedule = True
            self.adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
            if len(self.adults_in_house) > 0:
                self.adult_to_follow_today = self.rng.choice(self.adults_in_house, size=1).item()
                self.adult_to_follow_today.mobility_planner.inverted_supervision.add(self.human)
            else:
                self.follows_adult_schedule = False
                log(f"Improper housing allocation has led to {self.human} living without adult. MobilityPlanner will not keep them supervised.", self.human.city.logfile)

        if not self.follows_adult_schedule:
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
                to_schedule.append(Activity(None, does_work[i].item(), "work", self.human.workplace, self.human, tentative_date))
                to_schedule.append(Activity(None, does_socialize[i].item(), "socialize", None, self.human, tentative_date))
                to_schedule.append(Activity(None, does_grocery[i].item(), "grocery", None, self.human, tentative_date))
                to_schedule.append(Activity(None, does_exercise[i].item(), "exercise", None, self.human, tentative_date))

                # adds idle and sleep acivities too
                schedule = _patch_schedule(self.human, last_activity, to_schedule, self.conf)
                last_activity = schedule[-1]
                full_schedule.append(schedule)
                # (debug)
                if last_activity.duration == 0:
                    warnings.warn(f"{self.human} has 0 duration {last_activity}\nschedule:{schedule}\npenultimate:{full_schedule[-2]}")

            assert all(schedule[-1].name == "sleep" for schedule in full_schedule), "sleep not found as last element in a schedule"
            assert len(full_schedule) == n_days, "not enough schedule prepared"

            # fill the schedule with sleep if there is some time left at the end
            time_left_to_simulation_end = (full_schedule[-1][-1].end_time -  self.env.timestamp).total_seconds()
            assert time_left_to_simulation_end > SECONDS_PER_DAY, "A full day's schedule has not been planned"
            if time_left_to_simulation_end < n_days * SECONDS_PER_DAY:
                filler_schedule = deque([Activity(full_schedule[-1][-1].end_time, time_left_to_simulation_end, "sleep", self.human.household, self.human, prepend_name="filler")])
                full_schedule.append(filler_schedule)

            self.full_schedule = deque(full_schedule)

    def get_schedule(self, for_kids=False):
        """
        Moves the schedule pointer to the schedule for the current simulation day.

        Args:
            force_range (start_time, end_time): returns schedule that spans over all the acitivities across start_time and end_time
            force_end_time (datetime.datetime): return all the activities until first "sleep" which have end_time greater than force_end_time
        Returns:
            schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        """
        today = self.env.timestamp.date()

        if for_kids:
            assert not self.follows_adult_schedule, "kids do not have preplanned schedule"
            # on the last simulation day, at the time of this function call, adult might not have next_schedule.
            next_schedule = []
            if len(self.full_schedule) > 0:
                next_schedule = list(self.full_schedule[0])
            return [self.current_activity]  + list(self.schedule_for_day) + next_schedule

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
        self.current_activity = self._modify_activity_location_if_needed(self.current_activity)
        return self.current_activity

    def invite(self, activity, connections):
        """
        Sends `activity` to connections.

        Args:
            activity (Activity):  activity object defining the activity.
            connections (list): list of humans that are to be sent a request
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."

        # don't simulate gatherings which will not impact any message passing or transmissions
        if activity.duration < min(self.conf['MIN_MESSAGE_PASSING_DURATION'], self.conf['INFECTION_DURATION']):
            return None

        group = set()
        for human in connections:
            if human == self.human:
                continue
            if human.mobility_planner.receive(activity):
                group.add(human)

        # print(self.human, "invited", len(group), "others")
        activity.rsvp = group

    def receive(self, activity):
        """
        Receives the invite from `human`.

        Args:
            activity (Activity): attributes of activity

        Returns:
            (bool): True if `self` adds `activity` to its schedule. False o.w.
        """
        assert activity.name == "socialize", "coordination for other activities is not implemented."
        today = self.env.timestamp.date()

        current_schedule = [self.current_activity] + list(self.schedule_for_day)
        if (
            not _can_accept_invite(today, self)
            or (
                # Feature NotImplmented - to Schedule an event that modifies both the current_schedule and the next_schedule
                # ignoring the equality (in second cond. of `and`) here will make `activity` to be the last `schedule_for_day` which breaks the invariant that `last_activity` in `schedule_for_day` should be sleep.
                activity.start_time < current_schedule[-1].end_time
                and activity.end_time >= current_schedule[-1].end_time
            )
            or ( # can't schedule an event before the current_event ends
                activity.start_time < self.current_activity.end_time
            )
            or (
                # can only modify remaining schedule (sleep) if currently human is not sleeping (not waking up early)
                self.current_activity.name == "sleep"
                and activity.end_time <= current_schedule[-1].end_time
            )
        ):
            return False

        self.invitation["received"].add(today)

        P_INVITATION_ACCEPTANCE = self.conf['P_INVITATION_ACCEPTANCE']
        if self.rng.random() < 1 - P_INVITATION_ACCEPTANCE:
            return False

        # invitations are sent on the day of the event
        # only accept this activity if it fits in the schedule of the day on which it is sent
        # and leave the current schedule unchanged

        update_next_schedule = False
        next_schedule = list(self.full_schedule[0])
        # find schedule such that the invitation activity ends before the schedule ends
        # that schedule needs to be updated
        if activity.end_time <= current_schedule[-1].end_time:
            # its a double check wrt to the above condition (kept it here for better readability)
            if self.current_activity.name == "sleep":
                return False
            else:
                remaining_schedule = [self.current_activity]
                new_schedule = list(self.schedule_for_day)
        else:
            update_next_schedule = True
            remaining_schedule = [self.current_activity] + list(self.schedule_for_day) # this is only [current_activity] if current_activity.name == "sleep"
            new_schedule = list(self.full_schedule[0])

        # /!\ by accepting the invite, `self` doesn't invite others to its social
        # thus, if there is a non-overlapping social on the schedule, `self` will go alone.
        new_activity = activity.clone(prepend_name="invitation", new_owner=self.human)
        new_activity.set_location_tracker(activity)

        new_schedule, valid = _modify_schedule(self.human, remaining_schedule, new_activity, new_schedule)
        if valid:
            self.invitation["accepted"].add(today)
            if update_next_schedule:
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
            adults = _can_supervise_kid(self.adults_in_house)
            # when all adults in the house are dead --
            if len(adults) == 0:
                household = _reallocate_residence(self.human, self.human.city.households, self.rng, self.conf)
                if household is None:
                    raise NotImplementedError(f"There is no house available to supervise {self.human}")
                #
                log(f"Transferring {self.human} to {household} because no adult at {self.human.household} is alive. Current residents at {household}: {household.residents}", self.human.city.logfile)
                index_case_history = self.human.household.remove_resident(self.human)
                self.human.assign_household(household)
                household.add_resident(self.human, index_case_history)
                #
                MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = self.conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']
                self.adults_in_house = [h for h in self.human.household.residents if h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION]
                adults = _can_supervise_kid(self.adults_in_house)


            adult = self.rng.choice(adults, size=1).item()
            adult_schedule = adult.mobility_planner.get_schedule(for_kids = True)

            work_activity = None
            if not self.human.does_not_work and self.env.timestamp.weekday() in self.human.working_days:
                work = _sample_activity_duration("work", self.conf, self.rng)
                work_activity = Activity(None, work, "work", self.human.workplace, self.human, self.env.timestamp.date())

            schedule = _patch_kid_schedule(self.human, adult_schedule, work_activity, self.current_activity, self.conf)

            # set up leader and follower through inverted_supervision
            # kid follows adult's location all the time except for when kid is hospitalized or has to stay_at_home
            # adult checks inverted_supervision everytime before finalizing the location
            # thus, if kid needs to be followed, adult follows kid to their location
            self.adult_to_follow_today.mobility_planner.inverted_supervision.remove(self.human)
            self.adult_to_follow_today = adult
            self.adult_to_follow_today.mobility_planner.inverted_supervision.add(self.human)

        else:
            schedule = self.full_schedule.popleft()
            self.schedule_day += 1

        return schedule

    def send_social_invites(self):
        """
        Sends invitation for "socialize" activity to `self.human.known_connections`.
        NOTE (IMPORTANT): To be called once per day at midnight. By calling it at midnight helps in modifying schedules of those who accept the invitation.
        """
        today = self.env.timestamp.date()

        # invite others
        if _can_send_invite(today, self):
            todays_activities = []
            all_activities = [self.current_activity] + list(self.schedule_for_day)
            all_activities += [] if len(self.full_schedule) == 0 else list(self.full_schedule[0])
            for activity in all_activities:
                if activity.start_time.day == today.day or activity.end_time.day == today.day:
                    todays_activities.append(activity)

            MIN_DURATION = min(self.conf['INFECTION_DURATION'], self.conf['MIN_MESSAGE_PASSING_DURATION'])
            socials = [x for x in todays_activities if x.name == "socialize"]
            assert len(socials) <=1, "more than one socials on one day are not allowed in preplanned scheduling"
            if socials and socials[0].duration >= MIN_DURATION:
                self.invitation["sent"].add(today)
                self.invite(socials[0], self.human.known_connections)

    def _modify_activity_location_if_needed(self, activity):
        """
        Modifies the location of `activity` (and schedule if hospitalized) given a certain condition.

        Args:
            activity (Activity): current_activity for which checks are asked for

        Returns:
            activity (Activity): current_activity with updated location based on `self.human`'s condition
        """
        assert self.death_timestamp is None, "processing activities for a dead human"

        # for `human`s dying because of `human.never_recovers` we check the flag here
        if self.human_dies_in_next_activity:
            self, human, activity = _human_dies(self, self.human, activity, self.env)
            # print(self.human,  "is dead because never recovers", activity)
            return activity

        #
        self.human.intervened_behavior.quarantine.reset_if_its_time()

        # (a) set back to normal routine if self.human was hospitalized
        # (b) if human is still recovering in hospital then return the activity as it is because these were determined at the time when hospitalization occured
        # Note 1: after recovery from hospitalization, infection_timestamp will always be None
        # Note 2: hospitalization_recovery_timestamp will take into account hospitalization recovery due to critical condition
        # Note 3: patient dies in hospital so while recovering it is necessary to check for `human_will_die_if_critical`
        if (
            (self.hospitalization_timestamp is not None
            or self.critical_condition_timestamp is not None)
            and not self.human_will_die_if_critical
            and self.env.timestamp >= self.hospitalization_recovery_timestamp
        ):
            self.hospitalization_timestamp = None
            self.critical_condition_timestamp = None
            self.human.check_covid_recovery()
            # Hospital can discharge patient once the symptoms are not severe. patient needs to be careful after treatment to not infect others and practice isolation for some time
            # assert self.human.infection_timestamp is None, f"{self.human} is out of hospital and still has COVID"

        elif (
            (self.hospitalization_timestamp is not None
            or self.critical_condition_timestamp is not None)
            and self.env.timestamp < self.hospitalization_recovery_timestamp
        ):
            return activity # while in hospital, these activities are predetermined until hospitalization recovery time

        AVERAGE_TIME_TO_HOSPITAL_GIVEN_SYMPTOMS = self.conf['AVERAGE_DAYS_TO_HOSPITAL_GIVEN_SYMPTOMS']
        AVERAGE_TIME_TO_CRITICAL_IF_HOSPITALIZED = self.conf['AVERAGE_DAYS_TO_CRITICAL_IF_HOSPITALIZED']
        AVERAGE_TIME_DEATH_IF_CRITICAL = self.conf['AVERAGE_DAYS_DEATH_IF_CRITICAL']

        # hospitalization related checks
        # Note: because of the wide variance we have used averages only
        # TODO - P - Find out which distribution these parameters belong to.
        # 1. hospitalized given symptoms
        if (
            self.human_will_be_hospitalized
            and self.human.infection_timestamp is not None
            and self.human.covid_symptom_start_time is not None
            and self.hospitalization_timestamp is None
            and (self.env.timestamp - self.human.covid_symptom_start_time).total_seconds() >= AVERAGE_TIME_TO_HOSPITAL_GIVEN_SYMPTOMS * SECONDS_PER_DAY
        ):
            self.human.city.tracker.track_hospitalization(self.human) # track
            self.hospitalization_timestamp = self.env.timestamp
            hospital = _select_location(self.human, "hospital", self.human.city, self.rng, self.conf)
            if hospital is None:
                self, human, activity = _human_dies(self, self.human, activity, self.env)
                # print(self.human,  "died because of the lack of hospital capacity")
                return activity

            activity, self = _move_relevant_activities_to_hospital(self.human, self, activity, self.rng, self.conf, hospital, critical=False)
            # print(self.human,  "is hospitalized", activity)
            return activity

        # 2. critical given hospitalized
        # self.human is moved from hospital to its ICU
        if (
            self.human_will_be_critical_if_hospitalized
            and self.human.infection_timestamp is not None
            and self.hospitalization_timestamp is not None
            and self.critical_condition_timestamp is None
            and (self.env.timestamp - self.hospitalization_timestamp).total_seconds() >= AVERAGE_TIME_TO_CRITICAL_IF_HOSPITALIZED * SECONDS_PER_DAY
        ):
            self.human.city.tracker.track_hospitalization(self.human, "icu") # track
            self.critical_condition_timestamp = self.env.timestamp
            ICU = _select_location(self.human, "hospital-icu", self.human.city, self.rng, self.conf)
            if ICU is None:
                self, human, activity = _human_dies(self, self.human, activity, self.env)
                # print(self.human,  "died because of the lack of ICU capacity")
                return activity

            activity, self = _move_relevant_activities_to_hospital(self.human, self, activity, self.rng, self.conf, ICU, critical=True)
            # print(self.human,  "is critical", activity)
            return activity

        # 3. death given critical
        if (
            self.human_will_die_if_critical
            and self.human.infection_timestamp is not None
            and self.hospitalization_timestamp is not None
            and self.critical_condition_timestamp is not None
            and self.death_timestamp is None
            and (self.env.timestamp - self.critical_condition_timestamp).total_seconds() >= AVERAGE_TIME_DEATH_IF_CRITICAL * SECONDS_PER_DAY
        ):
            self, human, activity = _human_dies(self, self.human, activity, self.env)
            print(self.human,  "is dead because of the critical condition", activity)
            return activity

        # 4. for adults, if there is a kid that needs supervision because the kid has to stay at home or is hospitalized,
        # /!\ It doesn't let a single adult attend to two kids: one in hospital and another in house or 3 kids: in different hospitals etc..
        for kid in self.inverted_supervision:
            assert not self.follows_adult_schedule, "a kid should not go into inverted supervision"
            if (
                kid.mobility_planner.location_of_hospitalization is not None
                or kid.mobility_planner.hospitalization_timestamp is not None
                or kid.mobility_planner.critical_condition_timestamp is not None
            ):
                location = kid.mobility_planner.location_of_hospitalization # in hospitalization, kid's activities have location
                reason = "inverted-supervision-hospitalization"
                activity.cancel_and_go_to_location(reason=reason, location=location) # Note: this activity can't be supervised
                # print(self.human, activity, "for hospitalization of", kid)
                return activity

            # if kid is quarantined
            if kid.intervened_behavior.is_quarantining():
                location = kid.household
                reason = "inverted-supervision-quarantined"
                activity.cancel_and_go_to_location(reason=reason, location=location)
                return activity

            # if activity is already at home - no need to update
            # /!\ if for some reason, this condition is not valid while the kid's activity was canceled due to sickness
            # then this activity will call refresh_location and raise AssertionError due to prepend_name
            kid_to_stay_at_home = kid.mobility_planner.human_to_rest_at_home
            if (
                kid_to_stay_at_home
                and activity.location is not None
                and activity.location != kid.household
            ):
                location = kid.household #
                reason = "inverted-supervision-stay-at-home"
                activity.cancel_and_go_to_location(reason=reason, location=location)
                # print(self.human, activity, "for sick", kid)
                return activity

        # 5. if human is quarantined
        activity, quarantined = self._intervention_related_behavior_changes(activity)
        if quarantined:
            return activity

        
        # 6. health / intervention related checks; set the location to household
        # rest_at_home needs to be checked everytime. It is different from hospitalization which is for prespecified period of time
        rest_at_home = self._update_rest_at_home()
        if (
            rest_at_home
            and activity.location is not None
            and activity.location != self.human.household
        ):
            activity.cancel_and_go_to_location(reason="sick-rest-at_home", location=self.human.household)
            # print(self.human,  "is sick", activity)
            return activity

        # 7. (a) check the status of the parent_pointer (it has to be the invitation or supervised activity)
        # otherwise (b) find a location. if there is no location available, cancel the activity and stay at home
        if activity.parent_activity_pointer is not None:
            activity.refresh_location()
            return activity

        if activity.location is None:
            activity.location = _select_location(self.human, activity.name, self.human.city, self.rng, self.conf)
            if activity.location is None:
                activity.cancel_and_go_to_location(reason="no-location", location=self.human.household)
                print(self.human,  "can't get a location", activity)

        return activity

    def _update_rest_at_home(self):
        """
        Runs check on `human` to decide if saying at home is the right course of action.
        """
        if (
            self.hospitalization_timestamp is not None
            or self.critical_condition_timestamp is not None
        ):
            self.human_to_rest_at_home = False
            return self.human_to_rest_at_home

        self.human_to_rest_at_home = self.rng.random() < 1 - _get_likelihood_to_go_out(self.human, self.conf)
        return self.human_to_rest_at_home

    def _intervention_related_behavior_changes(self, activity):
        """
        Determines location of activity based on the restrictions imposed on `self.human` via interventions
        Implements basic intervention related behavior changes by modifying `human`s network profile (presence of `human` in various networks)

        Args:
            activity (Activity): activity for which location needs to be determined

        Returns:
            activity (Activity): activity with a new location
        """
        # Quarantine / Max behavior restriction
        # (assumption) only the last level changes the mobility pattern i.e. network presence of humans
        if self.human.intervened_behavior.is_quarantining():
            reason = self.human.intervened_behavior.current_behavior_reason[-1]
            activity.cancel_and_go_to_location(reason=f"quarantine-{reason}", location=self.human.household)
            return activity, True

        # econ
        # if activity is work
        if (self.human.conf['ECON_RUN'] == True) and (activity.name == 'work'):

            ## determine new work hour (say x in seconds)
            estimated_prevalence_dict = self.human.city.tracker.get_estimated_covid_prevalence()

            # use estimation by number of positive tests in the last 14 days
            estimated_prevalence = estimated_prevalence_dict["estimation_by_test"] # change to hospitalizations

            # compute the reduced number of hours worked by the human
            x = reduced_workload(self.human, activity.duration, estimated_prevalence) # change x

            activity.duration = x 

            current_schedule = self.get_schedule()

            next_scheduled_activity = current_schedule[0]

            remaining_time = (next_scheduled_activity.start_time - activity.end_time).total_seconds()

            new_activity = Activity(activity.end_time, remaining_time, "stay-home-prevalence", self.human.household, self.human, None, prepend_name="", append_name="-cancel-prevalence")
                                    #  start_time    , duration      , name                  , location            , owner     , tentative_date=None)

            self.current_schedule = current_schedule.appendleft(new_activity)

            return activity, True

        return activity, False

    def cancel_all_events(self):
        """
        Empties the remaining schedule and removes human from the residence.
        """
        self.schedule_for_day = []
        self.current_activity = None
        while len(self.full_schedule) > 0:
            schedule = self.full_schedule.popleft()
            while len(schedule) > 0:
                activity = schedule.popleft()

def _move_relevant_activities_to_hospital(human, mobility_planner, current_activity, rng, conf, hospital, critical=False):
    """
    Changes the schedule so that `human`s future activities are at a hospital.
    Note 1: [IMPORTANT]: human.recovery_days and mobility_planner.location_of_hospitalization is updated here too
    If the recovery time after being critical is less than the previous hospitalized recovery time, then activities are put back to normal routine.

    Note 2: This function should be called only at the beginning of hospitalzation because it also discharges and admits the patient
    """
    # modify the schedule until the recovery time
    AVERAGE_DAYS_RECOVERY = conf['AVERAGE_DAYS_RECOVERY_IF_HOSPITALIZED']
    if critical:
        AVERAGE_DAYS_RECOVERY = conf['AVERAGE_DAYS_RECOVERY_IF_CRITICAL']

    AVERAGE_TIME_RECOVERY = AVERAGE_DAYS_RECOVERY * SECONDS_PER_DAY
    recovery_time = human.env.timestamp + datetime.timedelta(seconds=AVERAGE_TIME_RECOVERY)
    mobility_planner.hospitalization_recovery_timestamp = recovery_time
    human.recovery_days = AVERAGE_DAYS_RECOVERY + (human.env.timestamp - human.infection_timestamp).total_seconds() / SECONDS_PER_DAY

    # admit this human to the list of patients at the hospital/ICU
    # this assumes that call to this function is only at the beginning and the end of hospitalization / ICU
    if mobility_planner.location_of_hospitalization is not None:
        mobility_planner.location_of_hospitalization.discharge(human)
    hospital.admit_patient(human, until=recovery_time)
    mobility_planner.location_of_hospitalization = hospital

    acitivities_to_revert_back_to_normal = [] # if critical, change in recovery time will need previously modified activities to change back to normal
    activities_to_modify = []
    for activity in [current_activity] + list(mobility_planner.schedule_for_day):
        if activity.end_time < recovery_time:
            activities_to_modify.append(activity)

    for schedule in mobility_planner.full_schedule:
        for activity in schedule:
            if (
                activity.start_time > recovery_time
                and "Hospitalized" in activity.append_name
            ):
                acitivities_to_revert_back_to_normal.append(activity)

            elif (
                activity.start_time > recovery_time
                and "Hospitalized" not in activity.append_name
            ):
                break

            activities_to_modify.append(activity)

    for activity in activities_to_modify:
        reason = "Hospitalized" if not critical else "ICU"
        activity.cancel_and_go_to_location(reason=reason, location=hospital)

    for activity in acitivities_to_revert_back_to_normal:
        activity._add_to_append_name("-recovered")
        activity.location = None
        if activity.name == "work":
            activity.location = human.workplace

    return current_activity, mobility_planner

def _get_likelihood_to_go_out(human, conf):
    """
    Checks for human's condition and recommends the likelhihood to go out

    Args:
        human (covid19sim.human.Human): `human` for whom mobility reduction is to be checked

    Returns:
        (float): likelihood to go out of home
    """
    current_symptoms = human.symptoms
    if len(current_symptoms) == 0:
        return 1.0

    ## reduction due to symtpoms
    # 1.
    SEVERE_SYMPTOMS = conf['SEVERE_SYMPTOMS']
    P_MOBILE_GIVEN_SEVERE_SYMPTOMS = conf['P_MOBILE_GIVEN_SEVERE_SYMPTOMS']

    if any(symptom in SEVERE_SYMPTOMS for symptom in current_symptoms):
        return P_MOBILE_GIVEN_SEVERE_SYMPTOMS

    # 2.
    MODERATE_SYMPTOMS = conf['MODERATE_SYMPTOMS']
    P_MOBILE_GIVEN_MODERATE_SYMPTOMS = conf['P_MOBILE_GIVEN_MODERATE_SYMPTOMS']

    if any(symptom in MODERATE_SYMPTOMS for symptom in current_symptoms):
        return P_MOBILE_GIVEN_MODERATE_SYMPTOMS

    # 3.
    MILD_SYMPTOMS = conf['MILD_SYMPTOMS']
    P_MOBILE_GIVEN_MILD_SYMPTOMS = conf['P_MOBILE_GIVEN_MILD_SYMPTOMS']

    if any(symptom in MILD_SYMPTOMS for symptom in current_symptoms):
        return P_MOBILE_GIVEN_MILD_SYMPTOMS

    return 1.0

def _modify_schedule(human, remaining_schedule, new_activity, new_schedule):
    """
    Finds space for `new_activity` while keeping `remaining_schedule` unchanged and maintaining its alignment with the `new_schedule`

    Args:
        human (covid19sim.human.Human): `human` for whom this modification needs be done
        remaining_schedule (list): list of `Activity`s to precede the new schedule
        new_activity (Activity): activity which needs to be fit in `new_schedule`
        new_schedule (list): list of activities that follow `remaining_schedule`

    Returns:
        schedule (deque): a deque of `Activity`s where the activities are arranged in increasing order of their starting time.
        valid (bool): True if its possible to safely edit the current schedule
    """
    assert len(remaining_schedule) > 0, "Empty remaining_schedule. Human should be doing something all the time."
    assert remaining_schedule[-1].end_time == new_schedule[0].start_time, "two schedules are not aligned"

    valid = True
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

    # fit the new_activity into the schedule
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
                # discard, but if both ends are equal, add new_activity before discarding or there will be a gap
                if new_activity not in partial_schedule:
                    partial_schedule.append(new_activity)
                continue

            if new_activity.end_time <= activity.start_time:
                partial_schedule.append(activity)
                continue

            # ...--.-- ==> .....--- (cut left only)
            cut_left = True

        if cut_right:
            partial_schedule.append(activity.align(new_activity, cut_left=False, prepend_name="modified-cut-right", new_owner=human))

        if new_activity not in partial_schedule:
            partial_schedule.append(new_activity)

        if cut_left:
            partial_schedule.append(activity.align(new_activity, cut_left=True, prepend_name="modified-cut-left", new_owner=human))

    full_schedule = [x for idx, x in enumerate(new_schedule) if idx < work_activity_idx and x.duration > 0]
    full_schedule += [x for x in partial_schedule if x.duration > 0 or x.name == "sleep"]

    assert remaining_schedule[-1].end_time == full_schedule[0].start_time, "times do not align"
    assert full_schedule[-1].name == "sleep", f"sleep not found as the last activity. \nfull_schedule:\n{full_schedule}\nnew_schedule:{new_schedule}\nremaining_schedule:{remaining_schedule}"

    # comment for rigorous checks or use -O as an option to run python scripts to avoid checking for assertions
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
    if (
        work_activity is not None
        and human.mobility_planner.hospitalization_timestamp is None
        and human.mobility_planner.critical_condition_timestamp is None
        and human.mobility_planner.death_timestamp is None
    ):
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
            new_activity = activity.align(last_activity, cut_left=True, prepend_name="supervised", new_owner=human)

        # 3.
        else:
            new_activity = activity.clone(prepend_name="supervised", new_owner=human)

        new_activity.set_location_tracker(activity) # this way location of this activity is decided only at the time of activity
        schedule, last_activity, awake_duration = _add_to_the_schedule(human, schedule, new_activity, last_activity, awake_duration)

        if awake_duration > max_awake_duration:
            break

    # finally, close the schedule by adding sleep
    schedule, last_activity, awake_duration = _add_sleep_to_schedule(human, schedule, last_sleep_activity, last_activity, human.rng, conf, awake_duration, max_awake_duration=max_awake_duration)

    # if kid is hospitalized, all the activities should take place at the hospital
    if human.mobility_planner.location_of_hospitalization is not None:
        for activity in schedule:
            activity.location = human.mobility_planner.location_of_hospitalization
            activity._add_to_append_name("-patched-hospitalized")

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
    sleep_activity = Activity(start_time, sleep_duration, "sleep", human.household, human)

    if sleep_activity.start_time >= last_activity.end_time:
        schedule, last_activity, awake_duration = _add_idle_activity(human, schedule, sleep_activity, last_activity, awake_duration)
        return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

    sleep_activity, last_activity = _make_hard_changes_to_activity_for_scheduling(sleep_activity, last_activity)
    return _add_to_the_schedule(human, schedule, sleep_activity, last_activity, awake_duration)

def _make_hard_changes_to_activity_for_scheduling(next_activity, last_activity, keep_last_unchanged=False):
    """
    Makes changes to either of the activities if next_activity starts before last_activity.

    Args:
        next_activity (Activity): activity that is in conflict with the last activity
        last_activity (Activity): previous activity that starts after `next_activity`
        keep_last_unchanged (bool): if True, doesn't affect last_activity

    Returns:
        next_activity (Activity): activity in alignment with `last_activity`
        last_activity (Activity): activity in alignment with `next_activity`
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
    idle_activity = Activity(last_activity.end_time, duration, "idle", human.household, human)

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
    elif activity_name in ["sleep", "idle"]:
        location_type = "HOUSEHOLD"
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
            if hospital.n_patients < hospital.capacity:
                return hospital
        return None

    elif activity == "hospital-icu":
        for hospital in sorted(filter_open(city.hospitals), key=lambda x:compute_distance(human.location, x)):
            if hospital.icu.n_patients < hospital.icu.capacity:
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

    elif activity in ["sleep", "idle"]:
        return human.household

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

def _can_accept_invite(today, mobility_planner):
    """
    Decides if `mobility_planner` can accept invite of a socialize activity

    Args:
        today (datetime.date): date on which an invite is received
        mobility_planner (MobilityPlanner): schedule planning object

    Returns:
        (bool): True if mobility_planner can accommodate an invite
    """
    # health related checks
    if (
        mobility_planner.human_to_rest_at_home
        or mobility_planner.hospitalization_timestamp is not None
        or mobility_planner.critical_condition_timestamp is not None
        or mobility_planner.death_timestamp is not None
    ):
        return False

    # behavior related checks
    if (
        mobility_planner.follows_adult_schedule
        or today in mobility_planner.invitation["accepted"]
        or today in mobility_planner.invitation["sent"]
        or today in mobility_planner.invitation["received"]
    ):
        return False

    # intervention related checks
    if mobility_planner.human.intervened_behavior.is_quarantining():
        return False

    return True

def _can_send_invite(today, mobility_planner):
    """
    Decides if `mobility_planner` can send invites to other `human`s

    Args:
        today (datetime.date): date on which an invite is to be sent
        mobility_planner (MobilityPlanner): schedule planning object

    Returns:
        (bool): True if mobility_planner can send an invite to other `human`s
    """
    # can accept is treated as can send as well
    send = _can_accept_invite(today, mobility_planner)
    return send

def _can_supervise_kid(adults):
    """
    Filters out adults that are a best fit to supervise kids

    Args:
        adults (list): list of `Human`s who live at the same household and are eligile to supervise kids

    Returns:
        (list): list of filtered adults who can take a kid for supervision
    """
    valid_adults = []
    for adult in adults:

        assert not adult.mobility_planner.follows_adult_schedule, "invlaid adult to consider for supervision"

        if (
            not adult.mobility_planner.human_to_rest_at_home
            and adult.mobility_planner.hospitalization_timestamp is None
            and adult.mobility_planner.death_timestamp is None
            and adult.mobility_planner.critical_condition_timestamp is None
        ):
            valid_adults.append(adult)

    # if there is no option, then the kid has to stay with someone
    if len(valid_adults) == 0:
        return [adult for adult in adults if adult.mobility_planner.death_timestamp is None]

    return valid_adults

def _human_dies(mobility_planner, human, activity, env):
    """
    Implements cleaning up of `human` object and setting up relevant flags before `human` timesout for infinity

    Args:
        mobility_planner (MobilityPlanner): schedule planning object
        human (covid19sim.human.Human): `human` object that needs to timeout for infinity (dies)
        activity (Activity): activity that human is about to do
        env (simpy.Environment): event scheduling object

    Returns:
        mobility_planner (MobilityPlanner): schedule planning object with relevant flags updated
        human (covid19sim.human.Human): `human` object that needs to timeout for infinity (dies)
        activity (Activity): activity with a flag to indicate `human` should be timedout for infinity
    """
    mobility_planner.death_timestamp = env.timestamp
    activity.human_dies = True
    if mobility_planner.location_of_hospitalization is not None:
        mobility_planner.location_of_hospitalization.discharge(human)
    mobility_planner.location_of_hospitalization = None
    return mobility_planner, human, activity

def _reallocate_residence(human, households, rng, conf):
    """
    Finds a random house for a kid.

    Args:
        human (covid19sim.human.Human): `human` for whom a house with adult needs to be searched for
        households (list): list of `Household` that are available for search
        rng (np.random.RandomState): Random number generator
        conf (dict): yaml configuration of the experiment

    Returns:
        household (covid19sim.locations.Household or None): None if the search is unsuccessful else a relevant household is returned
    """
    MAX_AGE_CHILDREN_WITHOUT_SUPERVISION = conf['MAX_AGE_CHILDREN_WITHOUT_PARENT_SUPERVISION']

    idxs = rng.permutation(len(households)).tolist()
    for idx in idxs:
        household = households[idx]
        if any(h.age > MAX_AGE_CHILDREN_WITHOUT_SUPERVISION for h in household.residents):
            return household

    return None

def reduced_workload(human, old_work_duration, estimated_prevalence):
    # modify work duration as a function of prevalence, covid sensitivity
    # returns a new work duration modified based on prevalence

    if human.work_covid_sensitivity == 'sensitive':
        new_work_duration = old_work_duration * (1/(1+(estimated_prevalence*25)))
        return new_work_duration

    elif human.work_covid_sensitivity == 'neutral':
        new_work_duration = old_work_duration * (1/(1+(estimated_prevalence*5)))
        return new_work_duration

    else:
        raise ValueError('Agent is neither covid sensitive nor neutral. Needs to be one or the other')
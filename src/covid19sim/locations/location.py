"""
This module contains several classes of location which are used in the simulator. These locations can contain people, have a spatial position, area,
and many other attributes.
"""

import simpy
import typing
import datetime
from orderedset import OrderedSet
from collections import namedtuple
import numpy as np
import warnings

from covid19sim.utils.utils import _sample_positive_normal
from covid19sim.epidemiology.p_infection import get_environment_human_p_transmission
from covid19sim.epidemiology.viral_load import compute_covid_properties

from covid19sim.interventions.tracing_utils import get_household_quarantine_duration
from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, AGE_BIN_WIDTH_5, ALL_LOCATIONS, WEEKDAYS, ALL_DAYS, SECONDS_PER_DAY
from covid19sim.utils.constants import QUARANTINE_UNTIL_TEST_RESULT, QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT
from covid19sim.utils.constants import QUARANTINE_HOUSEHOLD

DistanceProfile = namedtuple("DistanceProfile", ['encounter_term', 'social_distancing_term', 'packing_term', 'distance'])

def _extract_attrs(human, candidate, location):
    return (
        candidate,
        candidate.age_bin_width_5.index,
        candidate in human.known_connections,
        human.intervened_behavior.daily_interaction_reduction_factor(location)
    )

def _adjust_surveyed_contacts_to_regional_contacts(MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP, conf, MEAN_DAILY_KNOWN_CONTACTS=None):
    """
    Computes mean daily interactions for the population given mean daily interactions broken down by age groups.

    Args:
        MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP (np.ndarray): 1-D array containing mean daily interactions for each age group
        MEAN_DAILY_KNOWN_CONTACTS (float): daily known contacts for the region
        conf (dict): yaml configuration of the experiment

    Returns:
        (float): population wide mean daily interactions. No adjustment if MEAN_DAILY_KNOWN_CONTACTS is None
    """
    if MEAN_DAILY_KNOWN_CONTACTS is None:
        return MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP

    age_proportion = np.array([x[2] for x in conf['P_AGE_REGION']])
    surveyed_mean = np.sum(age_proportion * MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP)
    return MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP * MEAN_DAILY_KNOWN_CONTACTS / surveyed_mean


class Location(simpy.Resource):
    """
    Class representing generic locations used in the simulator
    """
    id_counter = 0
    district = None

    def __init__(self, env, rng, conf, area, name, location_type, lat, lon, capacity):
        """
        Locations are created with city.create_location(), not instantiated directly

        Args:
            env (covid19sim.Env): Shared environment
            rng (np.random.RandomState): Random number generator
            conf (dict): yaml configuration of the experiment
            area (float): Area of the location
            name (str): The location's name
            type (str): Location's type, see
            lat (float): Location's latitude
            lon (float): Location's longitude
            social_contact_factor (float): how much people are close to each other
                see contamination_probability() (this scales the contamination pbty)
            capacity (int): Daily intake capacity for the location (infinity if None).
            surface_prob (float): distribution of surface types in the Location. As
                different surface types have different contamination probabilities
                and virus "survival" durations, this will influence the contamination
                of humans at this location.
                Surfaces: aerosol, copper, cardboard, steel, plastic
        """

        assert location_type in ALL_LOCATIONS, "not a valid location"
        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet()  # OrderedSet instead of set for determinism when iterating
        self.conf = conf
        self.name = name
        self.id: int = Location.id_counter
        Location.id_counter += 1
        self.rng = np.random.RandomState(rng.randint(2 ** 16))
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.max_day_contamination = 0
        self.is_open_for_business = True
        self.binned_humans = {bin:OrderedSet() for bin in AGE_BIN_WIDTH_5}
        self.social_contact_factor = conf[f'{location_type}_CONTACT_FACTOR']
        self.contaminated_surface_probability = conf[f'{location_type}_SURFACE_PROB']

        # occupation related constants
        OPEN_CLOSE_TIMES = conf[f'{location_type}_OPEN_CLOSE_HOUR_MINUTE']
        OPEN_DAYS = conf[f'{location_type}_OPEN_DAYS']
        # /!\ opening and closing time are in seconds relative to midnight
        self.opening_time = OPEN_CLOSE_TIMES[0][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[0][1]
        self.closing_time = OPEN_CLOSE_TIMES[1][0] * SECONDS_PER_HOUR +  OPEN_CLOSE_TIMES[1][1]
        self.open_days = OPEN_DAYS

        # parameters related to sampling contacts
        if location_type in ["SENIOR_RESIDENCE", "HOUSEHOLD"]:
            key = "HOUSEHOLD"
        elif location_type == "SCHOOL":
            key = "SCHOOL"
        elif location_type in ["WORKPLACE", "HOSPITAL"]:
            key = "WORKPLACE"
        else:
            key = "OTHER"

        # contact related constants
        self.MEAN_DAILY_KNOWN_CONTACTS = conf.get(f'{key}_MEAN_DAILY_INTERACTIONS', None)
        self.P_CONTACT = np.array(conf[f'P_CONTACT_MATRIX_{key}'])
        self.ADJUSTED_CONTACT_MATRIX = np.array(conf[f'ADJUSTED_CONTACT_MATRIX_{key}'])
        self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP = self.ADJUSTED_CONTACT_MATRIX.sum(axis=0)
        self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP = _adjust_surveyed_contacts_to_regional_contacts(self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP, conf, self.MEAN_DAILY_KNOWN_CONTACTS)

        # duration matrices
        # self.MEAN_DAILY_CONTACT_DURATION_SECONDS = np.array(conf[f'{key}_CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX'])
        # self.STDDEV_DAILY_CONTACT_DURATION_SECONDS = np.array(conf[f'{key}_CONTACT_DURATION_NORMAL_SIGMA_SECONDS_MATRIX'])
        self.MEAN_DAILY_CONTACT_DURATION_SECONDS = np.array(conf[f'CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX'])
        self.STDDEV_DAILY_CONTACT_DURATION_SECONDS = np.array(conf[f'CONTACT_DURATION_NORMAL_MEAN_SECONDS_MATRIX'])

        for matrix in [self.P_CONTACT, self.ADJUSTED_CONTACT_MATRIX]:
            assert matrix.shape[0] == matrix.shape[1], "contact matrix is not square"

    def infectious_human(self):
        """
        Returns:
            bool: Is there an infectious human currently at that location
        """
        for h in self.humans:
            if h.is_infectious:
                return True
        return False

    def __repr__(self):
        """
        Returns:
            str: Representation of the Location
        """
        return f"{self.name} - occ:{len(self.humans)}/{self.capacity} - I:{self.is_contaminated}"

    def add_human(self, human):
        """
        Adds a human instance to the OrderedSet of humans at the location.

        Args:
            human (covid19sim.human.Human): The human to add.
        """
        self.humans.add(human)
        self.binned_humans[human.age_bin_width_5.bin].add(human)

    def remove_human(self, human):
        """
        Remove a given human from location.human
        If they are infectious, then location.contamination_timestamp is set to the
        env's timestamp and the duration of this contamination is set
        (location.max_day_contamination) according to the distribution of surfaces
        (location.contaminated_surface_probability) and the survival of the virus
        per surface type (MAX_DAYS_CONTAMINATION)
        /!\ Human is not returned

        Args:
            human (covid19sim.human.Human): The human to remove
        """
        if human in self.humans:
            if human.is_infectious:
                self.contamination_timestamp = self.env.timestamp
                rnd_surface = float(self.rng.choice(
                    a=human.conf.get("MAX_DAYS_CONTAMINATION"),
                    size=1,
                    p=self.contaminated_surface_probability
                ))
                self.max_day_contamination = max(self.max_day_contamination, rnd_surface)
            self.humans.remove(human)
            self.binned_humans[human.age_bin_width_5.bin].remove(human)

    @property
    def is_contaminated(self):
        """
        Is the location currently contaminated? It is if one of these two
        conditions is true :
        - an infectious human is present at the location (location is
          considered constantly reinfected by the human)
        - there are no infectious humans but the location was contaminated
          recently (see remove_human()). This depends on the time the last
          infectious human left, the current time and the duration of the
          contamination (location.max_day_contamination).

        Returns:
            bool: Is the place currently contaminating?
        """
        if self.infectious_human():
            return True
        else:
            return (self.env.timestamp - self.contamination_timestamp <=
                    datetime.timedelta(days=self.max_day_contamination))

    @property
    def contamination_probability(self):
        """
        Contamination depends on the time the virus has been sitting on a given surface
        (location.max_day_contamination) and is linearly decayed over time.
        Then it is scaled by location.social_contact_factor

        If not location.is_contaminated, return 0.0

        Returns:
            float: probability that a human is contaminated when going to this location.
        """
        if self.is_contaminated:
            if self.infectious_human():
                # Location constantly reinfected by the infectious human
                p_infection = 1.0
            else:
                # Linear decay of p_infection depending on time since last infection
                lag = (self.env.timestamp - self.contamination_timestamp)
                lag /= datetime.timedelta(days=1)
                p_infection = 1 - lag / self.max_day_contamination
        else:
            p_infection = 0.0

        return p_infection

    def _sample_interactee(self, type, human, n=1):
        """
        Samples encounter partner of `type` for `human`

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human
            n (int): number of `other_human`s to sample

        Returns:
            other_human (covid19sim.human.Human): `human` with whom this `human` will interact
        """
        if len(self.humans) == 1:
            return [None]

        PREFERENTIAL_ATTACHMENT_FACTOR = self.conf['_CURRENT_PREFERENTIAL_ATTACHMENT_FACTOR']

        if type == "known":
            if len(self.humans) - 1 == n:
                return [h for h in self.humans if h != human and h in human.known_connections]

            human_bin = human.age_bin_width_5.index
            candidate_humans = [h for h in self.humans if human != h]
            other_humans, h_vector, known_vector, reduction_factor = list(zip(*[_extract_attrs(human, h, self) for h in candidate_humans]))

            p_contact = self.P_CONTACT[h_vector, human_bin] * (1 - PREFERENTIAL_ATTACHMENT_FACTOR)
            p_contact += np.array(known_vector) * self.P_CONTACT[h_vector, human_bin] * PREFERENTIAL_ATTACHMENT_FACTOR
            # reduction factor is due to mutual interaction sampling where other_human's reduction factor is taken into account
            p_contact *= (1 - np.array(reduction_factor))
            if p_contact.sum() == 0:
                return [None]

            p_contact /= p_contact.sum()

        elif type == "unknown":
            if len(self.humans) - 1 == n:
                return [h for h in self.humans if h != human]

            other_humans = [x for x in self.humans if x != human]
            p_contact = np.ones_like(other_humans, dtype=np.float) / len(other_humans)

        else:
            raise

        return self.rng.choice(other_humans, size=n, p=p_contact, replace=True).tolist()

    def _sample_interaction_with_type(self, type, human):
        """
        Samples interactions of type `type` for `human`.

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human

        Returns:
            interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to have `type` of interaction
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which these two humans met (cms)
                duration (float): duration for which this encounter took place (seconds)
        """

        if type == "known":
            mean_daily_interactions = self.MEAN_DAILY_KNOWN_CONTACTS_FOR_AGEGROUP[human.age_bin_width_5.index]
            mean_daily_interactions *= (1 - human.intervened_behavior.daily_interaction_reduction_factor(self))
            min_dist_encounter = self.conf['MIN_DIST_KNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_KNOWN_CONTACT']
            mean_interaction_time = None
        elif type == "unknown":
            mean_daily_interactions = self.conf['_MEAN_DAILY_UNKNOWN_CONTACTS']
            min_dist_encounter = self.conf['MIN_DIST_UNKNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_UNKNOWN_CONTACT']
            mean_interaction_time = self.conf["GAMMA_UNKNOWN_CONTACT_DURATION"]
        else:
            raise

        mean_daily_interactions += 1e-6 # to avoid error in sampling with 0 mean from negative binomial
        scale_factor_interaction_time = self.conf['SCALE_FACTOR_CONTACT_DURATION']
        # (assumption) maximum allowable distance is when humans are uniformly spaced
        packing_term = 100 * np.sqrt(self.area/len(self.humans))

        interactions = []
        n_interactions = min(len(self.humans) - 1, self.rng.negative_binomial(mean_daily_interactions, 0.5))
        interactees = self._sample_interactee(type, human, n=n_interactions)
        for other_human in interactees:
            if other_human is None:
                continue

            assert other_human != human, "sampling with self is not allowed"
            # sample duration of encounter (seconds)
            t_overlap = (min(human.location_leaving_time, other_human.location_leaving_time) -
                         max(human.location_start_time,   other_human.location_start_time))

            # if the overlap duration is less than a relevant duration for infection, it is of no use.
            if t_overlap < min(self.conf['MIN_MESSAGE_PASSING_DURATION'], self.conf['INFECTION_DURATION']):
                continue

            # sample distance of encounter
            encounter_term = self.rng.uniform(min_dist_encounter, max_dist_encounter)
            distance = np.clip(encounter_term, a_min=0, a_max=packing_term)
            distance_profile = DistanceProfile(encounter_term=encounter_term, packing_term=packing_term, social_distancing_term=None, distance=distance)

            if type == "known":
                age_bin = human.age_bin_width_5.index
                other_bin = other_human.age_bin_width_5.index
                mean_duration = self.MEAN_DAILY_CONTACT_DURATION_SECONDS[other_bin, age_bin]
                sigma_duration = self.STDDEV_DAILY_CONTACT_DURATION_SECONDS[other_bin, age_bin]
                # surveyed data gives us minutes per day. Here we use it to sample rate of minutes spend per second of overlap in an encounter.
                # duration = (_sample_positive_normal(mean_duration, sigma_duration, self.rng) / SECONDS_PER_DAY) * t_overlap * SECONDS_PER_MINUTE
                duration = _sample_positive_normal(mean_duration, sigma_duration, self.rng, upper_limit=t_overlap)

            elif type == "unknown":
                duration = self.rng.gamma(mean_interaction_time/scale_factor_interaction_time, scale_factor_interaction_time)

            else:
                raise ValueError(f"Unknown interaction type: {type}")

            # (debug)
            # if self.location_type ==  "HOUSEHOLD" and type == "known" and human.mobility_planner.current_activity.name != "socialize":# and human.workplace != self:
            #     print(human, "-->", other_human, "for", duration / SECONDS_PER_MINUTE, "tota humans", len(self.humans), "t_overlap", t_overlap / SECONDS_PER_MINUTE, human.mobility_planner.current_activity)

            # add to the list
            interactions.append((other_human, distance_profile, duration))

        return interactions

    def sample_interactions(self, human, unknown_only=False):
        """
        samples how `human` interacts with other `human`s at this location (`self`) at this time.

        Args:
            human (covid19sim.human.Human): human for whom interactions need to be sampled
            unknown_only (bool): whether to sample interactions of type `unknown` only

        Returns:
            known_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to interact
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which this encounter took place (cms)
                duration (float): duration for which this encounter took place (minutes)
            unknown_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human who was nearby and unknown to `human`
                distance_profile (covid19sim.locations.location.DistanceProfile): distance from which this encounter took place (cms)
                duration (float): duration for which this encounter took place (minutes)
        """
        # only `human` is at this location. There will be no interactions.
        if len(self.humans) == 1:
            assert human == self.humans[0]
            return [], []

        known_interactions = []
        unknown_interactions = self._sample_interaction_with_type("unknown", human)
        if not unknown_only:
            known_interactions = self._sample_interaction_with_type("known", human)

        return known_interactions, unknown_interactions

    def check_environmental_infection(self, human):
        """
        Determines whether `human` gets infected due to the virus in environment.

        Environmental infection is modeled via surfaces. We consider the following study -
        https://www.nejm.org/doi/pdf/10.1056/NEJMc2004973?articleTools=true
        It shows the duration for which virus remains on a surface. Following surfaces are considered -
        aerosol    copper      cardboard       steel       plastic

        We sample a surface using surface_prob and infect the surface for MAX_DAYS_CONTAMINATION[surface_index] days.
        NOTE: self.surface_prob is experimental, and there is no data on which surfaces are dominant at a location.
        NOTE: our final objective is to make sure that environmental infection is around 10-15% of all transmissions to comply with the data.
        We do that via ENVIRONMENTAL_INFECTION_KNOB.

        Args:
            human (covid19sim.human.Human): `human` for whom to check environmental infection.

        Returns:
            (bool): whether `human` was infected via environmental contamination.
        """
        p_contamination = self.contamination_probability
        if not (human.is_susceptible and p_contamination > 0):
            return

        #
        p_transmission = get_environment_human_p_transmission(
                                        p_contamination,
                                        human,
                                        self.conf.get("_ENVIRONMENTAL_INFECTION_KNOB"),
                                        )

        x_environment =  self.rng.random() < p_transmission

        # track infection related stats
        self.district.tracker.track_infection(source="environment",
                                    from_human=None,
                                    to_human=human,
                                    location=self,
                                    timestamp=self.env.timestamp,
                                    p_infection=p_transmission,
                                    success=x_environment
                                )
        if x_environment:
            human._get_infected(initial_viral_load=self.rng.random())

        return

    def is_open(self, date):
        """
        Checks if `self` is open on `date`.

        Args:
            date (datetime.date): date for which this is to be checked.

        Returns:
            (bool): True if its open
        """
        return date.weekday() in self.open_days

    def __hash__(self):
        """
        Hash of the location is the hash of its name

        Returns:
            int: hash
        """
        return hash(self.name)

    def serialize(self):
        """
        This function serializes the location object by deleting
        non-serializable keys

        Returns:
            dict: serialized location
        """
        s = self.__dict__
        if s.get('env'):
            del s['env']
        if s.get('rng'):
            del s['rng']
        if s.get('_env'):
            del s['_env']
        if s.get('contamination_timestamp'):
            del s['contamination_timestamp']
        if s.get('residents'):
            del s['residents']
        if s.get('humans'):
            del s['humans']
        return s

    def __reduce__(self):
        """
        Helper function for pickling
        """
        args = (self.location_type, self.id)
        return (self._reconstruct, args)

    @classmethod
    def _reconstruct(
        cls,
        location_type: str,
        location_id: str
        ):
        """
        Find and return location object from city by name
        """
        try:
            # locations present in current district
            return next(
                loc for loc in getattr(cls.district, f"{location_type.lower()}s") \
                if loc.id == location_id
                )
        except StopIteration:
            # initialize an unintialized location
            loc_instance = cls.__new__(cls)
            loc_instance.location_type = location_type
            loc_instance.id = location_id
            return loc_instance
 

class Household(Location):
    """
    Household location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(Household, self).__init__(**kwargs)
        self.residents = []
        self.allocation_type = None

        location_type = kwargs['location_type']
        # for seniors, social common room serves as a workplace where they spend time hanging out with others
        if location_type == "SENIOR_RESIDENCE":
            self.n_nurses = 0
            name = kwargs.get("name")
            self.social_common_room = Location(
                env=kwargs.get("env"),
                rng=kwargs.get("rng"),
                name=f"SCR: {name}",
                conf=kwargs.get('conf'),
                location_type=location_type,
                lat=kwargs.get("lat"),
                lon=kwargs.get("lon"),
                area=kwargs.get("area"),
                capacity=None
            )

        # quarantine related attrs for residents who need to quarantine due to index cases
        self.index_cases = {}
        self.quarantine_start_timestamp = None
        self.quarantine_end_timestamp = None
        self.max_behavior_level = -1

    def remove_resident(self, human):
        """
        Removes `human` from the list of residents and clears other attrs. Happens on death or reallocation of residence.

        Args:
            human (covid19sim.human.Human): `human` who needs to be remove to this house

        Returns:
            index_case_history (dict): current index case value for `human` in this house.
        """
        # if `human` is being removed without any past triggers, there will be no entry in index_cases
        index_case_history = None
        if human in self.index_cases:
            index_case_history = self.index_cases.pop(human)
        self.residents.remove(human.human_id) # remove from the house
        return index_case_history

    def add_resident(self, human, index_case_history=None):
        """
        Adds a new resident to the house. Happens when a kid has to be reallocated a residence because no adult at their previous house is alive.

        Args:
            human (covid19sim.human.Human): `human` who needs to be added to this house
            index_case_history (dict): current index case value for `human` in the previous house. Defaults to None.
        """
        self.residents.append(human.human_id)
        if index_case_history:
            self.index_cases[human] = index_case_history

    def reset_index_case(self, human):
        """
        Resets the keys for `index_cases` corresponding to `human`.

        Args:
            human (covid19sim.human.Human): `human` who needs to be added to the index cases of the household
        """
        assert human in self.residents, "non-resident being added to index_cases"
        self.index_cases[human] = {
            "reasons": [],
            "suggested_quarantine_end_timestamp": None
        }

    def add_to_index_case(self, human, trigger):
        """
        Index cases for household are defined as those residents wwho have reason to quarantine.
        All others are secondary cases who needs to quarantine because of index cases.

        Adds `human` to index cases.

        Args:
            human (covid19sim.human.Human): `human` who needs to be added to the index cases of the household
            trigger (str): reason for adding to the index case.
        """
        assert human.human_id in self.residents, f"{human} does not reside at {self}. Trigger: {trigger}, residents: {self.residents}"

        # (happens only once at the firs trigger of `human`)
        if human not in self.index_cases:
            self.index_cases[human] = {
                "reasons": [],
                "suggested_quarantine_end_timestamp": None
            }

        # test results are conclusive, so we don't add any other triggers
        # Note 1: QUARANTINE_X_TEST_RESULT will always be the last element in the list if it existis
        # Note 2: Only one of QUARANTINE_X_TEST_RESULT will be in the list
        if (
            QUARANTINE_UNTIL_TEST_RESULT in self.index_cases[human]
            or QUARANTINE_DUE_TO_POSITIVE_TEST_RESULT in self.index_cases[human]
        ):
            return

        # update quarantine requirements for secondary cases
        if (
            self.quarantine_start_timestamp is None
            or self.quarantine_end_timestamp < self.env.timestamp
        ):
            self.quarantine_start_timestamp = self.env.timestamp

        self.index_cases[human]['reasons'].append(trigger)
        if trigger == QUARANTINE_UNTIL_TEST_RESULT:
            self.index_cases[human]['suggested_quarantine_end_timestamp'] = self.env.timestamp + datetime.timedelta(seconds=human.time_to_test_result * SECONDS_PER_DAY)

        else:
            duration = get_household_quarantine_duration(self.index_cases[human]['reasons'], self.conf)
            self.index_cases[human]['suggested_quarantine_end_timestamp'] = self.quarantine_start_timestamp + datetime.timedelta(seconds=duration*SECONDS_PER_DAY)

        # update quarantine timestamps
        quarantine_end_timestamp = self.env.timestamp
        secondary_cases = []
        for other_human, attrs in self.index_cases.items():
            triggers = attrs['reasons']
            suggested_quarantine_end_timestamp = attrs['suggested_quarantine_end_timestamp']

            # find maximum duration for which to quarantine secondary cases
            if suggested_quarantine_end_timestamp is not None:
                if suggested_quarantine_end_timestamp > quarantine_end_timestamp:
                    quarantine_end_timestamp = suggested_quarantine_end_timestamp

            # secondary cases
            if len(triggers) == 0:
                secondary_cases.append(other_human)

        self.quarantine_end_timestamp = quarantine_end_timestamp
        for other_human in secondary_cases:
            other_human.intervened_behavior.quarantine.update(QUARANTINE_HOUSEHOLD)

    def __reduce__(self):
        """
        Helper function for pickling households
        """
        args = (self.location_type, self.id, self.residents)
        return (self._reconstruct, args)

    @classmethod
    def _reconstruct(
        cls,
        location_type: str,
        location_id: str,
        household_residents: typing.List[int]
        ):
        """
        Find and return household object from city by name
        """
        try:
            # look through households present in current district
            return next(
                loc for loc in getattr(cls.district, f"{location_type.lower()}s") \
                if loc.id == location_id
                )
        except StopIteration:
            # initialize an unintialized household
            household_instance = cls.__new__(cls)
            household_instance.location_type = location_type
            household_instance.id = location_id
            household_instance.residents = household_residents
            return household_instance

class School(Location):
    """
    School location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(School, self).__init__(**kwargs)
        self.n_students = 0
        self.n_teachers = 0

    def __repr__(self):
        return self.name + f"| Students:{self.n_students} Teachers:{self.n_teachers}"


class WorkplaceA(Location):
    """
    Stores location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(WorkplaceA, self).__init__(**kwargs)
        self.workers = set()
        self.n_workers = 0

    def assign_worker(self, human):
        """
        Adds `human` to the set of workers.

        Args:
            human (covi19sim.human.Human): `human` to add to the set of workers
        """
        self.workers.add(human.human_id)
        self.n_workers += 1

    def __repr__(self):
        return self.name + f"| {self.n_workers} workers"


class WorkplaceB(Location):
    """
    Stores location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super(WorkplaceB, self).__init__(**kwargs)
        self.workers = set()
        self.n_workers = 0

    def assign_worker(self, human):
        """
        Adds `human` to the set of workers.

        Args:
            human (covi19sim.human.Human): `human` to add to the set of workers
        """
        self.workers.add(human.human_id)
        self.n_workers += 1

    def __repr__(self):
        return self.name + f"| {self.n_workers} workers"

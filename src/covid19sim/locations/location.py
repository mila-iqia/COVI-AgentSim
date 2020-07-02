import simpy
import datetime
from orderedset import OrderedSet
import numpy as np
from covid19sim.utils.constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR
from covid19sim.epidemiology.p_infection import get_environment_human_p_transmission
from covid19sim.epidemiology.viral_load import compute_covid_properties

class Location(simpy.Resource):
    """
    Class representing generic locations used in the simulator
    """

    def __init__(self, env, rng, conf, type, area, name, location_type, lat, lon,
                 social_contact_factor, capacity, surface_prob):
        """
        Locations are created with city.create_location(), not instantiated directly

        Args:
            env (covid19sim.Env): Shared environment
            rng (np.random.RandomState): Random number generator
            area (float): Area of the location
            name (str): The location's name
            location_type (str): Location's type, see
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

        if capacity is None:
            capacity = simpy.core.Infinity

        super().__init__(env, capacity)
        self.humans = OrderedSet()  # OrderedSet instead of set for determinism when iterating
        self.conf = conf
        self.type = type
        self.name = name
        self.rng = np.random.RandomState(rng.randint(2 ** 16))
        self.lat = lat
        self.lon = lon
        self.area = area
        self.location_type = location_type
        self.social_contact_factor = social_contact_factor
        self.env = env
        self.contamination_timestamp = datetime.datetime.min
        self.contaminated_surface_probability = surface_prob
        self.max_day_contamination = 0
        self.is_open_for_business = True

        self.MEAN_DAILY_KNOWN_INTERACTIONS = conf['LOCATION_DISTRIBUTION']['type']['mean_daily_interactions']
        self.MEAN_DAILY_UNKNOWN_INTERACTIONS = conf['MEAN_DAILY_UNKNOWN_CONTACTS']
        self.binned_humans = defaultdict(lambda :OrderedSet())

    def infectious_human(self):
        """
        Returns:
            bool: Is there an infectious human currently at that location
        """
        return any([h.is_infectious for h in self.humans])

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
        self.binned_humans[human.age_bin_width_5].add(human)

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
            self.binned_humans[human.age_bin_width_5].remove(human)

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

    def _sample_interactee(self, type, human):
        """
        returns a function that samples encounter partner for a human

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human

        Returns:
            other_human (covid19sim.human.Human): `human` with whom this `human` will interact
        """

        if type == "known":
            human_bin = human.age_bin_width_5
            # sample human whom to interact with
            other_bin = self.rng.choice(range(self.TOTAL_BINS), self.P_CONTACT[human_bin])
            # what if there is no human in this bin?????
            other_human = self.rng.choice(self.binned_humans[other_bin])
        elif type == "unknown":
            other_human = self.rng.choice(self.humans)
        else:
            raise

        return other_human


    def _sample_interaction_with_type(self, type, human):
        """
        Samples interactions of type `type` for `human`.

        Args:
            type (string): type of interaction to sample. expects "known", "unknown"
            human (covid19sim.human.Human): human who will interact with the sampled human

        Returns:
            interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to have `type` of interaction
                distance (float): distance from which this encounter took place (cm)
                duration (float): duration for which this encounter took place (minutes)
        """

        if type == "known":
            mean_daily_interactions = self.MEAN_DAILY_KNOWN_CONTACTS
            min_dist_encounter = self.conf['MIN_DIST_KNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_KNOWN_CONTACT']
            mean_interaction_time = None
        elif type == "unknown":
            mean_daily_interactions = self.MEAN_DAILY_UNKNOWN_CONTACTS
            min_dist_encounter = self.conf['MIN_DIST_UNKNOWN_CONTACT']
            max_dist_encounter = self.conf['MAX_DIST_UNKNOWN_CONTACT']
            mean_interaction_time = self.conf["GAMMA_UNKNOWN_CONTACT_DURATION"]
        else:
            raise

        scale_factor_interaction_time = self.conf['SCALE_FACTOR_CONTACT_DURATION']
        # (assumption) maximum allowable distance is when humans are uniformly spaced
        packing_term = 100 * np.sqrt(self.area/len(self.location.humans))
        interactee_sampler = _sample_interactee(type)

        interactions = []
        # what if there is no human as of now (solo house)
        n_interactions = self.rng.negative_binomial(mean_daily_interactions, 0.5)
        for i in range(n_interactions):
            # sample other human
            other_human = self._sample_interactee(type, human)

            # sample distance of encounter
            encounter_term = self.rng.uniform(min_dist_encounter, max_dist_encounter)
            social_distancing_term = np.mean([self.maintain_extra_distance, h.maintain_extra_distance]) #* self.rng.rand()
            distance = np.clip(encounter_term + social_distancing_term, a_min=0, a_max=packing_term)

            # if distance == packing_term:
            #     city.tracker.track_encounter_distance(
            #         "A\t1", packing_term, encounter_term,
            #         social_distancing_term, distance, location)
            # else:
            #     city.tracker.track_encounter_distance(
            #         "A\t0", packing_term, encounter_term,
            #         social_distancing_term, distance, location)

            # sample duration of encounter
            t_overlap = (min(human.location_leaving_time, other_human.location_leaving_time) -
                         max(human.location_start_time,   other_human.location_start_time)) / SECONDS_PER_MINUTE

            if type =="known":
                mean_interaction_time = self.TIME_DURATION_MATRIX[human.age_bin_width_5, other_human.age_bin_width_5]

            duration = min(t_overlap, self.rng.gamma(mean_interaction_time/scale_factor_interaction_time, scale_factor_interaction_time))

            # add to the list
            interactions.append((other_human, distance, duration))

        return interactions


    def sample_interactions(self, human):
        """
        samples how `human` interacts with other `human`s at this location at this time.

        Args:
            human (covid19sim.human.Human): human for whom interactions need to be sampled

        Returns:
            known_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human with whom to interact
                distance (float): distance from which this encounter took place (cm)
                duration (float): duration for which this encounter took place (minutes)
            unknown_interactions (list): each element is as follows -
                human (covid19sim.human.Human): other human who was nearby and unknown to `human`
                distance (float): distance from which this encounter took place (cm)
                duration (float): duration for which this encounter took place (minutes)
        """
        # only `human` is at this location. There will be no interactions.
        if len(self.humans) == 1:
            assert human == self.humans[0]
            return [], []

        known_interactions = self._sample_interaction_with_type("known", human)
        unknown_interactions = self._sample_interaction_with_type("unknown", human)

        return known_interactions, unknown_interactions
        #
        # scale_factor_interaction_time = self.conf['SCALE_FACTOR_INTERACTION_TIME']
        # packing_term = self._compute_maximum_distance_allowed_between_two_humans()
        #
        # known_interactions = []
        # # what if there is no human as of now (solo house)
        # n_interactions = self.rng.negative_binomial(self.MEAN_DAILY_KNOWN_INTERACTIONS, 0.5)
        # for i in range(n_interactions):
        #     human_bin = human.age_bin
        #
        #     # sample human whom to interact with
        #     other_bin = self.rng.choice(range(self.TOTAL_BINS), self.P_CONTACT[human_bin])
        #     # what if there is no human in this bin?????
        #     other_human = self.rng.choice(self.binned_humans[other_bin])
        #
        #     # sample distance of encounter
        #     encounter_term = self.rng.uniform(self.conf['MIN_DIST_INTERACTION'], self.conf['MAX_DIST_INTERACTION'])
        #     social_distancing_term = np.mean([self.maintain_extra_distance, h.maintain_extra_distance]) #* self.rng.rand()
        #     distance = np.clip(encounter_term + social_distancing_term, a_min=0, a_max=packing_term)
        #
        #
        #     # sample duration of encounter
        #     t_overlap = (min(human.location_leaving_time, other_human.location_leaving_time) -
        #                  max(human.location_start_time,   other_human.location_start_time)) / SECONDS_PER_MINUTE
        #     mean_interaction_time = self.TIME_DURATION_MATRIX[human_bin, other_bin]
        #     duration = min(t_overlap, self.rng.gamma(mean_interaction_time/scale_factor_interaction_time, scale_factor_interaction_time))
        #
        #     # add to the list
        #     known_interactions.append((other_human, distance, duration))
        #
        # # random interactions
        # unknown_interactions = []
        # n_interactions = self.rng.negative_binomial(self.MEAN_DAILY_UNKNOWN_INTERACTIONS, 0.5)
        # for i in range(n_interactions):
        #     other_human = self.rng.choice(self.humans)
        #     distance = self.rng.uniform(self.conf['MIN_DIST_UNKNOWN_INTERACTION'], self.conf['MAX_DIST_UNKNOWN_INTERACTION'])
        #     # time_already_spent_in_interactions = cumulative_time_spent[human]
        #     t_overlap = (min(human.location_leaving_time, other_human.location_leaving_time) -
        #                  max(human.location_start_time,   other_human.location_start_time)) / SECONDS_PER_MINUTE
        #     duration = min(t_overlap, self.rng.gamma(self.conf["GAMMA_UNKOWN_INTERACTION_DURATION"]/scale_factor_interaction_time, scale_factor_interaction_time))
        #     unknown_interactions.append((other_human, distance, duration))
        #
        # return known_interactions, unknown_interactions

    def check_environmental_infection(self, human):
        """
        determines whether `human` gets infected due to the environment.

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

        p_transmission = get_environment_human_p_transmission(
                                        self.contamination_probability,
                                        human,
                                        self.conf.get("ENVIRONMENTAL_INFECTION_KNOB"),
                                        self.conf['MASK_EFFICACY_FACTOR'],
                                        self.conf['HYGIENE_EFFICACY_FACTOR'],
                                        )

        x_environment = self.contamination_probability > 0 and self.rng.random() < p_transmission
        if x_environment and human.is_susceptible:
            human.infection_timestamp = human.env.timestamp
            compute_covid_properties(human)
            human.city.tracker.track_infection('env', from_human=None, to_human=human, location=self, timestamp=self.env.timestamp)
            Event.log_exposed(self.conf.get('COLLECT_LOGS'), human, self, p_transmission, self.env.timestamp)

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

        location_type = kwargs.get("location_type", "household")
        # for seniors, social common room serves as a workplace where they spend
        # time hanging out with others
        if location_type == "senior_residency":
            name = kwargs.get("name")
            self.social_common_room = Location(
                env=kwargs.get("env"),
                rng=kwargs.get("rng"),
                name=f"SCR: {name}",
                location_type="SCR",
                lat=kwargs.get("lat"),
                lon=kwargs.get("lon"),
                area=kwargs.get("area"),
                social_contact_factor=kwargs.get("social_contact_factor"),
                capacity=None,
                surface_prob=kwargs.get("surface_prob")
            )

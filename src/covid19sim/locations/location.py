import simpy
import datetime
from orderedset import OrderedSet
import numpy as np


class Location(simpy.Resource):
    """
    Class representing generic locations used in the simulator
    """

    def __init__(self, env, rng, area, name, location_type, lat, lon,
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

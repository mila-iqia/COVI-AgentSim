import math
from covid19sim.locations.location import Location

class Hospital(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    ICU_AREA = 0.10
    ICU_CAPACITY = 0.10
    def __init__(self, **kwargs):
        """
        Create the Hospital and its ICU

        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        env = kwargs.get('env')
        rng = kwargs.get('rng')
        capacity = kwargs.get('capacity')
        name = kwargs.get("name")
        lat = kwargs.get('lat')
        lon = kwargs.get('lon')
        area = kwargs.get('area')
        surface_prob = kwargs.get('surface_prob')
        social_contact_factor = kwargs.get('social_contact_factor')

        super(Hospital, self).__init__( env=env,
                                        rng=rng,
                                        area=area * (1-self.ICU_AREA),
                                        name=name,
                                        location_type="hospital",
                                        lat=lat,
                                        lon=lon,
                                        social_contact_factor=social_contact_factor,
                                        capacity=math.ceil(capacity* (1- self.ICU_CAPACITY)),
                                        surface_prob=surface_prob,
                                        )
        self.location_contamination = 1
        self.icu = ICU( env=env,
                        rng=rng,
                        area=area * (self.ICU_AREA),
                        name=f"{name}-icu",
                        location_type="hospital-icu",
                        lat=lat,
                        lon=lon,
                        social_contact_factor=social_contact_factor,
                        capacity=math.ceil(capacity* (self.ICU_CAPACITY)),
                        surface_prob=surface_prob,
                        )

    def add_human(self, human):
        """
        Add a human to the Hospital's OrderedSet through the Location's
        default add_human() method + set the human's obs_hospitalized attribute
        is set to True

        Args:
            human (covid19sim.human.Human): human to add
        """
        human.obs_hospitalized = True
        super().add_human(human)

    def remove_human(self, human):
        """
        Remove a human from the Hospital's Ordered set.
        On top of Location.remove_human(), the human's obs_hospitalized attribute is
        set to False

        Args:
            human (covid19sim.human.Human): human to remove
        """
        human.obs_hospitalized = False
        super().remove_human(human)

class ICU(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    def __init__(self, **kwargs):
        """
        Create a Hospital's ICU Location

        Args:
            kwargs (dict): all the args necessary for a Location's init
        """
        super().__init__(**kwargs)

    def add_human(self, human):
        """
        Add a human to the ICU's OrderedSet through the Location's
        default add_human() method + set the human's obs_hospitalized and
        obs_in_icu attributes are set to True

        Args:
            human (covid19sim.human.Human): human to add
        """
        human.obs_hospitalized = True
        human.obs_in_icu = True
        super().add_human(human)

    def remove_human(self, human):
        """
        Remove a human from the ICU's Ordered set.
        On top of Location.remove_human(), the human's obs_hospitalized and
        obs_in_icu attributes are set to False

        Args:
            human (covid19sim.human.Human): human to remove
        """
        human.obs_hospitalized = False
        human.obs_in_icu = False
        super().remove_human(human)

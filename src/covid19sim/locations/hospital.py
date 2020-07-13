import math
from covid19sim.locations.location import Location

class Hospital(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    ICU_AREA = 0.10

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
        n_icu_beds = kwargs.get('icu_capacity')


        super(Hospital, self).__init__( env=env,
                                        rng=rng,
                                        conf=kwargs.get('conf'),
                                        area=area * (1-self.ICU_AREA),
                                        name=name,
                                        location_type="HOSPITAL",
                                        lat=lat,
                                        lon=lon,
                                        capacity=capacity,
                                        )

        self.icu = ICU( env=env,
                        rng=rng,
                        conf=kwargs.get('conf'),
                        area=area * (self.ICU_AREA),
                        name=f"{name}-icu",
                        location_type="HOSPITAL",
                        lat=lat,
                        lon=lon,
                        capacity=n_icu_beds,
                    )
        self.hospital_bed_occupany = kwargs.get("hospital_bed_occupany")
        self.icu_bed_occupancy = kwargs.get("icu_bed_occupancy")

        self.doctors = set()
        self.n_doctors = 0
        self.nurses = set()
        self.n_nurses = 0

    def assign_worker(self, human, doctor):
        """
        Adds `human` to the list of doctors or nurses

        Args:
            human (covi19sim.human.Human): `human` to add to the list of doctors
            doctor (bool): add `human` to doctor's list if True, nurse's list otherwise
        """
        if doctor:
            self.doctors.add(human)
            self.n_doctors += 1
            return
        self.nurses.add(human)
        self.n_nurses += 1

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

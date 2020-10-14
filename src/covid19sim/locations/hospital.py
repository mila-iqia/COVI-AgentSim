"""
Hospitalization model is stil a WIP.
"""
import math
from covid19sim.locations.location import Location

class Hospital(Location):
    """
    Hospital location class, inheriting from covid19sim.base.Location
    """
    ICU_AREA = 0.10

    def __init__(self, env, rng, conf, name, lat, lon, area, hospital_bed_capacity, icu_bed_capacity):
        """
        Create the Hospital and its ICU

        Args:
            kwargs (dict): all the args necessary for a Location's init
        """

        super(Hospital, self).__init__( env=env,
                                        rng=rng,
                                        conf=conf,
                                        area=area * (1-self.ICU_AREA),
                                        name=name,
                                        location_type="HOSPITAL",
                                        lat=lat,
                                        lon=lon,
                                        capacity=hospital_bed_capacity,
                                        )

        self.icu = ICU(
                        env=env,
                        rng=rng,
                        conf=conf,
                        area=area * (self.ICU_AREA),
                        name=f"{name}-ICU",
                        lat=lat,
                        lon=lon,
                        icu_bed_capacity=icu_bed_capacity,
                        hospital=self
                    )

        self.bed_capacity = hospital_bed_capacity
        self.bed_occupany = conf['HOSPITAL_BEDS_OCCUPANCY']
        self.patients = {}
        self.doctors = set()
        self.nurses = set()
        self.nurses_on_duty = set()
        self.doctors_on_duty = set()

        self.n_nurses = 0
        self.n_doctors = 0

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

    @property
    def n_covid_patients(self):
        count = 0
        for patient, until in self.patients.items():
            if self.env.timestamp < until and (patient.state[1] or patient.state[2]) and not patient.is_dead:
                count += 1
        return count

    @property
    def n_patients(self):
        count = 0
        for patient, until in self.patients.items():
            if self.env.timestamp < until:
                count += 1
        return count

    def admit_patient(self, human, until):
        if self.n_patients == self.bed_capacity:
            raise NotImplementedError(f"{self} is at capacity. Can't admit {human}!")

        self.patients[human] = until

    def discharge(self, human):
        self.patients.pop(human)

    def __repr__(self):
        return f"{self.name}. Occupancy: {self.n_patients}/{self.capacity}"

    def add_human(self, human):
        """
        Add a human to the Hospital's OrderedSet through the Location's
        default add_human() method + set the human's obs_hospitalized attribute
        is set to True

        Args:
            human (covid19sim.human.Human): human to add
        """

        if human in self.doctors:
            self.doctors_on_duty.add(human)

        if human in self.nurses:
            self.nurses_on_duty.add(human)

        # also add to self.humans for interaction sampling
        super().add_human(human)

    def remove_human(self, human):
        """
        Remove a human from the Hospital's Ordered set.
        On top of Location.remove_human(), the human's obs_hospitalized attribute is
        set to False

        Args:
            human (covid19sim.human.Human): human to remove
        """
        if human in self.doctors_on_duty:
            self.doctors_on_duty.remove(human)

        if human in self.nurses_on_duty:
            self.nurses_on_duty.remove(human)

        super().remove_human(human)


class ICU(Location):
    """
    A separate class for ICU is needed to sample interactions independently.
    This is still WIP. # we do not add doctors and nurses to ICU as of now
    """
    def __init__(self, env, rng, conf, name, lat, lon, area, hospital, icu_bed_capacity):

        super(ICU, self).__init__(
            env=env,
            rng=rng,
            conf=conf,
            name=name,
            location_type="HOSPITAL",
            lat=lat,
            lon=lon,
            area=area,
            capacity=icu_bed_capacity
        )

        self.bed_occupany = conf['ICU_BEDS_OCCUPANCY']
        self.hospital = hospital
        self.bed_capacity = icu_bed_capacity

        self.patients = {}

    def __repr__(self):
        return f"{self.name}. Occupancy: {self.n_patients}/{self.capacity}"

    @property
    def n_patients(self):
        count = 0
        for patient, until in self.patients.items():
            if self.env.timestamp < until:
                count += 1
        return count

    def admit_patient(self, human, until):
        if self.n_patients == self.bed_capacity:
            raise NotImplementedError(f"{self} is at capacity. Can't admit {human}!")

        self.patients[human] = until

    def discharge(self, human):
        self.patients.pop(human)

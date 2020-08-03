"""
Implements human behavior/government policy changes.

- * Behaviors * change the behavior of a human, specifically they can modify the human's propensity to take actions which
reduce their liklihood of infecting others (wearing a mask, washing hands); they can choose to go out less
(reducing gamma and rho); or they can take more distance / reduce time spent with others while out.

"""
import typing

if typing.TYPE_CHECKING:
    from covid19sim.human import Human


class Behavior(object):
    """
    A base class to modify human behavior based on an intervention.
    """

    def modify(self, human):
        """
        Changes the behavior attributes of `Human`.

        This function should NOT add hidden attributes inside human, and instead save the
        old values (to revert the behavior later) inside this object directly.

        Args:
            human (Human): `Human` object.
        """
        raise NotImplementedError

    def revert(self, human):
        """
        Reverts the behavior attributes of `Human`.

        This function should restore the backed-up attribute value held inside this
        behavior object, and NOT rely on hidden attributes inside the human class.

        Args:
            human (Human): `Human` object.
        """
        raise NotImplementedError


class Unmitigated(Behavior):
    """An 'unmitigated' behavior means no one is paying attention to the virus and it spreads at full speed."""
    def modify(self, human):
        pass

    def revert(self, human):
        pass


class StandApart(Behavior):
    """With this behavior, humans will try to maintain an extra distance with other people."""
    def __init__(self, default_distance=25):
        self.default_distance = default_distance
        self.old_maintain_extra_distance = None

    def modify(self, human):
        assert self.old_maintain_extra_distance is None
        self.old_maintain_extra_distance = human.maintain_extra_distance
        human.maintain_extra_distance = self.default_distance + 100 * (human.carefulness - 0.5)

    def revert(self, human):
        assert self.old_maintain_extra_distance is not None
        human.maintain_extra_distance = self.old_maintain_extra_distance
        self.old_maintain_extra_distance = None

    def __repr__(self):
        return f"StandApart: {self.default_distance}cm"


class WashHands(Behavior):
    """With this behavior, humans will increase their hygiene, and affect their infection likelihood."""

    def __init__(self):
        self.old_hygiene = None

    def modify(self, human):
        assert self.old_hygiene is None
        self.old_hygiene = human.hygiene
        human.hygiene = human.rng.uniform(min(human.carefulness, 1), 1)

    def revert(self, human):
        assert self.old_hygiene is not None
        human.hygiene = self.old_hygiene
        self.old_hygiene = None


class Quarantine(Behavior):
    """
    Implements quarantining for `Human`. Following is included -
        1. work from home (changes `Human.workplace` to `Human.household`)
        2. rest at home (not go out unless)
        3. stay at home unless hospitalized (so there can still be household infections)
        4. go out with a reduce probability of 0.10 to stores/parks/miscs, but every time `Human` goes out
            they do not explore i.e. do not go to more than one location. (reduce RHO and GAMMA)

    """
    _RHO = 0.1 #0.01
    _GAMMA = 1 #0.01

    def __init__(self):
        self.old_RHO = None
        self.old_GAMMA = None
        self.old_workplace = None
        self.old_rest_at_home = None

    def modify(self, human):
        assert self.old_workplace is None
        self.old_RHO = human.rho
        self.old_GAMMA = human.gamma
        self.old_workplace = human.workplace
        self.old_rest_at_home = human.rest_at_home
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.workplace = human.household
        human.rest_at_home = True

    def revert(self, human):
        assert self.old_workplace is not None
        human.rho = self.old_RHO
        human.gamma = self.old_GAMMA
        human.workplace = self.old_workplace
        human.rest_at_home = self.old_rest_at_home
        self.old_RHO = None
        self.old_GAMMA = None
        self.old_workplace = None
        self.old_rest_at_home = None


class SocialDistancing(Behavior):
    """
    Implements social distancing. Following is included -
        1. maintain a distance of 200 cms with other people.
        2. Reduce the time of encounter by 0.5 than what one would do without this intervention.
        3. Reduced mobility (using RHO and GAMMA)

    """
    def __init__(self, default_distance=100, time_encounter_reduction_factor=0.5):
        self.default_distance = default_distance  # cm
        self.time_encounter_reduction_factor = time_encounter_reduction_factor
        self._RHO = 0.2 # 0.1
        self._GAMMA = 0.5 # 0.25
        self.old_RHO = None
        self.old_GAMMA = None
        self.old_maintain_extra_distance = None
        self.old_time_encounter_reduction_factor = None

    def modify(self, human):
        assert self.old_maintain_extra_distance is None
        self.old_RHO = human.rho
        self.old_GAMMA = human.gamma
        self.old_maintain_extra_distance = human.maintain_extra_distance
        self.old_time_encounter_reduction_factor = human.time_encounter_reduction_factor
        human.rho = self._RHO
        human.gamma = self._GAMMA
        human.maintain_extra_distance = self.default_distance + 100 * (human.carefulness - 0.5)
        human.time_encounter_reduction_factor = self.time_encounter_reduction_factor

    def revert(self, human):
        assert self.old_maintain_extra_distance is not None
        human.rho = self.old_RHO
        human.gamma = self.old_GAMMA
        human.maintain_extra_distance = self.old_maintain_extra_distance
        human.time_encounter_reduction_factor = self.old_time_encounter_reduction_factor
        self.old_RHO = None
        self.old_GAMMA = None
        self.old_maintain_extra_distance = None
        self.old_time_encounter_reduction_factor = None

    def __repr__(self):
        return f"SocialDistancing: {self.default_distance}cm + {self.time_encounter_reduction_factor}t"


class WearMask(Behavior):
    """With this behavior, the human will be encounraged to wear a mask when going out. This affects
    the outcome in Human.compute_mask_efficacy()."""

    def __init__(self, available=None):
        assert available is None, \
            "as of 2020/07/06, using a capped mask reserve is disabled because the implementation " \
            "that was used was buggy w/ the per-human modify --- we need to reimplement if we want a cap again"
        self.old_will_wear_mask = None

    def modify(self, human):
        assert self.old_will_wear_mask is None
        self.old_will_wear_mask = human.will_wear_mask
        human.will_wear_mask = True

    def revert(self, human):
        assert self.old_will_wear_mask is not None
        human.will_wear_mask = self.old_will_wear_mask
        self.old_will_wear_mask = None


class GetTested(Behavior):
    """With this behavior, the human will actively seek a COVID-19 test."""

    def __init__(self):
        self.old_test_recommended = None

    def modify(self, human):
        assert self.old_test_recommended is None
        self.old_test_recommended = human._test_recommended
        human._test_recommended = True

    def revert(self, human):
        assert self.old_test_recommended is not None
        human._test_recommended = self.old_test_recommended
        self.old_test_recommended = None

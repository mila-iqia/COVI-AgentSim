from enum import Enum, IntEnum
import dataclasses
import typing
from itertools import product
from collections import defaultdict
import numpy as np


class Severity(Enum):
    """Severity of the disease -- these should be adjectives."""

    UNDEFINED = -1
    MILD = 0
    MODERATE = 1
    HEAVY = 2
    SEVERE = 3
    EXTREMELY_SEVERE = 4


class BaseSymptom(Enum):
    """Base symptoms the human can have."""

    NO_SYMPTOMS = -1
    FEVER = 0
    CHILLS = 1
    GASTRO = 2
    DIARRHEA = 3
    NAUSEA_VOMITTING = 4
    FATIGUE = 5
    UNUSUAL = 6
    HARD_TIME_WAKING_UP = 7
    HEADACHE = 8
    CONFUSED = 9
    LOST_CONSCIOUSNESS = 10
    TROUBLE_BREATHING = 11
    SNEEZING = 12
    COUGH = 13
    RUNNY_NOSE = 14
    SORE_THROAT = 15
    CHEST_PAIN = 16
    LOSS_OF_TASTE = 17
    ACHES = 18


class DiseaseContext(IntEnum):
    """
    Phase (context?) of the disease. Supports comparisons, for instance:
        CovidContext.INCUBATION < CovidContext.ONSET evaluates to True.
        FluContext.FLU_FIRST_DAY < FluContext.FLU_LAST_DAY evaluates to True.
    """

    pass


class CovidContext(DiseaseContext):
    INCUBATION = 0
    ONSET = 1
    PLATEAU = 2
    POST_PLATEAU_1 = 3
    POST_PLATEAU_2 = 4


class AllergyContext(DiseaseContext):
    ALLERGY = 0


class ColdContext(DiseaseContext):
    COLD = 0
    COLD_LAST_DAY = 1


class FluContext(DiseaseContext):
    FLU_FIRST_DAY = 0
    FLU = 1
    FLU_LAST_DAY = 2


class Disease(Enum):
    COVID = CovidContext
    ALLERGY = AllergyContext
    COLD = ColdContext
    FLU = FluContext


@dataclasses.dataclass(unsafe_hash=True)
class Symptom(object):
    """
    This class uniquely defines a symptom. A few gotchas:

    # Comparisons work:
    >>> Symptom(Severity.MILD, BaseSymptom.FEVER) == Symptom(Severity.MILD, BaseSymptom.FEVER)
    # Returns True
    >>> Symptom(Severity.SEVERE, BaseSymptom.FEVER) == Symptom(Severity.MILD, BaseSymptom.FEVER)
    # Returns False (duh)

    But `is` comparison doesn't work:
    >>> Symptom(Severity.MILD, BaseSymptom.FEVER) is Symptom(Severity.MILD, BaseSymptom.FEVER)
    # Returns False
    """

    severity: Severity
    base_symptom: BaseSymptom

    @classmethod
    def all_possible_symptoms(cls):
        """Return a list of all possible symptoms."""
        return [
            Symptom(severity, base_symptom)
            for severity, base_symptom in product(Severity, BaseSymptom)
        ]

    def __str__(self):
        return f"{self.severity.name} {self.base_symptom.name}"


@dataclasses.dataclass(unsafe_hash=True, init=False)
class DiseasePhase(object):
    """
    This class uniquely defines a phase in the disease.

    The following works:
    >>> DiseasePhase(Disease.COVID, CovidContext.ONSET)
    The resulting object specifies the onset of COVID. The following throws an error:
    >>> DiseasePhase(Disease.COVID, ColdContext.COLD)
    AssertionError: Context COLD is not valid for disease COVID.
    """

    disease: Disease
    context: DiseaseContext

    def __init__(self, disease: Disease, context: DiseaseContext):
        super(DiseasePhase, self).__init__()
        self.disease = disease
        self.context = context
        assert (
            self.is_valid
        ), f"Context {context.name} is not valid for disease {disease.name}."

    @property
    def is_valid(self):
        return self.context in self.disease.value

    def __str__(self):
        return f"{self.disease.name} ({self.context.name})"


@dataclasses.dataclass(unsafe_hash=True)
class HealthState(object):
    symptom: Symptom
    disease_phase: DiseasePhase

    def __str__(self):
        return f"{str(self.symptom)} at {str(self.disease_phase)}"


@dataclasses.dataclass
class TransitionRule(object):
    """
    Specifies one step in the progression of the disease together with its probability.
    """

    from_health_state: HealthState
    to_health_state: HealthState

    proba_value: float = None
    proba_fn: typing.Union[typing.Callable, None] = None

    def get_proba(
        self,
        preexisting_conditions=None,
        age=None,
        carefulness=None,
        viral_load_curve=None,
    ) -> float:
        if self.proba_fn is not None:
            return self.proba_fn(
                preexisting_conditions=preexisting_conditions,
                age=age,
                carefuless=carefulness,
                viral_load_curve=viral_load_curve,
            )
        elif self.proba_value is not None:
            return self.proba_value
        else:
            return 0.0

    def __hash__(self):
        return hash((self.from_health_state, self.to_health_state))


class TransitionRuleSet(object):
    def __init__(self):
        self.rule_set = defaultdict(set)

    def add_rule(self, transition_rule: TransitionRule):
        self.rule_set[transition_rule.from_health_state].add(transition_rule)
        return self

    def sample_next_health_state(
        self,
        current_health_state: HealthState,
        rng: np.random.RandomState = np.random,
        preexisting_conditions=None,
        age=None,
        carefulness=None,
        viral_load_curve=None,
    ) -> HealthState:
        if current_health_state not in self.rule_set:
            raise ValueError(
                f"No forward rule defined for health-state {current_health_state}."
            )
        # Given the current state, pull all possible transition rules that tell
        # us what next state to go to, and compute the probability of going to the
        # said next state.
        candidate_next_health_states, next_state_probas = zip(
            *[
                (
                    transition_rule.to_health_state,
                    transition_rule.get_proba(
                        preexisting_conditions=preexisting_conditions,
                        age=age,
                        carefulness=carefulness,
                        viral_load_curve=viral_load_curve,
                    ),
                )
                for transition_rule in self.rule_set[current_health_state]
            ]
        )
        # Make sure the next state probas sum to one?
        next_state_probas = np.array(next_state_probas)
        next_state_probas = next_state_probas / next_state_probas.sum()
        # Sample the next state
        next_health_state = rng.choice(
            candidate_next_health_states, p=next_state_probas
        )
        # ... and done.
        return next_health_state


if __name__ == "__main__":
    default_rules = TransitionRuleSet()
    # Define a few dummy rules
    default_rules.add_rule(
        TransitionRule(
            from_health_state=HealthState(
                symptom=Symptom(Severity.UNDEFINED, BaseSymptom.NO_SYMPTOMS),
                disease_phase=DiseasePhase(Disease.COVID, CovidContext.INCUBATION),
            ),
            to_health_state=HealthState(
                symptom=Symptom(Severity.MILD, BaseSymptom.FEVER),
                disease_phase=DiseasePhase(Disease.COVID, CovidContext.INCUBATION),
            ),
            proba_value=0.2,
            proba_fn=None,  # This can be a lambda function ;-)
        )
    )
    default_rules.add_rule(
        TransitionRule(
            from_health_state=HealthState(
                symptom=Symptom(Severity.UNDEFINED, BaseSymptom.NO_SYMPTOMS),
                disease_phase=DiseasePhase(Disease.COVID, CovidContext.INCUBATION),
            ),
            to_health_state=HealthState(
                symptom=Symptom(Severity.UNDEFINED, BaseSymptom.NO_SYMPTOMS),
                disease_phase=DiseasePhase(Disease.COVID, CovidContext.INCUBATION),
            ),
            proba_value=0.6,
            proba_fn=None,
        )
    )
    # Now sample a next health state
    next_health_state = default_rules.sample_next_health_state(
        current_health_state=HealthState(
            symptom=Symptom(Severity.UNDEFINED, BaseSymptom.NO_SYMPTOMS),
            disease_phase=DiseasePhase(Disease.COVID, CovidContext.INCUBATION),
        )
    )
    print(next_health_state)
    # Prints one of:
    #   UNDEFINED NO_SYMPTOMS at COVID (INCUBATION)
    #   MILD FEVER at COVID (INCUBATION)

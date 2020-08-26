import numpy as np


def oracle(human, conf):
    human_infectiousnesses = np.asarray(human.infectiousnesses)
    if conf.get("ORACLE_MUL_NOISE") > 0 or conf.get("ORACLE_ADD_NOISE") > 0:
        rng = np.random.RandomState(human.oracle_noise_random_seed)
        mul_noise_mask = rng.uniform(-conf.get("ORACLE_MUL_NOISE"),
                                     +conf.get("ORACLE_MUL_NOISE"),
                                     size=human_infectiousnesses.shape)
        mul_noise_mask = 1 + mul_noise_mask
        add_noise_mask = rng.uniform(0., conf.get("ORACLE_ADD_NOISE", 0.),
                                     size=human_infectiousnesses.shape)
    else:
        mul_noise_mask = 1
        add_noise_mask = 0
    # return ground truth infectiousnesses
    risk_history = ((human_infectiousnesses + add_noise_mask) * mul_noise_mask).tolist()
    return risk_history

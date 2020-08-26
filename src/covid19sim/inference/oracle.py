import numpy as np


def oracle(human, conf):
    human_infectiousnesses = np.asarray(human.infectiousnesses)
    if conf.get("ORACLE_NOISE") > 0:
        rng = np.random.RandomState(human.oracle_noise_random_seed)
        noise_mask = rng.uniform(-conf.get("ORACLE_NOISE"), conf.get("ORACLE_NOISE"),
                                 size=human_infectiousnesses.shape)
        noise_mask = 1 + noise_mask
    else:
        noise_mask = 1
    # return ground truth infectiousnesses
    risk_history = (human_infectiousnesses * noise_mask).tolist()
    return risk_history

import numpy as np


def messages_to_np(messages):
    ms_enc = np.zeros((len(messages), 12))
    for idx, message in enumerate(messages):
        uid = np.array(message.uid.tolist())
        risk = np.array(message.risk.tolist())
        try:
            old_risk = np.array(message.old_risk.tolist())
        except Exception:
            old_risk = np.zeros(4)

        m_enc = np.concatenate([uid, risk, old_risk])
        ms_enc[idx] = m_enc
    return ms_enc

def symptoms_to_np(all_symptoms, all_possible_symptoms):
    rolling_window = 14
    symptoms_enc = np.zeros((rolling_window, len(all_possible_symptoms)+1))
    for day, symptoms in enumerate(all_symptoms[:14]):
        for s_idx, symptom in enumerate(symptoms):
            symptoms_enc[day, s_idx] = 1.
    return symptoms_enc

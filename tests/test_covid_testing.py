import datetime
from collections import Counter, deque

import numpy as np

from covid19sim.run import simulate
from covid19sim.utils.utils import extract_tracker_data
from tests.utils import get_test_conf


def print_dict(title, dic, is_sorted=None):
    if not is_sorted:
        items = dic.items()
    else:
        items = sorted(dic.items(), key=lambda x: x[1])
        if is_sorted == "desc":
            items = reversed(items)
    ml = max([len(k) for k in dic.keys()] + [0]) + 2
    aligned = "{:" + str(ml) + "}"
    print(
        "{}:\n   ".format(title),
        "\n    ".join((aligned + ": {}").format(k, v) for k, v in items),
    )


def date_str(date):
    return str(date).split()[0]


def compute_positivity_rate(positive_tests_per_day, negative_tests_per_day, average=3):
    rates = []
    days = []
    q = deque(maxlen=average)
    d = deque(maxlen=average)
    for i, day in enumerate(positive_tests_per_day):
        pt = positive_tests_per_day[day]
        nt = negative_tests_per_day[day]
        if pt + nt:
            q.append(pt / (nt + pt))
        else:
            q.append(0)
        d.append(day)
        rates.append(np.mean(q))
        days.append(" - ".join([_.replace("2020-", "").replace("-", "/") for _ in d]))
    return {d: "{:.3f}%".format(r * 100) for d, r in zip(days, rates)}


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # https://coronavirus.jhu.edu/testing/testing-positivity
    # https://www.canada.ca/content/dam/phac-aspc/documents/services/diseases/2019-novel-coronavirus-infection/surv-covid19-epi-update-eng.pdf
    conf = get_test_conf("test_covid_testing.yaml")
    # test_covid_test = no intervention
    outfile = None

    # ----------------------------
    # -----  Run Simulation  -----
    # ----------------------------
    n_people = 100
    simulation_days = 20
    init_fraction_sick = 0.01
    start_time = datetime.datetime(2020, 2, 28, 0, 0)
    city = simulate(
        n_people=n_people,
        start_time=start_time,
        simulation_days=simulation_days,
        outfile=outfile,
        init_fraction_sick=init_fraction_sick,
        out_chunk_size=500,
        conf=conf,
    )

    # ---------------------------
    # -----  Retreive Data  -----
    # ---------------------------
    data = extract_tracker_data(city.district.tracker, conf)
    tm = data["test_monitor"]
    days = [
        date_str(start_time + datetime.timedelta(days=i))
        for i in range(simulation_days)
    ]

    # -------------------------
    # -----  Daily Stats  -----
    # -------------------------
    cum_R = np.cumsum(data["r"])
    i_for_day = {days[i]: d for i, d in enumerate(data["i"])}

    # ------------------------------
    # -----  Positivity Rates  -----
    # ------------------------------
    positive_tests_per_day = {d: 0 for d in days}
    negative_tests_per_day = {d: 0 for d in days}
    for m in tm:
        if m["result"] == "positive":
            positive_tests_per_day[date_str(m["timestamp"])] += 1
        if m["result"] == "negative":
            negative_tests_per_day[date_str(m["timestamp"])] += 1
    tests_per_human = Counter([m["name"] for m in tm])
    tests_per_day = Counter([date_str(m["timestamp"]) for m in tm])
    positivity_rates = compute_positivity_rate(
        positive_tests_per_day, negative_tests_per_day, average=3
    )
    positives = len([m for m in tm if m["result"] == "positive"])
    negatives = len([m for m in tm if m["result"] == "negative"])
    positive_rate = positives / (negatives + positives)

    # ----------------------------
    # -----  Multiple Tests  -----
    # ----------------------------
    people_with_several_tests = {k: v for k, v in tests_per_human.items() if v > 1}
    multi_test_dates = {}
    for m in tm:
        if m["name"] in people_with_several_tests:
            old = multi_test_dates.get(m["name"], [])
            new = "{} : {}".format(date_str(m["timestamp"]), m["result"])
            multi_test_dates[m["name"]] = old + [new]

    # ----------------------
    # -----  Symptoms  -----
    # ----------------------
    symptoms = Counter([s for m in tm for s in m["symptoms"]])
    symptoms_positive = Counter(
        [s for m in tm for s in m["symptoms"] if m["result"] == "positive"]
    )
    symptoms_negative = Counter(
        [s for m in tm for s in m["symptoms"] if m["result"] == "negative"]
    )

    # --------------------
    # -----  Prints  -----
    # --------------------
    print("\n" + "-" * 50 + "\n" + "-" * 50)
    print("Cumulative removed per day: ", cum_R)
    print("Test events: ", len(tm))
    print("Individuals: ", len(set(m["name"] for m in tm)))
    print_dict("Tests per day", tests_per_day)
    print(
        "Results last day ( N | P | P / (N+P) ): ", negatives, positives, positive_rate
    )
    print("For reference, May 14th in Canada: ", 1104855, 66536, 66536 / 1172796)
    print_dict("Positivity rates", positivity_rates)
    print_dict("All symptoms", symptoms, is_sorted="desc")
    print_dict("Symptoms when positive tests", symptoms_positive, is_sorted="desc")
    print_dict("Symptoms when negative tests", symptoms_negative, is_sorted="desc")

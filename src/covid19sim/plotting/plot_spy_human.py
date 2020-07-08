import pandas as pd
import dill
import os
from collections import defaultdict
import datetime
import matplotlib
import matplotlib.pyplot as plt
import random


def load_humans(data):
    humans = {}
    for i in data["human_monitor"].keys():
        row = defaultdict(list)
        hm = sorted(
            data["human_monitor"][i], key=lambda x: int(x["name"].split(":")[-1])
        )
        for c, j in enumerate(hm):
            for k in j.keys():
                row[k].append(j[k])

        # index = (timestamp, attr)
        for k in row.keys():
            humans[(i, k)] = row[k]

    humans = pd.DataFrame.from_dict(humans, orient="index")
    humans.index = pd.MultiIndex.from_tuples(humans.index)
    humans.columns = [f"human:{x}" for x in range(1, humans.shape[1] + 1)]
    return humans


def _color_cell(val):
    val = int(val.split()[0].split(":")[-1])

    color = "white"
    if val == 0:
        color = "#b3e8a6"  # "green"
    elif val == 1:
        color = "#ffde22"  # "yellow"
    elif val == 2:
        color = "#ff8928"  # "orange"
    elif val == 3:
        color = "#f0aea3"  # "red"
    return f"background-color: {color}"


def _color_cell_raw(val):
    color = "white"
    if val == "positive":
        color = "red"
    return f"background-color: {color}"


# format the content of a cell
def summarize(
    x,
    human_risk_each_day,
    to_human_max_msg_per_day,
    infectee_location,
    infector_infectee_update_messages,
):
    location_html = {
        "household": "&#127968",
        "hospital": "&#127973",
        "senior_residency": "&#128116",
        "school": "&#127979",
        "park": "&#127794",
        "store": "&#128722",
        "workplace": "&#127970",
    }

    risk_levels = x.xs("risk_level", level=1)
    rec_levels = x.xs("rec_level", level=1)
    test_results = x.xs("test_result", level=1)
    reported_test_results = x.xs("reported_test_result", level=1)
    n_symptoms = x.xs("n_symptoms", level=1)
    n_reported_symptoms = x.xs("n_reported_symptoms", level=1)
    infection_timestamp = x.xs("infection_timestamp", level=1)
    risks = x.xs("risk", level=1)

    columns = risk_levels.columns
    y = pd.DataFrame(columns=columns, index=x.index.unique(level=0))

    today = x.index.unique(level=0).item().date()
    max_risk_level = -1
    for idx in range(len(columns)):
        # this person is assumed to be an infector for the next column
        infector_column = infector_name = columns[idx]  # infector
        RL = human_risk_each_day[today][infector_name]
        max_RL_received_today = -1
        if (
            infector_name in to_human_max_msg_per_day
            and today in to_human_max_msg_per_day[infector_name]
        ):
            max_RL_received_today = to_human_max_msg_per_day[infector_name][today]

        Rec = rec_levels[infector_column].item()
        test = test_results[infector_column].item()
        reported_test_result = reported_test_results[infector_column].item()
        symptoms = n_symptoms[infector_column].item()
        cell = f"Rec:{Rec} R:{RL}"
        if max_risk_level >= 0:
            # use the carried message here
            cell += f" &#9889:{max_risk_level}"

        if max_RL_received_today > 0:
            cell += f" &#128229:{max_RL_received_today}"

        test_content = ""
        if reported_test_result == "positive":
            test_content = " Test+"
        elif reported_test_result == "negative":
            test_content = " Test-"
        elif test == "positive":
            test_content = " (Test+)"
        elif test == "negative":
            test_content = " (Test-)"
        cell += test_content

        if symptoms > 0:
            cell += f"  S:{symptoms}"

        infectors_infection_timestamp = infection_timestamp[infector_column].item()
        if (
            infectors_infection_timestamp
            and infectors_infection_timestamp.date() == today
        ):
            # when this person was an infectee
            location_type = infectee_location[infector_name]
            location = location_html.get(location_type, location_type)
            cell += f" &#9763 &#9763 &#9763 {location}"

        # arrows
        if idx + 1 < len(columns):
            infectee_column = infectee_name = columns[idx + 1]
            next_infection_timestamp = infection_timestamp[infectee_column].item()
            max_risk_level = -1

            # update messages arrow
            messages = sorted(
                infector_infectee_update_messages[infector_name][infectee_name].items(),
                key=lambda x: x[0],
            )
            for timestamp, payload in messages:
                if timestamp.date() == today:
                    content = ""
                    if payload["contact"]:
                        count = payload["contact"]["count"]
                        content = f"C{count}"
                        max_risk_level = max(
                            max_risk_level, payload["contact"]["new_risk_level"]
                        )

                    if payload["unknown"]:
                        count = payload["unknown"]["count"]
                        content += f"U{count}"
                        max_risk_level = max(
                            max_risk_level, payload["unknown"]["new_risk_level"]
                        )
                    if content:
                        cell += f" \u2709 {content} \u2b95"
                    else:
                        raise
            # content of the message (can be more than one in a single day)
            # carry max_risk_level for the next person in the chain
            # no need to do anything here

        y[infector_column] = cell

    y.reset_index(drop=True, inplace=True)
    return y


def spy(ids, humans):
    ids = [x - 1 for x in ids]
    pd.set_option("display.max_rows", None)
    df = humans.loc[
        (
            slice(None),
            (
                "infection_timestamp",
                "risk",
                "test_result",
                "n_symptoms",
                "risk_level",
                "rec_level",
            ),
        ),
        ids,
    ]
    df = df.style.applymap(_color_cell_raw)
    return df


def spy_pro(
    ids,
    humans,
    human_risk_each_day,
    to_human_max_msg_per_day,
    infectee_location,
    human_has_app,
    infector_infectee_update_messages,
    human_is_asymptomatic,
):
    df = humans.loc[
        (
            slice(None),
            (
                "infection_timestamp",
                "risk",
                "test_result",
                "n_symptoms",
                "risk_level",
                "rec_level",
                "reported_test_result",
                "n_reported_symptoms",
            ),
        ),
        ids,
    ]

    # display until here
    notnull = pd.notnull(df[ids[-1]].unstack().infection_timestamp)
    last_index = notnull[::-1].idxmax() + datetime.timedelta(days=2)

    df = df.groupby(level=0, axis=0).apply(
        lambda x: summarize(
            x,
            human_risk_each_day,
            to_human_max_msg_per_day,
            infectee_location,
            infector_infectee_update_messages,
        )
    )
    df.reset_index(level=1, drop=True, inplace=True)
    df.index = df.index.date
    df = df[df.index <= last_index]

    has_app = ["&#128241" if x in human_has_app else "" for x in df.columns]
    is_asymptomatic = ["A" if x in human_is_asymptomatic else "" for x in df.columns]
    df.columns = [
        f"{x} ({is_asymptomatic[idx]}, {has_app[idx]})"
        for idx, x in enumerate(df.columns)
    ]
    df = df.style.applymap(_color_cell).set_properties(
        **{
            "font-weight": 600,
            "font-size": "150%",
            "border-color": "black",
            "text-align": "left",
            "border": "2px solid #4f2121",
        }
    )

    def remove_rec(x):
        return " ".join(x.split()[1:])

    df.format(remove_rec)

    # bells and whistles
    def hover(hover_color="#ffff99"):
        return dict(
            selector="tr:hover", props=[("background-color", "%s" % hover_color)]
        )

    styles = [
        hover(),
        dict(selector="th", props=[("font-size", "150%"), ("text-align", "center")]),
        dict(
            selector="caption",
            props=[
                ("caption-side", "bottom"),
                ("font-size", "150%"),
                ("font-weight", 600),
            ],
        ),
    ]

    df = df.set_table_styles(styles).set_caption(
        "Risk Propagation among individuals in a chain"
    )
    return df


def get_random_chain(infection_chain, depth=0):
    max_depth = 5
    start = random.choice(range(len(infection_chain)))

    # location contamination
    if not infection_chain[start]["from"]:
        return get_random_chain(infection_chain)

    infector_chain = []
    infector_chain.append(infection_chain[start]["from"])
    infectee = infection_chain[start]["to"]
    # /!\ It only considers the first infection by a human in the chain
    # TODO: randomize above
    for idx in range(start + 1, len(infection_chain)):
        if infection_chain[idx]["from"] == infectee:
            infector_chain.append(infectee)
            infectee = infection_chain[idx]["to"]

        if len(infector_chain) > 10:
            return infector_chain

    if len(infector_chain) < 3 and depth <= max_depth:
        return get_random_chain(infection_chain, depth=depth + 1)

    if depth > max_depth:
        print("maximum recursion depth reached...")

    return infector_chain


def plot(data, output_file):
    human_risk_each_day = defaultdict(lambda: defaultdict(lambda: -1))
    for x in data["risk_attributes"]:
        date = x["timestamp"].date()
        name = x["name"]
        risk_level = x["risk_level"]
        old_risk_level = human_risk_each_day[date][name]
        human_risk_each_day[date][name] = max(old_risk_level, risk_level)

    humans = load_humans(data)
    infection_chain = data["infection_monitor"]
    infector_infectee_update_messages = data["infector_infectee_update_messages"]
    human_has_app = data["human_has_app"]
    to_human_max_msg_per_day = data["to_human_max_msg_per_day"]

    human_is_asymptomatic = set()
    for x in data["infection_monitor"]:
        if x["from"] and x["from_is_asymptomatic"]:
            human_is_asymptomatic.add(x["from"])
        elif x["to"] and x["to_is_asymptomatic"]:
            human_is_asymptomatic.add(x["to"])
        else:
            pass

    infectee_location = {}
    _infected_humans = set()
    for x in infection_chain:
        if x["from"]:
            _infected_humans.add(x["from"])
        _infected_humans.add(x["to"])
        infectee_location[x["to"]] = x["location_type"]

    init_infected = _infected_humans - set(infectee_location.keys())
    for x in init_infected:
        infectee_location[x] = "unknown"

    from IPython.display import HTML

    ids = get_random_chain(infection_chain)
    df = spy_pro(
        ids,
        humans,
        human_risk_each_day,
        to_human_max_msg_per_day,
        infectee_location,
        human_has_app,
        infector_infectee_update_messages,
        human_is_asymptomatic,
    )

    with open(output_file, "w") as fo:
        fo.write(df.render())


def run(data, path, comparison_key, wandb=False, num_chains=1):

    # Options:
    # 1. "num_chains" is an integer, specifying how many infection chains we want to generate.

    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    for label, pkls in label2pkls:
        for k in range(num_chains):
            rand_index = random.randint(0, len(pkls) - 1)
            pkl = pkls[rand_index]
            output_file = os.path.join(path, f"infection_chain_{label}_chain:{k}.html")
            plot(pkl, output_file)

import pandas as pd
import dill
import os
from collections import defaultdict
import datetime
import random
import weasyprint
from PIL import Image
from covid19sim.plotting.utils import construct_infection_tree, get_infected

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
    humans.columns = [f"human:{x}" for x in range(0, humans.shape[1])]
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
    symptom_severity = x.xs("symptom_severity", level=1)
    reported_symptom_severity = x.xs("reported_symptom_severity", level=1)

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
        obs_symptoms = n_reported_symptoms[infector_column].item()
        symptoms = n_symptoms[infector_column].item()
        severity = symptom_severity[infector_column].item()
        reported_severity = reported_symptom_severity[infector_column].item()
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
            cell += f"  S:{obs_symptoms}/{symptoms}  Sev:{reported_severity}/{severity}"

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


def make_df(
    ids,
    humans,
    human_risk_each_day,
    to_human_max_msg_per_day,
    infectee_location,
    human_has_app,
    infector_infectee_update_messages,
    human_is_asymptomatic,
    caption,
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
                "symptom_severity",
                "reported_symptom_severity",
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
                ("caption-side", "top"),
                ("font-size", "200%"),
                ("font-weight", 600),
            ],
        ),
    ]

    df = df.set_table_styles(styles).set_caption(caption)
    return df



def plot(data, output_path, num_chains=10, init_infected={}):
    all_ids = set()
    caption = "heuristicv1 adoption 100%"
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
    init_infected, _infected_humans, infectee_locations = get_infected(infection_chain)
    tree, paths = construct_infection_tree(infection_chain, init_infected=init_infected, output_path=output_path)

    # get the longest 50% of lists
    candidates = paths[:int(len(paths) / 2)]

    for chain in range(num_chains):
        fig_path = os.path.join(output_path, f'chain_{chain}.png')
        ids = random.choice(candidates)
        candidates.remove(ids) # we are not sampling with replacement

        df = make_df(
            ids,
            humans,
            human_risk_each_day,
            to_human_max_msg_per_day,
            infectee_locations,
            human_has_app,
            infector_infectee_update_messages,
            human_is_asymptomatic,
            caption,
        )

        css_path = os.path.join(os.path.dirname(__file__), 'png.css')
        with open(css_path, "w") as css_file:
            width = len(ids) * 5
            height = 15
            css = f"@page {{ size: {width}in {height}in; margin: 0in 0.44in 0.2in 0.44in;}}"
            css_file.write(css)

        png = weasyprint.HTML(string=df.render()).write_png(stylesheets=[css_path], presentational_hints=True)
        with open(fig_path, "wb") as fo:
            fo.write(png)
        os.remove(css_path)
        img = Image.open(fig_path)
        pixels = img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if pixels[x,y][3] == 0:    # check alpha
                    pixels[x,y] = (255, 255, 255, 255)
        img.save(fig_path)
        all_ids = all_ids.union(ids)

    return all_ids

def run(data, path, comparison_key, adoption_rate=None, num_chains=5):

    # Options:
    # 1. "num_chains" is an integer, specifying how many infection chains we want to generate.
    for method, pkl in data.items():
        if str(method)[-1] == '/':
            m = str(method).split('/')[-2]
        else:
            m = str(method).split('/')[-1]

        dir_path = (
            path
            / "infection_chain" / m / ("adoption_" + str(adoption_rate))
        )
        os.makedirs(dir_path, exist_ok=True)

        for chain in range(num_chains):
            fig_path = os.path.join(dir_path, f'{m}_{adoption_rate}_{chain}.png')

            print(fig_path)
            caption = f"{m}, Adoption: {adoption_rate * 100}%"
            plot(pkl, fig_path, caption=caption)

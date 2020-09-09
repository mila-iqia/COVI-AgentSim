

def run(data, path, compare="app_adoption"):
    label2pkls = list()
    for method in data:
        for key in data[method]:
            label = f"{method}_{key}"
            pkls = [r["pkl"] for r in data[method][key].values()]
            label2pkls.append((label, pkls))

    # Define aggregate output variables
    agg_qaly = []

    for label, pkls in label2pkls:
        for pkl in pkls:

            # find the people who were afflicted (infection_monitor)

            # get their demographic information (pre-existing conditions, age)
            demos = pkl['humans_demographics']

            # get the symptoms they experienced and their hospitalization (or ICU) (also human_monitor)
            for date, humans in pkl['human_monitor']:
                for human in humans:
                    pass

                    # determine if they died (it's in the human_monitor)

            # put into the same QALY unit

            # add to aggregate output variables

    # print a table with mean / std
    print(agg_qaly)
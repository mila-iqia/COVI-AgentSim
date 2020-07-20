def main(conf):

    for plot in plots:
        func = all_plots[plot].run
        print_header(plot)
        try:
            # -------------------------------
            # -----  Run Plot Function  -----
            # -------------------------------

            # For infection_chains, we randomly load a file to print.
            if plot == "infection_chains":
                # Get all the methods.
                method_path = Path(root_path).resolve()
                assert method_path.exists()
                methods = [
                    m
                    for m in method_path.iterdir()
                    if m.is_dir()
                       and not m.name.startswith(".")
                       and len(list(m.glob("**/tracker*.pkl"))) > 0
                ]

                # For each method, load a random pkl file and plot the infection chains.
                for m in methods:
                    all_runs = [
                        r
                        for r in m.iterdir()
                        if r.is_dir()
                           and not r.name.startswith(".")
                           and len(list(r.glob("tracker*.pkl"))) == 1
                    ]
                    adoption_rate2runs = dict()
                    for run in all_runs:
                        adoption_rate = float(run.name.split("_uptake")[-1].split("_")[0][1:])
                        if adoption_rate not in adoption_rate2runs:
                            adoption_rate2runs[adoption_rate] = list()
                        adoption_rate2runs[adoption_rate].append(run)
                    for adoption_rate, runs in adoption_rate2runs.items():
                        rand_index = random.randint(0, len(runs) - 1)
                        rand_run = runs[rand_index]
                        with open(str(list(rand_run.glob("tracker*.pkl"))[0]), "rb") as f:
                            pkl = pickle.load(f)
                        func(dict([(m, pkl)]), plot_path, None, adoption_rate, **options[plot])
                print("Done.")

import os
import pickle
import zipfile
import argparse

parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
parser.add_argument('--data_path', type=str, default="output/data.pkl")
args = parser.parse_args()
all_data_for_day = []
with zipfile.ZipFile(f"{args.data_path}.zip", mode='a', compression=zipfile.ZIP_STORED) as zf:
    for day_path in os.listdir(args.data_path):
        days_data = []
        data_dir_path = os.path.join(args.data_path, day_path)
        for pkl in os.listdir(data_dir_path):
            with open(os.path.join(data_dir_path, pkl, "daily_human.pkl"), 'rb') as f:
                data = pickle.load(f)
                zf.writestr(f"{day_path}-{pkl}.pkl", pickle.dumps(data))

# add risks for plotting
#     todays_date = start + datetime.timedelta(days=current_day)
#     daily_risks.extend([(np.e ** human.risk, human.is_infectious(todays_date)[0], human.name) for human in hd.values()])
#     if args.plot_daily:
#         hist_plot(daily_risks, f"{args.plot_path}day_{str(current_day).zfill(3)}.png")
#     all_risks.extend(daily_risks)
# if args.save_training_data:
#     pickle.dump(all_outputs, open(args.output_file, 'wb'))
#
# dist_plot(all_risks,  f"{args.plot_path}all_risks.png")
#
# # make a gif of the dist output
# process = subprocess.Popen(f"convert -delay 50 -loop 0 {args.plot_path}/*.png {args.plot_path}/risk.gif".split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
#
# # write out the clusters to be processed by privacy_plots
# clusters = []
# for human in hd.values():
#     clusters.append(human.M)
# json.dump(clusters, open(args.cluster_path, 'w'))

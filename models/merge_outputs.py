import os
import pickle
import zipfile
import argparse

parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
parser.add_argument('--data_path', type=str, default="output/data.pkl")
parser.add_argument('--output_path', type=str, default="output/data.pkl")
args = parser.parse_args()
all_data_for_day = []
with zipfile.ZipFile(f"{args.output_path}", mode='a', compression=zipfile.ZIP_STORED) as zf:
    for day_path in os.listdir(args.data_path):
        days_data = []
        data_dir_path = os.path.join(args.data_path, day_path)
        try:
            for pkl in os.listdir(data_dir_path):
                with open(os.path.join(data_dir_path, pkl, "daily_human.pkl"), 'rb') as f:
                    data = pickle.load(f)
                    zf.writestr(f"{day_path}-{pkl}.pkl", pickle.dumps(data))
        except Exception:
            print(data_dir_path)
            print(os.path.join(data_dir_path, pkl, "daily_human.pkl"))

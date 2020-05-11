"""This file should be run on the output of the simulator's risk predictions to merge the output day/human pickle files"""
import os
import pickle
import zipfile
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Run Risk Models and Plot results')
parser.add_argument('--data_path', type=str, default="output/data.pkl")
parser.add_argument('--output_path', type=str, default="output/data.pkl")
args = parser.parse_args()
all_data_for_day = []
with zipfile.ZipFile(f"{args.output_path}", mode='a', compression=zipfile.ZIP_STORED) as zf:
    for day_path in tqdm(os.listdir(args.data_path)):
        days_data = []
        data_dir_path = os.path.join(args.data_path, day_path)
        for human_path in os.listdir(data_dir_path):
            for output_file in os.listdir(os.path.join(data_dir_path, human_path)):
                with open(os.path.join(data_dir_path, human_path, output_file), 'rb') as f:
                    data = pickle.load(f)
                    zf.writestr(f"{day_path}-{human_path}-{output_file}.pkl", pickle.dumps(data))

import argparse
import pickle
import os


def generate_human_centric_plots(debug_data, output_folder):
    pass


def generate_location_centric_plots(debug_data, output_folder):
    import pdb; pdb.set_trace()


def generate_debug_plots(debug_data, output_folder):
    generate_human_centric_plots(debug_data, output_folder)
    generate_location_centric_plots(debug_data, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_data")
    parser.add_argument("--output_folder")
    args = parser.parse_args()

    # Load the debug data
    with open(args.debug_data, "rb") as f:
        debug_data = pickle.load(f)

    # Ensure that the output folder does exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    generate_debug_plots(debug_data, args.output_folder)

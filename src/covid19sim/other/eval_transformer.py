from pathlib import Path
import argparse
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type="str", default=".", help="experimental directory")
    opts = parser.parse_args()

    path = Path(opts.path).resolve()
    model = path.parent.name

    candidates = [d for d in path.iterdir() if d.is_dir()]

    for c in candidates:
        data_path = list(c.glob("*.pkl"))[0]
        with data_path.open("rb") as f:
            data = pickle.load(f)


# DASHING-UNIVERSE vs BDT
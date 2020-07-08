from pathlib import Path

from imgurpython import ImgurClient
import json


def init_imgur_client():
    with (Path(__file__).parent / ".imgur.conf").open("r") as f:
        imgurconfig = json.load(f)
    assert "client_id" in imgurconfig
    assert "client_secret" in imgurconfig
    return ImgurClient(imgurconfig["client_id"], imgurconfig["client_secret"])


def rprint(*args):
    print(" " * 120, end="\r")
    print(*args, end="\r")


def get_plots(plot_path):
    return [
        str(f)
        for f in plot_path.glob("**/*")
        if f.is_file() and f.suffix in {".png", ".html"}
    ]


def upload_plots(plots, client):
    addresses = {}
    for p in plots:
        rprint("Uploading", p)
        addresses[p] = client.upload_from_path(p).get("link")
    print("\nDone.")
    return addresses


def make_report(plot_path):
    path = Path(plot_path).resolve()
    plots = get_plots(path)
    client = init_imgur_client()
    addresses = upload_plots(plots, client)
    print("\n".join("{}: {}".format(k, v) for k, v in addresses.items()))
    return

print("Loading imports...", end="", flush=True)
from pathlib import Path
import time
from collections import defaultdict
from imgurpython import ImgurClient
import json
import hydra
from omegaconf import OmegaConf

print("Ok.")
HYDRA_CONF_PATH = Path(__file__).parent.parent / "configs" / "plot"


def get_markdown(path, plots, addresses):
    with open(Path(__file__).parent / "experiment_template.md", "r") as f:
        template = f.read()

    plots_md = defaultdict(list)

    for p in plots:
        relative_path = p.relative_to(path)
        plot_type = relative_path.parts[0]
        plots_md[plot_type].append(p)

    md = ""
    for pt in sorted(plots_md.keys()):
        md += "\n## {}\n".format(pt.capitalize())
        for p in sorted(plots_md[pt]):
            md += f"[{p.name}]({str(p) if addresses is None else addresses[p]})\n"

    return template.format(exp_title=path.parent.name, plots=md)


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
        str(f.resolve())
        for f in plot_path.glob("**/*")
        if f.is_file() and f.suffix in {".png", ".html"}
    ]


def upload_plots(plot_path, plots, client):
    addresses = {}
    uploaded = 0
    for p in plots:
        rprint("Uploading (allow up to 10 seconds)", p)
        try:
            link = client.upload_from_path(p).get("link")
            addresses[p] = link
            uploaded += 1
            time.sleep(8)
        except Exception as e:
            print("\n*** Error:")
            print(e)
            print("Ignoring", p)
    print(f"\nDone. Uploaded {uploaded} images:")
    print("\n".join("{}: {}".format(k, v) for k, v in addresses.items()))
    with (plot_path / "imgur_uploads.json").open("w") as f:
        json.dump(addresses, f)
    return addresses


def make_report(plot_path, upload=False):
    path = Path(plot_path).resolve()
    plots = get_plots(path)

    addresses = None
    if upload:
        client = init_imgur_client()
        addresses = upload_plots(plot_path, plots, client)

    md = get_markdown(plot_path, plots, addresses)
    return


def delete_imgs(path):
    with path.open("r") as f:
        addresses = json.load(f)
    client = init_imgur_client()

    ask_pin = False
    inp = input("A pin is required to delete data on Imgur. Do you have a pin? [y/n]: ")
    if "y" in inp:
        ask_pin = True
    else:
        authorization_url = client.get_auth_url("pin")
        print("The account which uploaded the images must be the one to delete them")
        print(f"The owner of that account should get a pin at: {authorization_url}")
        print("This pin will be valid for 60 minutes.")
        ask_pin = "y" in input(
            "Did you get a pin from that url ^ and want to input it now? [y/n]: "
        )

    if ask_pin:
        pin = str(input("Write Pin: "))
        assert pin
        credentials = client.authorize(pin, "pin")
        client.set_user_auth(credentials["access_token"], credentials["refresh_token"])
        print()
        for i, (imp, img) in enumerate(addresses.items()):
            rprint(f"({i + 1} / {len(addresses)}) Deleting {imp}")
            im_id = img.split("/")[-1].split(".")[0]
            client.delete_image(im_id)
            Path(imp).unlink()
        print("Done")


@hydra.main(config_path=str(HYDRA_CONF_PATH.resolve() / "config.yaml"), strict=False)
def main(conf):
    conf = OmegaConf.to_container(conf)
    if not conf.get("delete_imgs"):
        assert "path" in conf
        make_report(conf["path"], conf.get("upload", False))
    else:
        assert Path(conf["delete_imgs"]).exists()
        delete_imgs(Path(conf["delete_imgs"]))


if __name__ == "__main__":
    main()

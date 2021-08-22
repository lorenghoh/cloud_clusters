import json

from pathlib import Path


# Resolve file path for project working directory
pwd = Path(__file__).absolute().parents[1]


def create_template():
    with open(pwd / "config.json", "w") as json_file:
        config = {
            "pwd": pwd.as_posix(),
            "src": "",
            "dst": "",
        }
        json.dump(config, json_file, indent=4)


def read_config():
    try:
        with open(pwd / "config.json", "r") as json_file:
            config = json.load(json_file)
            validate(config)

        return config
    except FileNotFoundError:
        create_template()

        print("Created default configuration file config.json. Modify as needed. \n")
    except ValueError:
        Path(pwd / "config.json").unlink()

        print("Could not resolve config.json. Re-run configuration.")


def validate(config, output=False):
    try:
        flag = True

        for item in config:
            flag = flag and Path(config[item]).exists()

        if flag is False:
            raise FileNotFoundError

        # Symlink and check .pq data location
        # Replaces any existing symlink
        pq_path = Path(pwd / "data")

        if pq_path.exists():
            pq_path.unlink()
        pq_path.symlink_to(config["src"])

        if Path(config["src"]).exists():
            if output is True:
                print("Found the following data entries: ")

                for item in sorted(pq_path.glob("*")):
                    print(f"\t {item.name}")

                print()
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Config dictionary contains unresolvable path(s). \n")


if __name__ == "__main__":
    result = read_config()

    if isinstance(result, dict):
        print(json.dumps(result, indent=4))

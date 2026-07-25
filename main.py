from pathlib import Path

import yaml


def load_club_identity() -> dict:
    config_path = Path("config/club_identity.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Club identity file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    identity = load_club_identity()

    project = identity["project"]
    manager = identity["manager"]

    print("=" * 48)
    print(project["name"])
    print(project["tagline"])
    print("=" * 48)
    print(f"Version: {project['version']}")
    print(f"Manager: {manager['name']}")
    print(f"Objective: {manager['objective']}")
    print(
        "Planning horizon: "
        f"{manager['planning_horizon_gameweeks']} Gameweeks"
    )


if __name__ == "__main__":
    main()
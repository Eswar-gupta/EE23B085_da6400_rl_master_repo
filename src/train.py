"""
Main training entry-point.

Usage:
    python src/train.py --config experiments/configs/default.yaml

Each team member can add their own config file under experiments/configs/
and run their experiments without touching this file.
"""

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config: dict):
    """Run a training loop described by *config*."""
    from src.utils.helpers import set_seed
    set_seed(config.get("seed", 42))

    env_name = config.get("env", "CartPole-v1")
    agent_type = config.get("agent", "random")
    total_steps = config.get("total_steps", 10_000)

    print(f"[train] env={env_name}  agent={agent_type}  steps={total_steps}")
    # TODO: instantiate env & agent, run loop, log results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent")
    parser.add_argument(
        "--config",
        default="experiments/configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)

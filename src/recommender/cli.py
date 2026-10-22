"""Command-line entry points for training (replay), evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.data import (
    build_deskdrop_log,
    build_rounds,
    load_articles,
    load_interactions,
    temporal_split,
)
from recommender.evaluation import evaluate_frozen, evaluate_ips, replay_train
from recommender.features import build_feature_tables, load_feature_spec
from recommender.models import LinUCBShared, PopularityGreedy, RandomPolicy, ThompsonSamplingShared

ROOT = Path(__file__).resolve().parents[2]
POOL_SIZE = 8
ALPHA = 1.0


def _load():
    interactions = load_interactions(ROOT / "data" / "interactions.csv")
    articles = load_articles(ROOT / "data" / "articles.csv")
    log = build_deskdrop_log(interactions, articles)
    split = temporal_split(log)
    spec = load_feature_spec(ROOT / "conf" / "features.yaml")
    tables = build_feature_tables(log, spec)
    rounds = build_rounds(log, tables.article_vectors, pool_size=POOL_SIZE)
    return log, split, tables, rounds


def train_command(_: argparse.Namespace) -> None:
    _, split, _, rounds = _load()
    policy = LinUCBShared(n_features=rounds.n_features, alpha=ALPHA)
    rewards = replay_train(policy, rounds, split.train_indices)
    print(
        json.dumps(
            {
                "n_rounds_trained": int(rewards.shape[0]),
                "mean_training_reward": round(float(rewards.mean()), 4),
                "last_100_mean_reward": round(float(rewards[-100:].mean()), 4),
            }
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    _, split, _, rounds = _load()
    indices = split.validation_indices if args.partition == "validation" else split.test_indices

    policies = {
        "LinUCB": LinUCBShared(n_features=rounds.n_features, alpha=ALPHA),
        "ThompsonSampling": ThompsonSamplingShared(n_features=rounds.n_features, random_state=0),
        "PopularityGreedy": PopularityGreedy(),
        "Random": RandomPolicy(),
    }

    frozen_rows, ips_rows = [], []
    for name, policy in policies.items():
        if hasattr(policy, "update") and name not in ("PopularityGreedy", "Random"):
            replay_train(policy, rounds, split.train_indices)
        frozen_rows.append(evaluate_frozen(policy, rounds, indices, name).as_row())
        ips_rows.append(evaluate_ips(policy, rounds, indices, name).as_row())

    print(json.dumps({"replay_evaluation": frozen_rows, "ips_evaluation": ips_rows}, indent=2))


def recommend_command(args: argparse.Namespace) -> None:
    import numpy as np

    log, split, _, rounds = _load()
    policy = LinUCBShared(n_features=rounds.n_features, alpha=ALPHA)
    replay_train(policy, rounds, split.train_indices)

    round_idx = args.round_index
    contexts = rounds.context[round_idx]
    position = policy.select_arm(contexts)
    content_index = rounds.candidate_content_index[round_idx]

    theta = np.linalg.inv(policy.A) @ policy.b
    scores = contexts @ theta
    order = np.argsort(-scores)

    payload = [
        {
            "rank": rank,
            "content_id": int(log.content_ids[content_index[pos]]),
            "score": round(float(scores[pos]), 4),
        }
        for rank, pos in enumerate(order, start=1)
    ]
    print(json.dumps({"chosen_position": position, "ranking": payload}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.set_defaults(func=train_command)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.set_defaults(func=evaluate_command)

    recommend = commands.add_parser("recommend")
    recommend.add_argument("--round-index", type=int, required=True)
    recommend.set_defaults(func=recommend_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

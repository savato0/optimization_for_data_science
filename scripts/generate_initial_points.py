from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simplex_qp import (
    DEFAULT_PROBLEM_CONFIG_PATH,
    generate_initial_points,
    load_problem_config,
    partition_from_config,
    save_initial_points,
    save_partition_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate initial_points.npz from problem_config.json.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_PROBLEM_CONFIG_PATH),
        help="Problem config JSON path.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to <data_folder>/initial_points.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_problem_config(args.config)
    partition = partition_from_config(config)
    data_folder = config.data_folder
    data_folder.mkdir(parents=True, exist_ok=True)
    save_partition_metadata(
        data_folder,
        partition,
        extra={
            "source": "problem_config",
            "problem_config": str(Path(args.config)),
        },
    )

    points = generate_initial_points(
        partition,
        dimension=config.n,
        num_canonical_vertices=config.initial_points.num_canonical_vertices,
        num_random_vertices=config.initial_points.num_random_vertices,
        num_random_feasible=config.initial_points.num_random_feasible,
        seed=config.initial_points.seed,
    )

    output_path = Path(args.output) if args.output else data_folder / "initial_points.npz"
    save_initial_points(output_path, points)
    print(
        f"Saved {len(points)} initial point(s) to {output_path}. "
        f"Keys: {', '.join(points.keys())}",
        flush=True,
    )


if __name__ == "__main__":
    main()

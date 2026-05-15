from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .problem import Partition


DEFAULT_PROBLEM_CONFIG_PATH = Path("config/problem_config.json")


@dataclass(frozen=True, slots=True)
class InitialPointsConfig:
    num_canonical_vertices: int = 10
    num_random_vertices: int = 20
    num_random_feasible: int = 0
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ProblemConfig:
    name: str
    data_root: Path
    n: int
    k: int
    seed: int = 42
    initial_points: InitialPointsConfig = InitialPointsConfig()

    @property
    def block_size(self) -> int:
        if self.n % self.k != 0:
            raise ValueError("n must be divisible by k for equal contiguous simplex blocks.")
        return self.n // self.k

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return (self.block_size,) * self.k

    @property
    def data_folder(self) -> Path:
        return self.data_root / self.name


def load_problem_config(path: str | Path = DEFAULT_PROBLEM_CONFIG_PATH) -> ProblemConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    n = int(payload["n"])
    k = int(payload["k"])
    if n <= 0:
        raise ValueError("Problem config field 'n' must be strictly positive.")
    if k <= 0:
        raise ValueError("Problem config field 'k' must be strictly positive.")
    if n % k != 0:
        raise ValueError("Problem config requires n divisible by k.")

    initial_payload = payload.get("initial_points", {})
    initial_points = InitialPointsConfig(
        num_canonical_vertices=int(initial_payload.get("num_canonical_vertices", 10)),
        num_random_vertices=int(initial_payload.get("num_random_vertices", 20)),
        num_random_feasible=int(initial_payload.get("num_random_feasible", 0)),
        seed=int(initial_payload.get("seed", 0)),
    )
    if initial_points.num_canonical_vertices < 0:
        raise ValueError("num_canonical_vertices must be non-negative.")
    if initial_points.num_random_vertices < 0:
        raise ValueError("num_random_vertices must be non-negative.")
    if initial_points.num_random_feasible < 0:
        raise ValueError("num_random_feasible must be non-negative.")

    name = str(payload.get("name", _default_problem_name(n, k)))
    data_root = Path(payload.get("data_root", "private/data"))
    return ProblemConfig(
        name=name,
        data_root=data_root,
        n=n,
        k=k,
        seed=int(payload.get("seed", 42)),
        initial_points=initial_points,
    )


def partition_from_config(config: ProblemConfig) -> Partition:
    return Partition.from_block_sizes(config.block_sizes)


def _default_problem_name(n: int, k: int) -> str:
    return f"dim_n{n}_k{k}"

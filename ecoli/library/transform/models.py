import abc
import os
from abc import ABC
from enum import StrEnum
import itertools
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any, Callable

import duckdb
import polars as pl
import numpy as np

from wholecell.utils.unit_struct_array import UnitStructArray
from ecoli.library.transform import REPO_ROOT


@dataclass
class PartitionSpecification:
    variant: int
    seed: int
    generation: int
    agent_id: str


@dataclass
class AvailablePartitions:
    experiment_id: str
    variants: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    generations: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._initialize()

    def _initialize(self):
        self.variants = AvailablePartitions.get_variants(self.experiment_id)
        self.seeds = sorted(np.array([AvailablePartitions.get_seeds(self.experiment_id, varid) for varid in self.variants]).flatten().tolist())

        gens = []
        for variant in self.variants:
            for seed in self.seeds:
                gen = AvailablePartitions.get_gens(self.experiment_id, variant, seed)
                gens.append(gen)
        self.generations = sorted(list(set(
            np.array(gens).flatten().tolist()
        )))

        agents = []
        for variant in self.variants:
            for seed in self.seeds:
                for gen in self.generations:
                    agents_i = AvailablePartitions.get_agents(self.experiment_id, variant, seed, gen)
                    agents.append(agents_i)
        agents = list(set(np.array(agents).flatten().tolist()))
        a = []
        indices = [
            (i, len(a)) for i, a in enumerate(agents)
        ]
        for i, length in pl.DataFrame({'i': [i for i, l in indices], 'l': [l for i, l in indices]}).sort(pl.col('l')).to_numpy().tolist():
            a.append(agents[i])
        self.agents = a

    @property
    def combinations(self) -> list[PartitionSpecification]:
        combos = []
        c = list(itertools.product(self.variants, self.seeds, self.generations, self.agents))
        for i, d in enumerate(c):
            combos.append(
                PartitionSpecification(
                    **dict(zip(['variant', 'seed', 'generation', 'agent_id'], d))
                )
            )
        return combos

    @staticmethod
    def get_variants(exp_id, outdir=str(REPO_ROOT / "out")):
        try:
            vars_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_id,
                    "history",
                    f"experiment_id={exp_id}",
                )
            )
            variant_folders = [
                folder for folder in vars_ls if not folder.startswith(".")
            ]
            variants = [int(var.split("variant=")[1]) for var in variant_folders]
        except (FileNotFoundError, TypeError):
            variants = ["N/A"]
        return variants

    @staticmethod
    def get_seeds(exp_id, var_id, outdir=str(REPO_ROOT / "out")):
        try:
            seeds_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_id,
                    "history",
                    f"experiment_id={exp_id}",
                    f"variant={var_id}",
                )
            )
            seed_folders = [folder for folder in seeds_ls if not folder.startswith(".")]
            seeds = [int(seed.split("lineage_seed=")[1]) for seed in seed_folders]
        except (FileNotFoundError, TypeError):
            seeds = ["N/A"]
        return seeds

    @staticmethod
    def get_gens(exp_id, var_id, seed_id, outdir=str(REPO_ROOT / "out")) -> list[str]:
        try:
            gens_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_id,
                    "history",
                    f"experiment_id={exp_id}",
                    f"variant={var_id}",
                    f"lineage_seed={seed_id}",
                )
            )
            gen_folders = [folder for folder in gens_ls if not folder.startswith(".")]
            gens = [int(gen.split("generation=")[1]) for gen in gen_folders]
        except (FileNotFoundError, TypeError):
            gens = ["N/A"]
        return gens

    @staticmethod
    def get_agents(
            exp_id, var_id, seed_id, gen_id, outdir=str(REPO_ROOT / "out")
    ):
        try:
            agents_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_id,
                    "history",
                    f"experiment_id={exp_id}",
                    f"variant={var_id}",
                    f"lineage_seed={seed_id}",
                    f"generation={gen_id}",
                )
            )
            agent_folders = [
                folder for folder in agents_ls if not folder.startswith(".")
            ]
            agents = [agent.split("agent_id=")[1] for agent in agent_folders]
        except (FileNotFoundError, TypeError):
            agents = ["N/A"]
        return agents


@dataclass
class SimulationConfigData:
    _df: pl.DataFrame

    def __init__(self, query: str):
        self._df = duckdb.sql(query).pl()

    def __getattr__(self, attr):
        if attr != "get":
            return getattr(self._df, attr)
        return getattr(self, attr)

    def get(self, attr: str) -> Any:
        value = self._df[[attr]].to_numpy().flatten()
        if len(value) != 1:
            raise ValueError(f"There is more than one configuration value for {attr} somehow!! I don't know how to handle this yet.")
        return value[0]


@dataclass
class DatasetLabels:
    bulk_ids: list[str]
    bulk_ids_biocyc: list[str]
    bulk_names_unique: list[str]
    bulk_common_names: list[str]
    reaction_ids: list[str]
    mrna_cistron_ids: list[str]
    mrna_cistron_names: list[str]
    cistron_data: UnitStructArray
    _common_names: list[str] | None = None

    def __post_init__(self):
        self._common_names = None

    @property
    def common_names(self):
        return self._common_names

    @common_names.setter
    def common_names(self, names: list[str] | None):
        self._common_names = names


class DataTransformExportFormat(StrEnum):
    PARQUET = "parquet"
    NDJSON = "ndjson"
    CSV = "csv"


@dataclass
class DataTransformation:
    x: pl.DataFrame
    y: pl.DataFrame
    function: Callable | FunctionType

    @property
    def f(self) -> dict[str, str]:
        return {
            "id": self.function.__name__,
            "source": self.function.__module__
        }

    @property
    def cardinality(self) -> tuple[float, float]:
        nx, ny = list(map(lambda df: len(df.rows()), [self.x, self.y]))
        dx, dy = list(map(lambda df: len(df.columns), [self.x, self.y]))
        return (
            (ny / nx), (dy / dx)
        )


class DataTransformerBase(ABC):
    x: pl.DataFrame | None = None
    y: pl.DataFrame | None = None

    def __init__(self, params: dict | None = None):
        self.params = params

    @abc.abstractmethod
    def f(self, x: pl.DataFrame, params: dict | None = None) -> pl.DataFrame:
        pass

    @property
    def cardinality(self) -> tuple[float, float]:
        nx, ny = list(map(lambda df: len(df.rows()), [self.x, self.y]))
        dx, dy = list(map(lambda df: len(df.columns), [self.x, self.y]))
        return (
            (ny / nx), (dy / dx)
        )

    def __call__(self, x: pl.DataFrame, params: dict | None = None):
        y = self.f(x, params)
        self.x = x
        self.y = y
        return y




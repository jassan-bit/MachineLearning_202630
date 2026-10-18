"""Budget-matched finite searches for a caller-owned, train-only objective.

All methods minimize the same callable. This module never loads data, splits a
sample or fits preprocessing: the caller must rebuild its entire pipeline inside
each inner training fold. A budget counts *unique objective evaluations*, not
duplicate proposals. Numeric candidate lists remain discrete categorical domains;
their ordering does not imply a continuous or logarithmic prior for TPE.

The Bayesian method uses Optuna's TPE density-ratio acquisition (expected
improvement), not a Gaussian-process surrogate. The evolutionary method uses
DEAP's actual tournament, uniform crossover and integer mutation operators.
References: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/
optuna.samplers.TPESampler.html and https://deap.readthedocs.io/en/master/api/tools.html.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, prod
import random
from threading import RLock
from time import perf_counter
from typing import Any


Scalar = str | int | float | bool | None
SearchSpace = Mapping[str, Sequence[Scalar]]
_DEAP_RANDOM_LOCK = RLock()
_ALIASES = {"grid": "grid", "grid_search": "grid", "random": "random",
            "random_search": "random", "bayesian": "bayesian", "bayes": "bayesian",
            "optuna": "bayesian", "tpe": "bayesian", "genetic": "genetic", "deap": "genetic"}


class _Space:
    """Encode a finite Cartesian product without materializing the whole grid."""

    def __init__(self, space: SearchSpace):
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a non-empty mapping of names to candidate lists")
        if any(not isinstance(key, str) or not key for key in space):
            raise ValueError("parameter names must be non-empty strings")
        self.names = tuple(sorted(space))
        self.values = []
        for name in self.names:
            candidates = space[name]
            if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
                raise ValueError(f"{name}: provide a non-empty list/tuple of scalar candidates")
            if not candidates:
                raise ValueError(f"{name}: candidate list is empty")
            values = tuple(candidates)
            for index, value in enumerate(values):
                if type(value) not in (str, int, float, bool, type(None)):
                    raise ValueError(f"{name}: unsupported candidate type {type(value).__name__}")
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError(f"{name}: candidates must be finite")
                # Optuna regards equal categorical choices (e.g. True and 1) as
                # identical. Reject ambiguity rather than silently losing type.
                if value in values[:index]:
                    raise ValueError(f"{name}: repeated/equal candidate {value!r}")
            self.values.append(values)
        self.sizes = tuple(map(len, self.values))
        self.total = prod(self.sizes)

    def decode(self, code: int) -> tuple[int, ...]:
        genes = [0] * len(self.names)
        for index in range(len(genes) - 1, -1, -1):
            code, genes[index] = divmod(code, self.sizes[index])
        return tuple(genes)

    def params(self, genes: Sequence[int]) -> dict[str, Scalar]:
        return {name: values[gene] for name, values, gene in zip(self.names, self.values, genes)}

    def genotype(self, params: Mapping[str, Scalar]) -> tuple[int, ...]:
        return tuple(values.index(params[name]) for name, values in zip(self.names, self.values))

    def unseen(self, used: set[tuple[int, ...]], rng: random.Random) -> tuple[int, ...]:
        """Return an unseen point, with a bounded random phase and exact fallback."""
        if len(used) >= self.total:
            raise ValueError("finite search space is exhausted")
        for _ in range(32):
            genes = self.decode(rng.randrange(self.total))
            if genes not in used:
                return genes
        start = rng.randrange(self.total)
        for offset in range(self.total):
            genes = self.decode((start + offset) % self.total)
            if genes not in used:
                return genes
        raise RuntimeError("unreachable: no unseen search configuration")


class _Recorder:
    def __init__(self, space: _Space, objective: Callable[[dict[str, Scalar]], float],
                 method: str, seed: int):
        self.space, self.objective, self.method, self.seed = space, objective, method, seed
        self.cache: dict[tuple[int, ...], float] = {}
        self.trials: list[dict[str, Any]] = []
        self.started = perf_counter()

    def evaluate(self, genes: Sequence[int], **extra: Any) -> float:
        key = tuple(genes)
        if key in self.cache:
            return self.cache[key]
        params = self.space.params(key)
        started = perf_counter()
        value = float(self.objective(dict(params)))
        seconds = perf_counter() - started
        if not isfinite(value):
            raise ValueError(f"objective returned non-finite value for {params!r}: {value}")
        self.cache[key] = value
        self.trials.append({"params": params, "value": value,
                            "best_so_far": min(self.cache.values()), "seconds": seconds,
                            "elapsed_seconds": perf_counter() - self.started,
                            "trial_number": len(self.trials), "seed": self.seed,
                            "method": self.method, **extra})
        return value


def _bayesian(space: _Space, record: _Recorder, limit: int, rng: random.Random) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian search requires optuna in the experiment environment") from error
    startup = min(4, max(1, limit // 2))
    sampler = optuna.samplers.TPESampler(seed=record.seed, n_startup_trials=startup,
                                        n_ei_candidates=24, multivariate=False)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    duplicate_hits = fallback_count = repeated = 0
    while len(record.cache) < limit:
        # TPE can concentrate on an already evaluated category. Preserve its
        # observations without charging another model fit, then inject an unseen
        # candidate after eight repeats so completion is guaranteed.
        if repeated >= 8:
            study.enqueue_trial(space.params(space.unseen(set(record.cache), rng)))
            fallback_count += 1
            repeated = 0
        trial = study.ask()
        params = {name: trial.suggest_categorical(name, values)
                  for name, values in zip(space.names, space.values)}
        genes = space.genotype(params)
        if genes in record.cache:
            duplicate_hits += 1
            repeated += 1
            study.tell(trial, record.cache[genes])
            continue
        value = record.evaluate(genes, proposal_number=trial.number)
        study.tell(trial, value)
        repeated = 0
    return {"library": "optuna", "library_version": optuna.__version__,
            "surrogate": "Tree-structured Parzen Estimator (independent categorical densities)",
            "acquisition": "expected improvement via good/bad density ratio l(x)/g(x)",
            "n_startup_trials": startup, "n_ei_candidates": 24,
            "multivariate": False, "duplicate_cache_hits": duplicate_hits,
            "fallback_unique_count": fallback_count,
            "proposals_including_cached": len(study.trials),
            "exploration_exploitation": "random startup, density-guided EI thereafter; unseen fallback after 8 repeats",
            "reference": "https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html"}


@contextmanager
def _deap_rng(rng: random.Random):
    """Give DEAP operators a private stream and restore stdlib random afterwards."""
    with _DEAP_RANDOM_LOCK:
        previous = random.getstate()
        random.setstate(rng.getstate())
        try:
            yield
        finally:
            rng.setstate(random.getstate())
            random.setstate(previous)


def _genetic(space: _Space, record: _Recorder, limit: int, rng: random.Random) -> dict[str, Any]:
    try:
        import deap
        from deap import base, tools
    except ImportError as error:
        raise ImportError("Genetic search requires deap in the experiment environment") from error

    class FitnessMin(base.Fitness):
        weights = (-1.0,)

    class Individual(list):
        def __init__(self, genes):
            super().__init__(genes)
            self.fitness = FitnessMin()

    toolbox = base.Toolbox()
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0] * len(space.names),
                     up=[size - 1 for size in space.sizes], indpb=1.0 / len(space.names))
    population_size = min(4, limit)
    population = []
    for _ in range(population_size):
        individual = Individual(space.unseen(set(record.cache), rng))
        individual.fitness.values = (record.evaluate(individual, generation=0),)
        population.append(individual)
    history = []
    immigrant_count = 0

    def log_generation(generation, members):
        unique = len({tuple(individual) for individual in members})
        # Share of distinct possible alleles represented, averaged over genes.
        # A fixed gene has diversity 1, since all its possible values are present.
        per_gene = [len({individual[j] for individual in members}) / size
                    for j, size in enumerate(space.sizes)]
        history.append({"generation": generation, "population_size": len(members),
                        "unique_genotypes": unique, "unique_fraction": unique / len(members),
                        "gene_diversity": sum(per_gene) / len(per_gene),
                        "per_gene_diversity": per_gene,
                        "evaluations": len(record.cache), "best_value": min(record.cache.values())})

    generation = 0
    log_generation(generation, population)
    while len(record.cache) < limit:
        generation += 1
        elite = deepcopy(tools.selBest(population, 1)[0])
        slots = min(population_size - 1, limit - len(record.cache))
        with _deap_rng(rng):
            offspring = list(map(deepcopy, toolbox.select(population, slots)))
            for index in range(0, len(offspring) - 1, 2):
                if random.random() < 0.5:
                    toolbox.mate(offspring[index], offspring[index + 1])
            for child in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(child)
        for child in offspring:
            # Re-evaluating clones wastes model fits and can stall a finite GA.
            # Explicit random immigrants repair duplicates; their count is part
            # of the record so the evolutionary/search mixture is transparent.
            if tuple(child) in record.cache:
                child[:] = space.unseen(set(record.cache), rng)
                immigrant_count += 1
            child.fitness.values = (record.evaluate(child, generation=generation),)
        population = [elite] + offspring
        log_generation(generation, population)
    try:
        deap_version = version("deap")
    except PackageNotFoundError:  # Source checkouts need not have wheel metadata.
        deap_version = getattr(deap, "__revision__", deap.__version__)
    return {"library": "deap", "library_version": deap_version,
            "population_size": population_size, "selection": "tournament", "tournament_size": 2,
            "crossover": "uniform", "crossover_probability": 0.5,
            "crossover_gene_probability": 0.5, "mutation": "uniform integer candidate index",
            "mutation_probability": 0.3, "mutation_gene_probability": 1.0 / len(space.names),
            "elitism": 1, "duplicate_repair": "unseen random immigrant",
            "immigrant_count": immigrant_count, "diversity_history": history,
            "reference": "https://deap.readthedocs.io/en/master/api/tools.html"}


def run_search(space: SearchSpace, objective: Callable[[dict[str, Scalar]], float],
               method: str, budget: int = 8, seed: int = 42) -> dict[str, Any]:
    """Minimize a fold-owned objective with a fixed number of unique model fits.

    Parameters
    ----------
    space : mapping of str to non-empty sequence of scalar
        Finite candidate values. Python scalar types are preserved; names are
        sorted for stable proposals independently of mapping insertion order.
    objective : callable
        Receives a fresh parameter dictionary and returns a finite loss. It must
        fit preprocessing and model only on the caller's inner training data.
        Exceptions propagate, so a broken pipeline cannot appear as a poor fit.
    method : {'grid', 'random', 'bayesian', 'genetic'}
        Grid follows Cartesian candidate order; a limited budget is a prefix,
        not an exhaustive grid. Random samples without replacement. TPE and GA
        reuse cached duplicates and explicitly repair proposal exhaustion.
    budget : int, default 8
        Maximum unique objective calls. When larger than the finite space, every
        point is evaluated once. Cost comparisons should also use recorded time.
    seed : int, default 42
        Search randomness, independent of the caller's estimator seed. The
        caller must separately propagate its documented seed to model fitting.

    Returns
    -------
    dict
        ``best_params``, ``best_value``, unique evaluation ``trials`` and method
        ``details``. Trials include incumbent loss and elapsed wall-clock time.
        Ties retain the first evaluated configuration. No external files change.
    """
    if not isinstance(method, str) or method.lower() not in _ALIASES:
        raise ValueError("method must be grid, random, bayesian or genetic")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not callable(objective):
        raise TypeError("objective must be callable")
    method = _ALIASES[method.lower()]
    domain = _Space(space)
    limit = min(budget, domain.total)
    record = _Recorder(domain, objective, method, seed)
    rng = random.Random(seed)
    details: dict[str, Any] = {}
    if method == "grid":
        for code in range(limit):
            record.evaluate(domain.decode(code))
        details = {"order": "Cartesian prefix, sorted parameter names; input candidate order",
                   "exhaustive": limit == domain.total}
    elif method == "random":
        for _ in range(limit):
            record.evaluate(domain.unseen(set(record.cache), rng))
        details = {"replacement": False}
    elif method == "bayesian":
        details = _bayesian(domain, record, limit, rng)
    elif method == "genetic":
        details = _genetic(domain, record, limit, rng)
    best = min(record.trials, key=lambda trial: trial["value"])
    details.update({"requested_budget": budget, "effective_budget": limit,
                    "search_space_size": domain.total, "evaluations": len(record.trials),
                    "objective_seconds": sum(trial["seconds"] for trial in record.trials),
                    "total_seconds": perf_counter() - record.started, "direction": "minimize",
                    "parameter_names": list(domain.names), "seed": seed, "method": method})
    return {"best_params": dict(best["params"]), "best_value": best["value"],
            "trials": record.trials, "details": details}

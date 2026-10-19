"""Data generators. Everything here may return oracle information; nothing in
``polyid.estimators`` or ``polyid.bounds`` may import this package."""

from polyid.generators.planted import PlantedConfig, PlantedDataset, Oracle, generate, population_moments

__all__ = ["PlantedConfig", "PlantedDataset", "Oracle", "generate", "population_moments"]

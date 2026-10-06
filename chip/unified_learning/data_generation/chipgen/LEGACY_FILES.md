# Legacy Files Organization

This document explains the organization of files in `unified_learning/data_generation/chipgen/`.

## Active V5 Files (Current Algorithm)

These files are actively used by the V5 pipeline (e.g. `scripts/generate_chipgen_training_dataset.py`, `scripts/generate_v5_dataset.py`):

- **`v5_config.py`** - V5 configuration dataclass
- **`v5.py`** - Main V5 algorithm (instance generation, edge generation, terminal assignment)
- **`v5_distributions.py`** - Distribution functions for V5
- **`placement_v5b1.py`** - V5B1 placement algorithm (macro-first, grid-based stdcell placement)
- **`spatial_hash.py`** - Spatial hash data structure for placement
- **`occupancy_grid.py`** - Occupancy grid for placement collision detection
- **`__init__.py`** - Package initialization

## Legacy Files (Moved to `legacy/` folder)

These files are from older algorithms and are not used by the main V5 generation path:

- **`config.py`** - Old configuration system (replaced by `v5_config.py`)
- **`degrees.py`** - Old degree sampling (replaced by V5's degree generation)
- **`endpoints.py`** - Old endpoint sampling (replaced by V5's edge generation)
- **`export_pyg.py`** - Old PyG export (V5 uses PyG Data directly)
- **`hierarchy.py`** - Old hierarchy generation (not used in V5)
- **`instances.py`** - Old instance generation (replaced by V5's `_generate_bimodal_instances`)
- **`pickle_to_hetero.py`** - Old HeteroData conversion (not needed for V5)
- **`pins.py`** - Old pin generation (replaced by V5's terminal assignment)
- **`placement_config.py`** - Old placement config (replaced by `v5_config.py`)
- **`placement_engine.py`** - Old placement engine (replaced by `placement_v5b1.py`)
- **`placement_first.py`** - Old placement-first algorithm (replaced by V5)
- **`runner.py`** - Old runner (replaced by `generate_v5.py`)
- **`sampler_sizes.py`** - Old size sampler (replaced by V5's instance generation)
- **`validate.py`** - Old validation (V5 has built-in validation)

## Scripts That Use Legacy Code

Scripts in `scripts/` that import from `legacy/`:
- **`generate_v5_dataset.py`** - Uses `legacy.pickle_to_hetero.data_to_heterodata`
- **`train_chipgen_diffusion.py`** - Uses `legacy.pickle_to_hetero.convert_pickle_to_hetero` when loading legacy pickle files

All other generation and training use the V5 pipeline.

## Dependency Graph (V5)

```
scripts/generate_chipgen_training_dataset.py, generate_v5_dataset.py
  └─> unified_learning.data_generation.chipgen.v5_config.V5Config
  └─> unified_learning.data_generation.chipgen.v5.V5
       ├─> v5_config.V5Config
       ├─> v5_distributions.get_distribution
       └─> placement_v5b1.V5B1Placer
            ├─> spatial_hash.SpatialHash2D
            ├─> occupancy_grid.OccupancyGrid
            └─> v5_config.V5Config
```

## Cleanup Summary

- **Active**: 7 files in `chipgen/` (V5 + helpers)
- **Legacy**: 14 files in `chipgen/legacy/`
- Legacy is kept for loading old pickle data only; new data should be generated with V5.

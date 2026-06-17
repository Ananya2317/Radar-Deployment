# Radar Placement Optimization System

A terrain-aware radar deployment optimizer that finds the best positions for multiple radars across a large geographic area, maximizing combined line-of-sight (LOS) coverage using a two-stage Dendritic Growth Optimization (DGO) algorithm.


What It Does

Given a procedurally generated 10,000×10,000 cell terrain (representing a 1,000×1,000 km area at 100m resolution), the system determines where to place N radars so that their combined visible area is as large as possible — accounting for hills, ridges, and terrain obstructions that block radar signals.

Pipeline Overview


Terrain Generation — Synthesizes realistic elevation data using multi-scale Gaussian filtering with three octaves (coarse ridges, mid-scale hills, fine detail), producing a float32 heightmap saved as a .npy checkpoint.
Viewshed / Line-of-Sight (LOS) — For each candidate radar position, ray-marching is performed across 720 angular directions. A cell is marked visible if the slope from the radar to that cell is monotonically non-decreasing — meaning no terrain between the radar and the target blocks the signal. This runs on GPU via a Numba CUDA kernel, with a CPU fallback for environments without a GPU.
Coverage Model — Combines the LOS mask with a circular range mask (max radar range in cells) to produce a per-radar boolean coverage footprint. A disk mask cache prevents recomputation.
Fitness Function — Scores a radar layout as the fraction of total terrain cells covered by at least one radar, minus penalties for radars placed too close together (below minimum separation) or out of bounds.
Two-Stage DGO Optimization — Described in detail below.
Visualization — Hillshaded terrain basemap with per-radar LOS coverage overlays, Gaussian glow borders, max-range dashed circles, and radar position markers. Outputs a high-resolution PNG.



Two-Stage DGO Algorithm

Stage 1 — Coarse Search

The full 10,000×10,000 terrain is downsampled by a factor of 4, producing a 2,500×2,500 coarse grid. All spatial parameters (max range, minimum separation, angular resolution) are scaled proportionally.

Before DGO runs, a greedy initialization seeds the starting positions. It evaluates 1,000 candidate positions (the highest-elevation cells, since elevated positions tend to have more visibility) and places radars one at a time, each time picking the position that maximizes marginal new coverage not already seen by previously placed radars. This gives DGO a far better starting point than random initialization.

The DGO optimizer then runs for 150 iterations on the coarse grid. Each iteration generates candidate positions by perturbing the current best solution with random noise. The perturbation window shrinks over time (exploration → exploitation). If the best score fails to improve for 5 consecutive iterations, a stagnation boost is triggered: the number of candidate branches and the perturbation radius are multiplied by 2 for 5 iterations, forcing the search to escape local optima. If 5 consecutive boosts produce no improvement, the run terminates early.

Stage 2 — Fine Refinement

The best coarse positions are mapped back to full-grid coordinates (multiplied by the downsample factor of 4). A jitter search then tests 13 small offsets around each mapped position on the full grid, picking the offset that maximizes the full-grid fitness score. This corrects for quantization error introduced by the coarse downsampling.

These jitter-refined positions are used as the starting point for a second DGO run on the full 10,000×10,000 terrain at full resolution (300 iterations, 40 branches). This fine stage makes precise local adjustments that the coarse grid was too low-resolution to discover.

Results (best positions, convergence history) are saved as .npy checkpoints after each stage so the run can be resumed or visualized without re-running the optimizer.


Technologies


Platform: Kaggle Notebooks (GPU T4 accelerator)
Language: Python 3


Libraries

numpy, numba, scipy, matplotlib, seaborn, joblib, tqdm, pandas, pathlib, psutil


Configuration (Cell 2)

All tunable parameters live in a single cell:

GRID_SIZE, CELL_SIZE_M, N_RADARS, RADAR_HEIGHT_M, MAX_RANGE_M, MIN_SEPARATION_M, N_ANGLES, DGO_ITER, DGO_BRANCHES, STAGNATION_LIMIT

Output


plots/radar_coverage_map_<timestamp>.png — hillshaded coverage map
plots/convergence_<timestamp>.png — DGO fitness curves (coarse + fine)
checkpoints/best_positions_fine_flat.npy — optimal radar positions
checkpoints/hist_coarse.npy / hist_fine.npy — convergence histories
results/radar_optimization_results_<timestamp>.txt — summary stats

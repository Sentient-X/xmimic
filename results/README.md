# Results

The evaluation files behind every number and figure in the README, one file per training run, all on H100 NVL GPUs.
`runs.csv` lists them: method, label, GPUs, seed and file. `python tools/make_figures.py` turns them into the figures and
tables.

Each file has one row per evaluated checkpoint (every 100 iterations): the iteration, the training time in hours
(`train_wall_h`, compilation excluded), the normalized evaluation return and the number of failed clips (both averaged
over 4 replicas of all 40 clips), and the names of the clips that failed in any replica.

- `xmimic_*`: XMimic runs, scored by `python -m xmimic.evaluate <run> --reps 4`. XMimic compiles before its clock starts.
  `xmimic_1gpu` has 8 runs from seeds 0 to 6 (seed 0 twice, in two separate jobs).
- `blinkmimic`, `instanttrack`: the baselines, scored by their own `tools/eval_batch.py --reps 4`. They compile inside
  their training loop; `seed*.csv` have that one-time cost removed by `tools/exclude_compile.py` from the files in `raw/`
  (their unchanged evaluation files and their train logs).

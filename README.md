# FADE-RIPE: Transmon Quantum-Gate Optimizer (Lindbladian, leakage- and filter-aware)

Reference implementation used for the simulations reported in:

> S. Satapathy, "FADE-RIPE: A Robust Quantum Optimal Control Algorithm for
> Superconducting Transmons with Realistic Hardware Constraints" (2026).
> TODO: add journal/arXiv reference and DOI once assigned -- see `CITATION.cff`.

**Author:** Sashreek Satapathy, Independent Researcher, Vadodara, Gujarat, India

## What this is

`fade_realistic.py` implements **FADE-RIPE**, a gradient-based quantum
optimal control method for a single-qubit gate on a 4-level (N=4) transmon,
and benchmarks it against **GRAPE** and **CRAB** (via `qutip_qtrl`). The
model and optimizer include:

1. **Realistic transmon physics** -- an N=4 Hilbert space with Kerr/anharmonic
   `H0`, a subspace-projected 2x2 target gate, and an explicit penalty for
   population leaking into levels |2>, |3>.
2. **Exact Lindbladian propagation and gradient** -- piecewise-constant
   Lindblad master-equation propagation with an *exact* analytic gradient
   (via the augmented-generator/Frechet-derivative trick), not a
   finite-difference or first-order approximation.
3. **AWG bandwidth filter** -- a Butterworth low-pass filter on the control
   pulses, represented as an exact linear operator so the gradient chain
   rule through it is exact.
4. **Filter-function noise penalty** -- a numerical filter function F(omega)
   of the optimized pulse, penalized against a 1/f (flux) or thermal-photon
   power spectral density via an overlap integral.
5. **Thermal initial-state fidelity** -- the reported "as-built" fidelity
   starts from a Boltzmann-weighted thermal state rather than the pure
   ground state.

Every gradient in the code is checked against central finite differences at
startup (`_selftest_gradients()`); the script aborts if this check fails.

## Repository contents

| File | Description |
|---|---|
| `fade_realistic.py` | Main script: physics model, optimizer, GRAPE/CRAB comparison, and analysis/plotting pipeline. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT license. |
| `CITATION.cff` | Machine-readable citation metadata (software + paper). |
| `.zenodo.json` | Additional Zenodo deposit metadata (keywords, related paper DOI). |
| `results/` | The paper's Table I/II and Figs. 1-6 are all from run `paper_final_v3` (`--gate-time 180 --run-id paper_final_v3 --mode compare-qtrl`, `Nt=360`, 8 multi-start restarts). |

## Requirements

- Python >= 3.9
- See `requirements.txt`. `numpy`, `scipy`, `matplotlib`, and `pandas` are
  required for all modes. `qutip` and `qutip_qtrl` are only required for
  `--mode compare-qtrl` (the GRAPE/CRAB baseline comparison); the script
  will print a warning and skip that comparison if they are not installed.

Install with:
```bash
pip install -r requirements.txt
```

## Usage

Run the built-in self-test (checks every analytic gradient against finite
differences; this also runs automatically at the start of every other mode):
```bash
python fade_realistic.py --mode selftest
```

Run a single FADE-RIPE optimization for a given gate time:
```bash
python fade_realistic.py --mode single --gate-time 180
```

Run FADE-RIPE together with the full analysis/plotting pipeline (writes
figures, CSVs, and JSON summaries under `results/`):
```bash
python fade_realistic.py --mode analyze --gate-time 180 --run-id my_run
```

Run FADE-RIPE alongside the GRAPE/CRAB (`qutip_qtrl`) baseline comparison
(requires `qutip`/`qutip_qtrl`):
```bash
python fade_realistic.py --mode compare-qtrl --gate-time 180 --run-id my_run
```

Command-line options:

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `single` | One of `single`, `selftest`, `compare-qtrl`, `analyze`. |
| `--gate-time` | `180.0` | Gate duration `Tg` in nanoseconds. |
| `--nt` | auto (`gate_time / 0.5`) | Number of piecewise-constant time slices. |
| `--run-id` | `default` | Label used in output filenames under `results/`. |
| `--workers` | auto (`min(4, cpu_count)`) | Number of parallel worker processes; override with `--workers N` or the `FADE_WORKERS` environment variable. |

On a multi-core machine or HPC node, increase parallelism with, e.g.:
```bash
FADE_WORKERS=32 python fade_realistic.py --mode analyze --gate-time 180
```

## Output

`--mode analyze` and `--mode compare-qtrl` write:
- `results/raw/checkpoint_<run_id>.npz` -- raw optimized control pulses
  (checkpointed *before* the analysis/plotting pipeline runs, so a failure
  in post-processing never loses the optimization result itself).
- `checkpoint_optimization_info_<run_id>.json` -- fidelity, leakage, and
  runtime summary.
- Additional figures (PDF/PNG), CSVs, and JSON metadata produced by the
  analysis pipeline (Bloch-sphere trajectories, time-dependent fidelity,
  leakage, noise-robustness sweeps, filter-function overlap, AWG-bandwidth
  sensitivity, method comparison, and run metadata/provenance).

## Known limitations (stated in-code)

- The filter-function gradient (item 4 above) costs O(Nt^2) per frequency
  and is evaluated on a coarser, log-spaced frequency grid
  (`Config.ff_n_freqs`) rather than the full pulse-time resolution; an
  O(Nt) adjoint reformulation is possible but not implemented here.
- The GRAPE/CRAB comparison uses `qutip_qtrl`'s own internal gradient in
  its `GEN_MAT` (generic-generator) propagation mode, which has not been
  independently re-derived or verified against FADE-RIPE's analytic
  gradient -- treat it as a baseline comparison, not as sharing FADE-RIPE's
  verified-gradient guarantee. `qutip`/`qutip_qtrl` were not installed in
  the environment this script was developed in, so run
  `python fade_realistic.py --mode selftest` (or `--mode compare-qtrl`)
  in an environment with `qutip_qtrl` installed to verify this path before
  relying on it.
- The filter function treatment is first-order (Magnus); it does not
  capture second-order/non-Gaussian noise effects.
- Baseline parameters (anharmonicity, thermal temperature, etc.) are
  documented inline in `Config` with the literature/assumption each is
  based on.

## Reproducibility

- Random seeds are fixed (`rng_global = np.random.default_rng(42)`,
  `_RESTART_SEED_BASE`) for the noise-ensemble sampling and multi-start
  perturbations, so re-running with the same `Config` and the same
  NumPy/SciPy versions should reproduce the reported numbers.
- `_selftest_gradients()` runs automatically before any optimization and
  will abort the run if any analytic gradient disagrees with its
  finite-difference check.

## Computational environment (as reported in the paper)

Per the paper's Sec. IV A: day-to-day development and gradient
verification were carried out on local Apple M4-class hardware; the
full benchmark reported in the paper (`run_id=paper_final_v3`, 8
multi-start restarts, K=128 noise-ensemble evaluations per restart) was
executed on a 96-physical-core AWS EC2 `hpc6a.48xlarge` instance, using
90 of the 96 cores for the worker pool (`--workers 90` or
`FADE_WORKERS=90`). That benchmark took 32,362.1 s (~8.99 h) for
FADE-RIPE, versus 8.6 s for GRAPE and 21.2 s for CRAB on the same
instance.

## License

MIT License

## Acknowledgements

From the paper's Acknowledgments section: the author thanks Dr. Boxi Li
and Dr. Veronika Kurth (RIKEN) for guidance on the QuTiP-based Lindbladian
and `qutip_qtrl` simulations used in this work, and R. F. Parcelas Resina
dos Santos for mentorship throughout the project. This research was
supported by the Lumiere Research Breakthrough Scholar Fellowship where the author recieved full financial aid. The author also thanks
Raul for early guidance on open quantum systems and quantum optimal
control.

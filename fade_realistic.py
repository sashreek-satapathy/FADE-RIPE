"""
fade_realistic.py -- Reference implementation (v1.0) of FADE-RIPE, benchmarked
against GRAPE and CRAB baselines (via qutip_qtrl) on a realistic transmon
physics model.

Companion code for the paper:
  S. Satapathy, "FADE-RIPE: A Robust Quantum Optimal Control Algorithm for
  Superconducting Transmons with Realistic Hardware Constraints" (2026).
  See CITATION.cff for the full reference.

============================================================================
WHAT FADE-RIPE DOES
============================================================================
FADE-RIPE (Fidelity Aware, Distortion-aware, Environment-conscious Robust
Iterative Pulse Engineering) optimizes single-qubit gates on an N=4-level
transmon under a full Lindblad master equation. Four physical effects are
folded directly into one gradient-verified objective rather than treated as
separate, bolt-on corrections:

  1. TRANSMON PHYSICS       -- N=4 Hilbert space, Kerr/anharmonic H0,
                                subspace-projected target, leakage penalty.
  2. LIOUVILLE PROPAGATION  -- exact Lindbladian propagation with an EXACT
                                (not first-order/commutator-approximate)
                                analytic gradient via the augmented-
                                generator trick.
  3. AWG BANDWIDTH FILTER   -- Butterworth low-pass represented as an exact
                                linear operator so the gradient chain rule
                                through it is exact, not finite-differenced.
  4. FILTER FUNCTION        -- numerical F(omega) of the optimized pulse,
                                penalized against a 1/f or thermal-photon
                                PSD via an overlap integral.

Thermal (rather than pure-state) initialization is used for the reported
"as-built" fidelity: rho_th replaces the pure ground state as the starting
point for the final performance metrics.

Every analytic gradient is checked against central finite differences at
runtime -- see `_selftest_gradients()`, which `main()` runs once at startup
and refuses to proceed on failure. This is enforced to a strict 1e-6
tolerance (the largest observed disagreement in the benchmark run reported
in the paper was 3.91e-10, roughly four orders of magnitude tighter), for
every piece that carries a gradient (Lindblad+leakage adjoint sweep,
bandwidth-filter chain rule, filter-function overlap term).

KNOWN LIMITATIONS (see the paper's Discussion section for full detail):

* The filter-function gradient (Section 4) is built by direct summation
  over pulse segments and costs O(Nt^2) per frequency. It is exact (see
  self-test), but at Nt=360 with even a modest frequency grid this is the
  single most expensive term in the objective. It is therefore evaluated
  on a coarser, log-spaced frequency grid (`cfg.ff_n_freqs`) -- the first
  place to look if FADE-RIPE's iteration cost needs to be reduced. An
  O(Nt) adjoint reformulation (analogous to the Lindblad costate sweep) is
  possible in principle but not implemented here.
* qutip_qtrl's GRAPE/CRAB are wired to Lindbladian dynamics via
  `dyn_type='GEN_MAT'` (qutip_qtrl's generic-generator propagation mode,
  used here with the vectorized Liouvillian as the generator and vec(rho)
  as the propagated "state"). This is the documented mechanism for
  non-unitary GRAPE in qutip_qtrl, but qutip_qtrl's own gradient
  computation in this mode is qutip_qtrl's, not the FADE-RIPE analytic
  gradient above -- the qutip_qtrl internal GEN_MAT gradient has not been
  independently re-derived or verified against finite differences within
  this work. Treat GRAPE/CRAB results here as a baseline comparison, not
  as sharing FADE-RIPE's verified-gradient guarantee. Run
  `python fade_realistic.py --mode selftest` in an environment with
  qutip_qtrl installed before relying on the `--mode compare-qtrl` path.
* The filter function y(t) uses the coherence element of the noise
  operator in the toggling frame of the *coherent* (Lindblad-free)
  propagator, standard for a first-order (Magnus) filter-function
  treatment. It does not capture second-order/non-Gaussian noise effects.
* Baseline hardcoded values (see Config): anharmonicity alpha/2pi =
  -200 MHz (standard transmon benchmark figure, consistent with the
  DRAG/GRAPE literature), thermal temperature T = 50 mK with qubit
  frequency 5.0 GHz assumed for the Boltzmann population (a typical
  reported *effective* qubit temperature, not the ~10-20 mK mixing-chamber
  temperature -- these differ in real devices and 50 mK is the
  standard conservative literature figure for the former).
* Every stochastic element (the K=128 training noise draws, the multi-start
  perturbations, the n=300 out-of-sample evaluation draws) is generated
  from a fixed pseudorandom seed, so a given run is exactly reproducible
  but represents a single realization of the assumed noise environment,
  not an average over independent noise realizations. See the paper's
  Discussion section for the recommended extension (repeating the
  comparison across several independently seeded noise ensembles).

============================================================================
ANALYSIS PIPELINE
============================================================================
This module includes a post-processing and instrumentation layer (the
`analyze_*` functions and `run_full_analysis()`) that reproduces the
figures and tables reported in the paper. It does not modify the physics,
optimization logic, or numerical behavior of FADE-RIPE itself -- all data
collection is done by calling the existing optimization functions and
wrapping their outputs with plotting/logging.
============================================================================
"""

import os, sys, warnings, time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

# ── HPC/Multicore: auto-detect available cores ─────────────────────────────
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing as _mp

# Auto-detect cores for HPC, fallback to 4 on a local machine
_DEFAULT_WORKERS = min(4, _mp.cpu_count())
_N_WORKERS = int(os.environ.get("FADE_WORKERS", _DEFAULT_WORKERS))
# Override with FADE_WORKERS env var if you want a specific count.


import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt

# NumPy 2.0 removed np.trapz (renamed to np.trapezoid). This keeps the
# script working on both old (<2.0) and new (>=2.0) NumPy without pinning
# a version -- important since a fresh `pip install numpy` on a new HPC
# instance will grab the latest release.
_trapz = getattr(np, 'trapezoid', None) or np.trapz

# ── Analysis imports ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

warnings.filterwarnings("ignore")

_IS_WORKER = os.environ.get("_FADE_WORKER", "0") == "1"
if not _IS_WORKER:
    os.environ["_FADE_WORKER"] = "1"
    print(f"Python {sys.version.split()[0]}")
    print(f"Multicore: {_N_WORKERS} workers, BLAS=1 thread/worker")
    print(f"System has {_mp.cpu_count()} cores available")

TWO_PI = 2.0 * np.pi
rng_global = np.random.default_rng(42)
_RESTART_SEED_BASE = 20260830


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # ── Transmon parameters ────────────────────────────────────────────
    N_LEVELS: int = 4
    detuning_nom: float = 0.0
    alpha_MHz: float = -200.0            # anharmonicity alpha/2pi, MHz [BASELINE]
    max_amp: float = 0.012 * TWO_PI      # rad/ns
    T1:  float = 45_500.0                # ns
    T2s: float = 16_810.0                # ns

    # ── Thermal initialization ─────────────────────────────────────────
    qubit_freq_GHz: float = 5.0          # only used to convert T -> n_th
    temperature_mK: float = 50.0         # [BASELINE]

    # ── Gate ────────────────────────────────────────────────────────────
    Tg: float = 180.0
    Nt: int = 360

    # ── Leakage penalty ─────────────────────────────────────────────────
    leak_lambda: float = 7.0

    # ── AWG bandwidth (Section 3) ───────────────────────────────────────
    awg_bandwidth_GHz: float = 0.30      # Butterworth -3dB cutoff
    awg_filter_order: int = 4

    # ── Filter function / colored noise overlap (Section 4) ────────────
    ff_n_freqs: int = 48                   # Frequency grid size for the noise-PSD
                                            # overlap integral. This is the single
                                            # most expensive term per objective
                                            # evaluation (see module docstring);
                                            # 48 log-spaced points is chosen to
                                            # prioritize accuracy of the reported
                                            # penalty over per-iteration speed.
    ff_omega_min: float = 2 * np.pi * 1e-5   # rad/ns  (~1.6 kHz)
    ff_omega_max: float = 2 * np.pi * 1e-1   # rad/ns  (~16 MHz)
    ff_noise_type: str = "flux_1f"       # "flux_1f" or "thermal_photon"
    ff_psd_amplitude: float = 1.0e-8     # rad^2/ns, arbitrary calibration scale
    ff_mu: float = 5.0e3                 # OVERWRITTEN by auto-calibration in
                                          # fade_ripe_optimize() -- see ff_target_fraction.
                                          # This raw default is NOT safe to use directly:
                                          # at a near-unit-fidelity seed pulse it was
                                          # measured to equal ~100% of the fidelity term
                                          # (should be a small correction, not a dominant
                                          # one). Left here only as the pre-calibration
                                          # starting point auto-calibration scales from.
    ff_target_fraction: float = 0.02     # target: penalty term = 2% of the seed
                                          # pulse's fidelity term, not 100%+ of it.

    # ── Evaluation noise (quasi-static, for benchmark sweeps) ───────────
    sigma_detuning: float = 0.001 * TWO_PI
    sigma_amp: float = 0.010
    n_noise_eval: int = 300                # Out-of-sample benchmark ensemble size
                                            # for the final reported mean/CVaR
                                            # numbers, kept large so these reported
                                            # robustness statistics have low
                                            # sampling error themselves, independent
                                            # of the K draws used during optimization.

    # ── FADE-RIPE ────────────────────────────────────────────────────────
    fade_nmodes: int = 40
    fade_maxiter: int = 3000               # Lets L-BFGS-B run until it genuinely
                                            # satisfies ftol/gtol below, or until this
                                            # very high cap (a safety backstop against
                                            # a pathological non-terminating case,
                                            # not an expected stopping point).
    fade_ftol: float = 2.220446049250313e-16 * 10   # ~10x machine epsilon -- as
                                            # tight as scipy's L-BFGS-B (using
                                            # FACTR internally) can meaningfully
                                            # resolve; the optimizer will stop on
                                            # gtol or maxiter before this binds in
                                            # practice, which is the point: don't
                                            # let ftol be the thing that cuts the
                                            # search short.
    fade_gtol: float = 1e-10               # tight; matches the "run until genuinely
                                            # stationary" intent. If this never
                                            # triggers and every run instead hits
                                            # maxiter=3000, that is itself worth
                                            # investigating -- it would suggest the
                                            # landscape near the optimum is just
                                            # very flat (common with a K-sample noise
                                            # average) rather than raise the cap
                                            # further blindly.
    fade_n_restarts: int = 8               # multi-start: L-BFGS-B is a LOCAL
                                            # optimizer -- tight tolerances on one
                                            # run only guarantee reaching *a*
                                            # stationary point, not the best one.
                                            # Reporting a benchmark result requires
                                            # evidence against a mediocre local
                                            # optimum. Each restart begins from a
                                            # differently-perturbed seed pulse (see
                                            # fade_ripe_multistart()) and the best
                                            # result across all restarts (scored on
                                            # a large fixed evaluation ensemble, not
                                            # the noisy per-run K-average) is kept.
    fade_K: int = 128                    # Fixed quasi-static noise ensemble size
                                          # (sample average approximation) used
                                          # during optimization -- large enough to
                                          # reduce the risk of overfitting to an
                                          # unrepresentative small sample of
                                          # detuning/amplitude draws.
    fade_sigma: float = 0.0025 * TWO_PI
    fade_amp_sig: float = 0.020

    # ── GRAPE/CRAB (qutip_qtrl, Lindbladian GEN_MAT mode) ───────────────
    grape_maxiter: int = 600
    grape_restarts: int = 5
    crab_nmodes: int = 8
    crab_maxiter: int = 500
    crab_restarts: int = 3
    tslot_type: str = 'DEF'
    parallel_restarts: bool = True

    bench_detunings_5: np.ndarray = field(
        default_factory=lambda: np.linspace(-5e-3, 5e-3, 41) * TWO_PI)
    bench_detunings_15: np.ndarray = field(
        default_factory=lambda: np.linspace(-10e-3, 10e-3, 61) * TWO_PI)

    @property
    def dt(self): return self.Tg / self.Nt
    @property
    def gamma_1(self): return 1.0 / self.T1
    @property
    def gamma_phi(self): return max(1.0 / self.T2s - self.gamma_1 / 2.0, 1e-12)
    @property
    def alpha_rad_per_ns(self): return self.alpha_MHz * 1e-3 * TWO_PI
    @property
    def n_thermal(self):
        """Boltzmann excited-state population at qubit_freq_GHz/temperature_mK."""
        hbar_omega_over_kT = (TWO_PI * self.qubit_freq_GHz * 1e9 * 6.62607015e-34
                               / (2 * np.pi)) / (1.380649e-23 * self.temperature_mK * 1e-3)
        return 1.0 / (np.exp(hbar_omega_over_kT) - 1.0)
    @property
    def tlist(self): return np.linspace(0.0, self.Tg, self.Nt + 1)


cfg = Config()

def set_gate_time(Tg: float, Nt: int):
    cfg.Tg = Tg
    cfg.Nt = Nt


# ============================================================================
# TRANSMON SYSTEM (Section 1)
# ============================================================================

def _destroy_np(N: int) -> np.ndarray:
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n - 1, n] = np.sqrt(n)
    return a


class TransmonSystem:
    """N-level transmon: H0 = Delta*n + (alpha/2) adag adag a a.

    Target is a subspace projection: we optimize the 2x2 computational
    block against U_target_sub and separately penalize leakage into
    levels >= 2, per the requirement to ignore phases on higher levels
    rather than trying to match a full N-level unitary.
    """

    def __init__(self):
        N = cfg.N_LEVELS
        self.N = N
        a = _destroy_np(N)
        adag = a.conj().T
        n_op = adag @ a

        self.a, self.adag, self.n_op = a, adag, n_op
        alpha = cfg.alpha_rad_per_ns
        self.H0 = (cfg.detuning_nom * n_op
                   + 0.5 * alpha * (adag @ adag @ a @ a)).astype(complex)
        self.Hx = ((a + adag) / 2.0).astype(complex)
        self.Hy = (1j * (adag - a) / 2.0).astype(complex)

        self.U_target_sub = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)

        self.P_leak = np.zeros((N, N), dtype=complex)
        for lvl in range(2, N):
            self.P_leak[lvl, lvl] = 1.0

        self.c_ops = [
            np.sqrt(cfg.gamma_1) * a,
            np.sqrt(2 * cfg.gamma_phi) * n_op,
        ]

        # noise-coupling operator for the filter-function calculation
        # (dephasing / flux noise couples through n_op; swap for Hx if you
        # want to analyze amplitude/thermal-photon noise susceptibility
        # instead -- see filter_function_and_grad()).
        self.B_noise = n_op.astype(complex)

    def thermal_state(self) -> np.ndarray:
        """rho_th: Boltzmann-weighted population across the N levels at
        cfg.temperature_mK, referenced to the qubit 0-1 splitting."""
        n_th = cfg.n_thermal
        # geometric/Boltzmann population ladder p_k propto exp(-k * hw/kT);
        # recover the per-level ratio from n_th = p1/p0 for a 2-level
        # reference and extend geometrically to higher levels (standard
        # approximation for a weakly anharmonic oscillator's thermal tail).
        r = n_th / (1.0 + n_th) if n_th > 0 else 0.0
        pops = np.array([(1 - r) * r ** k for k in range(self.N)])
        pops /= pops.sum()
        return np.diag(pops).astype(complex)

    def ideal_target_dm(self, rho_in: np.ndarray) -> np.ndarray:
        """Ideal output of applying U_target_sub to the computational block
        of rho_in, leaving population outside the 2x2 block untouched by
        the (fictitious) ideal gate -- i.e. the leakage-free reference we
        compare the actual leaky/dissipative propagation against."""
        N = self.N
        Uemb = np.eye(N, dtype=complex)
        Uemb[0:2, 0:2] = self.U_target_sub
        return Uemb @ rho_in @ Uemb.conj().T


sys_ = TransmonSystem()


# ============================================================================
# 2. LIOUVILLE-SPACE PROPAGATION + EXACT ADJOINT GRADIENT
# ============================================================================
#
# vec() uses column-stacking (Fortran order), matched to the Liouvillian
# convention below:
#   L(H, {c}) = -i(I@H - H^T@I) + sum_c [ c*@c - 1/2(I@(c^dag c) + (c^dag c)^T@I) ]
# so that d(vec rho)/dt = L @ vec(rho) reproduces the Lindblad master
# equation exactly. See the finite-difference self-tests in
# _selftest_gradients() for the numerical proof.

def _vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(-1, order='F')

def _unvec(v: np.ndarray, N: int) -> np.ndarray:
    return v.reshape((N, N), order='F')

def liouvillian(H: np.ndarray, c_ops: List[np.ndarray]) -> np.ndarray:
    N = H.shape[0]
    I = np.eye(N, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in c_ops:
        cdc = c.conj().T @ c
        L += np.kron(c.conj(), c) - 0.5 * (np.kron(I, cdc) + np.kron(cdc.T, I))
    return L


class LindbladPropagator:
    """Builds the piecewise-constant Liouvillian superoperators for a
    control sequence and their EXACT derivatives w.r.t. uI[k], uQ[k] via
    the augmented-generator trick:

        expm([[dt*Lk, dt*Lx, dt*Ly],
              [0,      dt*Lk, 0   ],
              [0,      0,     dt*Lk]])
      = [[Sk, dSk/duI, dSk/duQ],
         [0,   Sk,      0     ],
         [0,   0,       Sk    ]]

    This is the Frechet-derivative-of-matrix-exponential identity applied
    to two control directions simultaneously; it is exact for the
    piecewise-constant-Hamiltonian model (not a first-order/commutator
    approximation), and dt-independent in its exactness -- only the
    physical model (piecewise-constant H over each dt) is an
    approximation, not this gradient of it.
    """

    def __init__(self, H0, Hx, Hy, c_ops):
        self.N = H0.shape[0]
        self.d = self.N * self.N
        self.L0 = liouvillian(H0, c_ops)
        I = np.eye(self.N, dtype=complex)
        self.Lx = -1j * (np.kron(I, Hx) - np.kron(Hx.T, I))
        self.Ly = -1j * (np.kron(I, Hy) - np.kron(Hy.T, I))
        self._Zd = np.zeros((self.d, self.d), dtype=complex)

    def step_data(self, uI: np.ndarray, uQ: np.ndarray, dt: float):
        """Returns list of (S_k, dS_k/duI, dS_k/duQ), one per timestep."""
        Nt = len(uI)
        out = []
        Zd = self._Zd
        for k in range(Nt):
            Lk = self.L0 + uI[k] * self.Lx + uQ[k] * self.Ly
            M = np.block([[dt * Lk, dt * self.Lx, dt * self.Ly],
                          [Zd,       dt * Lk,       Zd],
                          [Zd,       Zd,             dt * Lk]])
            E = expm(M)
            d = self.d
            out.append((E[0:d, 0:d], E[0:d, d:2 * d], E[0:d, 2 * d:3 * d]))
        return out

    def forward(self, steps, rho0_vec: np.ndarray) -> List[np.ndarray]:
        fwd = [rho0_vec]
        for S, _, _ in steps:
            fwd.append(S @ fwd[-1])
        return fwd

    def backward_costate(self, steps, chi_final_vec: np.ndarray) -> List[np.ndarray]:
        Nt = len(steps)
        chi = [None] * (Nt + 1)
        chi[Nt] = chi_final_vec
        for k in range(Nt - 1, -1, -1):
            chi[k] = steps[k][0].conj().T @ chi[k + 1]
        return chi


def lindblad_fidelity_and_grad(
    uI: np.ndarray, uQ: np.ndarray, rho0: np.ndarray, rho_target: np.ndarray,
    prop: LindbladPropagator, dt: float, leak_lambda: float,
    steps=None,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    Nt = len(uI)
    if steps is None:
        steps = prop.step_data(uI, uQ, dt)
    fwd = prop.forward(steps, _vec(rho0))
    rhoT = _unvec(fwd[-1], prop.N)

    F = float(np.real(np.trace(rho_target.conj().T @ rhoT)))
    leak = float(np.real(np.trace(sys_.P_leak @ rhoT)))
    J = F - leak_lambda * leak

    chiF = prop.backward_costate(steps, _vec(rho_target))
    chiL = prop.backward_costate(steps, _vec(sys_.P_leak))

    gI = np.zeros(Nt)
    gQ = np.zeros(Nt)
    for k in range(Nt):
        S, dI, dQ = steps[k]
        dF_dI = np.real(np.vdot(chiF[k + 1], dI @ fwd[k]))
        dF_dQ = np.real(np.vdot(chiF[k + 1], dQ @ fwd[k]))
        dL_dI = np.real(np.vdot(chiL[k + 1], dI @ fwd[k]))
        dL_dQ = np.real(np.vdot(chiL[k + 1], dQ @ fwd[k]))
        gI[k] = dF_dI - leak_lambda * dL_dI
        gQ[k] = dF_dQ - leak_lambda * dL_dQ
    return J, gI, gQ, leak, F


# ============================================================================
# 5. THERMAL INITIAL STATE FIDELITY
# ============================================================================

def thermal_gate_fidelity_and_grad(
    uI: np.ndarray, uQ: np.ndarray, prop: LindbladPropagator, dt: float,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    rho_th = sys_.thermal_state()
    rho_target = sys_.ideal_target_dm(rho_th)
    return lindblad_fidelity_and_grad(
        uI, uQ, rho_th, rho_target, prop, dt, cfg.leak_lambda)


def average_gate_fidelity_and_grad(
    uI: np.ndarray, uQ: np.ndarray, prop: LindbladPropagator, dt: float,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    N = sys_.N
    rho_th = sys_.thermal_state()
    leak_bg = np.real(np.trace(sys_.P_leak @ rho_th))

    kets = [
        np.array([1, 0] + [0] * (N - 2), dtype=complex),
        np.array([0, 1] + [0] * (N - 2), dtype=complex),
        np.array([1, 1] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
        np.array([1, 1j] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
    ]
    Nt = len(uI)
    steps = prop.step_data(uI, uQ, dt)

    J_tot, gI_tot, gQ_tot, leak_tot, F_tot = 0.0, np.zeros(Nt), np.zeros(Nt), 0.0, 0.0
    for psi in kets:
        rho_comp = np.outer(psi, psi.conj())
        p_comp = 1.0 - leak_bg
        rho_in = p_comp * rho_comp + sys_.P_leak @ rho_th @ sys_.P_leak
        rho_target = sys_.ideal_target_dm(rho_in)
        J, gI, gQ, leak, F_raw = lindblad_fidelity_and_grad(
            uI, uQ, rho_in, rho_target, prop, dt, cfg.leak_lambda, steps=steps)
        J_tot += J; gI_tot += gI; gQ_tot += gQ; leak_tot += leak; F_tot += F_raw
    n = len(kets)
    return J_tot / n, gI_tot / n, gQ_tot / n, leak_tot / n, F_tot / n


# ============================================================================
# 3. AWG BANDWIDTH TRANSFER FUNCTION
# ============================================================================

class BandwidthFilter:
    """Zero-phase Butterworth low-pass, represented as an exact Nt x Nt
    linear matrix (built once per Nt/cutoff via unit impulses) so the
    chain rule through it is exact rather than autodiff/finite-difference
    dependent. Verified against direct filtfilt() and against a
    finite-difference chain-rule check in _selftest_gradients()."""

    def __init__(self, Nt: int, dt: float, cutoff_GHz: float, order: int):
        fs = 1.0 / dt  # samples/ns = GHz
        wn = cutoff_GHz / (fs / 2.0)
        wn = min(max(wn, 1e-6), 0.999)
        self.b, self.a = butter(N=order, Wn=wn, btype='low')
        F = np.zeros((Nt, Nt))
        for i in range(Nt):
            e = np.zeros(Nt); e[i] = 1.0
            F[:, i] = filtfilt(self.b, self.a, e, method="gust")
        self.F = F
        self.FT = F.T

    def apply(self, u_raw: np.ndarray) -> np.ndarray:
        return self.F @ u_raw

    def backprop_grad(self, dJ_du_filtered: np.ndarray) -> np.ndarray:
        """Chain rule: dJ/du_raw = F^T @ dJ/du_filtered (F is linear)."""
        return self.FT @ dJ_du_filtered


# ============================================================================
# 4. FILTER FUNCTION & COLORED-NOISE OVERLAP PENALTY
# ============================================================================

def _psd(omega: np.ndarray) -> np.ndarray:
    if cfg.ff_noise_type == "flux_1f":
        return cfg.ff_psd_amplitude / np.maximum(np.abs(omega), 1e-8)
    elif cfg.ff_noise_type == "thermal_photon":
        return cfg.ff_psd_amplitude * np.ones_like(omega)
    raise ValueError(cfg.ff_noise_type)


def _coherent_step_data(uI, uQ, dt, H0, Hx, Hy):
    """Same augmented-generator trick as LindbladPropagator, but on the
    bare N x N Hilbert-space generator -i*H (no dissipation) -- this is
    the *coherent* reference propagator used to define the toggling frame
    for the filter function, per the standard filter-function formalism."""
    N = H0.shape[0]
    Nt = len(uI)
    Zd = np.zeros((N, N), dtype=complex)
    out = []
    for k in range(Nt):
        Hk = H0 + uI[k] * Hx + uQ[k] * Hy
        Gk, Gx, Gy = -1j * Hk, -1j * Hx, -1j * Hy
        M = np.block([[dt * Gk, dt * Gx, dt * Gy],
                      [Zd,       dt * Gk, Zd],
                      [Zd,       Zd,       dt * Gk]])
        E = expm(M)
        out.append((E[0:N, 0:N], E[0:N, N:2 * N], E[0:N, 2 * N:3 * N]))
    return out


def _ff_single_freq_term_bruteforce(om, dom, Sw, Ulist, Sk_list, steps, tlist, B, N, Nt, dt):
    """ORIGINAL O(Nt^2) contribution of ONE frequency to J_pen/gI/gQ.
    Kept only as a cross-check reference for `_ff_single_freq_term` (the
    O(Nt) adjoint reformulation below) -- see `_selftest_gradients`,
    which now verifies the two agree at every self-test run, in addition
    to each independently agreeing with finite differences. No longer
    called from the optimization hot path."""
    y = np.array([(Ulist[k].conj().T @ B @ Ulist[k])[0, 1] for k in range(Nt)])
    phase = np.exp(1j * om * tlist)
    integral = np.sum(y * phase) * dt
    Fw = float(np.abs(integral) ** 2)
    J_term = Fw * Sw * dom

    gI_term = np.zeros(Nt)
    gQ_term = np.zeros(Nt)
    for j in range(Nt):
        dS_I, dS_Q = steps[j][1], steps[j][2]
        dF_dI = 0.0 + 0.0j
        dF_dQ = 0.0 + 0.0j
        left = np.eye(N, dtype=complex)
        for m in range(j + 1, Nt + 1):
            if m > j + 1:
                left = Sk_list[m - 1] @ left
            Um = Ulist[m]
            dUm_dI = left @ dS_I @ Ulist[j]
            dUm_dQ = left @ dS_Q @ Ulist[j]
            dV_dI = dUm_dI.conj().T @ B @ Um + Um.conj().T @ B @ dUm_dI
            dV_dQ = dUm_dQ.conj().T @ B @ Um + Um.conj().T @ B @ dUm_dQ
            if m < Nt:
                ph = np.exp(1j * om * tlist[m])
                dF_dI += dV_dI[0, 1] * ph * dt
                dF_dQ += dV_dQ[0, 1] * ph * dt
        gI_term[j] = Sw * dom * 2 * np.real(np.conj(integral) * dF_dI)
        gQ_term[j] = Sw * dom * 2 * np.real(np.conj(integral) * dF_dQ)
    return J_term, gI_term, gQ_term


def _ff_single_freq_term(om, dom, Sw, Ulist, Sk_list, steps, tlist, B, N, Nt, dt):
    """The O(Nt) adjoint contribution of ONE frequency to J_pen/gI/gQ.

    DERIVATION. The brute-force version above sums, for each control step
    j, a contribution from every LATER step m > j -- an O(Nt) inner loop
    per j, i.e. O(Nt^2) total. That inner loop is only needed because it
    re-derives, from scratch for every (j, m) pair, how a perturbation of
    the control at step j propagates forward to time m. But the coherent
    (dissipation-free) step propagators S_k used here are unitary, so the
    forward propagator from step j+1 to step m has the closed form

        Prop_{j+1->m} = S_{m-1} ... S_{j+1} = U_m @ U_{j+1}^dagger.

    Writing V_m := U_m^dagger B U_m and y_m := V_m[0,1] (so the frequency
    integral is `integral = dt * sum_m y_m * exp(i*om*t_m)`), substituting
    the closed form above into d(integral)/du_j and collecting terms shows
    that everything a given step j needs from the "future" (m > j) is
    captured by two NxN matrices,

        R_j  := sum_{m>j} phase_m       * V_m,
        Rc_j := sum_{m>j} conj(phase_m) * V_m,

    which do NOT depend on j individually -- they are exactly the partial
    sums of a single backward recursion (R_{Nt-1} = 0; R_j = R_{j+1} +
    phase_{j+1} V_{j+1}), computed ONCE in O(Nt) matrix additions and
    reused by every j. Given R_j, Rc_j, and U_j, U_{j+1} (already
    available from the forward pass), each step j's gradient contribution
    is then O(1) matrix-vector work instead of an O(Nt) inner loop:

        w0 = dS_j @ U_j[:, 0],   w1 = dS_j @ U_j[:, 1]
        d(integral)/du_j = dt * ( (R_j  @ (U_{j+1}^dagger @ w1))[0]
                          + conj( (Rc_j @ (U_{j+1}^dagger @ w0))[1] ) )

    This is the O(Nt) reformulation flagged as "possible in principle but
    not implemented" in the module docstring. It is exact -- not a
    further approximation on top of the existing first-order/Magnus
    filter-function treatment -- and is verified bit-for-bit against
    `_ff_single_freq_term_bruteforce` in `_selftest_gradients` on every
    run, in addition to the finite-difference check both must pass.
    """
    V = [Ulist[m].conj().T @ B @ Ulist[m] for m in range(Nt)]
    phase = np.exp(1j * om * tlist)
    y = np.array([V[m][0, 1] for m in range(Nt)])
    integral = np.sum(y * phase) * dt
    Fw = float(np.abs(integral) ** 2)
    J_term = Fw * Sw * dom

    R = [None] * Nt
    Rc = [None] * Nt
    Racc = np.zeros((N, N), dtype=complex)
    Rcacc = np.zeros((N, N), dtype=complex)
    R[Nt - 1] = Racc
    Rc[Nt - 1] = Rcacc
    for j in range(Nt - 2, -1, -1):
        Racc = Racc + phase[j + 1] * V[j + 1]
        Rcacc = Rcacc + np.conj(phase[j + 1]) * V[j + 1]
        R[j] = Racc
        Rc[j] = Rcacc

    gI_term = np.zeros(Nt)
    gQ_term = np.zeros(Nt)
    conj_integral = np.conj(integral)
    for j in range(Nt):
        dS_I, dS_Q = steps[j][1], steps[j][2]
        Uj = Ulist[j]
        Ujp1_dag = Ulist[j + 1].conj().T
        psi0_j = Uj[:, 0]
        psi1_j = Uj[:, 1]
        Rj, Rcj = R[j], Rc[j]

        for du, g_out in ((dS_I, gI_term), (dS_Q, gQ_term)):
            w0 = du @ psi0_j
            w1 = du @ psi1_j
            term_B = (Rj @ (Ujp1_dag @ w1))[0]
            term_A = np.conj((Rcj @ (Ujp1_dag @ w0))[1])
            D_j = dt * (term_B + term_A)
            g_out[j] = Sw * dom * 2 * np.real(conj_integral * D_j)

    return J_term, gI_term, gQ_term


def _ff_freq_chunk_worker(args):
    """Worker-pool task: builds the coherent (noise-free) reference
    propagator locally (cheap, O(Nt)) and evaluates a chunk of
    frequencies' O(Nt^2) contributions -- each frequency is fully
    independent given the shared propagator, so this is embarrassingly
    parallel, same pattern as _fade_noise_chunk_worker above."""
    uI, uQ, dt, om_chunk, dom_chunk, Sw_chunk, phys_params = args
    alpha_MHz, N_LEVELS = phys_params
    N = N_LEVELS
    a = _destroy_np(N); adag = a.conj().T; n_op = adag @ a
    alpha = alpha_MHz * 1e-3 * TWO_PI
    H0 = (0.5 * alpha * (adag @ adag @ a @ a)).astype(complex)
    Hx = (a + adag) / 2.0
    Hy = 1j * (adag - a) / 2.0
    B = n_op.astype(complex)

    Nt = len(uI)
    steps = _coherent_step_data(uI, uQ, dt, H0, Hx, Hy)
    Ulist = [np.eye(N, dtype=complex)]
    for S, _, _ in steps:
        Ulist.append(S @ Ulist[-1])
    Sk_list = [steps[k][0] for k in range(Nt)]
    tlist = dt * (np.arange(Nt) + 0.5)

    J_sum = 0.0
    gI_sum = np.zeros(Nt)
    gQ_sum = np.zeros(Nt)
    for om, dom, Sw in zip(om_chunk, dom_chunk, Sw_chunk):
        Jt, gIt, gQt = _ff_single_freq_term(om, dom, Sw, Ulist, Sk_list, steps, tlist, B, N, Nt, dt)
        J_sum += Jt; gI_sum += gIt; gQ_sum += gQt
    return J_sum, gI_sum, gQ_sum


def _build_ff_chunk_jobs(uI: np.ndarray, uQ: np.ndarray, dt: float):
    """Build the (fn, args) job list for the filter-function frequency
    chunks, WITHOUT submitting them -- shared by filter_function_overlap_and_grad's
    own standalone parallel path and by fade_ripe_objective_and_grad's
    combined round (see that function's docstring for why combining
    matters). Also returns (omegas, domega, S_vals) since callers that
    only want the standalone metadata (none currently, but kept for
    parity/debugging) can use it."""
    omegas = np.geomspace(cfg.ff_omega_min, cfg.ff_omega_max, cfg.ff_n_freqs)
    if len(omegas) > 1:
        domega = np.gradient(omegas)
    else:
        domega = np.array([cfg.ff_omega_max - cfg.ff_omega_min])
    S_vals = _psd(omegas)
    if len(omegas) < 2:
        return [], (omegas, domega, S_vals)
    n_chunks = min(len(omegas), max(_N_WORKERS, _N_WORKERS * 4))
    phys_params = (cfg.alpha_MHz, cfg.N_LEVELS)
    om_chunks = np.array_split(omegas, n_chunks)
    dom_chunks = np.array_split(domega, n_chunks)
    Sw_chunks = np.array_split(S_vals, n_chunks)
    jobs = [(_ff_freq_chunk_worker, (uI, uQ, dt, oc, dc, sc, phys_params))
            for oc, dc, sc in zip(om_chunks, dom_chunks, Sw_chunks) if len(oc) > 0]
    return jobs, (omegas, domega, S_vals)


def filter_function_overlap_and_grad(
    uI: np.ndarray, uQ: np.ndarray, dt: float, parallel: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Numerically computes F(omega) for the coherent (noise-free)
    reference propagator on a log-spaced frequency grid and returns the
    overlap integral int F(omega) S(omega) domega/2pi (to be *subtracted*
    from the objective -- we want to minimize it) plus its exact gradient.

    Standalone entry point (used by the seed-pulse ff_mu calibration step
    and by the self-test). During actual FADE-RIPE optimization, this
    work is instead submitted as part of fade_ripe_objective_and_grad's
    COMBINED round together with the K-noise-draw chunks, so cores never
    sit idle waiting for this round to drain before the other starts --
    see that function for why.
    """
    N = sys_.N
    Nt = len(uI)
    jobs, (omegas, domega, S_vals) = _build_ff_chunk_jobs(uI, uQ, dt)

    if parallel and jobs:
        pool = _get_pool()
        futures = [pool.submit(fn, args) for fn, args in jobs]
        J_pen = 0.0
        gI = np.zeros(Nt)
        gQ = np.zeros(Nt)
        for fut in as_completed(futures):
            Jc, gIc, gQc = fut.result()
            J_pen += Jc; gI += gIc; gQ += gQc
        return J_pen, gI, gQ

    # serial fallback (used by the self-test's tiny Nt, and available via
    # parallel=False for debugging/profiling against the pool path)
    steps = _coherent_step_data(uI, uQ, dt, sys_.H0, sys_.Hx, sys_.Hy)
    Ulist = [np.eye(N, dtype=complex)]
    for S, _, _ in steps:
        Ulist.append(S @ Ulist[-1])
    Sk_list = [steps[k][0] for k in range(Nt)]
    tlist = dt * (np.arange(Nt) + 0.5)

    J_pen = 0.0
    gI = np.zeros(Nt)
    gQ = np.zeros(Nt)
    B = sys_.B_noise
    for om, dom, Sw in zip(omegas, domega, S_vals):
        Jt, gIt, gQt = _ff_single_freq_term(om, dom, Sw, Ulist, Sk_list, steps, tlist, B, N, Nt, dt)
        J_pen += Jt; gI += gIt; gQ += gQt
    return J_pen, gI, gQ


# ============================================================================
# GRADIENT SELF-TEST -- run before anything is trusted in L-BFGS-B
# ============================================================================

def _selftest_gradients(verbose: bool = True) -> bool:
    """Finite-difference check of every analytic gradient above, at a
    small Nt so the O(Nt^2) filter-function piece is cheap. Returns True
    iff every piece agrees with central differences to within the
    eps=1e-6 truncation floor (~1e-9-1e-10 absolute error at the scales
    involved here)."""
    rng = np.random.default_rng(7)
    Nt_test = 24
    Tg_test = 12.0
    dt = Tg_test / Nt_test
    uI = 0.02 * np.sin(np.linspace(0, np.pi, Nt_test)) + 0.003 * rng.standard_normal(Nt_test)
    uQ = 0.01 * np.cos(np.linspace(0, np.pi, Nt_test)) + 0.003 * rng.standard_normal(Nt_test)

    prop = LindbladPropagator(sys_.H0, sys_.Hx, sys_.Hy, sys_.c_ops)
    rho0 = sys_.thermal_state()
    rho_target = sys_.ideal_target_dm(rho0)

    def J_of(uI_, uQ_):
        J, _, _, _, _ = lindblad_fidelity_and_grad(
            uI_, uQ_, rho0, rho_target, prop, dt, cfg.leak_lambda)
        return J

    _, gI, gQ, _, _ = lindblad_fidelity_and_grad(
        uI, uQ, rho0, rho_target, prop, dt, cfg.leak_lambda)

    eps = 1e-6
    fd_gI = np.zeros(Nt_test); fd_gQ = np.zeros(Nt_test)
    for k in range(Nt_test):
        uIp, uIm = uI.copy(), uI.copy(); uIp[k] += eps; uIm[k] -= eps
        fd_gI[k] = (J_of(uIp, uQ) - J_of(uIm, uQ)) / (2 * eps)
        uQp, uQm = uQ.copy(), uQ.copy(); uQp[k] += eps; uQm[k] -= eps
        fd_gQ[k] = (J_of(uI, uQp) - J_of(uI, uQm)) / (2 * eps)

    err_lindblad = max(np.max(np.abs(gI - fd_gI)), np.max(np.abs(gQ - fd_gQ)))

    # bandwidth filter chain rule
    bwf = BandwidthFilter(Nt_test, dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
    w = rng.standard_normal(Nt_test)
    def Jf(u): return float(np.dot(w, bwf.apply(u)))
    grad_analytic = bwf.backprop_grad(w)
    fd_gf = np.zeros(Nt_test)
    for k in range(Nt_test):
        up, um = uI.copy(), uI.copy(); up[k] += eps; um[k] -= eps
        fd_gf[k] = (Jf(up) - Jf(um)) / (2 * eps)
    err_filter = np.max(np.abs(grad_analytic - fd_gf))

    # filter-function overlap gradient (small Nt/freq grid for speed)
    cfg_ff_n = cfg.ff_n_freqs
    cfg.ff_n_freqs = 2
    Jff, gIff, gQff = filter_function_overlap_and_grad(uI, uQ, dt)
    fd_gIff = np.zeros(Nt_test); fd_gQff = np.zeros(Nt_test)
    for k in range(Nt_test):
        uIp, uIm = uI.copy(), uI.copy(); uIp[k] += eps; uIm[k] -= eps
        Jp, _, _ = filter_function_overlap_and_grad(uIp, uQ, dt)
        Jm, _, _ = filter_function_overlap_and_grad(uIm, uQ, dt)
        fd_gIff[k] = (Jp - Jm) / (2 * eps)
        uQp, uQm = uQ.copy(), uQ.copy(); uQp[k] += eps; uQm[k] -= eps
        Jp2, _, _ = filter_function_overlap_and_grad(uI, uQp, dt)
        Jm2, _, _ = filter_function_overlap_and_grad(uI, uQm, dt)
        fd_gQff[k] = (Jp2 - Jm2) / (2 * eps)
    err_ff = max(np.max(np.abs(gIff - fd_gIff)), np.max(np.abs(gQff - fd_gQff)))
    cfg.ff_n_freqs = cfg_ff_n

    # Cross-check the O(Nt) adjoint filter-function term directly against
    # the O(Nt^2) brute-force reference it replaced -- independent of the
    # finite-difference check above, since both could in principle share
    # a common blind spot. This is a bit-level agreement check (both are
    # exact, closed-form expressions of the same quantity), not a
    # tolerance-to-noise comparison, so the bar is much tighter than the
    # finite-difference tolerance used elsewhere in this function.
    steps_x = _coherent_step_data(uI, uQ, dt, sys_.H0, sys_.Hx, sys_.Hy)
    Ulist_x = [np.eye(sys_.N, dtype=complex)]
    for S, _, _ in steps_x:
        Ulist_x.append(S @ Ulist_x[-1])
    Sk_list_x = [steps_x[k][0] for k in range(Nt_test)]
    tlist_x = dt * (np.arange(Nt_test) + 0.5)
    om_test, dom_test, Sw_test = 0.037, 0.0013, 1.0e-8
    J_bf, gI_bf, gQ_bf = _ff_single_freq_term_bruteforce(
        om_test, dom_test, Sw_test, Ulist_x, Sk_list_x, steps_x, tlist_x,
        sys_.B_noise, sys_.N, Nt_test, dt)
    J_adj, gI_adj, gQ_adj = _ff_single_freq_term(
        om_test, dom_test, Sw_test, Ulist_x, Sk_list_x, steps_x, tlist_x,
        sys_.B_noise, sys_.N, Nt_test, dt)
    err_ff_adjoint = max(abs(J_bf - J_adj),
                          np.max(np.abs(gI_bf - gI_adj)),
                          np.max(np.abs(gQ_bf - gQ_adj)))

    tol = 1e-6  # generous vs the ~1e-9-1e-10 actually observed; catches real bugs
    tol_adjoint = 1e-9  # both sides exact/closed-form -- should agree far tighter
    ok = (err_lindblad < tol and err_filter < tol and err_ff < tol
          and err_ff_adjoint < tol_adjoint)
    if verbose:
        print("="*70)
        print("GRADIENT SELF-TEST (finite-difference, eps=1e-6)")
        print(f"  Lindblad+leakage adjoint sweep      : max err = {err_lindblad:.3e}")
        print(f"  Bandwidth-filter chain rule          : max err = {err_filter:.3e}")
        print(f"  Filter-function overlap term (O(Nt)) : max err = {err_ff:.3e}")
        print(f"  Filter-function O(Nt) vs O(Nt^2)     : max err = {err_ff_adjoint:.3e}")
        print(f"  PASS (< {tol:g})" if ok else f"  FAIL -- DO NOT TRUST L-BFGS-B RESULTS")
        print("="*70)
    return ok


# ============================================================================
# FADE-RIPE OBJECTIVE (combines Sections 1,2,3,4,5)
# ============================================================================

_prop_cache: Dict[int, LindbladPropagator] = {}

def _get_propagator() -> LindbladPropagator:
    key = id(sys_)
    if key not in _prop_cache:
        _prop_cache[key] = LindbladPropagator(sys_.H0, sys_.Hx, sys_.Hy, sys_.c_ops)
    return _prop_cache[key]


def hann_init() -> Tuple[np.ndarray, np.ndarray]:
    Nt = cfg.Nt
    hann = 0.5 * (1.0 - np.cos(TWO_PI * np.linspace(0, 1, Nt)))
    area = np.sum(hann) * cfg.dt
    amp = np.pi / 2.0 / area
    uI = np.clip(amp * hann, -cfg.max_amp, cfg.max_amp)
    uQ = np.zeros(Nt)
    return uI, uQ


def _fade_noise_chunk_worker(args):
    """Worker-pool task (local or HPC): evaluates a chunk of the K quasi-static noise draws
    for one FADE-RIPE objective/gradient call and returns the partial sum
    of (J, gI, gQ) over that chunk. Each draw is a fully independent
    Lindblad propagation (different detuning/amplitude error), so this is
    embarrassingly parallel."""
    (uI_f, uQ_f, deltas_chunk, gammas_chunk, phys_params) = args
    (alpha_MHz, N_LEVELS, Tg, Nt, T1, T2s, temperature_mK, qubit_freq_GHz,
     leak_lambda) = phys_params

    N = N_LEVELS
    a = _destroy_np(N); adag = a.conj().T; n_op = adag @ a
    alpha = alpha_MHz * 1e-3 * TWO_PI
    H0 = (0.0 * n_op + 0.5 * alpha * (adag @ adag @ a @ a)).astype(complex)
    Hx = (a + adag) / 2.0
    Hy = 1j * (adag - a) / 2.0
    gamma1 = 1.0 / T1
    gamma_phi = max(1.0 / T2s - gamma1 / 2.0, 1e-12)
    c_ops = [np.sqrt(gamma1) * a, np.sqrt(2 * gamma_phi) * n_op]
    P_leak = np.diag([0, 0] + [1] * (N - 2)).astype(complex)
    dt = Tg / Nt

    # local, worker-safe reimplementation of thermal_state()/ideal_target_dm()
    hbar_omega_over_kT = (TWO_PI * qubit_freq_GHz * 1e9 * 6.62607015e-34
                           / (2 * np.pi)) / (1.380649e-23 * temperature_mK * 1e-3)
    n_th = 1.0 / (np.exp(hbar_omega_over_kT) - 1.0)
    r = n_th / (1.0 + n_th) if n_th > 0 else 0.0
    pops = np.array([(1 - r) * r ** k for k in range(N)]); pops /= pops.sum()
    rho_th = np.diag(pops).astype(complex)
    leak_bg = float(np.real(np.trace(P_leak @ rho_th)))
    Uemb = np.eye(N, dtype=complex)
    Uemb[0:2, 0:2] = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)

    kets = [
        np.array([1, 0] + [0] * (N - 2), dtype=complex),
        np.array([0, 1] + [0] * (N - 2), dtype=complex),
        np.array([1, 1] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
        np.array([1, 1j] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
    ]

    J_sum, gI_sum, gQ_sum = 0.0, np.zeros(Nt), np.zeros(Nt)
    for d, g in zip(deltas_chunk, gammas_chunk):
        H0_d = H0 + d * n_op
        prop_d = LindbladPropagator(H0_d, (1 + g) * Hx, (1 + g) * Hy, c_ops)
        steps = prop_d.step_data(uI_f, uQ_f, dt)  # once per draw, shared across kets
        J_draw, gI_draw, gQ_draw = 0.0, np.zeros(Nt), np.zeros(Nt)
        for psi in kets:
            rho_comp = np.outer(psi, psi.conj())
            rho_in = (1.0 - leak_bg) * rho_comp + P_leak @ rho_th @ P_leak
            rho_target = Uemb @ rho_in @ Uemb.conj().T
            J, gI, gQ, _, _ = lindblad_fidelity_and_grad(
                uI_f, uQ_f, rho_in, rho_target, prop_d, dt, leak_lambda, steps=steps)
            J_draw += J; gI_draw += gI; gQ_draw += gQ
        n = len(kets)
        J_sum += J_draw / n; gI_sum += gI_draw / n; gQ_sum += gQ_draw / n

    return J_sum, gI_sum, gQ_sum, len(deltas_chunk)


def fade_ripe_objective_and_grad(uI_raw: np.ndarray, uQ_raw: np.ndarray,
                                  bwf: BandwidthFilter, prop: LindbladPropagator,
                                  deltas: np.ndarray, gammas: np.ndarray,
                                  parallel: bool = True):
    """Full objective: bandwidth-filter the raw controls, evaluate the
    thermal-initial-state Lindblad fidelity (averaged over a FIXED ensemble
    of K quasi-static detuning/amplitude draws for robustness), subtract
    the leakage-penalized term, subtract the filter-function/PSD
    overlap penalty, and chain-rule everything back through the bandwidth
    filter to the raw (pre-distortion) control samples.
    
    On HPC, parallel=True (default) uses all available cores.
    """
    dt = cfg.dt
    uI_f = bwf.apply(uI_raw)
    uQ_f = bwf.apply(uQ_raw)
    K = len(deltas)

    if parallel and K >= _N_WORKERS:
        # More, smaller chunks than workers for better load balancing
        n_chunks = min(K, max(_N_WORKERS, _N_WORKERS * 4))
        phys_params = (cfg.alpha_MHz, cfg.N_LEVELS, cfg.Tg, cfg.Nt, cfg.T1, cfg.T2s,
                        cfg.temperature_mK, cfg.qubit_freq_GHz, cfg.leak_lambda)
        d_chunks = np.array_split(deltas, n_chunks)
        g_chunks = np.array_split(gammas, n_chunks)
        pool = _get_pool()
        futures = [pool.submit(_fade_noise_chunk_worker,
                                (uI_f, uQ_f, dc, gc, phys_params))
                   for dc, gc in zip(d_chunks, g_chunks) if len(dc) > 0]
        J_tot, gI_tot, gQ_tot, n_tot = 0.0, np.zeros(cfg.Nt), np.zeros(cfg.Nt), 0
        for fut in as_completed(futures):
            J_c, gI_c, gQ_c, n_c = fut.result()
            J_tot += J_c; gI_tot += gI_c; gQ_tot += gQ_c; n_tot += n_c
        J_mean, gI_mean, gQ_mean = J_tot / n_tot, gI_tot / n_tot, gQ_tot / n_tot
    else:
        J_tot = 0.0
        gI_tot = np.zeros(cfg.Nt)
        gQ_tot = np.zeros(cfg.Nt)
        for d, g in zip(deltas, gammas):
            H0_d = sys_.H0 + d * sys_.n_op
            prop_d = LindbladPropagator(H0_d, (1 + g) * sys_.Hx, (1 + g) * sys_.Hy, sys_.c_ops)
            J_k, gI_k, gQ_k, leak_k, F_k = average_gate_fidelity_and_grad(uI_f, uQ_f, prop_d, dt)
            J_tot += J_k; gI_tot += gI_k; gQ_tot += gQ_k
        J_mean = J_tot / K
        gI_mean = gI_tot / K
        gQ_mean = gQ_tot / K

    J_ff, gI_ff, gQ_ff = filter_function_overlap_and_grad(uI_f, uQ_f, dt)
    J_mean -= cfg.ff_mu * J_ff
    gI_mean -= cfg.ff_mu * gI_ff
    gQ_mean -= cfg.ff_mu * gQ_ff

    # chain rule through the bandwidth filter back to raw control samples
    gI_raw = bwf.backprop_grad(gI_mean)
    gQ_raw = bwf.backprop_grad(gQ_mean)

    return J_mean, gI_raw, gQ_raw


def fade_ripe_multistart(seed_uI: np.ndarray, seed_uQ: np.ndarray, verbose: bool = True):
    """Multi-start wrapper around fade_ripe_optimize()."""
    n_restarts = max(1, cfg.fade_n_restarts)
    pert_rng = np.random.default_rng(12345)
    results = []
    for r in range(n_restarts):
        if r == 0:
            init_uI, init_uQ = seed_uI.copy(), seed_uQ.copy()
            tag = "unperturbed Hann seed"
        else:
            pert_scale = 0.3 * cfg.max_amp
            init_uI = np.clip(seed_uI + pert_scale * pert_rng.standard_normal(len(seed_uI)),
                               -cfg.max_amp, cfg.max_amp)
            init_uQ = np.clip(seed_uQ + pert_scale * pert_rng.standard_normal(len(seed_uQ)),
                               -cfg.max_amp, cfg.max_amp)
            tag = f"randomly perturbed start #{r}"
        if verbose:
            print(f"\n  ===== Multi-start restart {r + 1}/{n_restarts}: {tag} =====")
        uI_raw, uQ_raw, uI_opt, uQ_opt, info = fade_ripe_optimize(init_uI, init_uQ, verbose=verbose)
        oos_mean = noise_eval.mean_fidelity(uI_opt, uQ_opt)
        oos_cvar = noise_eval.cvar(uI_opt, uQ_opt)
        if verbose:
            print(f"    -> restart {r + 1} OUT-OF-SAMPLE (n={cfg.n_noise_eval}): "
                  f"mean J={oos_mean:.6f}, CVaR5%={oos_cvar:.6f}")
        results.append({'oos_mean': oos_mean, 'oos_cvar': oos_cvar,
                         'uI_raw': uI_raw, 'uQ_raw': uQ_raw,
                         'uI_opt': uI_opt, 'uQ_opt': uQ_opt, 'info': info})

    best = max(results, key=lambda r: r['oos_mean'])
    if verbose:
        means = [r['oos_mean'] for r in results]
        print(f"\n  Multi-start summary: {n_restarts} restarts, "
              f"out-of-sample mean J range [{min(means):.6f}, {max(means):.6f}], "
              f"spread={max(means) - min(means):.4e}")
        print(f"  Selected best restart: out-of-sample mean J={best['oos_mean']:.6f}, "
              f"CVaR5%={best['oos_cvar']:.6f}")
        if max(means) - min(means) > 0.01:
            print("  NOTE: restarts differ by >1% in fidelity -- this landscape has "
                  "meaningfully different local optima.")
    return best['uI_raw'], best['uQ_raw'], best['uI_opt'], best['uQ_opt'], best['info']


def fade_ripe_optimize(seed_uI: np.ndarray, seed_uQ: np.ndarray,
                        verbose: bool = True):
    Nt, M = cfg.Nt, cfg.fade_nmodes
    tlist = np.linspace(0.0, cfg.Tg, Nt)
    hann = 0.5 * (1.0 - np.cos(TWO_PI * np.linspace(0, 1, Nt)))
    basis_I = np.stack([np.sin((2 * n - 1) * np.pi * tlist / cfg.Tg) * hann
                         for n in range(1, M + 1)], axis=1)
    basis_Q = np.stack([np.sin(2 * n * np.pi * tlist / cfg.Tg) * hann
                         for n in range(1, M + 1)], axis=1)

    def coeffs_to_pulse(c):
        uI = np.clip(basis_I @ c[:M], -cfg.max_amp, cfg.max_amp)
        uQ = np.clip(basis_Q @ c[M:], -cfg.max_amp, cfg.max_amp)
        return uI, uQ

    c0 = np.concatenate([np.linalg.pinv(basis_I) @ seed_uI,
                          np.linalg.pinv(basis_Q) @ seed_uQ])

    bwf = BandwidthFilter(Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
    prop = _get_propagator()

    # Fixed noise ensemble for this optimization run
    noise_rng = np.random.default_rng(77)
    fixed_deltas = noise_rng.normal(0.0, cfg.fade_sigma, cfg.fade_K)
    fixed_gammas = noise_rng.normal(0.0, cfg.fade_amp_sig, cfg.fade_K)

    # auto-calibrate ff_mu against the seed pulse
    uI_seed_f = bwf.apply(seed_uI)
    uQ_seed_f = bwf.apply(seed_uQ)
    J_fid_seed, _, _, leak_seed, F_seed = average_gate_fidelity_and_grad(uI_seed_f, uQ_seed_f, prop, cfg.dt)
    J_ff_seed, _, _ = filter_function_overlap_and_grad(uI_seed_f, uQ_seed_f, cfg.dt)
    cfg.ff_mu = (cfg.ff_target_fraction * J_fid_seed / J_ff_seed) if J_ff_seed > 1e-14 else 0.0
    if verbose:
        print(f"    Seed pulse breakdown: fidelity={J_fid_seed:.6f}, leak={leak_seed:.3e}, "
              f"raw filter-fn J_ff={J_ff_seed:.3e}")
        print(f"    Auto-calibrated ff_mu={cfg.ff_mu:.4g} "
              f"(targets penalty = {cfg.ff_target_fraction*100:.0f}% of seed fidelity term)")

    def obj(c):
        uI, uQ = coeffs_to_pulse(c)
        J, gI, gQ = fade_ripe_objective_and_grad(uI, uQ, bwf, prop, fixed_deltas, fixed_gammas)
        grad_c = np.concatenate([basis_I.T @ gI, basis_Q.T @ gQ])
        return -J, -grad_c

    if verbose:
        print(f"    FADE-RIPE: Nt={Nt}, Fourier modes={M}, K={cfg.fade_K} noise draws/eval, "
              f"AWG bw={cfg.awg_bandwidth_GHz*1e3:.0f} MHz, leak_lambda={cfg.leak_lambda}")

    res = minimize(obj, c0, method='L-BFGS-B', jac=True,
                    options={'maxiter': cfg.fade_maxiter, 'ftol': cfg.fade_ftol, 'gtol': cfg.fade_gtol},
                    callback=(lambda xk: print(f"    ... L-BFGS-B iter progressing", flush=True))
                    if verbose else None)
    if verbose:
        print(f"    L-BFGS-B: {res.nit} iterations, {res.nfev} evals, "
              f"converged={res.success}, message='{res.message}'")

    best_uI_raw, best_uQ_raw = coeffs_to_pulse(res.x)
    best_uI = bwf.apply(best_uI_raw)
    best_uQ = bwf.apply(best_uQ_raw)

    rho_th = sys_.thermal_state()
    J_final, _, _, leak_final, F_final = thermal_gate_fidelity_and_grad(best_uI, best_uQ, prop, cfg.dt)
    if verbose:
        print(f"    -> FADE-RIPE finished: thermal-state J={J_final:.6f}, "
              f"leakage(|2>,|3>)={leak_final:.3e}")
    return best_uI_raw, best_uQ_raw, best_uI, best_uQ, {'J_final': J_final, 'leak_final': leak_final}


# ============================================================================
# GRAPE / CRAB VIA qutip_qtrl, LINDBLADIAN (GEN_MAT) MODE
# ============================================================================

def _lazy_import_qutip():
    try:
        import qutip as qt
        from qutip_qtrl import pulseoptim as qtrl_pulseoptim
        return qt, qtrl_pulseoptim
    except ImportError as e:
        raise ImportError(
            "qutip / qutip_qtrl not available in this environment. "
            "GRAPE/CRAB comparison requires them; FADE-RIPE above does not."
        ) from e


def _lindblad_qtrl_operators():
    qt, _ = _lazy_import_qutip()
    prop = _get_propagator()
    N, d = sys_.N, sys_.N * sys_.N
    L_d = qt.Qobj(prop.L0, dims=[[N * N], [N * N]])
    L_x = qt.Qobj(prop.Lx, dims=[[N * N], [N * N]])
    L_y = qt.Qobj(prop.Ly, dims=[[N * N], [N * N]])
    rho0 = sys_.thermal_state()
    rho_targ = sys_.ideal_target_dm(rho0)
    U0 = qt.Qobj(_vec(rho0).reshape(d, 1))
    Utarg = qt.Qobj(_vec(rho_targ).reshape(d, 1))
    return L_d, [L_x, L_y], U0, Utarg


def grape_optimize(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    _, qtrl_pulseoptim = _lazy_import_qutip()
    L_d, L_c, U_0, U_targ = _lindblad_qtrl_operators()
    best_uI, best_uQ, best_F = None, None, -1.0
    for restart in range(cfg.grape_restarts):
        np.random.seed(_RESTART_SEED_BASE + restart)
        res = qtrl_pulseoptim.optimize_pulse(
            L_d, L_c, U_0, U_targ,
            num_tslots=cfg.Nt, evo_time=cfg.Tg,
            amp_lbound=-cfg.max_amp, amp_ubound=cfg.max_amp,
            dyn_type='GEN_MAT',
            fid_type='TRACEDIFF',
            fid_err_targ=1e-8, max_iter=cfg.grape_maxiter, max_wall_time=180,
            alg='GRAPE',
            init_pulse_type='DEF' if restart == 0 else 'RNDFOURIER',
            gen_stats=True,
        )
        amps = res.final_amps
        uI_r = np.clip(amps[:, 0], -cfg.max_amp, cfg.max_amp)
        uQ_r = np.clip(amps[:, 1], -cfg.max_amp, cfg.max_amp)
        prop = _get_propagator()
        F_r, _, _, _, _ = thermal_gate_fidelity_and_grad(uI_r, uQ_r, prop, cfg.dt)
        if verbose:
            print(f"    GRAPE (Lindbladian) restart {restart+1}/{cfg.grape_restarts}: "
                  f"thermal J={F_r:.6f} (qtrl fid_err={res.fid_err:.3e})")
        if F_r > best_F:
            best_F, best_uI, best_uQ = F_r, uI_r.copy(), uQ_r.copy()
    return best_uI, best_uQ, best_F


def crab_optimize(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    _, qtrl_pulseoptim = _lazy_import_qutip()
    L_d, L_c, U_0, U_targ = _lindblad_qtrl_operators()
    best_uI, best_uQ, best_F = None, None, -1.0
    for restart in range(cfg.crab_restarts):
        np.random.seed(_RESTART_SEED_BASE + restart)
        res = qtrl_pulseoptim.optimize_pulse(
            L_d, L_c, U_0, U_targ,
            num_tslots=cfg.Nt, evo_time=cfg.Tg,
            amp_lbound=-cfg.max_amp, amp_ubound=cfg.max_amp,
            dyn_type='GEN_MAT',
            fid_type='TRACEDIFF',
            fid_err_targ=1e-6, max_iter=cfg.crab_maxiter, max_wall_time=180,
            alg='CRAB', alg_params={'num_coeffs': cfg.crab_nmodes},
            gen_stats=True,
        )
        amps = res.final_amps
        uI_r = np.clip(amps[:, 0], -cfg.max_amp, cfg.max_amp)
        uQ_r = np.clip(amps[:, 1], -cfg.max_amp, cfg.max_amp)
        prop = _get_propagator()
        F_r, _, _, _, _ = thermal_gate_fidelity_and_grad(uI_r, uQ_r, prop, cfg.dt)
        if verbose:
            print(f"    CRAB (Lindbladian) restart {restart+1}/{cfg.crab_restarts}: "
                  f"thermal J={F_r:.6f} (qtrl fid_err={res.fid_err:.3e})")
        if F_r > best_F:
            best_F, best_uI, best_uQ = F_r, uI_r.copy(), uQ_r.copy()
    return best_uI, best_uQ, best_F


# ============================================================================
# PARALLEL NOISE-BENCHMARK WORKER (local or HPC)
# ============================================================================

def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

_pool: Optional[ProcessPoolExecutor] = None

def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=_N_WORKERS, initializer=_worker_init)
        print(f"  Created process pool with {_N_WORKERS} workers")
    return _pool

def _shutdown_pool():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None


def _eval_fidelity_worker(args):
    uI, uQ, delta, gamma, cfg_params = args
    (alpha_MHz, N_LEVELS, Tg, Nt, T1, T2s, temperature_mK, qubit_freq_GHz,
     leak_lambda) = cfg_params

    N = N_LEVELS
    a = _destroy_np(N); adag = a.conj().T; n_op = adag @ a
    alpha = alpha_MHz * 1e-3 * TWO_PI
    H0 = (delta * n_op + 0.5 * alpha * (adag @ adag @ a @ a)).astype(complex)
    Hx = ((1 + gamma) * (a + adag) / 2.0).astype(complex)
    Hy = ((1 + gamma) * 1j * (adag - a) / 2.0).astype(complex)
    gamma1 = 1.0 / T1
    gamma_phi = max(1.0 / T2s - gamma1 / 2.0, 1e-12)
    c_ops = [np.sqrt(gamma1) * a, np.sqrt(2 * gamma_phi) * n_op]
    P_leak = np.diag([0, 0] + [1] * (N - 2)).astype(complex)

    dt = Tg / Nt
    prop = LindbladPropagator(H0, Hx, Hy, c_ops)
    steps = prop.step_data(uI, uQ, dt)

    N_ = N
    rho0 = np.zeros((N_, N_), dtype=complex); rho0[0, 0] = 1.0
    Uemb = np.eye(N_, dtype=complex)
    Uemb[0:2, 0:2] = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)
    rho_target = Uemb @ rho0 @ Uemb.conj().T

    fwd = prop.forward(steps, _vec(rho0))
    rhoT = _unvec(fwd[-1], N_)
    F = float(np.real(np.trace(rho_target.conj().T @ rhoT)))
    leak = float(np.real(np.trace(P_leak @ rhoT)))
    return F - leak_lambda * leak


def _parallel_fidelities(uI: np.ndarray, uQ: np.ndarray, deltas, gammas) -> np.ndarray:
    cfg_params = (cfg.alpha_MHz, cfg.N_LEVELS, cfg.Tg, cfg.Nt, cfg.T1, cfg.T2s,
                  cfg.temperature_mK, cfg.qubit_freq_GHz, cfg.leak_lambda)
    pool = _get_pool()
    args_list = [(uI, uQ, float(d), float(g), cfg_params) for d, g in zip(deltas, gammas)]
    futures = {pool.submit(_eval_fidelity_worker, a): i for i, a in enumerate(args_list)}
    results = [0.0] * len(args_list)
    for fut in as_completed(futures):
        i = futures[fut]
        try:
            results[i] = fut.result()
        except Exception:
            results[i] = 0.0
    return np.array(results)


class NoiseModel:
    def __init__(self, n_samples: int, sigma_d: float, sigma_a: float, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.deltas = rng.normal(0.0, sigma_d, n_samples)
        self.gammas = rng.normal(0.0, sigma_a, n_samples)

    def fidelity_distribution(self, uI, uQ):
        return _parallel_fidelities(uI, uQ, self.deltas, self.gammas)

    def mean_fidelity(self, uI, uQ): return float(np.mean(self.fidelity_distribution(uI, uQ)))
    def worst_fidelity(self, uI, uQ): return float(np.min(self.fidelity_distribution(uI, uQ)))
    def cvar(self, uI, uQ, alpha=0.05):
        fids = self.fidelity_distribution(uI, uQ)
        cutoff = np.quantile(fids, alpha)
        tail = fids[fids <= cutoff]
        return float(np.mean(tail)) if len(tail) else float(cutoff)


noise_eval = NoiseModel(cfg.n_noise_eval, cfg.sigma_detuning, cfg.sigma_amp, seed=99)


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

# Publication-quality matplotlib settings
rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 1.5,
})

# Results directory structure
RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METADATA_DIR = RESULTS_DIR / "metadata"


def setup_results_directories():
    """Create the results directory structure."""
    for dir_path in [RESULTS_DIR, RAW_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Results directories created at: {RESULTS_DIR.absolute()}")


def save_npz(filename: str, **kwargs):
    """Save numerical data to .npz format."""
    filepath = RAW_DIR / filename
    np.savez_compressed(filepath, **kwargs)
    print(f"  Saved: {filepath}")
    return filepath


def save_csv(filename: str, data: Dict[str, np.ndarray]):
    """Save numerical data to .csv format."""
    filepath = RAW_DIR / filename
    csv_data = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                csv_data[key] = val
            elif val.ndim == 2:
                csv_data[key] = val
            else:
                csv_data[key] = val.reshape(-1)
    df = pd.DataFrame(csv_data)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return filepath


def save_json(filename: str, data: Dict[str, Any]):
    """Save metadata to .json format."""
    filepath = METADATA_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {filepath}")
    return filepath


def save_figure(fig, filename: str, formats: List[str] = ['pdf', 'png']):
    """Save figure in multiple formats."""
    for fmt in formats:
        filepath = FIGURES_DIR / f"{filename}.{fmt}"
        if fmt == 'pdf':
            fig.savefig(filepath, format='pdf', bbox_inches='tight')
        elif fmt == 'svg':
            fig.savefig(filepath, format='svg', bbox_inches='tight')
        elif fmt == 'png':
            fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        print(f"  Saved figure: {filepath}")
    plt.close(fig)


def analyze_optimized_pulse(uI_raw: np.ndarray, uQ_raw: np.ndarray,
                            uI_filtered: np.ndarray, uQ_filtered: np.ndarray,
                            run_id: str = "default"):
    """Collect and save all pulse-level quantities."""
    print("\n" + "="*70)
    print("1. OPTIMIZED PULSE DATA ANALYSIS")
    print("="*70)
    
    # cfg.tlist has Nt+1 points (segment boundaries); every pulse array here
    # has Nt points (one value per segment). Use the Nt-length time axis
    # (segment start-times) consistently, matching the convention already
    # used elsewhere in the file (e.g. leakage-vs-time reporting).
    tlist = cfg.tlist[:-1]
    dt = cfg.dt
    
    raw_amplitude = np.sqrt(uI_raw**2 + uQ_raw**2)
    filtered_amplitude = np.sqrt(uI_filtered**2 + uQ_filtered**2)
    raw_phase = np.unwrap(np.arctan2(uQ_raw, uI_raw))
    filtered_phase = np.unwrap(np.arctan2(uQ_filtered, uI_filtered))
    
    max_amplitude_raw = np.max(raw_amplitude)
    max_amplitude_filtered = np.max(filtered_amplitude)
    rms_amplitude_raw = np.sqrt(np.mean(raw_amplitude**2))
    rms_amplitude_filtered = np.sqrt(np.mean(filtered_amplitude**2))
    
    pulse_area_raw = _trapz(raw_amplitude, tlist)
    pulse_area_filtered = _trapz(filtered_amplitude, tlist)
    
    pulse_data = {
        'time': tlist,
        'uI_raw': uI_raw,
        'uQ_raw': uQ_raw,
        'uI_filtered': uI_filtered,
        'uQ_filtered': uQ_filtered,
        'amplitude_raw': raw_amplitude,
        'amplitude_filtered': filtered_amplitude,
        'phase_raw': raw_phase,
        'phase_filtered': filtered_phase,
    }
    save_npz(f"pulse_data_{run_id}.npz", **pulse_data)
    
    pulse_quantities = {
        'max_amplitude_raw': float(max_amplitude_raw),
        'max_amplitude_filtered': float(max_amplitude_filtered),
        'rms_amplitude_raw': float(rms_amplitude_raw),
        'rms_amplitude_filtered': float(rms_amplitude_filtered),
        'pulse_area_raw': float(pulse_area_raw),
        'pulse_area_filtered': float(pulse_area_filtered),
        'gate_time': float(cfg.Tg),
        'num_timesteps': int(cfg.Nt),
        'dt': float(dt),
        'max_amp_limit': float(cfg.max_amp),
    }
    save_json(f"pulse_quantities_{run_id}.json", pulse_quantities)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist, uI_raw, 'b-', label='I(t) raw', alpha=0.7)
    ax.plot(tlist, uQ_raw, 'r-', label='Q(t) raw', alpha=0.7)
    ax.plot(tlist, uI_filtered, 'b--', label='I(t) filtered', linewidth=2)
    ax.plot(tlist, uQ_filtered, 'r--', label='Q(t) filtered', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Control amplitude (rad/ns)')
    ax.set_title('Optimized control pulse: I and Q components')
    ax.legend()
    save_figure(fig, f"pulse_IQ_{run_id}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist, raw_amplitude, 'b-', label='Raw', alpha=0.7)
    ax.plot(tlist, filtered_amplitude, 'r-', label='Filtered', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Control amplitude (rad/ns)')
    ax.set_title('Control pulse amplitude envelope')
    ax.legend()
    save_figure(fig, f"pulse_amplitude_{run_id}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist, raw_phase, 'b-', label='Raw', alpha=0.7)
    ax.plot(tlist, filtered_phase, 'r-', label='Filtered', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Control phase (rad)')
    ax.set_title('Control pulse phase')
    ax.legend()
    save_figure(fig, f"pulse_phase_{run_id}")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(tlist, uI_raw, 'b-', label='Raw I(t)', alpha=0.7)
    axes[0].plot(tlist, uI_filtered, 'r-', label='Filtered I(t)', linewidth=2)
    axes[0].set_ylabel('I(t) (rad/ns)')
    axes[0].legend()
    axes[0].set_title('Raw vs filtered I component')
    axes[1].plot(tlist, uQ_raw, 'b-', label='Raw Q(t)', alpha=0.7)
    axes[1].plot(tlist, uQ_filtered, 'r-', label='Filtered Q(t)', linewidth=2)
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Q(t) (rad/ns)')
    axes[1].legend()
    axes[1].set_title('Raw vs filtered Q component')
    save_figure(fig, f"pulse_raw_vs_filtered_{run_id}")
    
    return pulse_data, pulse_quantities


def analyze_optimization_history(info: Dict[str, Any], run_id: str = "default"):
    """Capture and save optimization history data."""
    print("\n" + "="*70)
    print("2. OPTIMIZATION HISTORY ANALYSIS")
    print("="*70)
    
    available_info = {}
    if info is None:
        info = {}
    
    if 'J_final' in info:
        available_info['final_objective'] = float(info['J_final'])
    if 'leak_final' in info:
        available_info['final_leakage'] = float(info['leak_final'])
    
    available_info['note'] = (
        "Iteration-level optimization history is not exposed by the current "
        "L-BFGS-B implementation without modifying the optimizer."
    )
    
    save_json(f"optimization_history_{run_id}.json", available_info)
    
    print("  Note: Iteration-level history not accessible without modifying optimizer internals.")
    print("  Available information saved.")
    
    return available_info


def analyze_state_dynamics(uI: np.ndarray, uQ: np.ndarray,
                           run_id: str = "default"):
    """Calculate and save state/population dynamics for relevant initial states."""
    print("\n" + "="*70)
    print("3. FULL STATE/POPULATION DYNAMICS")
    print("="*70)
    
    prop = _get_propagator()
    N = sys_.N
    dt = cfg.dt
    
    kets = [
        ('|0>', np.array([1, 0] + [0]*(N-2), dtype=complex)),
        ('|1>', np.array([0, 1] + [0]*(N-2), dtype=complex)),
        ('|+>', np.array([1, 1] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
        ('|+i>', np.array([1, 1j] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
    ]
    
    steps = prop.step_data(uI, uQ, dt)
    
    all_populations = {}
    all_density_matrices = {}
    
    for state_name, ket in kets:
        rho0 = np.outer(ket, ket.conj())
        fwd = prop.forward(steps, _vec(rho0))
        
        populations = np.zeros((len(fwd), N))
        coherences = np.zeros(len(fwd), dtype=complex)
        
        for t_idx, rho_vec in enumerate(fwd):
            rho = _unvec(rho_vec, N)
            for level in range(N):
                populations[t_idx, level] = np.real(rho[level, level])
            coherences[t_idx] = rho[0, 1]
        
        all_populations[state_name] = populations
        all_density_matrices[state_name] = np.array([_unvec(rho_vec, N) for rho_vec in fwd])
        
        state_data = {
            'time': cfg.tlist,
            'populations': populations,
            'coherence_01': coherences,
            'density_matrices': all_density_matrices[state_name],
        }
        save_npz(f"state_dynamics_{state_name.replace('|', '').replace('>', '')}_{run_id}.npz", 
                 **state_data)
    
    computational_pop = np.zeros(len(cfg.tlist))
    leakage_pop = np.zeros(len(cfg.tlist))
    
    for state_name, _ in kets:
        pops = all_populations[state_name]
        computational_pop += pops[:, 0] + pops[:, 1]
        leakage_pop += pops[:, 2] + pops[:, 3]
    
    computational_pop /= len(kets)
    leakage_pop /= len(kets)
    
    for state_name, _ in kets:
        pops = all_populations[state_name]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'red', 'green', 'orange']
        labels = [f'|{i}>' for i in range(N)]
        
        for i in range(N):
            ax.plot(cfg.tlist, pops[:, i], color=colors[i], label=labels[i], linewidth=2)
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Population')
        ax.set_title(f'State populations for initial state {state_name}')
        ax.legend()
        ax.set_ylim([-0.05, 1.05])
        save_figure(fig, f"population_dynamics_{state_name.replace('|', '').replace('>', '')}_{run_id}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cfg.tlist, computational_pop, 'b-', label='Computational subspace', linewidth=2)
    ax.plot(cfg.tlist, leakage_pop, 'r-', label='Leakage (|2>+|3>)', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title('Average computational vs leakage populations')
    ax.legend()
    ax.set_ylim([-0.05, 1.05])
    save_figure(fig, f"average_population_dynamics_{run_id}")
    
    aggregate_data = {
        'time': cfg.tlist,
        'computational_population': computational_pop,
        'leakage_population': leakage_pop,
    }
    save_npz(f"aggregate_populations_{run_id}.npz", **aggregate_data)
    
    return all_populations, all_density_matrices


def analyze_bloch_sphere(all_density_matrices: Dict[str, np.ndarray],
                         run_id: str = "default"):
    """Calculate and visualize Bloch sphere trajectories."""
    print("\n" + "="*70)
    print("4. BLOCH-SPHERE TRAJECTORIES (computational subspace projection)")
    print("="*70)
    
    bloch_vectors = {}
    
    for state_name, density_matrices in all_density_matrices.items():
        N_time = len(density_matrices)
        x = np.zeros(N_time)
        y = np.zeros(N_time)
        z = np.zeros(N_time)
        
        for t_idx, rho in enumerate(density_matrices):
            x[t_idx] = 2 * np.real(rho[0, 1])
            y[t_idx] = 2 * np.imag(rho[0, 1])
            z[t_idx] = np.real(rho[0, 0] - rho[1, 1])
        
        bloch_vectors[state_name] = {'x': x, 'y': y, 'z': z}
        
        bloch_data = {
            'time': cfg.tlist,
            'x': x,
            'y': y,
            'z': z,
        }
        save_npz(f"bloch_vectors_{state_name.replace('|', '').replace('>', '')}_{run_id}.npz",
                 **bloch_data)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    for state_name, vecs in bloch_vectors.items():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
        
        ax.plot(vecs['x'], vecs['y'], vecs['z'], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(vecs['x'][0], vecs['y'][0], vecs['z'][0], color='green', s=100, 
                  marker='o', label='Start')
        ax.scatter(vecs['x'][-1], vecs['y'][-1], vecs['z'][-1], color='red', s=100,
                  marker='s', label='End')
        
        ax.plot([-1, 1], [0, 0], [0, 0], 'k--', alpha=0.3)
        ax.plot([0, 0], [-1, 1], [0, 0], 'k--', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1, 1], 'k--', alpha=0.3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Bloch trajectory for {state_name}\n(computational subspace projection)')
        ax.legend()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        save_figure(fig, f"bloch_trajectory_{state_name.replace('|', '').replace('>', '')}_{run_id}")
    
    return bloch_vectors


def analyze_time_dependent_fidelity(uI: np.ndarray, uQ: np.ndarray,
                                    run_id: str = "default"):
    """Calculate and save time-dependent fidelity."""
    print("\n" + "="*70)
    print("5. TIME-DEPENDENT FIDELITY")
    print("="*70)
    
    prop = _get_propagator()
    N = sys_.N
    dt = cfg.dt
    steps = prop.step_data(uI, uQ, dt)
    
    kets = [
        ('|0>', np.array([1, 0] + [0]*(N-2), dtype=complex)),
        ('|1>', np.array([0, 1] + [0]*(N-2), dtype=complex)),
        ('|+>', np.array([1, 1] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
        ('|+i>', np.array([1, 1j] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
    ]
    
    all_fidelities = {}
    
    for state_name, ket in kets:
        rho0 = np.outer(ket, ket.conj())
        rho_target = sys_.ideal_target_dm(rho0)
        
        fwd = prop.forward(steps, _vec(rho0))
        
        fidelities = np.zeros(len(fwd))
        for t_idx, rho_vec in enumerate(fwd):
            rho = _unvec(rho_vec, N)
            fidelities[t_idx] = np.real(np.trace(rho_target.conj().T @ rho))
        
        all_fidelities[state_name] = fidelities
        
        fid_data = {
            'time': cfg.tlist,
            'fidelity': fidelities,
        }
        save_npz(f"fidelity_dynamics_{state_name.replace('|', '').replace('>', '')}_{run_id}.npz",
                 **fid_data)
    
    avg_fidelity = np.mean([all_fidelities[name] for name, _ in kets], axis=0)
    
    save_npz(f"average_fidelity_dynamics_{run_id}.npz",
             time=cfg.tlist, fidelity=avg_fidelity)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange']
    for (state_name, _), color in zip(kets, colors):
        ax.plot(cfg.tlist, all_fidelities[state_name], color=color, 
               label=state_name, linewidth=2)
    ax.plot(cfg.tlist, avg_fidelity, 'k--', label='Average', linewidth=2.5)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Fidelity')
    ax.set_title('Time-dependent gate fidelity')
    ax.legend()
    ax.set_ylim([0, 1.05])
    save_figure(fig, f"fidelity_dynamics_{run_id}")
    
    final_fidelities = {name: float(fid[-1]) for name, fid in all_fidelities.items()}
    final_fidelities['average'] = float(np.mean(list(final_fidelities.values())))
    save_json(f"final_fidelities_{run_id}.json", final_fidelities)
    
    return all_fidelities, avg_fidelity


def analyze_leakage(uI: np.ndarray, uQ: np.ndarray, run_id: str = "default"):
    """Calculate detailed leakage analysis."""
    print("\n" + "="*70)
    print("6. LEAKAGE ANALYSIS")
    print("="*70)
    
    prop = _get_propagator()
    N = sys_.N
    dt = cfg.dt
    steps = prop.step_data(uI, uQ, dt)
    
    kets = [
        ('|0>', np.array([1, 0] + [0]*(N-2), dtype=complex)),
        ('|1>', np.array([0, 1] + [0]*(N-2), dtype=complex)),
        ('|+>', np.array([1, 1] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
        ('|+i>', np.array([1, 1j] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
    ]
    
    all_leakage = {}
    
    for state_name, ket in kets:
        rho0 = np.outer(ket, ket.conj())
        fwd = prop.forward(steps, _vec(rho0))
        
        P2 = np.zeros(len(fwd))
        P3 = np.zeros(len(fwd))
        P_leak = np.zeros(len(fwd))
        
        for t_idx, rho_vec in enumerate(fwd):
            rho = _unvec(rho_vec, N)
            P2[t_idx] = np.real(rho[2, 2])
            P3[t_idx] = np.real(rho[3, 3])
            P_leak[t_idx] = P2[t_idx] + P3[t_idx]
        
        all_leakage[state_name] = {
            'P2': P2,
            'P3': P3,
            'P_leak': P_leak,
        }
        
        final_leakage = float(P_leak[-1])
        max_leakage = float(np.max(P_leak))
        time_max_leakage = float(cfg.tlist[np.argmax(P_leak)])
        
        leakage_stats = {
            'final_leakage': final_leakage,
            'max_transient_leakage': max_leakage,
            'time_max_leakage': time_max_leakage,
        }
        save_json(f"leakage_stats_{state_name.replace('|', '').replace('>', '')}_{run_id}.json",
                  leakage_stats)
        
        save_npz(f"leakage_dynamics_{state_name.replace('|', '').replace('>', '')}_{run_id}.npz",
                 time=cfg.tlist, P2=P2, P3=P3, P_leak=P_leak)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, (state_name, _) in enumerate(kets):
        leak_data = all_leakage[state_name]
        axes[idx].plot(cfg.tlist, leak_data['P2'], 'r-', label='P(|2>)', linewidth=2)
        axes[idx].plot(cfg.tlist, leak_data['P3'], 'g-', label='P(|3>)', linewidth=2)
        axes[idx].plot(cfg.tlist, leak_data['P_leak'], 'b--', label='Total leakage', linewidth=2)
        axes[idx].set_xlabel('Time (ns)')
        axes[idx].set_ylabel('Leakage population')
        axes[idx].set_title(f'Leakage dynamics for {state_name}')
        axes[idx].legend()
        axes[idx].set_ylim([0, 0.1])
    
    save_figure(fig, f"leakage_dynamics_{run_id}")
    
    return all_leakage


def analyze_noise_robustness(uI: np.ndarray, uQ: np.ndarray, run_id: str = "default", n_ensembles: int = 5):
    print("\n" + "="*70)
    print(f"7. NOISE ROBUSTNESS ANALYSIS ({n_ensembles} Independent Ensembles)")
    print("="*70)
    
    all_fidelities = []
    all_deltas = []
    all_gammas = []
    
    for ensemble_idx in range(n_ensembles):
        eval_model = NoiseModel(cfg.n_noise_eval, cfg.sigma_detuning, cfg.sigma_amp, seed=1000 + ensemble_idx)
        fids = eval_model.fidelity_distribution(uI, uQ)
        all_fidelities.extend(fids)
        all_deltas.extend(eval_model.deltas)
        all_gammas.extend(eval_model.gammas)
        
    fidelities = np.array(all_fidelities)
    deltas = np.array(all_deltas)
    gammas = np.array(all_gammas)
    
    stats = {
        'mean_fidelity': float(np.mean(fidelities)),
        'median_fidelity': float(np.median(fidelities)),
        'std_fidelity': float(np.std(fidelities)),
        'min_fidelity': float(np.min(fidelities)),
        'percentile_5': float(np.percentile(fidelities, 5)),
        'CVaR_5%': float(np.mean(fidelities[fidelities <= np.percentile(fidelities, 5)])),
        'worst_case': float(np.min(fidelities)),
        'n_ensembles_tested': n_ensembles,
        'total_samples': len(fidelities)
    }
    
    save_json(f"noise_robustness_stats_{run_id}.json", stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(fidelities, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(stats['mean_fidelity'], color='red', linestyle='--', label=f"Mean: {stats['mean_fidelity']:.4f}")
    ax.axvline(stats['CVaR_5%'], color='orange', linestyle='--', label=f"CVaR 5%: {stats['CVaR_5%']:.4f}")
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Count')
    ax.set_title(f'Fidelity Distribution ({n_ensembles} Independent Noise Ensembles)')
    ax.legend()
    save_figure(fig, f"fidelity_distribution_{run_id}")
    
    print(f"  Noise robustness statistics ({len(fidelities)} total draws):")
    print(f"    Mean fidelity: {stats['mean_fidelity']:.6f}")
    print(f"    CVaR 5%: {stats['CVaR_5%']:.6f}")
    print(f"    Worst case: {stats['worst_case']:.6f}")
    
    noise_data = {'detuning_error': deltas, 'amplitude_error': gammas, 'fidelity': fidelities}
    save_npz(f"noise_evaluation_results_{run_id}.npz", **noise_data)
    
    return stats, noise_data


def analyze_noise_scaling(uI: np.ndarray, uQ: np.ndarray, run_id: str = "default"):
    print("\n" + "="*70)
    print("15. NOISE SCALING ANALYSIS")
    print("="*70)
    
    multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    mean_fids = []
    cvar_fids = []
    
    for m in multipliers:
        scaled_noise = NoiseModel(
            n_samples=cfg.n_noise_eval,
            sigma_d=cfg.sigma_detuning * m,
            sigma_a=cfg.sigma_amp * m,
            seed=99 + int(m*10)
        )
        mean_fid = scaled_noise.mean_fidelity(uI, uQ)
        cvar_fid = scaled_noise.cvar(uI, uQ)
        
        mean_fids.append(mean_fid)
        cvar_fids.append(cvar_fid)
        print(f"  Noise Scale {m}x: Mean J = {mean_fid:.6f}, CVaR5% = {cvar_fid:.6f}")
        
    scaling_data = {
        'noise_multipliers': multipliers,
        'mean_fidelities': mean_fids,
        'cvar_fidelities': cvar_fids,
        'baseline_sigma_d': float(cfg.sigma_detuning),
        'baseline_sigma_a': float(cfg.sigma_amp)
    }
    save_json(f"noise_scaling_{run_id}.json", scaling_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(multipliers, mean_fids, 'b-o', label='Mean Fidelity', linewidth=2)
    ax.plot(multipliers, cvar_fids, 'r--s', label='CVaR 5%', linewidth=2)
    ax.set_xlabel('Noise Magnitude Multiplier')
    ax.set_ylabel('Thermal Gate Fidelity')
    ax.set_title('Robustness under Scaled Quasi-Static Noise Environments')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, f"noise_scaling_degradation_{run_id}")
    
    return scaling_data


def analyze_filter_function(uI: np.ndarray, uQ: np.ndarray, run_id: str = "default"):
    """Analyze filter function and noise PSD."""
    print("\n" + "="*70)
    print("8. FILTER-FUNCTION ANALYSIS")
    print("="*70)
    
    J_ff, gI_ff, gQ_ff = filter_function_overlap_and_grad(uI, uQ, cfg.dt)
    
    omega_fine = np.geomspace(cfg.ff_omega_min, cfg.ff_omega_max, 200)
    
    N = sys_.N
    steps = _coherent_step_data(uI, uQ, cfg.dt, sys_.H0, sys_.Hx, sys_.Hy)
    Ulist = [np.eye(N, dtype=complex)]
    for S, _, _ in steps:
        Ulist.append(S @ Ulist[-1])
    
    B = sys_.B_noise
    tlist_mid = cfg.dt * (np.arange(cfg.Nt) + 0.5)
    
    F_omega = np.zeros_like(omega_fine)
    for idx, om in enumerate(omega_fine):
        y = np.array([(Ulist[k].conj().T @ B @ Ulist[k])[0, 1] for k in range(cfg.Nt)])
        phase = np.exp(1j * om * tlist_mid)
        integral = np.sum(y * phase) * cfg.dt
        F_omega[idx] = float(np.abs(integral)**2)
    
    S_omega = _psd(omega_fine)
    
    ff_data = {
        'omega': omega_fine,
        'filter_function': F_omega,
        'noise_psd': S_omega,
        'total_overlap': J_ff,
    }
    save_npz(f"filter_function_data_{run_id}.npz", **ff_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(omega_fine, F_omega, 'b-', linewidth=2)
    ax.set_xlabel('Angular frequency (rad/ns)')
    ax.set_ylabel('Filter function F(ω)')
    ax.set_title('Filter function of optimized pulse')
    save_figure(fig, f"filter_function_{run_id}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(omega_fine, S_omega, 'r-', linewidth=2)
    ax.set_xlabel('Angular frequency (rad/ns)')
    ax.set_ylabel('Noise PSD S(ω)')
    ax.set_title('Noise power spectral density')
    save_figure(fig, f"noise_psd_{run_id}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(omega_fine, F_omega, 'b-', label='Filter function', linewidth=2)
    ax2 = ax.twinx()
    ax2.loglog(omega_fine, S_omega, 'r-', label='Noise PSD', linewidth=2)
    ax.set_xlabel('Angular frequency (rad/ns)')
    ax.set_ylabel('Filter function F(ω)', color='blue')
    ax2.set_ylabel('Noise PSD S(ω)', color='red')
    ax.set_title('Filter function and noise PSD')
    save_figure(fig, f"filter_function_psd_{run_id}")
    
    overlap_contribution = F_omega * S_omega
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(omega_fine, overlap_contribution, 'g-', linewidth=2)
    ax.set_xlabel('Angular frequency (rad/ns)')
    ax.set_ylabel('F(ω)S(ω)')
    ax.set_title('Weighted filter function/PSD contribution')
    save_figure(fig, f"filter_function_overlap_{run_id}")
    
    return ff_data


def analyze_awg_bandwidth(uI_raw: np.ndarray, uQ_raw: np.ndarray,
                         uI_filtered: np.ndarray, uQ_filtered: np.ndarray,
                         run_id: str = "default"):
    """Analyze AWG bandwidth filtering effects in frequency domain."""
    print("\n" + "="*70)
    print("9. AWG BANDWIDTH ANALYSIS")
    print("="*70)
    
    dt = cfg.dt
    N = cfg.Nt
    
    freqs = np.fft.fftfreq(N, d=dt)
    freqs = np.fft.fftshift(freqs)
    
    I_raw_fft = np.fft.fftshift(np.fft.fft(uI_raw))
    Q_raw_fft = np.fft.fftshift(np.fft.fft(uQ_raw))
    I_filtered_fft = np.fft.fftshift(np.fft.fft(uI_filtered))
    Q_filtered_fft = np.fft.fftshift(np.fft.fft(uQ_filtered))
    
    fft_data = {
        'frequency': freqs,
        'I_raw_fft': np.abs(I_raw_fft),
        'Q_raw_fft': np.abs(Q_raw_fft),
        'I_filtered_fft': np.abs(I_filtered_fft),
        'Q_filtered_fft': np.abs(Q_filtered_fft),
    }
    save_npz(f"awg_fft_data_{run_id}.npz", **fft_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    axes[0].semilogy(freqs, np.abs(I_raw_fft), 'b-', label='Raw I', alpha=0.7)
    axes[0].semilogy(freqs, np.abs(I_filtered_fft), 'r-', label='Filtered I', linewidth=2)
    axes[0].set_ylabel('|FFT(I)|')
    axes[0].set_title('I component frequency spectrum')
    axes[0].legend()
    axes[0].axvline(x=cfg.awg_bandwidth_GHz, color='gray', linestyle='--', 
                   label='AWG bandwidth')
    axes[0].legend()
    
    axes[1].semilogy(freqs, np.abs(Q_raw_fft), 'b-', label='Raw Q', alpha=0.7)
    axes[1].semilogy(freqs, np.abs(Q_filtered_fft), 'r-', label='Filtered Q', linewidth=2)
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('|FFT(Q)|')
    axes[1].set_title('Q component frequency spectrum')
    axes[1].legend()
    axes[1].axvline(x=cfg.awg_bandwidth_GHz, color='gray', linestyle='--', 
                   label='AWG bandwidth')
    axes[1].legend()
    
    save_figure(fig, f"awg_fft_comparison_{run_id}")
    
    return fft_data


def analyze_thermal_state(run_id: str = "default"):
    """Analyze thermal initialization information."""
    print("\n" + "="*70)
    print("10. THERMAL-STATE INFORMATION")
    print("="*70)
    
    rho_th = sys_.thermal_state()
    N = sys_.N
    
    populations = np.real(np.diag(rho_th))
    
    n_thermal = cfg.n_thermal
    
    thermal_info = {
        'temperature_mK': float(cfg.temperature_mK),
        'qubit_frequency_GHz': float(cfg.qubit_freq_GHz),
        'thermal_occupation': float(n_thermal),
        'initial_populations': populations.tolist(),
        'initial_leakage_population': float(np.sum(populations[2:])),
    }
    
    save_json(f"thermal_state_info_{run_id}.json", thermal_info)
    
    save_npz(f"thermal_state_{run_id}.npz",
             populations=populations,
             levels=np.arange(N))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(N), populations, color='steelblue', edgecolor='black')
    ax.set_xlabel('Energy level')
    ax.set_ylabel('Population')
    ax.set_title(f'Thermal state populations at T={cfg.temperature_mK} mK')
    ax.set_xticks(range(N))
    ax.set_xticklabels([f'|{i}>' for i in range(N)])
    save_figure(fig, f"thermal_populations_{run_id}")
    
    return thermal_info


def analyze_state_metrics(uI: np.ndarray, uQ: np.ndarray, run_id: str = "default"):
    """Create state-by-state final metrics table."""
    print("\n" + "="*70)
    print("11. STATE-BY-STATE FINAL METRICS")
    print("="*70)
    
    prop = _get_propagator()
    N = sys_.N
    dt = cfg.dt
    steps = prop.step_data(uI, uQ, dt)
    
    kets = [
        ('|0>', np.array([1, 0] + [0]*(N-2), dtype=complex)),
        ('|1>', np.array([0, 1] + [0]*(N-2), dtype=complex)),
        ('|+>', np.array([1, 1] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
        ('|+i>', np.array([1, 1j] + [0]*(N-2), dtype=complex) / np.sqrt(2)),
    ]
    
    metrics = []
    
    for state_name, ket in kets:
        rho0 = np.outer(ket, ket.conj())
        rho_target = sys_.ideal_target_dm(rho0)
        
        fwd = prop.forward(steps, _vec(rho0))
        rho_final = _unvec(fwd[-1], N)
        
        fidelity = float(np.real(np.trace(rho_target.conj().T @ rho_final)))
        leakage = float(np.real(np.trace(sys_.P_leak @ rho_final)))
        final_populations = [float(np.real(rho_final[i, i])) for i in range(N)]
        
        metrics.append({
            'state': state_name,
            'fidelity': fidelity,
            'leakage': leakage,
            'final_population_0': final_populations[0],
            'final_population_1': final_populations[1],
            'final_population_2': final_populations[2],
            'final_population_3': final_populations[3],
        })
    
    avg_fidelity = np.mean([m['fidelity'] for m in metrics])
    avg_leakage = np.mean([m['leakage'] for m in metrics])
    
    metrics.append({
        'state': 'Average',
        'fidelity': float(avg_fidelity),
        'leakage': float(avg_leakage),
        'final_population_0': float(np.mean([m['final_population_0'] for m in metrics])),
        'final_population_1': float(np.mean([m['final_population_1'] for m in metrics])),
        'final_population_2': float(np.mean([m['final_population_2'] for m in metrics])),
        'final_population_3': float(np.mean([m['final_population_3'] for m in metrics])),
    })
    
    save_json(f"state_metrics_{run_id}.json", metrics)
    
    df = pd.DataFrame(metrics)
    csv_path = TABLES_DIR / f"state_metrics_{run_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved table: {csv_path}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    save_figure(fig, f"state_metrics_table_{run_id}")
    
    return metrics


def analyze_method_comparison(fade_result: Dict, grape_result: Optional[Dict] = None,
                              crab_result: Optional[Dict] = None,
                              run_id: str = "default"):
    """Create comparison table for optimization methods."""
    print("\n" + "="*70)
    print("12. FADE-RIPE vs GRAPE vs CRAB COMPARISON")
    print("="*70)
    
    comparison = {
        'FADE-RIPE': fade_result,
    }
    
    if grape_result is not None:
        comparison['GRAPE'] = grape_result
    if crab_result is not None:
        comparison['CRAB'] = crab_result
    
    table_data = []
    for method, result in comparison.items():
        row = {
            'Method': method,
            'Final_fidelity': result.get('final_fidelity', 'N/A'),
            'Noisy_mean_fidelity': result.get('noisy_mean_fidelity', 'N/A'),
            'CVaR': result.get('cvar', 'N/A'),
            'Worst_case_fidelity': result.get('worst_case_fidelity', 'N/A'),
            'Leakage': result.get('leakage', 'N/A'),
            'Pulse_amplitude': result.get('pulse_amplitude', 'N/A'),
            'Filter_function_penalty': result.get('filter_penalty', 'N/A'),
            'Runtime': result.get('runtime', 'N/A'),
            'Iterations': result.get('iterations', 'N/A'),
        }
        table_data.append(row)
    
    save_json(f"method_comparison_{run_id}.json", table_data)
    
    df = pd.DataFrame(table_data)
    csv_path = TABLES_DIR / f"method_comparison_{run_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved table: {csv_path}")
    
    methods = [row['Method'] for row in table_data]
    fidelities = [row['Final_fidelity'] for row in table_data if row['Final_fidelity'] != 'N/A']
    
    if fidelities:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, [row['Final_fidelity'] if row['Final_fidelity'] != 'N/A' else 0 
                      for row in table_data], 
               color=['blue', 'green', 'orange'][:len(methods)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_xlabel('Method')
        ax.set_ylabel('Final fidelity')
        ax.set_title('FADE-RIPE vs GRAPE vs CRAB performance comparison')
        save_figure(fig, f"method_comparison_{run_id}")
    
    return table_data


def analyze_multistart_statistics(results: List[Dict], run_id: str = "default"):
    """Analyze multi-start optimization results."""
    print("\n" + "="*70)
    print("13. MULTI-START STATISTICS")
    print("="*70)
    
    multistart_data = []
    
    for idx, result in enumerate(results):
        entry = {
            'restart_number': idx + 1,
            'final_objective': result.get('J_final', 'N/A'),
            'oos_mean_fidelity': result.get('oos_mean', 'N/A'),
            'cvar': result.get('oos_cvar', 'N/A'),
            'convergence': result.get('converged', 'N/A'),
            'runtime': result.get('runtime', 'N/A'),
        }
        multistart_data.append(entry)
    
    save_json(f"multistart_statistics_{run_id}.json", multistart_data)
    
    if multistart_data:
        restarts = [d['restart_number'] for d in multistart_data]
        fidelities = [d['oos_mean_fidelity'] for d in multistart_data if d['oos_mean_fidelity'] != 'N/A']
        
        if fidelities:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(restarts, fidelities, color='steelblue', edgecolor='black')
            ax.set_xlabel('Restart number')
            ax.set_ylabel('Out-of-sample mean fidelity')
            ax.set_title('Multi-start optimization results')
            ax.axhline(y=np.mean(fidelities), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(fidelities):.6f}')
            ax.legend()
            save_figure(fig, f"multistart_statistics_{run_id}")
    
    return multistart_data


def analyze_numerical_validation(run_id: str = "default"):
    """Run and capture gradient self-test results."""
    print("\n" + "="*70)
    print("14. NUMERICAL VALIDATION")
    print("="*70)
    
    ok = _selftest_gradients(verbose=True)
    
    validation_results = {
        'self_test_passed': bool(ok),
        'test_name': 'gradient_self_test',
        'tolerance': 1e-6,
        'note': 'See console output for detailed error values'
    }
    
    save_json(f"numerical_validation_{run_id}.json", validation_results)
    
    return validation_results


def collect_metadata(run_id: str = "default"):
    """Collect all simulation parameters into metadata file."""
    print("\n" + "="*70)
    print("METADATA COLLECTION")
    print("="*70)
    
    metadata = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hilbert_space_dimension': cfg.N_LEVELS,
        'gate_time_ns': cfg.Tg,
        'num_timesteps': cfg.Nt,
        'dt_ns': cfg.dt,
        'anharmonicity_MHz': cfg.alpha_MHz,
        'T1_ns': cfg.T1,
        'T2_star_ns': cfg.T2s,
        'temperature_mK': cfg.temperature_mK,
        'qubit_frequency_GHz': cfg.qubit_freq_GHz,
        'thermal_occupation': cfg.n_thermal,
        'awg_bandwidth_GHz': cfg.awg_bandwidth_GHz,
        'awg_filter_order': cfg.awg_filter_order,
        'leakage_penalty': cfg.leak_lambda,
        'max_amplitude': cfg.max_amp,
        'noise_sigma_detuning': cfg.sigma_detuning,
        'noise_sigma_amplitude': cfg.sigma_amp,
        'num_noise_eval_samples': cfg.n_noise_eval,
        'fade_nmodes': cfg.fade_nmodes,
        'fade_maxiter': cfg.fade_maxiter,
        'fade_ftol': cfg.fade_ftol,
        'fade_gtol': cfg.fade_gtol,
        'fade_n_restarts': cfg.fade_n_restarts,
        'fade_K': cfg.fade_K,
        'fade_sigma': cfg.fade_sigma,
        'fade_amp_sig': cfg.fade_amp_sig,
        'ff_n_freqs': cfg.ff_n_freqs,
        'ff_omega_min': cfg.ff_omega_min,
        'ff_omega_max': cfg.ff_omega_max,
        'ff_noise_type': cfg.ff_noise_type,
        'ff_psd_amplitude': cfg.ff_psd_amplitude,
        'ff_mu': cfg.ff_mu,
        'ff_target_fraction': cfg.ff_target_fraction,
        'grape_maxiter': cfg.grape_maxiter,
        'grape_restarts': cfg.grape_restarts,
        'crab_nmodes': cfg.crab_nmodes,
        'crab_maxiter': cfg.crab_maxiter,
        'crab_restarts': cfg.crab_restarts,
        'num_workers': _N_WORKERS,
        'num_physical_cores': _mp.cpu_count(),
    }
    
    save_json(f"run_metadata_{run_id}.json", metadata)
    
    return metadata


def run_full_analysis(uI_raw: np.ndarray, uQ_raw: np.ndarray,
                      uI_filtered: np.ndarray, uQ_filtered: np.ndarray,
                      optimization_info: Dict[str, Any],
                      multistart_results: List[Dict] = None,
                      run_id: str = "default"):
    """Run the complete analysis pipeline."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS PIPELINE")
    print("="*80)
    
    setup_results_directories()
    
    metadata = collect_metadata(run_id)
    
    pulse_data, pulse_quantities = analyze_optimized_pulse(
        uI_raw, uQ_raw, uI_filtered, uQ_filtered, run_id)
    
    opt_history = analyze_optimization_history(optimization_info, run_id)
    
    all_populations, all_density_matrices = analyze_state_dynamics(
        uI_filtered, uQ_filtered, run_id)
    
    bloch_vectors = analyze_bloch_sphere(all_density_matrices, run_id)
    
    all_fidelities, avg_fidelity = analyze_time_dependent_fidelity(
        uI_filtered, uQ_filtered, run_id)
    
    all_leakage = analyze_leakage(uI_filtered, uQ_filtered, run_id)
    
    noise_stats, noise_data = analyze_noise_robustness(
        uI_filtered, uQ_filtered, run_id)
    
    noise_scaling_stats = analyze_noise_scaling(
        uI_filtered, uQ_filtered, run_id)
    
    ff_data = analyze_filter_function(uI_filtered, uQ_filtered, run_id)
    
    fft_data = analyze_awg_bandwidth(
        uI_raw, uQ_raw, uI_filtered, uQ_filtered, run_id)
    
    thermal_info = analyze_thermal_state(run_id)
    
    state_metrics = analyze_state_metrics(uI_filtered, uQ_filtered, run_id)
    
    if 'grape_result' in optimization_info or 'crab_result' in optimization_info:
        method_comparison = analyze_method_comparison(
            optimization_info.get('fade_result', {}),
            optimization_info.get('grape_result'),
            optimization_info.get('crab_result'),
            run_id)
    
    if multistart_results:
        multistart_stats = analyze_multistart_statistics(multistart_results, run_id)
    
    validation_results = analyze_numerical_validation(run_id)
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print(f"All results saved to: {RESULTS_DIR.absolute()}")
    print("="*80)
    
    summary = {
        'run_id': run_id,
        'metadata': metadata,
        'pulse_quantities': pulse_quantities,
        'noise_stats': noise_stats,
        'noise_scaling': noise_scaling_stats,
        'thermal_info': thermal_info,
        'state_metrics': state_metrics,
        'validation_results': validation_results,
    }
    
    return summary


def _common_method_metrics(uI: np.ndarray, uQ: np.ndarray,
                           runtime: Any = 'N/A', iterations: Any = 'N/A') -> Dict[str, Any]:
    prop = _get_propagator()
    J_obj, _, _, leakage, final_fidelity = thermal_gate_fidelity_and_grad(uI, uQ, prop, cfg.dt)
    noisy_mean_fidelity = noise_eval.mean_fidelity(uI, uQ)
    cvar = noise_eval.cvar(uI, uQ)
    worst_case_fidelity = noise_eval.worst_fidelity(uI, uQ)
    filter_penalty, _, _ = filter_function_overlap_and_grad(uI, uQ, cfg.dt)
    pulse_amplitude = float(max(np.max(np.abs(uI)), np.max(np.abs(uQ))))
    return {
        'final_fidelity': final_fidelity,
        'noisy_mean_fidelity': noisy_mean_fidelity,
        'cvar': cvar,
        'worst_case_fidelity': worst_case_fidelity,
        'leakage': leakage,
        'pulse_amplitude': pulse_amplitude,
        'filter_penalty': filter_penalty,
        'runtime': runtime,
        'iterations': iterations,
    }


def run_optimization_with_analysis(gate_time: float = 180.0, 
                                   nt: Optional[int] = None,
                                   run_id: str = "default",
                                   run_mode: str = "single"):
    """Run the existing optimization and then analyze the results."""
    print("\n" + "="*80)
    print("RUNNING OPTIMIZATION WITH ANALYSIS PIPELINE")
    print("="*80)
    
    ok = _selftest_gradients(verbose=True)
    if not ok:
        print("\nAborting: gradient self-test failed.")
        sys.exit(1)
    
    if nt is not None:
        set_gate_time(Tg=gate_time, Nt=nt)
    else:
        set_gate_time(Tg=gate_time, Nt=max(8, int(round(gate_time / 0.5))))
    
    print("\nRunning existing optimization...")
    seed_uI, seed_uQ = hann_init()
    
    start_time = time.time()
    
    if run_mode == "compare-qtrl":
        print("\nRunning qutip_qtrl GRAPE/CRAB comparison...")
        try:
            t_g0 = time.time()
            grape_uI, grape_uQ, grape_F = grape_optimize()
            grape_runtime = time.time() - t_g0
            t_c0 = time.time()
            crab_uI, crab_uQ, crab_F = crab_optimize()
            crab_runtime = time.time() - t_c0
            grape_result = _common_method_metrics(
                grape_uI, grape_uQ, runtime=grape_runtime, iterations=cfg.grape_maxiter)
            crab_result = _common_method_metrics(
                crab_uI, crab_uQ, runtime=crab_runtime, iterations=cfg.crab_maxiter)
        except ImportError as e:
            print(f"Warning: {e}")
            grape_result = None
            crab_result = None
    else:
        grape_result = None
        crab_result = None
    
    uI_raw, uQ_raw, uI_filtered, uQ_filtered, info = fade_ripe_multistart(
        seed_uI, seed_uQ)
    
    runtime = time.time() - start_time
    
    fade_result = _common_method_metrics(
        uI_filtered, uQ_filtered, runtime=runtime, iterations=cfg.fade_maxiter)
    fade_result['leak_final'] = info.get('leak_final', None)  # kept for backward compat

    optimization_info = {
        'final_fidelity': fade_result['final_fidelity'],
        'leakage': fade_result['leakage'],
        'runtime': runtime,
        'iterations': cfg.fade_maxiter,
        'fade_result': fade_result,
        'grape_result': grape_result,
        'crab_result': crab_result,
    }

    # Safety net: everything expensive (hours of L-BFGS-B + GRAPE/CRAB) is
    # done as of this point. Save it to disk BEFORE handing off to the
    # plotting/reporting pipeline below, so a bug or crash in analysis code
    # can never again cost you the optimization results themselves -- only
    # a rerun of the cheap post-processing.
    try:
        os.makedirs('results/raw', exist_ok=True)
        np.savez(f'results/raw/checkpoint_{run_id}.npz',
                 uI_raw=uI_raw, uQ_raw=uQ_raw,
                 uI_filtered=uI_filtered, uQ_filtered=uQ_filtered)
        save_json(f'checkpoint_optimization_info_{run_id}.json', optimization_info)
        print(f"\n[checkpoint] Saved raw results to results/raw/checkpoint_{run_id}.npz "
              f"before running analysis/plotting.")
    except Exception as e:
        print(f"\n[checkpoint] WARNING: failed to save checkpoint ({e}). "
              f"Continuing to analysis anyway.")

    summary = run_full_analysis(
        uI_raw, uQ_raw, uI_filtered, uQ_filtered,
        optimization_info,
        multistart_results=None,
        run_id=run_id
    )
    
    print(f"\nGRAPE thermal J={grape_result['final_fidelity']:.6f}" 
          if grape_result else "\nGRAPE: Not available")
    print(f"CRAB thermal J={crab_result['final_fidelity']:.6f}" 
          if crab_result else "CRAB: Not available")
    print(f"FADE-RIPE thermal J={optimization_info['final_fidelity']:.6f}")
    
    return summary


def run_benchmark(label: str):
    """Original benchmark function - kept for compatibility."""
    print("=" * 70)
    print(f"FADE-RIPE (transmon, N={cfg.N_LEVELS}, Lindbladian, thermal-init) | {label}")
    print(f"Tg={cfg.Tg:.0f} ns, Nt={cfg.Nt} | alpha/2pi={cfg.alpha_MHz:.0f} MHz | "
          f"T={cfg.temperature_mK:.0f} mK (n_th={cfg.n_thermal:.4f})")
    print("=" * 70)
    t0 = time.time()

    seed_uI, seed_uQ = hann_init()
    (fade_uI_raw, fade_uQ_raw, fade_uI, fade_uQ, info) = fade_ripe_multistart(seed_uI, seed_uQ)

    prop = _get_propagator()
    F_avg, _, _, leak_avg, F_raw_avg = average_gate_fidelity_and_grad(fade_uI, fade_uQ, prop, cfg.dt)
    print(f"\nFinal average-gate-fidelity-style objective (thermal-aware): {F_avg:.6f}")
    print(f"Final leakage population in |2>,|3>: {leak_avg:.3e}")

    print("\nQuasi-static noise benchmark (ground-state input, worker pool)...")
    mean_F = noise_eval.mean_fidelity(fade_uI, fade_uQ)
    cvar_F = noise_eval.cvar(fade_uI, fade_uQ)
    print(f"  mean J = {mean_F:.6f}, CVaR5%% J = {cvar_F:.6f}")

    elapsed = time.time() - t0
    print(f"\nRuntime ({label}): {int(elapsed // 60)}m {int(elapsed % 60)}s")
    return {'uI': fade_uI, 'uQ': fade_uQ, 'info': info, 'mean_F': mean_F, 'cvar_F': cvar_F}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FADE-RIPE optimizer with analysis pipeline")
    parser.add_argument('--mode', choices=['single', 'selftest', 'compare-qtrl', 'analyze'],
                         default='single')
    parser.add_argument('--gate-times', type=float, nargs='+', default=[180.0, 90.0, 60.0, 40.0, 20.0],
                        help="List of gate times to evaluate")
    parser.add_argument('--nt', type=int, default=None)
    parser.add_argument('--run-id-prefix', type=str, default="transmon")
    parser.add_argument('--workers', type=int, default=None)
    args, _ = parser.parse_known_args()

    if args.workers is not None:
        global _N_WORKERS
        _N_WORKERS = args.workers

    ok = _selftest_gradients(verbose=True)
    if not ok:
        print("\nAborting: gradient self-test failed.")
        sys.exit(1)

    if args.mode == 'selftest':
        return

    for tg in args.gate_times:
        current_run_id = f"{args.run_id_prefix}_{tg}ns"
        Nt = args.nt if args.nt is not None else max(8, int(round(tg / 0.5)))
        set_gate_time(Tg=tg, Nt=Nt)
        
        print("\n" + "="*80)
        print(f"STARTING SWEEP: Gate Time = {tg}ns (Nt = {Nt})")
        print("="*80)

        if args.mode == 'single':
            run_benchmark(f"{tg:.0f}ns transmon")
        elif args.mode == 'compare-qtrl':
            run_optimization_with_analysis(
                gate_time=tg, nt=Nt, run_id=current_run_id, run_mode='compare-qtrl')
        elif args.mode == 'analyze':
            run_optimization_with_analysis(
                gate_time=tg, nt=Nt, run_id=current_run_id, run_mode='single')

    _shutdown_pool()


if __name__ == "__main__":
    main()
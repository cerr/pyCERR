# urOMT: MATLAB ↔ pyCERR implementation comparison

This document maps the pyCERR urOMT tool (`cerr/uromt/`) onto the original
MATLAB urOMT pipeline it was ported from
([xinan-nancy-chen/urOMT](https://github.com/xinan-nancy-chen/urOMT),
`driver_RatBrain.m`), function by function, with the theory behind each stage.

The two implementations solve the **same mathematical problem** with the
**same discretization and the same Gauss–Newton optimizer**. pyCERR differs in
plumbing (in-memory dicts + `planC` instead of MATLAB's file-based workflow), in
a few numerical accelerations that are provably equivalent, and in one added
regularizer (velocity H1 smoothness). Differences are called out explicitly in
[§6](#6-where-pycerr-deliberately-differs).

---

## 1. The model (theory)

urOMT = **unbalanced, regularized, dynamic Optimal Mass Transport**. It extends
the Benamou–Brenier dynamic-OMT formulation with (a) a **source/sink term** that
makes transport *unbalanced* (mass need not be conserved), and (b) **diffusion**,
giving an advection–diffusion–source PDE. Given two density images ρ₀ and ρ₁
(here, consecutive contrast-concentration frames of a DCE-MRI series), it finds a
time-dependent velocity field **v(x,t)** and relative source **r(x,t)** that
carry ρ₀ into (approximately) ρ₁.

**Governing PDE** (interpolation model), on `t ∈ [0, T]`:

```
∂ρ/∂t + ∇·(ρ v)  =  σ Δρ  +  ρ r
        (advection)   (diffusion)  (source)
```

**Objective** minimized per time interval:

```
Γ(v, r) =  ∫∫ ρ |v|²         dx dt      (Γ1: kinetic energy, Benamou–Brenier)
       + α ∫∫ ρ r²           dx dt      (Γ2: source penalty)
       + β ‖ρ(T) − ρ₁‖²                 (Γ3: final-image fidelity)
       + η ∫∫ |∇v|²          dx dt      (Γ4: velocity H1 smoothness — pyCERR only)
   subject to the PDE above with ρ(0) = ρ₀.
```

α, β weight the source and the data fidelity; σ is the diffusivity. The kinetic
term is **ρ-weighted**, so velocity is only constrained where mass is present —
this is why the raw recovered velocity is under-determined in low-density voxels
(the motivation for pyCERR's optional η term; see [§6](#6-where-pycerr-deliberately-differs)).

**Post-processing quantities** (both implementations):

- **Effective (flux) velocity**  `v_eff = v − σ ∇log ρ`  — advection minus the
  diffusive drift; this is the physically transported velocity.
- **Flux**  `Φ = ρ v_eff`.
- **Péclet number**  `Pe = |v| / |σ ∇log ρ|`  — ratio of advective to diffusive
  transport.
- **Lagrangian pathlines** — streamlines of `v_eff` seeded in the ROI, giving
  displacement/speed maps along transport paths.

---

## 2. Pipeline at a glance (Parts 1–5)

The MATLAB `driver_RatBrain.m` runs as five sequential "Parts". pyCERR keeps the
same five stages as importable functions:

| Part | Stage | MATLAB | pyCERR |
|------|-------|--------|--------|
| 1 | Data prep (ROI, concentration, smoothing) | `driver_RatBrain.m` Part 1 + `getData`/`getRange` | [`data.py::prepareData`](data.py) |
| 2 | Solve urOMT over intervals | `runUROMT.m` + `Inverse/GNblock_ur.m` | [`solver.py::runUROMT`](solver.py) + `gnBlockExact` |
| 3 | Eulerian maps (speed/rate/Pe/flux) | `runEULA.m` | [`analyze.py::runEULA`](analyze.py) / `runEULAIntervals` |
| 4 | Lagrangian pathlines | `runGLAD.m` | [`analyze.py::runGLAD`](analyze.py) |
| 5 | Visualization | MATLAB figures / `getFlowFrom*` | [`viz.py`](viz.py) (matplotlib + napari + Qt overlay) |

One-call driver: [`cerr/uromt/__init__.py::runUROMTPipeline`](__init__.py) chains
Parts 1–4 and stores the run on `planC.urOMT`.

---

## 3. Detailed file / function mapping

### 3.1 Configuration & parameters

| MATLAB | pyCERR | What it does (theory) |
|--------|--------|-----------------------|
| `getParams.m` (the `cfg` struct) | [`config.py::UROMTConfig`](config.py), `loadModelSettings` | Holds all model/algorithm parameters (σ, dt, nt, α, β, η, solver, smoothing, concentration). In MATLAB everything is one `cfg` struct; pyCERR splits **model params** (JSON `settings/uromt_model_settings.json`) from **data params** read straight from `planC`. |
| `cfg` fields (b) block | `settings/uromt_model_settings.json` | Serialized model settings, one-to-one with the MATLAB `cfg` algorithm fields. |
| `paramInitFunc.m` | [`numerics.py::paramInit`](numerics.py) | Builds the runtime `par` dict: grid `n`, spacing `h`, cell volume `hd`, gradient operator, diffusion solver, and scalar parameters. |

### 3.2 Grid & differential operators

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `getCellCenteredGradMatrix('ccn', …)` | [`numerics.py::neumannGrad`](numerics.py) (+ `_ddx1d`, `cellCenteredGrid`) | Cell-centered finite-difference **gradient** with Neumann (zero-flux) boundary conditions. `Gradᵀ·Grad` is the Neumann Laplacian used by diffusion and (in pyCERR) the H1 regularizer. |
| implicit-diffusion solve `B \ x`, `B = I + dt·σ·GradᵀGrad` | [`numerics.py::_DiffusionSolver`](numerics.py) | Backward-Euler (implicit) diffusion step. **Same operator B**; pyCERR diagonalizes it exactly with a 3-D DCT (`idctn(dctn(x)/eig)`) instead of MATLAB's sparse LU — O(N log N), machine-identical (see [§6](#6-where-pycerr-deliberately-differs)). |
| `dTrilinears3d.m` | [`numerics.py::_trilinear`](numerics.py) / `_trilinearApply` | Trilinear-interpolation matrix `S` (samples a field at departure points) **and** its spatial derivative `dS`, used by the semi-Lagrangian advection and by the velocity sensitivities. |

### 3.3 Forward model (advection–diffusion–source)

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `Inverse/SourceAdvecDiff.m` | [`numerics.py::sourceAdvecDiff`](numerics.py) | One interval's forward evolution: for each of `nt` sub-steps, **(i)** semi-Lagrangian advection (sample ρ at back-traced departure points `x − dt·v` via `S`), **(ii)** apply the relative source `(1 + dt·r)`, **(iii)** implicit diffusion `B⁻¹`. Returns the density trajectory ρ(:,k). |
| source-indicator `K` | `par["chi"]` (`cfg.chiStructNum`) | Optional mask restricting where the source `r` may act (χ = K in the MATLAB). χ = 1 everywhere by default. |

### 3.4 Objective & gradient

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `Inverse/get_Gamma.m` | [`numerics.py::getGamma`](numerics.py) | Evaluates Γ = Γ1(kinetic) + α·Γ2(source) + β·Γ3(fit) **+ η·Γ4(H1)**. Returns the components and the density trajectory. |
| adjoint of `get_Gamma` | [`numerics.py::gradGamma`](numerics.py) | Analytic **adjoint (reverse-mode) gradient** ∂Γ/∂(v,r): a backward sweep over the `nt` sub-steps propagating the cotangent through diffusion (Bᵀ = B), source, and advection. Finite-difference validated (`tests/test_uromt.py`). |

### 3.5 Sensitivities (for the exact Gauss–Newton Hessian)

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `Sensitivities/` operators (`Sensitivity`, `dSens`, adjoints) | [`numerics.py::forwardSensitivity`](numerics.py), `adjointSensitivity`, `precomputeSensDeriv` | The **tangent-linear model** `J = dρ_N/d(v,r)` and its adjoint `Jᵀ`. These implement the Gauss–Newton Hessian action `JᵀJ` matrix-free (no dense Jacobian). Adjoint validated against the tangent-linear model by the dot-product test. |

### 3.6 Inverse solver (per interval)

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `runUROMT.m` (interval loop) | [`solver.py::runUROMT`](solver.py) | Loops over consecutive frame pairs, warm-starting each interval from the previous velocity/source and (optionally) the previous evolved density (`reinitR`). Stores per-interval `u`, `r`, `rho`, `gamma`. |
| `Inverse/GNblock_ur.m` | [`solver.py::gnBlockExact`](solver.py) (default, `solver='gn'`) | **Gauss–Newton** step: form `H = H_reg + 2β·hd·JᵀJ (+ η H1) + λI`, solve `H dx = −grad` by **preconditioned CG** (Jacobi preconditioner from the regularization diagonal), then Armijo backtracking line search. Levenberg–Marquardt damping λ is adapted (↓ on success → toward pure GN, ↑ on failure). `maxUiter` few iterations = early-stopping regularization, exactly as the MATLAB. |
| (scipy alternative, no MATLAB analog) | [`solver.py::gnBlockUr`](solver.py) (`solver='lbfgs'`) | First-pass path: minimize the same Γ with L-BFGS-B on the adjoint gradient. Kept for reference/debugging; the GN block is the production solver. |

### 3.7 Part 1 — data preparation

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| Part-1 loop, `getData`, `getRange` | [`data.py::prepareData`](data.py) | Select longitudinal frames (time-ordered), build the ROI mask + bounding box, convert DCE signal → contrast concentration, crop, per-frame smooth, optional resize, mask. |
| `affine_diffusion_3d.m` | [`utils/image_proc.py::affineDiffusion3d`](../utils/image_proc.py) | Affine-invariant mean-curvature-flow smoothing of each frame (edge-preserving). `smooth_method='linear'` gives plain heat-flow; `'gaussian'` is a scipy stand-in. |
| signal→concentration (SPGR) | [`mri_metrics/dce_mri.py::normalizeToBaseline`](../mri_metrics/dce_mri.py), `intToConc`, `buildConcDict` | Convert DCE signal to contrast-agent concentration via the spoiled-gradient-echo (SPGR) relaxivity model (`normMethod='CC'`, needs TR/FA/T10/r1), or relative signal enhancement `S(t)/S₀` (`'RSE'`), or raw (`'none'`). Baseline S₀ = first frame(s), consumed in-sequence or supplied externally. This module is general pyCERR DCE code reused by urOMT. |
| `getRange` bbox padding | `cfg.bbox_pad`, `cfg.bbox_full_z` | Options to match MATLAB's ROI box (pad in-plane, use full z-extent). |

### 3.8 Part 3 — Eulerian analysis

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `runEULA.m` (+ `paramInitEULApar.m`) | [`analyze.py::runEULA`](analyze.py), `runEULAIntervals` | Time-average the fields over each interval and produce maps: advective **speed** `|v|`, **source rate** `r`, **Péclet** `|v|/|σ∇log ρ|`, and mean **flux** `ρ·v_eff`. `runEULAIntervals` writes one map set per interval (as MATLAB does per interval); `runEULA` gives the whole-run time-average. MATLAB `EulerS` ≡ pyCERR `effSpeed` = `|v_eff|`. |

### 3.9 Part 4 — Lagrangian analysis

| MATLAB | pyCERR | Theory |
|--------|--------|--------|
| `runGLAD.m` (+ `paramInitGLADpar.m`) | [`analyze.py::runGLAD`](analyze.py) | Integrate **pathlines** of the effective velocity `v_eff` from ROI seed points (sub-stepped Euler with `RegularGridInterpolator`), accumulating displacement, path speed and path Péclet. `direction=±1` traces transport forward or backward. Returns streamlines + per-path scalars. |

### 3.10 Part 5 — visualization

| MATLAB | pyCERR | Notes |
|--------|--------|-------|
| MATLAB figure scripts / flow renderers | [`viz.py`](viz.py) | Three back-ends: (a) matplotlib slice/3-D figures (`drawUROMTSlice`, `drawUROMT3D`); (b) napari vectors/tracks (`velocityVectors`, `pathlineTracks`, `showVelocity/Eulerian/Lagrangian`); (c) the embedded Qt overlay on the main pyCERR viewer (`drawUROMTOverlay`, `fieldToScan`, `eulerianMapToScan`, `pathlinesToScanVox`). No MATLAB equivalent — this is pyCERR's GUI integration. |
| — | [`export.py`](export.py) | Save Eulerian maps / vector fields as NIfTI on the scan grid (`saveEulerianMapsNii`, `saveVectorFieldNii`). pyCERR-only. |

### 3.11 Storage & driver

| MATLAB | pyCERR | Notes |
|--------|--------|-------|
| `.mat` files per interval on disk | `planC.urOMT` list of `UROMT` dataclasses ([`dataclasses/uromt.py`](../dataclasses/uromt.py)) | pyCERR keeps runs in-memory on the plan container (survives pickle), instead of MATLAB's file-per-interval outputs. |
| `driver_RatBrain.m` | [`__init__.py::runUROMTPipeline`](__init__.py) | The end-to-end driver. |

---

## 4. State-vector conventions (important for cross-reading)

Both codes use **Fortran (column-major) ordering** for flattened state, so index
math matches between the two:

- Grid `n = [nRow, nCol, nSlice]`; voxel flat index `= r + nRow·c + nRow·nCol·s`.
- Velocity per interval is stored flat as **`comp·N + voxel + 3N·k`** (the 3
  components stacked as N-blocks, per sub-step `k`). In pyCERR `runUROMT`
  unflattens this to `(3, N, nt)` via `reshape(N,3,nt,'F').transpose(1,0,2)`.
- Physical units: pyCERR runs the model in **millimetres** (voxel spacing read
  from `planC` in cm, ×10), matching the MATLAB mm grid, so σ, velocity, Péclet
  and displacement come out on the same scale.

---

## 5. Numerical equivalence (validated)

The pyCERR core is checked against first principles and, where data exists,
against MATLAB outputs (`tests/test_uromt.py`, `cerr/scripts/compare_uromt_*`):

- **Forward model**: identity when `v=0, r=0, σ=0`; diffusion solver inverts `B`
  to ~1e-15 vs the sparse operator on non-uniform spacing.
- **Gradient**: adjoint `gradGamma` matches central finite differences to
  ~1e-6 (interior) — including the new η H1 term.
- **Sensitivities**: adjoint/tangent-linear dot-product test to ~1e-14.
- **MATLAB similarity** (brain DCE, uptake interval): density magnitude matches
  well (median voxel ratio ≈ 0.97, spatial corr ≈ 0.78 with RSE), speed and
  Péclet correlate ≈ 0.47–0.58. Residual gaps are attributable to the different
  concentration model (no MATLAB source available to replicate its exact
  formula) and to velocity non-uniqueness, not to a discretization error.

---

## 6. Where pyCERR deliberately differs

These are intentional divergences, each equivalence-checked or additive:

1. **DCT diffusion solve** — MATLAB factorizes `B = I + dt·σ·GradᵀGrad` with
   sparse LU; pyCERR diagonalizes the (separable Neumann) Laplacian exactly with
   a 3-D DCT-II. Machine-identical result (`B` symmetric ⇒ serves forward and
   adjoint), O(N log N), ~9× faster.
2. **Matrix-free advection** — the forward-only line search applies the
   trilinear operator `S` as a gather+weighted-sum instead of building the sparse
   matrix. Bit-identical; ~2× faster. Gradient/sensitivity paths still build
   `S`/`Sᵀ` because they need the transpose and derivative.
3. **Velocity H1 smoothness (Γ4)** — *added, not in MATLAB.* Penalizes
   `η·hd·dt·Σ|∇v|²`, regularizing the velocity null space that the ρ-weighted
   misfit leaves under-determined in low-density voxels. `η = 0` recovers the
   original MATLAB objective exactly. On the brain data it lifts neighbouring-
   vector direction coherence from ~0.32 to ~0.87 with no loss (in fact a gain)
   in the density fit Γ3. Wired through `getGamma`/`gradGamma`/`gnBlockExact`
   (Hessian block + preconditioner diagonal).
4. **In-memory + planC** — runs live on `planC.urOMT` rather than `.mat` files;
   the concentration step reuses the general pyCERR `dce_mri` module.
5. **GUI/NIfTI integration** — napari, the embedded Qt overlay, the 3-D volume
   renderer, and NIfTI export have no MATLAB counterpart.

---

*Generated for the pyCERR urOMT tool (`cerr/uromt/`). MATLAB reference:
[xinan-nancy-chen/urOMT](https://github.com/xinan-nancy-chen/urOMT).*

# Phase-field cohesive fracture

FE-code for the paper *[F. Vicentini, J. Heinzmann, P. Carrara, L. De Lorenzis: Variational phase-field modeling of cohesive fracture with flexibly tunable strength surface, JMPS (2025)](https://doi.org/10.1016/j.jmps.2025.106424)*

Two implementations are provided:

| Directory | FEniCS version |
|-----------|---------------|
| `src/` | [FEniCSx](https://fenicsproject.org/) v0.9.0 (DOLFINx) |
| `src_legacy/` | [Legacy FEniCS](https://fenicsproject.org/download/archive/) 2019.1 (DOLFIN) |

---

## FEniCSx version (`src/`)

### Software requirements

This code was built for:

- [FEniCSx](https://fenicsproject.org/) v0.9.0
- [PETSc](https://petsc.org/release/) >=3.23 (for older versions the bisection line search algorithm is missing)

The easiest way to install everything is with [conda](https://anaconda.org/anaconda/conda), see the [download page of FEniCSx](https://fenicsproject.org/download/).
To install FEniCSx release 0.9.0, one has to run `conda install -c conda-forge fenics-dolfinx=0.9.0 mpich`.

### Usage

All of the codes can be found in `src/`.
The main scripts there are the files `phasefield_cohesive_1D.py` and `phasefield_cohesive_2D.py`.
The other files are:

- `dolfinx_jit_options.json`: JIT configuration for dolfinx
- `model.py`: all model definitions (energy contributions, strain, stress, ...)
- `petsc_interface.py`: classes that manage the interaction with PETSc SNES
- `petsc_options.yaml`: configuration for the PETSc solvers
- `problems.py`: classes containing the setup (mesh, parameters, boundary conditions, reaction force computation, ...) for each of the tests
- `solver.py`: class implementing the alternate minimization algorithm
- `utils.py`: convenience classes and functions for I/O

The 1D implementation can be run as:

```bash
python phasefield_cohesive_1D.py
```

while for the 2D implementation, several more options can be selected, such as the norm for the strength potential, or the target ratio $\ell/\ell_{\text{ch}}$.
For details on the available options, one can run:

```bash
python phasefield_cohesive_2D.py --help
```

The scripts will create the following output files:

- `solution_u.bp`, `solution_α.bp`, `solution_εσ.bp` (optional) and `solution_η.bp`: the resulting fields which can be viewed with [ParaView](https://www.paraview.org/) (it might be that ParaView has issues correctly displaying the greek symbols; in that case it is advised to replace them in [`src/utils.py`](./src/utils.py) lines 60,86,87 and [`src/phasefield_cohesive_1D.py`](./src/phasefield_cohesive_1D.py) line 92, as well as [`src/phasefield_cohesive_2D.py`](./src/phasefield_cohesive_2D.py) line 156),
- `damage.csv`: the maximum values of $\alpha$ and $\eta$ (or the trace and norm of the deviatoric part thereof) for each load step
- `energies.csv`: the energy contributions for each load step
- `reaction.csv`: the reaction forces for each load step
- `snes_*_convergence.csv`: the residual norms for each Newton iteration of the respective solver
- `staggered_convergence.csv`: the energies, the residual norm of the ($u,\eta$)-problem, the number of Newton iterations and function evaluations as well as the time spent for each staggered iteration

---

## Legacy FEniCS version (`src_legacy/`)

This is a port of the same implementation to [legacy FEniCS](https://fenicsproject.org/download/archive/) (DOLFIN, also known as FEniCS 2019.1).  The mathematical formulation, the staggered alternate-minimization scheme, the bound-constrained SNES solver and the PETSc linear algebra back-end are all identical to the FEniCSx version; only the Python API layer is adapted.

### Key differences from the FEniCSx version

| Aspect | FEniCSx (`src/`) | Legacy FEniCS (`src_legacy/`) |
|--------|-----------------|-------------------------------|
| Import | `dolfinx` | `dolfin` |
| Elements | `basix.ufl.element` | `FiniteElement / VectorElement / MixedElement` |
| Function spaces | `dolfinx.fem.functionspace` | `FunctionSpace` |
| Constants | `dolfinx.fem.Constant(domain, val)` | `Constant(val)` |
| Boundary conditions | `locate_dofs_topological/geometrical` + `dirichletbc` | `SubDomain` + `DirichletBC` |
| Assembly | `dolfinx.fem.petsc.assemble_vector/matrix` | `assemble(form, tensor=…)` |
| Output | ADIOS2 BP files via `VTXWriter` | XDMF files via `XDMFFile` |
| PETSc options file | YAML (requires PETSc ≥ 3.19) | Flat `.cfg` format |
| Line search | `bisection` (requires PETSc ≥ 3.23) | `bt` (back-tracking, all versions) |
| Topology dim | `mesh.topology.dim` | `mesh.topology().dim()` |
| Vector access | `function.x.array` | `function.vector().get_local()` |
| PETSc Vec access | `function.x.petsc_vec` | `as_backend_type(function.vector()).vec()` |

### Software requirements

- [Legacy FEniCS](https://fenicsproject.org/download/archive/) 2019.1
- PETSc ≥ 3.9 (any version shipped with FEniCS 2019.1 works)
- `mpi4py`, `petsc4py`, `numpy`

The easiest installation is:

```bash
conda install -c conda-forge fenics=2019.1
```

> **Note on the bisection line search:** The `bisection` line search type — which is particularly effective for the $r = \infty$-norm strength potential — requires PETSc ≥ 3.23.  Legacy FEniCS typically ships with an older PETSc, so the `bt` (back-tracking) line search is used by default in `src_legacy/petsc_options.cfg`.  If you build against PETSc ≥ 3.23 you can switch all `bt` entries to `bisection`.

### Usage

All codes are in `src_legacy/`. The main scripts are `phasefield_cohesive_1D.py` and `phasefield_cohesive_2D.py`.
The other files are:

- `model.py`: model definitions (energy contributions, strain, stress, ...)
- `petsc_interface.py`: classes that manage the interaction with PETSc SNES
- `petsc_options.cfg`: PETSc solver configuration (flat format)
- `problems.py`: mesh setup, boundary conditions, reaction force computation for each test
- `solver.py`: class implementing the alternate minimization algorithm
- `utils.py`: convenience classes and functions for I/O

Run the 1D implementation from within `src_legacy/`:

```bash
cd src_legacy
python phasefield_cohesive_1D.py
```

For 2D, with optional arguments:

```bash
python phasefield_cohesive_2D.py --help
python phasefield_cohesive_2D.py --rnorm 2 --theta 25
```

The scripts create the following output files (XDMF format instead of ADIOS2 BP):

- `solution_u.xdmf` / `solution_u.h5`: displacement field
- `solution_eta.xdmf` (1D) or `solution_etatr.xdmf` + `solution_etadev.xdmf` (2D): non-linear strain fields
- `solution_alpha.xdmf`: damage phase-field
- `solution_sigma_epsilon.xdmf` (optional): stress and strain projections
- `damage.csv`, `energies.csv`, `reaction.csv`, `snes_*_convergence.csv`, `staggered_convergence.csv`: same as the FEniCSx version

## Citation

If this code is useful for your research project, or you used it to obtain results for a publication, please cite our work as follows:

```bibtex
@article{vicentini_variational_2025,
  title = {Variational phase-field modeling of cohesive fracture with flexibly tunable strength surface},
  author = {Vicentini, F. and Heinzmann, J. and Carrara, P. and {De Lorenzis}, L.},
  year = {2025},
  journal = {Journal of the Mechanics and Physics of Solids},
  doi = {10.1016/j.jmps.2025.106424},
}
```

## Contact

For questions or requests, please contact [J. Heinzmann](mailto:jheinzmann@ethz.ch) or [F. Vicentini](mailto:fvicentini@ethz.ch).

## Acknowledgements

We gratefully acknowledge funding from the Swiss National Science Foundation (SNF) through Grant No. [200021-219407 'Phase-field modeling of fracture and fatigue: from rigorous theory to fast predictive simulations'](https://data.snf.ch/grants/grant/219407).

<small>© 2025 ETH Zürich</small>
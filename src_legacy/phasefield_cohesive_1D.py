# Copyright (C) 2025 ETH Zürich
# Creators: Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
#
# note: we solve the displacement and non-linear strain problem monolithically,
# following the same approach as in the FEniCSx version.

from mpi4py import MPI
import petsc4py

comm = MPI.COMM_WORLD
petsc4py.init(["-options_file", "./petsc_options.cfg"], comm=comm)
from petsc4py import PETSc

import numpy as np
from dolfin import (
    FiniteElement,
    MixedElement,
    FunctionSpace,
    Function,
    Constant,
    XDMFFile,
    assemble,
    as_backend_type,
    project,
)
import ufl
import time
import argparse

from petsc_interface import SNESProblem, SNESSolver
from solver import AlternateMinimizer
from utils import stdout, CSVWriter, MixedXDMFWriter
import problems
from model import CohesivePhaseField

# command line arguments
parser = argparse.ArgumentParser(
    description="1D implementation of the phase-field cohesive model (legacy FEniCS)"
)
parser.add_argument(
    "--narrow_element",
    action="store_true",
    help="insert a narrow element with ell=h/25 at the center of the domain, as mentioned in the caption of figure 20",
    default=False,
)
args = parser.parse_args()

# define solver parameters
solver_setup = {
    "tol_R": 1e-8,
    "nonzeroα_output": True,
    "εσ_output": True,
    "outdir": "./",
}

# =======================================================================================
# P R O B L E M   S E T U P
# =======================================================================================

# material parameters
L = 1.0
E = 10000.0
Gc = 0.001
ell = 0.05
σ_c = 5.0

ell_ch = Gc * E / (σ_c**2)
B = 2 * L / ell_ch

stdout(f"ell_ch = {ell_ch:.4e}, B = {B:.4e}")
stdout(f"ell/ ell_ch = {ell / ell_ch:.4e}")
stdout(f"σ_c = {σ_c:.4e}, E = {E:.4e}, Gc = {Gc:.4e}, ell = {ell:.4e}, L = {L:.4e}")
stdout(f"u_max = {5 * σ_c / E * 2 * L}")

Problem = problems.Bar(
    comm=comm,
    L=L,
    h=ell / 5,
    steps=500,
    u_max=5 * σ_c / E * 2 * L,
    narrow_element=args.narrow_element,
)

# create model
model = CohesivePhaseField(
    Problem.mesh,
    E=E,
    Gc=Gc,
    ell=ell,
    p_c=σ_c,
)

# create elements
# CG1 for displacement u (scalar in 1D), DG0 for non-linear strain η
u_element = FiniteElement("CG", Problem.mesh.ufl_cell(), 1)
η_element = FiniteElement("DG", Problem.mesh.ufl_cell(), 0)
uη_element = MixedElement([u_element, η_element])

α_element = FiniteElement("CG", Problem.mesh.ufl_cell(), 1)

# create function spaces
uη_space = FunctionSpace(Problem.mesh, uη_element)
uη = Function(uη_space)
(u, η) = ufl.split(uη)

α_space = FunctionSpace(Problem.mesh, α_element)
α = Function(α_space)
α.rename("alpha", "damage phase field")

# define upper / lower bound for damage and iteration state
α_lowerbound = Function(α_space)
α_upperbound = Function(α_space)

α_lowerbound.vector().zero()
α_upperbound.vector()[:] = 1.0

# define Dirichlet boundary conditions
bcs_uη, bcs_α = Problem.create_bcs_mixed(uη_space, α_space)

# =======================================================================================
# V A R I A T I O N A L   F O R M U L A T I O N
# =======================================================================================

# measure for integration
dx = ufl.Measure("dx", domain=Problem.mesh)

# energies
E_pot = model.ψ_el(model.ε(u), α, η) * dx
E_frac = model.ψ_frac(α) * dx
E_total = E_pot + E_frac

# residuals
E_uη = ufl.derivative(E_total, uη)
E_uηuη = ufl.derivative(E_uη, uη)

E_α = ufl.derivative(E_total, α)
E_αα = ufl.derivative(E_α, α)

# create test function for reaction force computation
uη_reaction = Function(uη_space)
virtual_work_form = ufl.action(E_uη, uη_reaction)

# =======================================================================================
# S O L V E R
# =======================================================================================

opts = PETSc.Options()

# solver for the displacement problem
problem_uη = SNESProblem(
    E_uη,
    E_uηuη,
    uη,
    bcs_uη,
    objective=E_pot,
)
solver_uη = SNESSolver(problem_uη, comm, "un_", convergence_monitoring=True)

# ── variable bounds for the (u, η) problem ──────────────────────────────────
# Collect η DOFs in parent-space ordering, sorted by their x-coordinate so
# that we can identify the central element (where η is allowed to be non-zero).
η_collapsed_space, η_to_parent = uη_space.sub(1).collapse()
η_parent_dofs = np.array(list(η_to_parent.values()), dtype=np.int32)
η_local_dofs = np.array(list(η_to_parent.keys()), dtype=np.int32)
η_coords_x = η_collapsed_space.tabulate_dof_coordinates()[:, 0]
sort_order = np.argsort(η_coords_x)
η_parent_dofs_sorted = η_parent_dofs[sort_order]

n_eta = len(η_parent_dofs_sorted)

# lower bound: η ≥ 0 everywhere, no lower bound on u
uη_lowerbound = solver_uη.x_vec.copy()
uη_lowerbound.set(PETSc.NINFINITY)
uη_lowerbound.setValues(η_parent_dofs_sorted, np.zeros(n_eta))
uη_lowerbound.assemble()

# upper bound for u: unconstrained; η = 0 except in the central element
uη_upperbound = solver_uη.x_vec.copy()
uη_upperbound.set(PETSc.INFINITY)
# zero out the upper bound for all η DOFs except the middle one
constrained_left = η_parent_dofs_sorted[: n_eta // 2]
constrained_right = η_parent_dofs_sorted[n_eta // 2 + 1 :]
if len(constrained_left) > 0:
    uη_upperbound.setValues(constrained_left, np.zeros(len(constrained_left)))
if len(constrained_right) > 0:
    uη_upperbound.setValues(constrained_right, np.zeros(len(constrained_right)))
uη_upperbound.assemble()

solver_uη.petsc_snes.setVariableBounds(uη_lowerbound, uη_upperbound)

# non-linear solver for the phase-field problem
problem_α = SNESProblem(E_α, E_αα, α, bcs_α)
solver_α = SNESSolver(problem_α, comm, "a_", convergence_monitoring=True)
solver_α.petsc_snes.setVariableBounds(
    as_backend_type(α_lowerbound.vector()).vec(),
    as_backend_type(α_upperbound.vector()).vec(),
)

# create staggered solver object
alternate_minimizer = AlternateMinimizer(
    comm,
    solver_uη,
    solver_α,
    E_pot,
    E_frac,
)

# =======================================================================================
# P O S T P R O C E S S I N G
# =======================================================================================

# open XDMF output files
xdmf_uη = MixedXDMFWriter(
    comm,
    [
        f"{solver_setup['outdir']}/solution_u.xdmf",
        f"{solver_setup['outdir']}/solution_eta.xdmf",
    ],
    uη,
)
xdmf_α = XDMFFile(comm, f"{solver_setup['outdir']}/solution_alpha.xdmf")

# strain and stress output (requires a scalar DG0 projection space in 1D)
if solver_setup["εσ_output"]:
    σ_space = FunctionSpace(Problem.mesh, "DG", 0)
    σ_nodes = Function(σ_space)
    σ_nodes.rename("sigma", "Cauchy stress")

    ε_space = FunctionSpace(Problem.mesh, "DG", 0)
    ε_nodes = Function(ε_space)
    ε_nodes.rename("epsilon", "strain")

    xdmf_εσ = XDMFFile(comm, f"{solver_setup['outdir']}/solution_sigma_epsilon.xdmf")

# initialize monitoring output
csv_energies = CSVWriter(
    f"{solver_setup['outdir']}/energies.csv", "step\tE_pot\tE_frac\tE_tot\n"
)
csv_reaction = CSVWriter(f"{solver_setup['outdir']}/reaction.csv", "step\tu\tF\n")
csv_damage = CSVWriter(
    f"{solver_setup['outdir']}/damage.csv", "step\talpha_max\teta_max\n"
)

# =======================================================================================
# I N C R E M E N T A L   S T A G G E R E D   S O L U T I O N
# =======================================================================================

# initial condition on the phase field
Problem.initial_α(solver_α, α, α_lowerbound)

# start timer
t_solution_start = time.time()

step = 0
while step <= Problem.steps:
    stdout("=" * 75 + f"\nstep: {step:3d} / {Problem.steps:3d}\n" + "=" * 75)

    # update Dirichlet BCs
    u_bar = Problem.update_bcs(step)

    # update the lower bound of the phase field (irreversibility)
    α_lowerbound.vector()[:] = α.vector()
    solver_α.petsc_snes.setVariableBounds(
        as_backend_type(α_lowerbound.vector()).vec(),
        as_backend_type(α_upperbound.vector()).vec(),
    )

    # staggered solution
    E_pot_step, E_frac_step = alternate_minimizer.solve(step)

    # write energies of converged state
    csv_energies.write(
        f"{step:04d}\t{E_pot_step:.8e}\t{E_frac_step:.8e}\t{E_pot_step + E_frac_step:.8e}\n",
    )

    # reaction forces
    F_react = Problem.compute_reaction(uη_reaction, virtual_work_form)
    csv_reaction.write(
        f"{step:04d}\t{u_bar:.8e}\t{F_react:.8e}\n",
    )

    # get maximum of the phase field (vector().max() is already MPI-reduced)
    α_max = α.vector().max()

    # output fields
    if (
        (not solver_setup["nonzeroα_output"] and α_max > 0.0)
        or (solver_setup["nonzeroα_output"])
        or (step == 1)
    ):
        xdmf_uη.write(step)
        xdmf_α.write(α, float(step))

        if solver_setup["εσ_output"]:
            σ_nodes.assign(project(model.σ(model.ε(u), α, η), σ_space))
            ε_nodes.assign(project(model.ε(u), ε_space))
            xdmf_εσ.write(σ_nodes, float(step))
            xdmf_εσ.write(ε_nodes, float(step))

    # get maximum of non-linear strains
    η_max = uη.split(deepcopy=True)[1].vector().max()
    csv_damage.write(
        f"{step:04d}\t{α_max:.8e}\t{η_max:.8e}\n",
    )

    # problem-dependent post-processing
    Problem.postprocess_step(step)

    # increment step
    step += 1

# stop timer
stdout(f"\ntotal time for solution: {time.time() - t_solution_start:.2f} s")

# close the output files
xdmf_uη.close()
xdmf_α.close()
if solver_setup["εσ_output"]:
    xdmf_εσ.close()

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
    VectorElement,
    FiniteElement,
    MixedElement,
    FunctionSpace,
    Function,
    Constant,
    XDMFFile,
    TensorFunctionSpace,
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
from model import CohesivePhaseField, elastic_constants

# command line arguments
parser = argparse.ArgumentParser(
    description="2D / 2D plane strain implementation of the phase-field cohesive model (legacy FEniCS)"
)
parser.add_argument(
    "--problem",
    type=str,
    help="problem to be computed, either ElasticDomainBlock (default) or CohesiveForcesBlockRigid",
    default="ElasticDomainBlock",
)
parser.add_argument(
    "--ell_ellch",
    type=float,
    help="target ratio of ell/ell_ch (default: 1/4)",
    default=1 / 4,
)
parser.add_argument(
    "--rnorm",
    type=str,
    help="norm for the strength potential, either 1, 2 (default) or inf",
    default="2",
)
parser.add_argument(
    "--theta",
    type=float,
    help="loading angle (default: 0.0)",
    default=0.0,
)
parser.add_argument(
    "--maxalpha",
    type=float,
    help="maximum alpha after which to stop the computations (default: 1e-5)",
    default=1e-5,
)
args = parser.parse_args()

# define solver parameters
solver_setup = {
    "tol_R": 1e-8,
    "nonzeroα_output": False,
    "εσ_output": False,
    "αmax_stop": args.maxalpha,
    "outdir": "./",
}

# =======================================================================================
# P R O B L E M   S E T U P
# =======================================================================================

# material parameters
E = 100
ν = 0.3
μ, _, κ = elastic_constants(3, E, ν)
Gc = 0.2
ell = 0.025

# ratio pc/τc
pc_τc = 1.0

# determine p_c and τ_c based on strain hardening limit
if args.ell_ellch == 1 / 4:
    args.ell_ellch -= 1e-6  # make sure we are slightly below the limit

if (args.rnorm == "2") or (args.rnorm == "inf"):
    w_1 = Gc / (2 * ell)
    p_c_lim = 2 * np.sqrt(args.ell_ellch) * np.sqrt(κ * w_1 / 2.0)
    τ_c_lim = 2 * np.sqrt(args.ell_ellch) * np.sqrt(μ * w_1)
    if pc_τc <= 1.0:
        p_c = pc_τc * min(p_c_lim, τ_c_lim)
        τ_c = min(p_c_lim, τ_c_lim)
    else:
        p_c = min(p_c_lim, τ_c_lim)
        τ_c = 1 / pc_τc * min(p_c_lim, τ_c_lim)

elif args.rnorm == "1":
    p_c_τ_c_lim = np.sqrt(2 * μ * κ * Gc / ((2 * μ + κ) * ell / args.ell_ellch))
    if pc_τc <= 1.0:
        p_c = pc_τc * p_c_τ_c_lim
        τ_c = p_c_τ_c_lim
    else:
        p_c = p_c_τ_c_lim
        τ_c = 1 / pc_τc * p_c_τ_c_lim

if args.problem == "ElasticDomainBlock":
    Problem = problems.ElasticDomainBlock(comm=comm, h=ell / 5, Θ=args.theta)
elif args.problem == "CohesiveForcesBlockRigid":
    Problem = problems.CohesiveForcesBlockRigid(comm=comm, h=ell / 5, Θ=args.theta)
else:
    raise ValueError(f"Unknown problem type: {args.problem}")

# =======================================================================================

# create model
model = CohesivePhaseField(
    Problem.mesh,
    E=E,
    ν=ν,
    Gc=Gc,
    ell=ell,
    p_c=p_c,
    τ_c=τ_c,
    stress_state="plane_strain",
    r_norm=args.rnorm,
)

# create elements
# CG1 vector for u, DG0 scalars for eta_tr and eta_dev, CG1 scalar for alpha
u_element = VectorElement("CG", Problem.mesh.ufl_cell(), 1, dim=2)
ηtr_element = FiniteElement("DG", Problem.mesh.ufl_cell(), 0)
ηdev_element = FiniteElement("DG", Problem.mesh.ufl_cell(), 0)
α_element = FiniteElement("CG", Problem.mesh.ufl_cell(), 1)

uη_element = MixedElement([u_element, ηtr_element, ηdev_element])

# create function spaces and functions
uη_space = FunctionSpace(Problem.mesh, uη_element)
uη = Function(uη_space)
(u, ηtr, ηdev) = ufl.split(uη)

α_space = FunctionSpace(Problem.mesh, α_element)
α = Function(α_space)
α.rename("alpha", "damage phase field")

# define upper / lower bound for damage and iteration state
α_lowerbound = Function(α_space)
α_upperbound = Function(α_space)

α_lowerbound.vector().zero()
α_upperbound.vector()[:] = 1.0

# boundary conditions
bcs_uη, bcs_α = Problem.create_bcs_mixed(uη_space, α_space)

# =======================================================================================
# V A R I A T I O N A L   F O R M U L A T I O N
# =======================================================================================

# measure for integration
dx = ufl.Measure("dx", domain=Problem.mesh)

# energies
E_pot = model.ψ_el(model.ε(u), α, ηtr, ηdev) * dx
E_frac = model.ψ_frac(α) * dx
E_total = E_pot + E_frac

# displacement problem
E_uη = ufl.derivative(E_total, uη)
E_uηuη = ufl.derivative(E_uη, uη)

# phase-field problem
E_α = ufl.derivative(E_total, α)
E_αα = ufl.derivative(E_α, α)

# create test function and virtual work form for reaction forces
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
solver_uη = SNESSolver(problem_uη, comm, "untrndev_", convergence_monitoring=True)

# ── variable bounds for the (u, η_tr, η_dev) problem ────────────────────────
# η_tr ≥ 0 and η_dev ≥ 0 everywhere; u is unconstrained
ηtr_collapsed_space, ηtr_to_parent = uη_space.sub(1).collapse()
ηtr_parent_dofs = np.array(list(ηtr_to_parent.values()), dtype=np.int32)

ηdev_collapsed_space, ηdev_to_parent = uη_space.sub(2).collapse()
ηdev_parent_dofs = np.array(list(ηdev_to_parent.values()), dtype=np.int32)

uη_lowerbound = solver_uη.x_vec.copy()
uη_lowerbound.set(PETSc.NINFINITY)
uη_lowerbound.setValues(ηtr_parent_dofs, np.zeros(len(ηtr_parent_dofs)))
uη_lowerbound.setValues(ηdev_parent_dofs, np.zeros(len(ηdev_parent_dofs)))
uη_lowerbound.assemble()

uη_upperbound = solver_uη.x_vec.copy()
uη_upperbound.set(PETSc.INFINITY)

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
    tol_Rnorm=solver_setup["tol_R"],
)

# =======================================================================================
# P O S T P R O C E S S I N G
# =======================================================================================

# open XDMF output files (three files: u, eta_tr, eta_dev)
xdmf_uη = MixedXDMFWriter(
    comm,
    [
        f"{solver_setup['outdir']}/solution_u.xdmf",
        f"{solver_setup['outdir']}/solution_etatr.xdmf",
        f"{solver_setup['outdir']}/solution_etadev.xdmf",
    ],
    uη,
)
xdmf_α = XDMFFile(comm, f"{solver_setup['outdir']}/solution_alpha.xdmf")

# strain and stress output
if solver_setup["εσ_output"]:
    if model.stress_state == "plane_strain":
        σε_dim = 3
    else:
        σε_dim = model.dim

    σ_space = TensorFunctionSpace(Problem.mesh, "DG", 0, shape=(σε_dim, σε_dim))
    σ_nodes = Function(σ_space)
    σ_nodes.rename("sigma", "Cauchy stress")

    ε_space = TensorFunctionSpace(Problem.mesh, "DG", 0, shape=(σε_dim, σε_dim))
    ε_nodes = Function(ε_space)
    ε_nodes.rename("epsilon", "strain")

    xdmf_εσ = XDMFFile(comm, f"{solver_setup['outdir']}/solution_sigma_epsilon.xdmf")

# initialize monitoring output
csv_energies = CSVWriter(
    f"{solver_setup['outdir']}/energies.csv", "step\tE_pot\tE_frac\tE_tot\n"
)
csv_reaction = CSVWriter(
    f"{solver_setup['outdir']}/reaction.csv", "step\tu_x\tu_y\tF_x\tF_y\n"
)
csv_damage = CSVWriter(
    f"{solver_setup['outdir']}/damage.csv",
    "step\talpha_max\tetatr_max\tetadev_max\n",
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
    stdout("=" * 75 + f"\nstep: {step:3d} / {Problem.steps - 1:3d}\n" + "=" * 75)

    # update Dirichlet BCs
    u_x, u_y = Problem.update_bcs(step)

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
    F_x, F_y = Problem.compute_reaction(uη_reaction, virtual_work_form)
    csv_reaction.write(
        f"{step:04d}\t{u_x:.8e}\t{u_y:.8e}\t{F_x:.8e}\t{F_y:.8e}\n",
    )

    # get maximum of the phase field (already MPI-reduced in dolfin)
    α_max = α.vector().max()

    # output fields
    if (
        (not solver_setup["nonzeroα_output"] and α_max > 0.0)
        or (solver_setup["nonzeroα_output"])
        or (step == 1)
        or (step == Problem.steps - 1)
    ):
        xdmf_uη.write(step)
        xdmf_α.write(α, float(step))

        if solver_setup["εσ_output"]:
            σ_nodes.assign(project(model.σ(model.ε(u), α, ηtr, ηdev), σ_space))
            ε_nodes.assign(project(model.ε(u), ε_space))
            xdmf_εσ.write(σ_nodes, float(step))
            xdmf_εσ.write(ε_nodes, float(step))

    # get maximum of non-linear strains
    ηtr_max = uη.split(deepcopy=True)[1].vector().max()
    ηdev_max = uη.split(deepcopy=True)[2].vector().max()
    csv_damage.write(
        f"{step:04d}\t{α_max:.8e}\t{ηtr_max:.8e}\t{ηdev_max:.8e}\n",
    )

    # problem-dependent post-processing
    Problem.postprocess_step(step)

    # stop the computations once crack has nucleated
    if α_max > solver_setup["αmax_stop"]:
        stdout(
            f"\nstopping due to alpha_max = {α_max:.3e} > {solver_setup['αmax_stop']:.3e}"
        )
        break

    step += 1

# stop timer
stdout(f"\ntotal time for solution: {time.time() - t_solution_start:.2f} s")

# close the output files
xdmf_uη.close()
xdmf_α.close()
if solver_setup["εσ_output"]:
    xdmf_εσ.close()

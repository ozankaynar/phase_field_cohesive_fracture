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

import ufl_legacy as ufl
from dolfin import (
    Function,
    assemble,
    as_backend_type,
    PETScVector,
    PETScMatrix,
    PETScOptions,
)
from mpi4py import MPI
from petsc4py import PETSc
import typing

from utils import CSVWriter


# =======================================================================================
# S N E S   P R O B L E M
# =======================================================================================
class SNESProblem:
    """
    Wraps a pair of UFL forms (residual F and Jacobian J) together with a
    Function (the unknown) and a list of DirichletBCs so that they can be
    handed to a raw PETSc SNES object.

    This mirrors the SNESProblem class in the FEniCSx version but uses
    legacy FEniCS (dolfin) assembly routines.
    """

    def __init__(
        self,
        F: ufl.Form,
        J: ufl.Form,
        u: Function,
        bcs: list,
        objective: typing.Optional[ufl.Form] = None,
    ) -> None:
        self.L = F
        self.a = J
        self.bcs = bcs
        self.u = u
        self.objective = objective

        # Pre-allocate dolfin PETSc containers.
        # The first assembly here is intentional: it populates the sparsity
        # pattern and creates correctly sized PETSc Vec / Mat objects.
        # All subsequent calls in the F() and J() callbacks reuse these same
        # objects via assemble(form, tensor=…), so the PETSc Vec / Mat
        # references registered with the SNES remain valid throughout the solve.
        self._b_dolfin = PETScVector()
        assemble(self.L, tensor=self._b_dolfin)

        self._A_dolfin = PETScMatrix()
        assemble(self.a, tensor=self._A_dolfin)

        if self.objective is not None:
            self._obj_form = self.objective

    def f(self, snes: PETSc.SNES, x: PETSc.Vec) -> float:
        """
        Evaluate the objective function (used by the bisection line search).

        snes  SNES object
        x     vector containing the current solution
        """
        # synchronise the dolfin Function with the SNES iterate x
        x.copy(as_backend_type(self.u.vector()).vec())
        as_backend_type(self.u.vector()).update_ghost_values()

        # assemble the scalar objective (already MPI-reduced by dolfin)
        obj = float(assemble(self._obj_form))

        return obj

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """
        Assemble the residual F into the PETSc vector b.

        snes  SNES object
        x     vector containing the current solution
        b     vector to assemble the residual into
        """
        # synchronise the dolfin Function with the SNES iterate x
        x.copy(as_backend_type(self.u.vector()).vec())
        as_backend_type(self.u.vector()).update_ghost_values()

        # assemble residual into the pre-allocated dolfin vector
        # (this also updates the underlying PETSc Vec b_petsc = b)
        assemble(self.L, tensor=self._b_dolfin)

        # apply Dirichlet BCs:
        # bc.apply(b_vec, x_vec) sets b[bc_dofs] = 0 for residual DOFs and
        # accounts for lifting of inhomogeneous BCs (equivalent to
        # apply_lifting + set_bc in FEniCSx)
        for bc in self.bcs:
            bc.apply(self._b_dolfin, self.u.vector())

        # copy the assembled + BC-corrected residual into the SNES buffer b
        as_backend_type(self._b_dolfin).vec().copy(b)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def J(
        self, snes: PETSc.SNES, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat
    ) -> None:
        """
        Assemble the Jacobian matrix.

        snes  SNES object
        x     vector containing the current solution
        A     matrix to assemble the Jacobian into
        """
        # synchronise the dolfin Function with the SNES iterate x
        x.copy(as_backend_type(self.u.vector()).vec())
        as_backend_type(self.u.vector()).update_ghost_values()

        # assemble Jacobian into the pre-allocated dolfin matrix
        assemble(self.a, tensor=self._A_dolfin)
        for bc in self.bcs:
            bc.apply(self._A_dolfin)

        # copy values into the SNES Jacobian matrix A
        # (they share the same sparsity pattern, so this is efficient)
        A.zeroEntries()
        as_backend_type(self._A_dolfin).mat().copy(A)
        A.assemble()


# =======================================================================================
# S N E S   S O L V E R
# =======================================================================================
class SNESSolver:
    """
    Solver class used for the above SNESProblem class.

    Mirrors the SNESSolver class in the FEniCSx version. Uses raw petsc4py
    SNES so that the same PETSc options (vinewtonrsls, variable bounds, …)
    work unchanged.
    """

    def __init__(
        self,
        snes_problem: SNESProblem,
        comm: MPI.Comm,
        options_prefix: str = "",
        convergence_monitoring: bool = False,
    ) -> None:
        self.snes_problem = snes_problem
        self.comm = comm
        self.options_prefix = options_prefix
        self.convergence_monitoring = convergence_monitoring

        # create SNES object
        self.petsc_snes = PETSc.SNES().create(comm=self.comm)

        # the solution vector is owned by the dolfin Function; copy it to get
        # a properly sized SNES iterate vector
        self.x_vec = as_backend_type(snes_problem.u.vector()).vec().copy()

        # create residual and Jacobian PETSc objects from the pre-assembled
        # dolfin containers (they carry the correct size and sparsity)
        self.b_vec = as_backend_type(snes_problem._b_dolfin).vec().copy()
        self.J_mat = as_backend_type(snes_problem._A_dolfin).mat().duplicate(
            copy=True
        )

        # register function and Jacobian with SNES
        if self.snes_problem.objective is not None:
            self.petsc_snes.setObjective(self.snes_problem.f)
        self.petsc_snes.setFunction(self.snes_problem.F, self.b_vec)
        self.petsc_snes.setJacobian(self.snes_problem.J, self.J_mat)

        # set options prefix, then read options from the PETSc database
        self.petsc_snes.setOptionsPrefix(options_prefix)
        self.petsc_snes.setFromOptions()
        self.petsc_snes.setErrorIfNotConverged(True)
        self.petsc_snes.logConvergenceHistory(norm=2)

        # initialise convergence history file
        if self.convergence_monitoring:
            self.csv_convergence = CSVWriter(
                f"snes_{self.options_prefix}convergence.csv",
                "step\tstagg_iter\tNR_iter\t||R||_2\n",
            )

    def solve(self, step: int = 0, iteration: int = 0) -> tuple:
        # reset convergence history
        self.petsc_snes.setConvergenceHistory()

        # initialise the SNES iterate from the current dolfin Function
        as_backend_type(self.snes_problem.u.vector()).vec().copy(self.x_vec)

        # solve
        self.petsc_snes.solve(None, self.x_vec)
        iterations = self.petsc_snes.getIterationNumber()
        funcevals = self.petsc_snes.getFunctionEvaluations()

        # copy SNES solution back into the dolfin Function
        self.x_vec.copy(as_backend_type(self.snes_problem.u.vector()).vec())
        as_backend_type(self.snes_problem.u.vector()).update_ghost_values()

        # output convergence history
        if self.convergence_monitoring:
            (R_norm_hist, _) = self.petsc_snes.getConvergenceHistory()
            for i in range(len(R_norm_hist)):
                self.csv_convergence.write(
                    f"{step:04d}\t{iteration:04d}\t{i:04d}\t{R_norm_hist[i]:.8e}\n",
                )

        return iterations, funcevals

    def residual_norm(self) -> float:
        # compute residual norm without performing any iterations by
        # temporarily setting max_it=0 (same trick as in the FEniCSx version)
        self.petsc_snes.setTolerances(max_it=0)
        self.petsc_snes.setErrorIfNotConverged(False)

        as_backend_type(self.snes_problem.u.vector()).vec().copy(self.x_vec)
        self.petsc_snes.solve(None, self.x_vec)

        R_norm = self.petsc_snes.getFunctionNorm()

        self.petsc_snes.setTolerances(
            max_it=PETSc.Options().getInt(f"{self.options_prefix}snes_max_it")
        )
        self.petsc_snes.setErrorIfNotConverged(True)

        return R_norm

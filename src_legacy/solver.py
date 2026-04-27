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

from mpi4py import MPI

from dolfin import assemble
import time
from utils import stdout, CSVWriter
from petsc_interface import SNESSolver


class AlternateMinimizer:
    def __init__(
        self,
        comm: MPI.Comm,
        solver_uη: SNESSolver,
        solver_α: SNESSolver,
        E_pot_form,
        E_frac_form,
        tol_Rnorm: float = 1e-6,
        max_iter: int = 100000,
        monitoring: bool = True,
    ):
        # store MPI communicator
        self.comm = comm

        # store the separate solvers
        self.solver_uη = solver_uη
        self.solver_α = solver_α

        # store UFL forms for energy contributions
        self.E_pot_form = E_pot_form
        self.E_frac_form = E_frac_form

        # store all parameters
        self.tol_Rnorm = tol_Rnorm
        self.max_iter = max_iter
        self.monitoring = monitoring

        # initialise the monitoring output
        if self.monitoring:
            self.csv_staggered_convergence = CSVWriter(
                "staggered_convergence.csv",
                "step\titeration\tE_pot\tE_frac\tE_tot\tR_uη_norm"
                "\tNR_iter_uη\tfuncevals_uη\tNR_iter_α\tfuncevals_α\ttime_staggered\n",
            )

    def solve(self, step: int):
        # start timer
        t_staggered_start = time.time()

        # loop over the staggered iterations
        for iteration in range(self.max_iter):
            if iteration > 0:
                stdout("_" * 75)

            stdout(f"staggered iteration: {iteration:3d}")

            # minimize wrt u
            stdout("\nminimizing wrt uη")
            solver_uη_iterations, solver_uη_funcevals = self.solver_uη.solve(
                step, iteration
            )

            # minimize wrt α
            stdout("\nminimizing wrt α")
            solver_α_iterations, solver_α_funcevals = self.solver_α.solve(
                step, iteration
            )

            # re-evaluate the uη residual
            R_uη_norm = self.solver_uη.residual_norm()

            # compute the energies (assemble already performs MPI reduction in dolfin)
            E_pot_iter = float(assemble(self.E_pot_form))
            E_frac_iter = float(assemble(self.E_frac_form))

            stdout(f"\n||R_uη||_2 = {R_uη_norm}")

            # output staggered solver monitoring
            if self.monitoring:
                self.csv_staggered_convergence.write(
                    f"{step:04d}\t{iteration:04d}\t{E_pot_iter:.8e}\t{E_frac_iter:.8e}\t{E_pot_iter + E_frac_iter:.8e}\t{R_uη_norm:.8e}"
                    f"\t{solver_uη_iterations:04d}\t{solver_uη_funcevals:04d}\t{solver_α_iterations:04d}\t{solver_α_funcevals:04d}"
                    f"\t{time.time() - t_staggered_start:.8e}\n",
                )

            # check convergence criteria and if fulfilled exit the staggered loop
            if R_uη_norm <= self.tol_Rnorm:
                break

        else:
            raise RuntimeError(
                f"convergence not reached after {iteration:3d} iterations"
            )

        # return the energies of the last converged state
        return E_pot_iter, E_frac_iter

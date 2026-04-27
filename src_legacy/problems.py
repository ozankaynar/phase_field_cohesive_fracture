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

import numpy as np
from mpi4py import MPI
from dolfin import (
    IntervalMesh,
    RectangleMesh,
    Point,
    Function,
    Constant,
    SubDomain,
    DirichletBC,
    assemble,
    near,
    DOLFIN_EPS,
)

from utils import CSVWriter


# =======================================================================================
# O N E   D I M E N S I O N A L   B A R
# =======================================================================================
class Bar:
    """
    1D bar which is clamped on the left end, and has a prescribed displacement on the right end,
    Section 5.2 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.005,
        steps: int = 500,
        u_max: float = 0.001,
        narrow_element: bool = False,
    ) -> None:
        # save geometric and mesh parameters
        self.L = L
        self.h = h

        # define mesh
        n = int(2 * L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at x=0

        if narrow_element:
            # insert one more element in the middle, which is then shrunk
            n += 1

        # note: the bar has a length of 2L, as in the theoretical explanations
        self.mesh = IntervalMesh(comm, n, -self.L, self.L)

        # locally change a single element to be much smaller
        if narrow_element:
            assert comm.size == 1, (
                "narrow element mesh adjustment only works in serial with the current implementation"
            )

            # get writable view of nodal coordinates
            coords = self.mesh.coordinates()

            # create array with original spacing without the small element
            x_h = np.linspace(-self.L, self.L, n)

            # replace the nodes with the original spacing except the middle one
            coords[: int(n / 2), 0] = x_h[: int(n / 2)]
            coords[int(n / 2 + 1) :, 0] = x_h[int(n / 2) :]

            # adjust one coordinate of the middle element to have size h/5
            coords[int(n / 2), 0] = coords[int(n / 2 + 1), 0] - h / 5

        # initialize load stepping
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

    def create_bcs_mixed(self, uη_space, α_space):
        # Get subspace of mixed space for displacement (sub-space 0)
        u_subspace = uη_space.sub(0)

        # left edge: fixed (zero) displacement
        left_boundary = CompiledSubDomainNear("near(x[0], val, tol)", val=-self.L)
        self.bc_u_left = Constant(0.0)
        bc_u_left = DirichletBC(u_subspace, self.bc_u_left, left_boundary)

        # right edge: prescribed displacement
        right_boundary = CompiledSubDomainNear("near(x[0], val, tol)", val=self.L)
        self.bc_u_right = Constant(0.0)
        bc_u_right = DirichletBC(u_subspace, self.bc_u_right, right_boundary)

        bcs_uη = [bc_u_left, bc_u_right]
        self.bcs_uη = bcs_uη

        # no Dirichlet BCs on the phase field
        bcs_α = []

        return bcs_uη, bcs_α

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # set everything to zero except the left BC
        uη_reaction.vector().zero()
        self.bc_u_left.assign(Constant(1.0))
        self.bc_u_right.assign(Constant(0.0))
        for bc in self.bcs_uη:
            bc.apply(uη_reaction.vector())

        # assemble virtual work; dolfin's assemble is already MPI-reduced
        F_x = float(assemble(virtual_work_form))

        # reset the left BC to zero
        self.bc_u_left.assign(Constant(0.0))

        return F_x

    def update_bcs(self, step):
        u_bar = self.loadsteps[step]
        self.bc_u_right.assign(Constant(u_bar))

        return u_bar

    def postprocess_step(self, step):
        pass


# =======================================================================================
# H E L P E R   S U B D O M A I N S
# =======================================================================================
# In legacy FEniCS, SubDomain.inside() must be a regular Python method (not a lambda).
# We define reusable helper classes here.

class CompiledSubDomainNear(SubDomain):
    """SubDomain that marks vertices whose x[0] coordinate is close to a target value."""

    def __init__(self, _expr_unused, val: float, tol: float = 1000 * DOLFIN_EPS):
        super().__init__()
        self.val = val
        self.tol = tol

    def inside(self, x, on_boundary):
        return near(x[0], self.val, self.tol)


class SubDomainAbsX(SubDomain):
    """SubDomain that marks vertices where |x[0]| is close to val (pointwise)."""

    def __init__(self, val: float, tol: float = 1000 * DOLFIN_EPS):
        super().__init__()
        self.val = val
        self.tol = tol

    def inside(self, x, on_boundary):
        return near(abs(x[0]), self.val, self.tol)


class SubDomainAbsY(SubDomain):
    """SubDomain that marks vertices where |x[1]| is close to val (pointwise)."""

    def __init__(self, val: float, tol: float = 1000 * DOLFIN_EPS):
        super().__init__()
        self.val = val
        self.tol = tol

    def inside(self, x, on_boundary):
        return near(abs(x[1]), self.val, self.tol)


class SubDomainAbsXorY(SubDomain):
    """SubDomain that marks vertices where |x[0]| or |x[1]| is close to val (pointwise)."""

    def __init__(self, val: float, tol: float = 1000 * DOLFIN_EPS):
        super().__init__()
        self.val = val
        self.tol = tol

    def inside(self, x, on_boundary):
        return near(abs(x[0]), self.val, self.tol) or near(abs(x[1]), self.val, self.tol)


class SubDomainLeftHalf(SubDomain):
    """SubDomain matching all vertices in the left half (x[0] < 0)."""

    def inside(self, x, on_boundary):
        return x[0] < 0.0


class SubDomainRightHalf(SubDomain):
    """SubDomain matching all vertices in the right half (x[0] > 0)."""

    def inside(self, x, on_boundary):
        return x[0] > 0.0


# =======================================================================================
# E L A S T I C   D O M A I N   B L O C K
# =======================================================================================
class ElasticDomainBlock:
    """
    quadratic block with a homogeneous deformation state, used to validate the strength surface,
    Section 5.3 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.01,
        Θ: float = 25.0,
        steps: int = 2500,
        u_max: float = 0.25,
    ):
        # save parameters
        self.L = L

        # create mesh
        n = int(self.L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at L/2

        self.mesh = RectangleMesh(
            comm,
            Point(-self.L / 2, -self.L / 2),
            Point(self.L / 2, self.L / 2),
            n,
            n,
        )

        # initialize load stepping
        self.Θ = Θ
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

        # prepare post-processing of each step (output of deformation state)
        self.csv_deformation = CSVWriter(
            "deformation.csv", "step\tε_xx\tε_yy\tεtr\tεdev\n"
        )

    def create_bcs_mixed(self, uη_space, α_space):
        # ── BCs on ux (x-displacement, sub(0).sub(0)) ──────────────────────
        ux_subspace = uη_space.sub(0).sub(0)

        self.bc_ux = Function(ux_subspace.collapse())
        self.bc_ux.vector().zero()

        bc_ux = DirichletBC(
            ux_subspace,
            self.bc_ux,
            SubDomainAbsX(self.L / 2),
            "pointwise",
        )

        # ── BCs on uy (y-displacement, sub(0).sub(1)) ──────────────────────
        uy_subspace = uη_space.sub(0).sub(1)

        self.bc_uy = Function(uy_subspace.collapse())
        self.bc_uy.vector().zero()

        bc_uy = DirichletBC(
            uy_subspace,
            self.bc_uy,
            SubDomainAbsY(self.L / 2),
            "pointwise",
        )

        bcs_upq = [bc_ux, bc_uy]

        # ── BCs on phase field α ────────────────────────────────────────────
        bc_α0 = DirichletBC(
            α_space,
            Constant(0.0),
            SubDomainAbsXorY(self.L / 2),
            "pointwise",
        )
        bcs_α = [bc_α0]

        return bcs_upq, bcs_α

    def update_bcs(self, step):
        loadstep_factor_x = self.loadsteps[step] * np.cos(np.pi * self.Θ / 180)
        loadstep_factor_y = self.loadsteps[step] * np.sin(np.pi * self.Θ / 180)

        # update DOF values using tabulated coordinates
        ux_coords = self.bc_ux.function_space().tabulate_dof_coordinates()
        self.bc_ux.vector()[:] = loadstep_factor_x * np.sign(ux_coords[:, 0])

        uy_coords = self.bc_uy.function_space().tabulate_dof_coordinates()
        self.bc_uy.vector()[:] = loadstep_factor_y * np.sign(uy_coords[:, 1])

        return loadstep_factor_x, loadstep_factor_y

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # note: computing a reaction force does not make sense for this test since
        # it has a symmetric prescribed displacement on opposing edges
        return 0.0, 0.0

    def postprocess_step(self, step):
        # get strain state in xx-yy space
        ε_xx = 2 * np.cos(np.pi * self.Θ / 180) * self.loadsteps[step]
        ε_yy = 2 * np.sin(np.pi * self.Θ / 180) * self.loadsteps[step]

        # get strain in volumetric-deviatoric space
        εtr = ε_xx + ε_yy
        εdev = np.sqrt(2 / 3 * (ε_xx**2 + ε_yy**2 - ε_xx * ε_yy))

        self.csv_deformation.write(
            f"{step:04d}\t{ε_xx:.8e}\t{ε_yy:.8e}\t{εtr:.8e}\t{εdev:.8e}\n"
        )


# =======================================================================================
# C O H E S I V E   F O R C E S   B L O C K
# =======================================================================================
class CohesiveForcesBlockRigid:
    """
    quadratic block with a pre-defined deformation and crack location, used to validate the cohesive law,
    Section 5.4 in the paper
    """

    def __init__(
        self,
        comm: MPI.Comm,
        L: float = 1.0,
        h: float = 0.01,
        Θ: float = 25.0,
        steps: int = 2500,
        u_max: float = 0.5,
        narrow_elementrow: bool = True,
    ):
        # save geometric parameters
        self.L = L

        # create mesh
        n = int(self.L / h)
        if n % 2 == 0:
            n += 1  # make sure to have no nodes at L/2

        self.n = n

        self.mesh = RectangleMesh(
            comm,
            Point(-self.L / 2, -self.L / 2),
            Point(self.L / 2, self.L / 2),
            n,
            n,
        )

        # locally change a single element row to be much smaller
        if narrow_elementrow:
            assert comm.size == 1, (
                "narrow element mesh adjustment only works in serial with the current implementation"
            )

            coords = self.mesh.coordinates()

            h_orig = self.L / self.n
            h_narrow = h_orig / 5
            a = (h_orig - h_narrow) / (h_orig - self.L)
            b = -a * self.L / 2

            right_idx = np.where(coords[:, 0] > 0)[0]
            coords[right_idx, 0] = (1 - a) * coords[right_idx, 0] - b

            left_idx = np.where(coords[:, 0] < 0)[0]
            coords[left_idx, 0] = (
                coords[left_idx, 0] - a * coords[left_idx, 0] + b
            )

        # initialize load stepping
        self.Θ = Θ
        self.steps = steps
        self.loadsteps = np.linspace(0, u_max, steps + 1)

    def create_bcs_mixed(self, uη_space, α_space):
        ux_subspace = uη_space.sub(0).sub(0)
        uy_subspace = uη_space.sub(0).sub(1)

        ux_collapsed = ux_subspace.collapse()
        uy_collapsed = uy_subspace.collapse()

        # ── left half: ux = 0 ───────────────────────────────────────────────
        self.bc_ux_fixed = Function(ux_collapsed)
        self.bc_ux_fixed.vector().zero()
        bc_ux_fixed = DirichletBC(
            ux_subspace,
            self.bc_ux_fixed,
            SubDomainLeftHalf(),
            "pointwise",
        )

        # ── right half: ux prescribed ───────────────────────────────────────
        self.bc_ux_prescribed = Function(ux_collapsed)
        self.bc_ux_prescribed.vector().zero()
        bc_ux_prescribed = DirichletBC(
            ux_subspace,
            self.bc_ux_prescribed,
            SubDomainRightHalf(),
            "pointwise",
        )

        # ── left half: uy = 0 ───────────────────────────────────────────────
        self.bc_uy_fixed = Function(uy_collapsed)
        self.bc_uy_fixed.vector().zero()
        bc_uy_fixed = DirichletBC(
            uy_subspace,
            self.bc_uy_fixed,
            SubDomainLeftHalf(),
            "pointwise",
        )

        # ── right half: uy prescribed ───────────────────────────────────────
        self.bc_uy_prescribed = Function(uy_collapsed)
        self.bc_uy_prescribed.vector().zero()
        bc_uy_prescribed = DirichletBC(
            uy_subspace,
            self.bc_uy_prescribed,
            SubDomainRightHalf(),
            "pointwise",
        )

        bcs_upq = [bc_ux_fixed, bc_uy_fixed, bc_ux_prescribed, bc_uy_prescribed]
        self.bcs_upq = bcs_upq

        # no Dirichlet BCs on the phase field
        bcs_α = []

        return bcs_upq, bcs_α

    def update_bcs(self, step):
        loadstep_factor_x = self.loadsteps[step] * np.cos(np.pi * self.Θ / 180)
        loadstep_factor_y = self.loadsteps[step] * np.sin(np.pi * self.Θ / 180)

        self.bc_ux_prescribed.vector()[:] = loadstep_factor_x
        self.bc_uy_prescribed.vector()[:] = loadstep_factor_y

        return loadstep_factor_x, loadstep_factor_y

    def initial_α(self, solver_α, α, α_lowerbound):
        pass

    def compute_reaction(self, uη_reaction, virtual_work_form):
        # ── x-direction reaction ────────────────────────────────────────────
        uη_reaction.vector().zero()
        self.bc_ux_fixed.vector()[:] = 1.0
        self.bc_uy_fixed.vector().zero()
        self.bc_ux_prescribed.vector().zero()
        self.bc_uy_prescribed.vector().zero()

        for bc in self.bcs_upq:
            bc.apply(uη_reaction.vector())

        F_x = float(assemble(virtual_work_form))

        # reset fixed ux
        self.bc_ux_fixed.vector().zero()

        # ── y-direction reaction ────────────────────────────────────────────
        uη_reaction.vector().zero()
        self.bc_ux_fixed.vector().zero()
        self.bc_uy_fixed.vector()[:] = 1.0
        self.bc_ux_prescribed.vector().zero()
        self.bc_uy_prescribed.vector().zero()

        for bc in self.bcs_upq:
            bc.apply(uη_reaction.vector())

        F_y = float(assemble(virtual_work_form))

        # reset fixed uy
        self.bc_uy_fixed.vector().zero()

        return F_x, F_y

    def postprocess_step(self, step):
        pass

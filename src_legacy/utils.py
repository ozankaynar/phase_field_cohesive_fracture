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
import sys

from dolfin import XDMFFile, Function


def stdout(msg) -> None:
    """convenience function for printing (parallelized)"""
    if MPI.COMM_WORLD.rank == 0:
        print(msg)
        sys.stdout.flush()


class CSVWriter:
    """convenience class for writing CSV files (in parallel)"""

    def __init__(self, filename: str, header: str) -> None:
        self.filename = filename

        # open new file in write mode and write header
        if MPI.COMM_WORLD.rank == 0:
            with open(self.filename, "w") as file:
                file.write(header)

    def write(self, filecontents: str) -> None:
        # open file in append mode and write contents
        if MPI.COMM_WORLD.rank == 0:
            with open(self.filename, "a") as file:
                file.write(filecontents)


class MixedXDMFWriter:
    """
    convenience class to ease output from mixed function spaces.
    Replaces the MixedVTXWriter from the FEniCSx version; uses XDMFFile
    instead of ADIOS2 VTX writers so that output can be read with ParaView.

    For the 1D case (2 sub-spaces: u, eta) filenames should be a list with
    two entries: [u_filename, eta_filename].

    For the 2D case (3 sub-spaces: u, eta_tr, eta_dev) filenames should be a
    list with three entries: [u_filename, etatr_filename, etadev_filename].
    """

    def __init__(self, comm: MPI.Comm, filenames: list, ueta: Function) -> None:
        self.ueta = ueta
        self.num_sub_spaces = ueta.function_space().num_sub_spaces()

        if self.num_sub_spaces == 2:
            # split mixed function and assign names
            u_out, eta_out = ueta.split(deepcopy=True)
            u_out.rename("u", "displacement")
            eta_out.rename("eta", "nonlinear strain")
            self._funcs = [u_out, eta_out]

            # open XDMF writers (two files: u and eta)
            self.xdmf_u = XDMFFile(comm, filenames[0])
            self.xdmf_eta = XDMFFile(comm, filenames[1])

        elif self.num_sub_spaces == 3:
            # split mixed function and assign names
            u_out, etatr_out, etadev_out = ueta.split(deepcopy=True)
            u_out.rename("u", "displacement")
            etatr_out.rename("eta_tr", "trace of nonlinear strain")
            etadev_out.rename("eta_dev", "deviatoric part of nonlinear strain")
            self._funcs = [u_out, etatr_out, etadev_out]

            # open XDMF writers (three files: u, eta_tr, eta_dev)
            self.xdmf_u = XDMFFile(comm, filenames[0])
            self.xdmf_etatr = XDMFFile(comm, filenames[1])
            self.xdmf_etadev = XDMFFile(comm, filenames[2])

        else:
            raise ValueError(
                f"MixedXDMFWriter supports 2 or 3 sub-spaces, got {self.num_sub_spaces}"
            )

    def write(self, t: float) -> None:
        # refresh function values from the mixed parent function
        sub_funcs = self.ueta.split(deepcopy=True)
        for i, f in enumerate(sub_funcs):
            self._funcs[i].vector()[:] = f.vector()

        if self.num_sub_spaces == 2:
            self.xdmf_u.write(self._funcs[0], float(t))
            self.xdmf_eta.write(self._funcs[1], float(t))

        elif self.num_sub_spaces == 3:
            self.xdmf_u.write(self._funcs[0], float(t))
            self.xdmf_etatr.write(self._funcs[1], float(t))
            self.xdmf_etadev.write(self._funcs[2], float(t))

    def close(self) -> None:
        self.xdmf_u.close()
        if self.num_sub_spaces == 2:
            self.xdmf_eta.close()
        elif self.num_sub_spaces == 3:
            self.xdmf_etatr.close()
            self.xdmf_etadev.close()

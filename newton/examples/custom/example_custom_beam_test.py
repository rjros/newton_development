# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Simple Soft Cantilever Beam Example
#
# Creates a soft tetrahedral mesh cantiliver beam, and checks the deformation
# from its bending momemnt. Compares it against different solvers#
###########################################################################

from pxr import Usd


import numpy as np
import warp as wp
import warp.optim
import warp.render

import newton
import newton.examples
from newton.tests.unittest_utils import most


@wp.kernel
def assign_param(params: wp.array(dtype=wp.float32), tet_materials: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    params_idx = 2 * wp.tid() % params.shape[0]
    tet_materials[tid, 0] = params[params_idx]
    tet_materials[tid, 1] = params[params_idx + 1]


@wp.kernel
def com_kernel(particle_q: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    point = particle_q[tid]
    a = point / wp.float32(particle_q.shape[0])

    # Atomically add the point coordinates to the accumulator
    wp.atomic_add(com, 0, a)


@wp.kernel
def loss_kernel(
    target: wp.vec3,
    com: wp.array(dtype=wp.vec3),
    pos_error: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    diff = com[0] - target
    pos_error[0] = wp.length(diff)
    loss[0] = wp.dot(diff, diff)


@wp.kernel
def enforce_constraint_kernel(lower_bound: wp.float32, upper_bound: wp.float32, x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if x[tid] < lower_bound:
        x[tid] = lower_bound
    elif x[tid] > upper_bound:
        x[tid] = upper_bound


class Example:
    def __init__(self, viewer, material_behavior="anisotropic", verbose=False):
        
        # setup simulation parameters first
        self.fps =60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 2.0  # 1.0 seconds
        
        self.sim_substeps = 32

        self.iterations=10
        self.sim_dt=self.frame_dt / self.sim_substeps

        # setup rendering
        self.viewer = viewer
        self.verbose = verbose
        
        # Material properties
        self.material_behavior = material_behavior
      
        # For Semi_Implicit model and ground properties
        builder = newton.ModelBuilder()

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e2
        ground_cfg.kd = 5.0e1
        builder.add_ground_plane(cfg=ground_cfg)
        
        # Create FEM MODEL
        builder.default_particle_radius = 0.001
        # setup grid parameters
        cell_dim = 2
        cell_size = 0.05
        # compute particle density
        total_mass = 0.2
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        # compute Lame parameters

        # TODO
        # System struggles with other young modulus, poisson ratio, and tri_ke,ka,kd
        # Tested with the semi_implicit solver
        young_mod = 1.5 * 1e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        # add soft grid to builder
        builder.add_soft_grid(
            pos=wp.vec3(0.0,0.0,1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=9,
            dim_y=2,
            dim_z=2,
            cell_x=0.01,
            cell_y=0.01,
            cell_z=0.01,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_right=True,
            )

        self.model= builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0
       
        # Select the solver (VBD, XPBD, MUJOCO etc)
        self.solver = newton.solvers.SolverSemiImplicit(model=self.model)

        # allocate sim states for trajectory, control and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        # rendering
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.01,-0.26,1.0),pitch=2.8,yaw=90.0)

        # capture forward/backward passes
        self.capture()


    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
           
           # Swap states
            self.state_0,self.state_1 = self.state_1, self.state_0 


    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt 

# TODO include logging of forces and contacts for plotting
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()



if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--material_behavior",
        default="anisotropic",
        choices=["anisotropic", "isotropic"],
        help="Set material behavior to be Anisotropic or Isotropic.",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, material_behavior=args.material_behavior, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)


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
# Simple Soft Cube Drop Example
#
# Creates a soft tetrahedral mesh cube and drops it onto a ground plane.
# No control or training - just passive physics simulation. Check contacts
# with the floor and print the values.
#
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
        self.fps =100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0  # 1.0 seconds
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # setup rendering
        self.viewer = viewer
        self.verbose = verbose
        
        # Material properties
        self.material_behavior = material_behavior

        # Create FEM model.
        self.model = self.create_model()

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # allocate sim states for trajectory, control and contacts
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        

        # Create a contact pipeline from the command line args 
        # self.collision_pipeline...


        self.contacts = self.model.collide(
                self.state_0,
                soft_contact_max=100,
                soft_contact_margin=0.001\
                )

        # Initialize material parameters to be optimized from model
        if self.material_behavior == "anisotropic":
            # Different Lame parameters for each tet
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[:, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )
        else:
            # Same Lame parameters for all tets
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[0, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )

        # setup hard bounds for material parameters
        self.hard_lower_bound = wp.float32(500.0)
        self.hard_upper_bound = wp.float32(4e6)

       # rendering
        self.viewer.set_model(self.model)

        # capture forward/backward passes
        self.capture()

    def create_model(self):
        # setup simulation scene
        scene = newton.ModelBuilder()
        scene.default_particle_radius = 0.0005

        # setup grid parameters
        cell_dim = 2
        cell_size = 0.1

        # compute particle density
        total_mass = 0.2
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        if self.verbose:
            print(f"Particle density: {particle_density}")

        # compute Lame parameters
        young_mod = 1.5 * 1e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        # add soft grid to scene
        scene.add_soft_grid(
            pos=wp.vec3(-0.5 * cell_size * cell_dim, -0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim*10,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            tri_ke=1e-4,
            tri_ka=1e-4,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        # add wall and ground plane to scene
        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
        mu = 0.2
        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # use `requires_grad=True` to create a model for differentiable simulation
        model = scene.finalize(requires_grad=True)

        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        model.soft_contact_restitution = 1.0

        return model
 
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

            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.001)
            self.solver.__init__
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
           
           # Swap states
            self.state_0,self.state_1 = self.state_1, self.state_0 


    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt 


    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
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

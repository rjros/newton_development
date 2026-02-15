# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cube in VBD and XPBD
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverXPBD,SolverVBD
from newton.utils import transform_twist

class Example:
    def __init__(self, viewer, verbose=False, cell_dim=3):
        
        # Simulation parameters
        # 
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 128
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.verbose = verbose
        
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Physics dt: {self.sim_dt:.6f} s")
        print()
        
        # Build model
        builder = newton.ModelBuilder()

        # Ground plane
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e4
        ground_cfg.kd = 1.0e2
        builder.add_ground_plane(cfg=ground_cfg)
        
        builder.default_particle_radius = 0.001
        
        # Setup grid parameters - NOW CONFIGURABLE
        # cell_dim = 3 → 4x4x4 = 64 particles
        # cell_dim = 4 → 5x5x5 = 125 particles
        # cell_dim = 5 → 6x6x6 = 216 particles
        cell_size = 0.05  # 5cm cells
        
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        print(f"Soft cube configuration:")
        print(f"  Grid: {cell_dim}×{cell_dim}×{cell_dim} cells")
        print(f"  Particles: {num_particles}")
        print(f"  Cell size: {cell_size*100:.0f}cm")
        print(f"  Total size: {cell_dim*cell_size*100:.0f}cm cube")
        print()
        
        # Compute Lame parameters
        young_mod = 1.5e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        k_damp = 100.0
        
        # Add soft grid
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * cell_size * cell_dim, -0.5 * cell_size * cell_dim, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )
        
        # Add rigid sphere
        sphere_mass = 0.5
        sphere_radius = 0.05
        sphere_height = 0.5
        
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0
        sphere_cfg.ke = 1.0e4
        sphere_cfg.kd = 1.0e2
        
        body_sphere = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.45, sphere_height), q=wp.quat_identity()),
            mass=sphere_mass
        )
        
        builder.add_shape_sphere(
            body=body_sphere,
            radius=sphere_radius,
            cfg=sphere_cfg,
        )

        self.model = builder.finalize()
        
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 5.5
       
        self.solver = newton.solvers.SolverSemiImplicit(model=self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        self.viewer.set_model(self.model)
        self.capture()
        
        self.frame_counter = 0
        self.sphere_mass = sphere_mass
        self.sphere_body_id = 0
        self.cell_dim = cell_dim
        self.cell_size = cell_size
 



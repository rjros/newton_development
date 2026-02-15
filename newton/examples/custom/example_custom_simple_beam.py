# Copyright (c) 2025# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Simple Cantilever Beam - Visualization Only
#
# Shows a soft beam anchored on one end, hanging under gravity
# Based on cloth_hanging example pattern
#
# Command: python simple_beam.py
#          python simple_beam.py --solver vbd
#          python simple_beam.py --resolution 20
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(
        self,
        viewer,
        solver_type: str = "semi_implicit",
        resolution: int = 10,
    ):
        # Setup simulation parameters
        self.solver_type = solver_type
        self.resolution = resolution
        
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        if self.solver_type == "semi_implicit":
            self.sim_substeps = 32
        else:
            self.sim_substeps = 10

        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Build model
        builder = newton.ModelBuilder()

        # Add ground plane
        if self.solver_type == "semi_implicit":
            ground_cfg = builder.default_shape_cfg.copy()
            ground_cfg.ke = 1.0e2
            ground_cfg.kd = 5.0e1
            builder.add_ground_plane(cfg=ground_cfg)
        else:
            builder.add_ground_plane()

        # Beam parameters - based on your working example
        cell_dim = 2
        cell_size = 0.05  # 5cm cells (half of your example for finer beam)
        
        # Make beam 10x longer in X direction (like your example: dim_x=cell_dim*10)
        beam_length_cells = cell_dim * 5  # 10 cells along length
        beam_cross_section_cells = cell_dim  # 2 cells in cross-section
        
        # Material: Dragon Skin 10
        E = 263824.0  # Pa
        nu = 0.4999
        density = 1070.0  # kg/m³
        mass_damping = 0.0  # No damping (like your example)
        
        # Convert to Lame parameters
        k_mu = 0.5 * E / (1.0 + nu)
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # Actual dimensions
        beam_length = beam_length_cells * cell_size
        beam_width = beam_cross_section_cells * cell_size
        beam_height = beam_cross_section_cells * cell_size
        
        print(f"\n{'='*70}")
        print(f"Simple Beam Visualization - {solver_type.upper()} Solver")
        print(f"{'='*70}")
        print(f"Beam dimensions:")
        print(f"  Length (X): {beam_length*100:.1f}cm")
        print(f"  Width  (Y): {beam_width*100:.1f}cm")
        print(f"  Height (Z): {beam_height*100:.1f}cm")
        print(f"Grid: {beam_length_cells} × {beam_cross_section_cells} × {beam_cross_section_cells} cells")
        print(f"Cell size: {cell_size*100:.1f}cm")
        
        # Position beam (like your example positioning)
        beam_x = -0.5 * cell_size * beam_length_cells  # Center in X
        beam_y = -0.5 * cell_size * beam_cross_section_cells  # Center in Y
        beam_z = 1.0  # 1m above ground (like your example)
        
        # Add soft grid (beam extending along +X axis)
        builder.add_soft_grid(
            pos=wp.vec3(beam_x, beam_y, beam_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=beam_length_cells,            # 10 cells - LONG dimension
            dim_y=beam_cross_section_cells,     # 2 cells - cross-section
            dim_z=beam_cross_section_cells,     # 2 cells - cross-section
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=mass_damping,
            fix_left=True,   # Fix leftmost particles (x=0)
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=1.0e-4,   # Like your example
            tri_ka=1.0e-4,
            tri_kd=1.0e-4,
            tri_drag=0.0,
            tri_lift=0.0,
        )

        # Color for VBD
        if self.solver_type == "vbd":
            builder.color(include_bending=False)

        # Finalize model
        self.model = builder.finalize()
        
        # Contact properties (exactly like your example)
        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
        mu = 0.2
        
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 0.0

        # Create solver
        if self.solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(model=self.model)
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.iterations,
            )
        else:  # vbd
            self.solver = newton.solvers.SolverVBD(
                model=self.model, 
                iterations=self.iterations
            )

        print(f"Particles: {self.model.particle_count}")
        print(f"Tets: {self.model.tet_count}")
        print(f"Triangles: {self.model.tri_count}")
        print(f"{'='*70}\n")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()
        
        self.frame = 0

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # Collide with margin (exactly like your example)
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.001)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame += 1
        
        # Print status every 60 frames
        if self.frame % 60 == 0:
            import numpy as np
            positions = self.state_0.particle_q.numpy()
            # Find tip (rightmost particle)
            tip_idx = np.argmax(positions[:, 0])
            tip_z = positions[tip_idx][2]
            print(f"Frame {self.frame:4d} | Time {self.sim_time:.2f}s | Tip height: {tip_z*100:.2f}cm")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Add solver-specific arguments
    parser.add_argument(
        "--solver",
        help="Type of solver",
        type=str,
        choices=["semi_implicit", "xpbd", "vbd"],
        default="semi_implicit",
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=10, 
        help="Mesh resolution (cells along beam length)"
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        solver_type=args.solver,
        resolution=args.resolution,
    )

    newton.examples.run(example, args)

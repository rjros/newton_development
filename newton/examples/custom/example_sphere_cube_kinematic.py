# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Kinematic Sphere Compressing Soft Cube
#
# Based on the working sphere_kinematic_test.py, now with soft cube added.
# The sphere animates with kinematic motion and compresses the soft cube.
#
# Command: python sphere_cube_kinematic.py
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        
        # Simulation parameters
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Motion parameters (slower for compression testing)
        self.frequency = 0.01  # Hz (100 second period)
        self.amplitude = 0.05  # m (5cm amplitude)
        
        print(f"Kinematic Sphere Compressing Soft Cube")
        print(f"{'='*70}")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Substeps: {self.sim_substeps}")
        print(f"  Motion: {self.frequency} Hz, ±{self.amplitude*100:.0f}cm")
        print(f"  Period: {1.0/self.frequency:.0f} seconds")
        print(f"{'='*70}\n")
        
        # === Build Model ===
        builder = newton.ModelBuilder(gravity=-9.81)
        
        # Ground plane
        builder.add_ground_plane()
        
        # === Add Soft Cube (VBD) ===
        cell_size = 0.05  # 5cm cells
        cell_dim = 3      # 3x3x3 grid
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        # Material properties
        young_mod = 1.5e4  # 15 kPa
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        k_damp = 100.0
        
        self.cube_height = cell_dim * cell_size  # 0.15m = 15cm
        
        builder.add_soft_grid(
            pos=wp.vec3(
                -0.5 * cell_size * cell_dim,
                -0.5 * cell_size * cell_dim,
                0.0
            ),
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
            tri_ke=1e2,
            tri_ka=1e2,
            tri_kd=1.5e-6,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
            particle_radius=0.005,  # 5mm
        )
        
        # === Add Kinematic Sphere ===
        self.sphere_radius = 0.05  # 5cm
        # Start sphere 10cm above cube top
        self.sphere_start_height = self.cube_height + 0.10  # 0.25m = 25cm
        
        # Create kinematic body
        body_sphere = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.sphere_start_height),
                q=wp.quat_identity()
            ),
            key="sphere"
        )
        
        # Kinematic sphere configuration
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0  # Kinematic (not affected by gravity)
        sphere_cfg.ke = 1.0e5
        sphere_cfg.kd = 1.0e-4
        sphere_cfg.mu = 0.5
        
        builder.add_shape_sphere(
            body_sphere,
            radius=self.sphere_radius,
            cfg=sphere_cfg,
        )
        
        # Store sphere body index
        self.sphere_body_index = 0
        
        # Color mesh (required for VBD)
        builder.color()
        
        # === Finalize Model ===
        self.model = builder.finalize()
        
        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_mu = 0.5
        
        # === Create VBD Solver ===
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,  # Single cube
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.003,
        )
        
        # === Create States ===
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # === Create Collision Pipeline ===
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=0.01,
        )
        self.contacts = self.collision_pipeline.contacts()
        
        # === Track Sphere Height (State Variable) ===
        self.sphere_current_height = self.sphere_start_height
        
        # === Set Viewer ===
        self.viewer.set_model(self.model)
        
        print(f"Soft Cube:")
        print(f"  Size: {self.cube_height*100:.0f}cm × {self.cube_height*100:.0f}cm × {self.cube_height*100:.0f}cm")
        print(f"  Particles: {num_particles}")
        print(f"  Young's modulus: {young_mod/1000:.0f} kPa")
        print(f"  Mass: {total_mass:.1f} kg")
        print(f"\nSphere:")
        print(f"  Radius: {self.sphere_radius*100:.0f}cm")
        print(f"  Start height: {self.sphere_start_height*100:.0f}cm")
        print(f"  Motion range: {(self.sphere_start_height-self.amplitude)*100:.0f}cm to {(self.sphere_start_height+self.amplitude)*100:.0f}cm")
        print(f"\nWatch the sphere compress the cube!")
        print(f"{'='*70}\n")
        
        # Disable CUDA graph (kinematic animation requires CPU-GPU transfers)
        self.graph = None
        
        self.frame_counter = 0
    
    def simulate(self):
        """Simulate with kinematic sphere animation"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Apply viewer forces
            self.viewer.apply_forces(self.state_0)
            
            # === Animate Kinematic Sphere ===
            # Calculate velocity: d/dt[A*sin(2πft)] = A*2πf*cos(2πft)
            velocity_z = self.amplitude * 2.0 * np.pi * self.frequency * np.cos(
                2.0 * np.pi * self.frequency * self.sim_time
            )
            
            # Update position incrementally
            self.sphere_current_height += velocity_z * self.sim_dt
            
            # Set sphere position in state
            body_q = self.state_0.body_q.numpy()
            body_q[self.sphere_body_index][0] = 0.0  # x
            body_q[self.sphere_body_index][1] = 0.0  # y
            body_q[self.sphere_body_index][2] = self.sphere_current_height  # z (animated!)
            self.state_0.body_q = wp.array(body_q, dtype=wp.transform)
            
            # Collision detection
            self.collision_pipeline.collide(self.state_0, self.contacts)
            
            # Solver step
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        """Single frame step"""
        self.simulate()
        
        # Increment time ONCE per frame
        self.sim_time += self.frame_dt
        
        self.frame_counter += 1
        
        # Print state every 60 frames (every second)
        if self.frame_counter % 60 == 0:
            self.print_state()
    
    def print_state(self):
        """Print current simulation state"""
        # Calculate target position from current time
        target_displacement = self.amplitude * np.sin(2.0 * np.pi * self.frequency * self.sim_time)
        target_height = self.sphere_start_height + target_displacement
        
        # Get actual sphere height (from our state variable)
        actual_height = self.sphere_current_height
        
        # Get sphere force (from contacts)
        body_f = self.state_0.body_f.numpy()
        sphere_force = body_f[self.sphere_body_index, 0:3]
        sphere_force_z = sphere_force[2]
        
        # Soft body state
        particle_q = self.state_0.particle_q.numpy()
        max_particle_z = np.max(particle_q[:, 2])
        min_particle_z = np.min(particle_q[:, 2])
        cube_height_current = max_particle_z - min_particle_z
        
        # Calculate compression
        compression_mm = (self.cube_height - cube_height_current) * 1000
        
        # Control tracking error
        error = actual_height - target_height
        
        # Calculate velocity for reference
        current_velocity = self.amplitude * 2.0 * np.pi * self.frequency * np.cos(
            2.0 * np.pi * self.frequency * self.sim_time
        )
        
        print(
            f"t={self.sim_time:.2f}s | "
            f"Target: {target_height:.4f}m | "
            f"Actual: {actual_height:.4f}m | "
            f"Error: {error:+.6f}m | "
            f"Force Z: {sphere_force_z:+.1f}N | "
            f"Compression: {compression_mm:.2f}mm"
        )
    
    def render(self):
        """Render current state"""
        if self.viewer is None:
            return
        
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
    
    def test_final(self):
        """Validate final simulation state"""
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        
        # Check velocity (should be reasonable)
        max_vel = np.max(np.linalg.norm(particle_qd, axis=1))
        assert max_vel < 1.0, f"Particles moving too fast: max_vel={max_vel:.4f} m/s"
        
        # Check bbox size is reasonable (not exploding)
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 1.0, f"Bounding box exploded: size={bbox_size:.2f}m"
        
        # Check no excessive penetration
        assert min_pos[2] > -0.1, f"Excessive penetration: z_min={min_pos[2]:.4f}m"
        
        # Check cube height is reasonable
        cube_height = max_pos[2] - min_pos[2]
        assert 0.10 < cube_height < 0.20, f"Cube height unreasonable: {cube_height*100:.1f}cm"
        
        print(f"\n✓ All validation tests passed!")


if __name__ == "__main__":
    # Parse arguments
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 seconds
    
    # Initialize viewer
    viewer, args = newton.examples.init(parser)
    
    # Create and run example
    example = Example(viewer, args)
    
    newton.examples.run(example, args)

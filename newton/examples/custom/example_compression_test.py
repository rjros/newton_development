# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Compression Test - Rigid Indenter on Soft Silicone Cube
#
# A 10mm diameter rigid sphere indenter compresses a 40mm silicone cube
# Material: Young's modulus = 50 kPa, Poisson's ratio = 0.49
# Test: 5mm penetration depth
#
###########################################################################

import numpy as np
import warp as wp
import csv

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False):
        
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 64  # High substeps for stability
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.verbose = verbose
        
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Frame dt: {self.frame_dt:.4f} s")
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
        
        # === SOFT SILICONE CUBE ===
        # Material properties
        youngs_modulus = 1.5e4  # 50 kPa = 50,000 Pa
        poisson_ratio = 0.3   # Nearly incompressible (silicone)
        
        # Compute Lame parameters
        k_mu = 0.5 * youngs_modulus / (1.0 + poisson_ratio)
        k_lambda = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        
        # Cube dimensions
        cube_size = 0.04  # 40mm = 0.04m
        cell_size = 0.01  # 10mm cells
        cell_dim = int(cube_size / cell_size)  # 4x4x4 cells
        
        # Mass calculation
        cube_mass = 2.0  # kg (doesn't affect compression much)
        num_particles = (cell_dim + 1) ** 3
        particle_mass = cube_mass / num_particles
        particle_density = particle_mass / (cell_size ** 3)
        
        print(f"Soft silicone cube:")
        print(f"  Size: {cube_size*1000:.1f}mm × {cube_size*1000:.1f}mm × {cube_size*1000:.1f}mm")
        print(f"  Young's modulus: {youngs_modulus/1000:.1f} kPa")
        print(f"  Poisson's ratio: {poisson_ratio:.2f}")
        print(f"  Grid: {cell_dim}×{cell_dim}×{cell_dim} cells ({cell_size*1000:.1f}mm)")
        print(f"  Particles: {num_particles}")
        print(f"  Density: {particle_density:.1f} kg/m³")
        print()
        
        # Add soft cube on ground
        builder.add_soft_grid(
            pos=wp.vec3(-cube_size/2.0, -cube_size/2.0, 0.0),
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
            k_damp=100.0,  # Material damping
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )
        
        # === RIGID INDENTER SPHERE ===
        indenter_diameter = 0.01  # 10mm
        indenter_radius = indenter_diameter / 2.0
        indenter_mass = 0.1  # 100g
        
        # Starting position: above cube top surface
        cube_top = cube_size
        indenter_start_height = cube_top + indenter_radius + 0.01  # 10mm clearance above cube
        
        print(f"Rigid indenter:")
        print(f"  Diameter: {indenter_diameter*1000:.1f}mm")
        print(f"  Mass: {indenter_mass*1000:.1f}g")
        print(f"  Start height: {indenter_start_height*1000:.1f}mm")
        print()
        
        # Create indenter with prismatic joint for controlled vertical motion
        indenter_cfg = builder.default_shape_cfg.copy()
        indenter_cfg.density = 0.0  # Mass from add_link only
        indenter_cfg.ke = 1.0e4
        indenter_cfg.kd = 1.0e2
        
        body_indenter = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, indenter_start_height), q=wp.quat_identity()),
            mass=indenter_mass,
            key="indenter"
        )
        
        builder.add_shape_sphere(
            body=body_indenter,
            radius=indenter_radius,
            cfg=indenter_cfg,
        )
        
        # Add prismatic joint for vertical motion
        joint_indenter = builder.add_joint_prismatic(
            parent=-1,
            child=body_indenter,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, indenter_start_height), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            limit_lower=-0.1,
            limit_upper=0.1,
            target_ke=1e4,
            target_kd=1e3,
            limit_ke=1e5,
            limit_kd=1e3,
            key="indenter_joint"
        )
        
        builder.add_articulation([joint_indenter], key="indenter_articulation")
        
        # Finalize model
        self.model = builder.finalize()
        
        # Contact properties
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 5.0
        
        # Solver
        #self.solver = newton.solvers.SolverXPBD(self.model, iterations=10) 
        self.solver = newton.solvers.SolverSemiImplicit(self.model, joint_attach_ke=1600.0, joint_attach_kd=20.0)
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Evaluate FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Store parameters
        self.indenter_body_id = body_indenter
        self.indenter_joint_id = 0
        self.indenter_start_height = indenter_start_height
        self.indenter_radius = indenter_radius
        self.cube_top = cube_top
        
        # Get joint index
        self.joint_q_start_idx = self.model.joint_q_start.numpy()[self.indenter_joint_id]
        
        # Compression test parameters
        self.penetration_depth = 0.005  # 5mm penetration
        self.compression_speed = 0.002  # 2mm/s (slow compression)
        self.target_height = cube_top - self.penetration_depth  # Target: 5mm into cube
        
        # Test phases
        self.phase = "approach"  # Phases: approach, compress, hold
        self.hold_start_time = None
        self.hold_duration = 2.0  # Hold for 2 seconds
        
        # Rendering
        self.viewer.set_model(self.model)
        
        # Camera position
        cam_distance = 0.15  # 15cm away
        cam_height = cube_size / 2.0  # Cube mid-height
        self.viewer.set_camera(
            pos=wp.vec3(cam_distance, 0.0, cam_height),
            pitch=0.0,
            yaw=180.0
        )
        
        # Capture
        self.capture()
        
        # Frame counter and logging
        self.frame_counter = 0
        self.log_data = []
        
        # Print test info
        print(f"{'='*70}")
        print(f"Compression Test Configuration")
        print(f"{'='*70}")
        print(f"Target penetration: {self.penetration_depth*1000:.1f}mm")
        print(f"Compression speed: {self.compression_speed*1000:.1f}mm/s")
        print(f"Target height: {self.target_height*1000:.1f}mm")
        print(f"{'='*70}\n")

    def capture(self):
        if wp.get_device().is_cuda:
            # Create control array for joint target
            control_array = wp.zeros(self.model.joint_coord_count, dtype=wp.float32)
            self.control.joint_target_pos = control_array
            
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            
            contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_control(self):
        """Update indenter position based on compression test phase"""
        # Get current indenter position
        body_q = self.state_0.body_q.numpy()
        current_z = body_q[self.indenter_body_id, 2]
        
        # Determine target based on phase
        if self.phase == "approach":
            # Move down slowly until contact with cube top
            if current_z > self.cube_top + self.indenter_radius:
                target_z = current_z - self.compression_speed * self.frame_dt
            else:
                # Contact made, switch to compress phase
                self.phase = "compress"
                target_z = current_z
                print(f"\n[t={self.sim_time:.2f}s] Contact made, starting compression...")
        
        elif self.phase == "compress":
            # Compress until target depth reached
            if current_z > self.target_height:
                target_z = current_z - self.compression_speed * self.frame_dt
                target_z = max(target_z, self.target_height)  # Don't overshoot
            else:
                # Target depth reached, switch to hold
                self.phase = "hold"
                self.hold_start_time = self.sim_time
                target_z = self.target_height
                print(f"\n[t={self.sim_time:.2f}s] Target depth reached, holding position...")
        
        elif self.phase == "hold":
            # Hold at target depth
            target_z = self.target_height
        
        else:
            target_z = self.indenter_start_height
        
        # Convert to joint coordinate (relative to start)
        joint_target = target_z - self.indenter_start_height
        
        # Update control
        target_array = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        target_array[self.joint_q_start_idx] = joint_target
        target_wp = wp.array(target_array, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos, target_wp)

    def step(self):
        # Update control before simulation
        self.update_control()
        
        # Run simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        # Log data
        self.log_forces()
        
        # Update time
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Print every 30 frames (0.5s)
        if self.frame_counter % 30 == 0:
            self.print_status()
    
    def log_forces(self):
        """Log force data every frame"""
        # Indenter forces
        body_forces = self.state_0.body_f.numpy()
        indenter_force = body_forces[self.indenter_body_id, 0:3]
        indenter_fz = indenter_force[2]
        indenter_force_mag = np.linalg.norm(indenter_force)
        
        # Indenter position
        body_q = self.state_0.body_q.numpy()
        indenter_z = body_q[self.indenter_body_id, 2]
        
        # Penetration depth (negative = into cube)
        penetration = self.cube_top - (indenter_z - self.indenter_radius)
        
        # Soft cube forces
        particle_forces = self.state_0.particle_f.numpy()
        total_cube_force = np.sum(particle_forces, axis=0)
        cube_fz = total_cube_force[2]
        
        # Cube compression (change in average height)
        particle_q = self.state_0.particle_q.numpy()
        cube_avg_z = np.mean(particle_q[:, 2])
        
        # Store data
        self.log_data.append([
            self.sim_time,
            indenter_z,
            penetration,
            indenter_fz,
            indenter_force_mag,
            cube_avg_z,
            cube_fz,
            self.phase
        ])
    
    def print_status(self):
        """Print test status"""
        # Get current data
        body_q = self.state_0.body_q.numpy()
        indenter_z = body_q[self.indenter_body_id, 2]
        
        body_forces = self.state_0.body_f.numpy()
        indenter_force = body_forces[self.indenter_body_id, 0:3]
        indenter_fz = indenter_force[2]
        
        penetration = self.cube_top - (indenter_z - self.indenter_radius)
        
        print(f"[t={self.sim_time:.2f}s] Phase: {self.phase:8s} | "
              f"Indenter z={indenter_z*1000:.2f}mm | "
              f"Penetration={penetration*1000:.2f}mm | "
              f"Force={indenter_fz:.2f}N")
    
    def save_log(self, filename="compression_test.csv"):
        """Save logged data to CSV"""
        header = ['time', 'indenter_z', 'penetration', 'indenter_force_z', 
                  'indenter_force_mag', 'cube_avg_z', 'cube_force_z', 'phase']
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.log_data)
        
        print(f"\nCompression test data saved to {filename}")
        print(f"  Frames logged: {len(self.log_data)}")
        print(f"  Duration: {self.sim_time:.2f}s")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print detailed status.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
    
    # Save log
    # example.save_log("compression_test.csv")

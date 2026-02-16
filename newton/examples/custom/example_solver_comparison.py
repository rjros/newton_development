# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Kinematic Sphere Compressing Soft Cube - SEMI-IMPLICIT SOLVER
#
# Same simulation as sphere_cube_stable.py but using SemiImplicit solver
# instead of VBD for direct comparison.
#
# Key differences from VBD version:
# - SolverSemiImplicit instead of SolverVBD
# - Can use more realistic soft material parameters
# - model.collide() instead of collision_pipeline
# - No builder.color() required
# - Single-phase simulation (no particle toggle)
# Force value is visible, plot and compare results with real material
# Need to make own custom plugin
# Command: python sphere_cube_semiimplicit.py
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
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 128  # SemiImplicit needs MORE substeps than VBD
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Motion parameter/cubs
        self.frequency = 0.10  # Hz (100 second period)
        self.amplitude = 0.10  # m (5cm amplitude)
        
        print(f"Kinematic Sphere Compression (SEMI-IMPLICIT SOLVER)")
        print(f"{'='*70}")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Substeps: {self.sim_substeps} (MORE than VBD!)")
        print(f"  dt: {self.sim_dt:.6f} s")
        print(f"  Motion: {self.frequency} Hz, ±{self.amplitude*100:.0f}cm")
        print(f"{'='*70}\n")
        
        # === Build Model ===
        builder = newton.ModelBuilder()
        
        # Ground plane
        builder.add_ground_plane()
        
        # === Add Soft Cube ===
        # Can use SMALLER cells and SOFTER materials with SemiImplicit!
        cell_size = 0.05  # 5cm cells (smaller than VBD version)
        cell_dim = 3      # 3x3x3 grid
        
        self.cube_height = cell_dim * cell_size  # 0.15m
        
        # Material properties - Can use REALISTIC soft rubber with SemiImplicit!
        young_mod = 1.5e4  # 15 kPa (soft rubber - realistic!)
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        k_damp = 100.0  # Higher damping OK with SemiImplicit
        
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
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
            particle_radius=0.005,
        )
        
        # === Add Kinematic Sphere ===
        self.sphere_radius = 0.05  # 5cm
        self.sphere_start_height = self.cube_height + 0.10  # 10cm above cube
        
        body_sphere = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.sphere_start_height),
                q=wp.quat_identity()
            ),
            key="sphere"
        )
        
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0  # Kinematic
        sphere_cfg.ke = 1.0e4     # Softer contact
        sphere_cfg.kd = 1.0e2
        sphere_cfg.mu = 0.5
        
        builder.add_shape_sphere(
            body_sphere,
            radius=self.sphere_radius,
            cfg=sphere_cfg,
        )
        
        self.sphere_body_index = 0
        
        # NO builder.color() - not needed for SemiImplicit!
        
        # === Finalize Model ===
        self.model = builder.finalize()
        
        # Contact parameters (softer than VBD)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.5
        
        # === Create SEMI-IMPLICIT Solver ===
        self.solver = newton.solvers.SolverSemiImplicit(
            model=self.model
        )
        
        # === Create States ===
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        self.contacts= self.model.contacts()
        # === Track Sphere Height ===
        self.sphere_current_height = self.sphere_start_height
        
        # Store initial cube top
        particle_q = self.state_0.particle_q.numpy()
        self.initial_cube_top = np.max(particle_q[:, 2])
        
        # === Set Viewer ===
        self.viewer.set_model(self.model)
        
        print(f"Soft Cube (REALISTIC PARAMETERS - SemiImplicit):")
        print(f"  Size: {self.cube_height*100:.0f}cm cube")
        print(f"  Cell size: {cell_size*100:.0f}cm")
        print(f"  Particles: {num_particles}")
        print(f"  k_mu: {k_mu:.0f} Pa = {k_mu/1000:.1f} kPa (SOFT - realistic!)")
        print(f"  k_lambda: {k_lambda:.0f} Pa = {k_lambda/1000:.1f} kPa")
        print(f"  k_damp: {k_damp:.0f} Pa·s (NORMAL damping)")
        print(f"  Young's modulus: {young_mod/1000:.0f} kPa (soft rubber)")
        print(f"  Density: {particle_density:.0f} kg/m³")
        print(f"\nSphere:")
        print(f"  Radius: {self.sphere_radius*100:.0f}cm")
        print(f"  Start height: {self.sphere_start_height*100:.0f}cm")
        print(f"\nSolver:")
        print(f"  SemiImplicit (force-based integration)")
        print(f"  {self.sim_substeps} substeps/frame")
        print(f"{'='*70}\n")
        
        # Force logging arrays
        self.log_time = []
        self.log_particle_force = []
        self.log_top_displacement = []
        
        # Disable CUDA graph (kinematic animation)
        self.graph = None
        
        self.frame_counter = 0
    
    def simulate(self):
        """Single-phase simulation (SemiImplicit)"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            self.viewer.apply_forces(self.state_0)
            
            # Animate kinematic sphere
            velocity_z = self.amplitude * 2.0 * np.pi * self.frequency * np.cos(
                2.0 * np.pi * self.frequency * self.sim_time
            )
            
            self.sphere_current_height += velocity_z * self.sim_dt
            
            body_q = self.state_0.body_q.numpy()
            body_q[self.sphere_body_index][0] = 0.0
            body_q[self.sphere_body_index][1] = 0.0
            body_q[self.sphere_body_index][2] = self.sphere_current_height
            self.state_0.body_q = wp.array(body_q, dtype=wp.transform)
            
            # Collision and solver (simple with SemiImplicit!)
            self.model.collide(self.state_0,self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        self.simulate()
        
        self.sim_time += self.frame_dt
        
        self.frame_counter += 1
        
        # Log forces every frame
        self.log_forces()
        
        if self.frame_counter % 60 == 0:
            self.print_state()
    
    def log_forces(self):
        """Log force data every frame"""
        particle_forces = self.state_0.particle_f.numpy()
        total_particle_force = np.sum(particle_forces, axis=0)
        particle_force_mag = np.linalg.norm(total_particle_force)
        
        particle_q = self.state_0.particle_q.numpy()
        cube_top_z = np.max(particle_q[:, 2])
        top_displacement = (self.initial_cube_top - cube_top_z) * 1000  # mm
        
        self.log_time.append(self.sim_time)
        self.log_particle_force.append(particle_force_mag)
        self.log_top_displacement.append(top_displacement)
    
    def print_state(self):
        target_displacement = self.amplitude * np.sin(2.0 * np.pi * self.frequency * self.sim_time)
        target_height = self.sphere_start_height + target_displacement
        
        actual_height = self.sphere_current_height
        
        particle_forces = self.state_0.particle_f.numpy()
        total_particle_force = np.sum(particle_forces, axis=0)
        particle_force_z = total_particle_force[2]
        particle_force_mag = np.linalg.norm(total_particle_force)
        
        particle_q = self.state_0.particle_q.numpy()
        max_particle_z = np.max(particle_q[:, 2])
        min_particle_z = np.min(particle_q[:, 2])
        cube_height_current = max_particle_z - min_particle_z
        
        compression_mm = (self.cube_height - cube_height_current) * 1000
        
        error = actual_height - target_height
        
        print(
            f"t={self.sim_time:.2f}s | "
            f"Target: {target_height:.4f}m | "
            f"Actual: {actual_height:.4f}m | "
            f"Error: {error:+.6f}m | "
            f"Particle Force: {particle_force_mag:+.2f}N (Z: {particle_force_z:+.2f}N) | "
            f"Compression: {compression_mm:.2f}mm"
        )
    
    def render(self):
        if self.viewer is None:
            return
        
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
    
    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        
        assert bbox_size < 1.0, f"Bounding box exploded: size={bbox_size:.2f}m"
        assert min_pos[2] > -0.1, f"Excessive penetration: z_min={min_pos[2]:.4f}m"
        
        print(f"\n✓ SemiImplicit simulation stable!")
        
        # Save data
        self.save_log("force_log_semiimplicit.csv")
        self.plot_results("compression_semiimplicit.png")
    
    def save_log(self, filename="force_log_semiimplicit.csv"):
        """Save time-series force data to CSV"""
        import csv
        
        print(f"\nSaving force time-series to CSV...")
        
        if len(self.log_time) == 0:
            print("No data to save!")
            return
        
        header = ['time', 'particle_force', 'top_displacement_mm']
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(self.log_time)):
                writer.writerow([
                    self.log_time[i],
                    self.log_particle_force[i],
                    self.log_top_displacement[i]
                ])
        
        print(f"Time-series data saved to {filename}")
        print(f"  Frames logged: {len(self.log_time)}")
        print(f"  Duration: {self.log_time[-1]:.2f}s")
        print(f"\nFinal Values:")
        print(f"  Particle Force: {self.log_particle_force[-1]:.2f} N")
        print(f"  Top Displacement: {self.log_top_displacement[-1]:.2f} mm")
    
    def plot_results(self, filename="compression_semiimplicit.png"):
        """Create time-series plots"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot")
            return
        
        print(f"\nGenerating visualization...")
        
        if len(self.log_time) == 0:
            print("No data logged - skipping plot")
            return
        
        time = np.array(self.log_time)
        particle_force = np.array(self.log_particle_force)
        top_displacement = np.array(self.log_top_displacement)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Particle Force
        ax1.plot(time, particle_force, 'b-', linewidth=2, label='Particle Force')
        ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Force Magnitude (N)', fontsize=12, fontweight='bold')
        ax1.set_title('Compression Force (SemiImplicit Solver)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        
        # Plot 2: Compression
        ax2.plot(time, top_displacement, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Top Face Displacement (mm)', fontsize=12, fontweight='bold')
        ax2.set_title('Cube Compression (Positive = Compressed)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        
        max_compression = np.max(top_displacement)
        max_idx = np.argmax(top_displacement)
        max_time = time[max_idx]
        
        ax2.annotate(f'Max: {max_compression:.2f} mm\nat t={max_time:.2f}s',
                    xy=(max_time, max_compression),
                    xytext=(max_time + 1, max_compression - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        fig.suptitle(f'SemiImplicit Solver - {self.model.particle_count} Particles', 
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        plt.close()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    
    viewer, args = newton.examples.init(parser)
    
    example = Example(viewer, args)
    
    newton.examples.run(example, args)
    
    # Save data after simulation
    example.test_final()

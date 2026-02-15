# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Sphere Falling onto Soft Cube with Force Logging
# - Configurable particle count
# - Simplified plots: forces and top face displacement
###########################################################################

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False, cell_dim=3):
        
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        
        # Adjust substeps based on cell dimension for stability
        # More cells = finer mesh = need more substeps
        # cell_dim=3 → 64 substeps, cell_dim=4 → 96, cell_dim=5 → 128
        self.sim_substeps = 32 * cell_dim
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.verbose = verbose
        
        print(f"Simulation configured:")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Physics substeps: {self.sim_substeps} (32 × cell_dim for stability)")
        print(f"  Physics dt: {self.sim_dt:.6f} s")
        print()
        
        # Build model
        builder = newton.ModelBuilder()

        # Ground plane
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e4
        ground_cfg.kd = 1.0e2
        builder.add_ground_plane(cfg=ground_cfg)
        
        # Setup grid parameters - NOW CONFIGURABLE
        # cell_dim = 3 → 4x4x4 = 64 particles
        # cell_dim = 4 → 5x5x5 = 125 particles
        # cell_dim = 5 → 6x6x6 = 216 particles
        cell_size = 0.05  # 5cm cells
        
        # Adjust particle radius based on cell size for stability
        # Rule of thumb: particle_radius = 0.2 * cell_size
        particle_radius = 0.2 * cell_size
        builder.default_particle_radius = particle_radius
        
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        print(f"Soft cube configuration:")
        print(f"  Grid: {cell_dim}×{cell_dim}×{cell_dim} cells")
        print(f"  Particles: {num_particles}")
        print(f"  Cell size: {cell_size*100:.0f}cm")
        print(f"  Particle radius: {particle_radius*1000:.1f}mm (0.2 × cell_size)")
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
            xform=wp.transform(p=wp.vec3(0.0, 0.0, sphere_height), q=wp.quat_identity()),
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
        self.model.soft_contact_mu = 5.0
       
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
        
        # Store initial cube top height
        particle_q = self.state_0.particle_q.numpy()
        self.initial_cube_top = np.max(particle_q[:, 2])
        
        # Force logging arrays
        self.log_time = []
        self.log_sphere_force = []
        self.log_cube_force = []
        self.log_top_displacement = []
        
        print(f"\n{'='*70}")
        print(f"Sphere Falling onto Soft Cube")
        print(f"{'='*70}")
        print(f"Soft box:")
        print(f"  Size: {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm × {cell_dim*cell_size*100:.0f}cm")
        print(f"  Particles: {self.model.particle_count}")
        print(f"  Tetrahedra: {self.model.tet_count}")
        print(f"  Mass: {total_mass:.3f} kg")
        print(f"  Expected weight: {total_mass * 9.81:.2f} N")
        print(f"Rigid sphere:")
        print(f"  Radius: {sphere_radius*100:.0f}cm")
        print(f"  Mass: {sphere_mass:.3f} kg")
        print(f"  Drop height: {sphere_height*100:.0f}cm")
        print(f"  Expected weight: {sphere_mass * 9.81:.2f} N")
        print(f"{'='*70}")
        print(f"NOTE: Force logging starts at t=5.0s (only positive displacement)")
        print(f"{'='*70}\n")

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
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0 

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Log forces every frame (try/except for CUDA graph safety)
        try:
            self.log_forces()
        except:
            pass
        
        # Print forces every 10 frames (only during non-graph simulation)
        if self.frame_counter % 10 == 0 and not self.graph:
            self.print_forces()
    
    def log_forces(self):
        """Log force data every frame (only after 5s and when displacement is positive)"""
        
        # Skip logging for first 5 seconds
        if self.sim_time < 5.0:
            return
        
        body_forces = self.state_0.body_f.numpy()
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        particle_forces = self.state_0.particle_f.numpy()
        total_particle_force = np.sum(particle_forces, axis=0)
        cube_force_mag = np.linalg.norm(total_particle_force)
        
        particle_q = self.state_0.particle_q.numpy()
        cube_top_z = np.max(particle_q[:, 2])
        
        # Displacement: initial_top - current_top
        # POSITIVE value = cube is compressed (top moved down)
        top_displacement = (self.initial_cube_top - cube_top_z) * 1000  # mm
        
        # Only log when displacement is positive (cube is actually compressed)
        if top_displacement <= 0:
            return
        
        # Store data
        self.log_time.append(self.sim_time)
        self.log_sphere_force.append(sphere_force_mag)
        self.log_cube_force.append(cube_force_mag)
        self.log_top_displacement.append(top_displacement)
    
    def save_log(self, filename="force_log.csv"):
        """Save time-series force data to CSV"""
        import csv
        
        print(f"\nSaving force time-series to CSV...")
        
        if len(self.log_time) == 0:
            print("No data to save! (logging requires t>5.0s and displacement>0)")
            return
        
        # Write configuration info as comments, then data
        with open(filename, 'w', newline='') as f:
            f.write(f"# Soft cube: {self.cell_dim}x{self.cell_dim}x{self.cell_dim} cells, {self.model.particle_count} particles\n")
            f.write(f"# Cell size: {self.cell_size*100:.0f}cm, Particle radius: {self.cell_size*0.2*1000:.1f}mm\n")
            f.write(f"# Substeps: {self.sim_substeps}\n")
            
            writer = csv.writer(f)
            header = ['time', 'sphere_force', 'cube_force', 'top_displacement_mm']
            writer.writerow(header)
            
            for i in range(len(self.log_time)):
                writer.writerow([
                    self.log_time[i],
                    self.log_sphere_force[i],
                    self.log_cube_force[i],
                    self.log_top_displacement[i]
                ])
        
        print(f"Time-series data saved to {filename}")
        print(f"  Frames logged: {len(self.log_time)}")
        print(f"  Time range: {self.log_time[0]:.2f}s - {self.log_time[-1]:.2f}s")
        print(f"\nSimulation Configuration:")
        print(f"  Particles: {self.model.particle_count} ({self.cell_dim}³ grid)")
        print(f"  Particle radius: {self.cell_size*0.2*1000:.1f}mm (0.2 × cell_size)")
        print(f"  Substeps/frame: {self.sim_substeps} (32 × cell_dim)")
        print(f"\nFinal Values:")
        print(f"  Sphere Force: {self.log_sphere_force[-1]:.2f} N")
        print(f"  Cube Force: {self.log_cube_force[-1]:.2f} N")
        print(f"  Top Displacement: {self.log_top_displacement[-1]:.2f} mm")
        
        # Calculate Hertzian prediction at final displacement
        E = 1.5e4  # Pa
        nu = 0.3
        R = 0.05   # m
        E_star = E / (1 - nu**2)
        delta_m = self.log_top_displacement[-1] / 1000  # convert mm to m
        F_hertz = (4/3) * E_star * np.sqrt(R) * (delta_m ** 1.5)
        
        error_percent = abs(self.log_sphere_force[-1] - F_hertz) / F_hertz * 100
        
        print(f"\nHertzian Theory Comparison:")
        print(f"  Hertzian Prediction: {F_hertz:.2f} N")
        print(f"  Simulation Result:   {self.log_sphere_force[-1]:.2f} N")
        print(f"  Error: {error_percent:.1f}%")
    
    def plot_results(self, filename="simulation_results.png"):
        """Create time-series plots of forces throughout simulation"""
        
        print(f"\nGenerating time-series visualization...")
        
        if len(self.log_time) == 0:
            print("No data logged - skipping plot (requires t>5.0s and displacement>0)")
            return
        
        # Convert to numpy arrays
        time = np.array(self.log_time)
        sphere_force = np.array(self.log_sphere_force)
        cube_force = np.array(self.log_cube_force)
        top_displacement = np.array(self.log_top_displacement)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(14, 12))
        
        # --- Plot 1: Forces over Time ---
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time, sphere_force, 'b-', linewidth=2, label='Sphere Force')
        ax1.plot(time, cube_force, 'g-', linewidth=2, label='Cube Force')
        
        ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Force Magnitude (N)', fontsize=12, fontweight='bold')
        ax1.set_title('Measured Forces During Simulation', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax1.set_ylim(bottom=0)  # Start from 0 force
        
        # Add stats box
        stats_text = f"""
Final Values:
Sphere: {sphere_force[-1]:.2f} N
Cube: {cube_force[-1]:.2f} N
Total: {sphere_force[-1] + cube_force[-1]:.2f} N
        """
        ax1.text(0.02, 0.98, stats_text.strip(),
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- Plot 2: Top Face Displacement over Time ---
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(time, top_displacement, 'r-', linewidth=2)
        
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Top Face Displacement (mm)', fontsize=12, fontweight='bold')
        ax2.set_title('Cube Compression During Simulation (Positive = Compressed)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax2.set_ylim(bottom=0)  # Start from 0 displacement
        
        # Add max compression annotation
        max_compression = np.max(top_displacement)
        max_idx = np.argmax(top_displacement)
        max_time = time[max_idx]
        
        ax2.annotate(f'Max: {max_compression:.2f} mm\nat t={max_time:.2f}s',
                    xy=(max_time, max_compression),
                    xytext=(max_time + 0.5, max_compression - 0.2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # --- Plot 3: Force-Displacement Curve with Hertzian Theory ---
        ax3 = plt.subplot(3, 1, 3)
        
        # Simulation data - color by time for trajectory visualization
        scatter = ax3.scatter(top_displacement, sphere_force, 
                             c=time, cmap='viridis', s=20, alpha=0.7,
                             edgecolors='black', linewidth=0.3,
                             label='Simulation', zorder=3)
        
        # Hertzian contact theory prediction
        # F = (4/3) * E* * sqrt(R) * δ^(3/2)
        # E* = E / (1 - ν²)
        E = 1.5e4  # Young's modulus (Pa)
        nu = 0.3   # Poisson's ratio
        R = 0.05   # Sphere radius (m)
        
        E_star = E / (1 - nu**2)
        
        # Create displacement range for analytical curve
        delta_range = np.linspace(0, np.max(top_displacement), 100)  # mm
        delta_m = delta_range / 1000  # convert to meters
        
        # Hertzian force (Newtons)
        F_hertz = (4/3) * E_star * np.sqrt(R) * (delta_m ** 1.5)
        
        # Plot Hertzian prediction
        ax3.plot(delta_range, F_hertz, 'r--', linewidth=3, 
                label='Hertzian Theory', zorder=2)
        
        ax3.set_xlabel('Displacement (mm)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sphere Force (N)', fontsize=12, fontweight='bold')
        ax3.set_title('Force-Displacement: Simulation vs Hertzian Theory\n' + 
                     r'$F = \frac{4}{3}E^*\sqrt{R}\delta^{3/2}$ where $E^* = \frac{E}{1-\nu^2}$',
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        
        # Set axes to start from 0
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=0)
        
        # Add legend for simulation and theory
        ax3.legend(fontsize=11, loc='best')
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Time (s)', fontsize=10)
        
        # Add material properties text box
        props_text = f"""
Material Properties:
E = {E/1000:.1f} kPa
ν = {nu}
E* = {E_star/1000:.1f} kPa
R = {R*100:.0f} cm

Mesh:
Particles: {self.model.particle_count}
Grid: {self.cell_dim}³
Cell: {self.cell_size*100:.0f}cm
r_p: {self.cell_size*0.2*1000:.1f}mm
        """
        ax3.text(0.98, 0.02, props_text.strip(),
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Overall title
        fig.suptitle(f'Force Analysis with Hertzian Contact Theory\n' + 
                    f'{self.model.particle_count} Particles ({self.cell_dim}³ grid, {self.sim_substeps} substeps/frame)', 
                    fontsize=15, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Time-series visualization saved to {filename}")
        print(f"  Logged {len(time)} frames (positive displacement only)")
        print(f"  Time range: {time[0]:.2f}s - {time[-1]:.2f}s")
        plt.close()
    
    def print_forces(self):
        """Print contact information"""
        body_forces = self.state_0.body_f.numpy()
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        body_q = self.state_0.body_q.numpy()
        sphere_z = body_q[self.sphere_body_id, 2]
        
        body_qd = self.state_0.body_qd.numpy()
        sphere_vz = body_qd[self.sphere_body_id, 2]
        
        particle_forces = self.state_0.particle_f.numpy()
        particle_q = self.state_0.particle_q.numpy()
        
        total_particle_force = np.sum(particle_forces, axis=0)
        total_particle_force_mag = np.linalg.norm(total_particle_force)
        
        cube_top_z = np.max(particle_q[:, 2])
        top_displacement = (self.initial_cube_top - cube_top_z) * 1000  # mm
        
        sphere_weight = self.sphere_mass * 9.81
        
        print(f"Frame {self.frame_counter:4d} | t={self.sim_time:.2f}s")
        print(f"  Sphere: z={sphere_z*100:.2f}cm, vz={sphere_vz:.3f}m/s, F={sphere_force_mag:7.2f}N", end="")
        if sphere_force_mag > 0.1:
            print(f" ({sphere_force_mag/sphere_weight:.1f}x)")
        else:
            print()
        print(f"  Cube:   top_z={cube_top_z*100:.2f}cm, compression={top_displacement:.2f}mm, F={total_particle_force_mag:.2f}N")
        print()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print additional status messages")
    parser.add_argument("--cell-dim", type=int, default=3, help="Grid dimensions (default: 3 = 64 particles)")
    
    viewer, args = newton.examples.init(parser)
    
    example = Example(viewer, verbose=args.verbose, cell_dim=args.cell_dim)
    
    newton.examples.run(example, args)
    
    example.save_log("force_log.csv")
    example.plot_results("simulation_results.png")

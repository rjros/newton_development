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
# Sphere Falling onto Soft Cube with Force Logging
#
# A rigid sphere falls onto a soft deformable cube - logs all forces to CSV
#
###########################################################################

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False):
        
        # Simulation parameters
        self.fps = 60  # Frames per second for visualization
        self.frame_dt = 1.0 / self.fps  # Time step per frame
        self.sim_time = 0.0
        
        # Physics substeps (higher = more stable but slower)
        self.sim_substeps = 64  # Number of physics substeps per frame
        self.sim_dt = self.frame_dt / self.sim_substeps  # Physics time step

        # Setup rendering
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

        # Ground plane configuration
        # STABILITY: Match contact properties with soft contact settings
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = 1.0e4  # Increased for stiffer contacts
        ground_cfg.kd = 1.0e2  # Increased for more damping
        builder.add_ground_plane(cfg=ground_cfg)
        
        # Create FEM MODEL
        builder.default_particle_radius = 0.001
        
        # Setup grid parameters for soft box
        cell_dim = 3  # 3x3x3 cells
        cell_size = 0.05  # 5cm cells
        
        # Compute particle density for 1.0 kg total mass
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        
        if self.verbose:
            print(f"Particle density: {particle_density}")
        
        # Compute Lame parameters
        young_mod = 1.5e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        
        # STABILITY: Add material damping to dissipate energy
        k_damp = 100.0  # Increased from 0.0 (damping stabilizes high-frequency oscillations)
        
        # Add soft grid (box) on ground
        # Position: bottom of box at z=0
        box_height = cell_dim * cell_size
        
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
            k_damp=k_damp,  # Material damping for stability
            tri_ke=1e-3,
            tri_ka=1e-3,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )
        
        # Add rigid sphere falling from above
        sphere_mass = 0.5  # kg
        sphere_radius = 0.05  # 5cm radius
        sphere_height = 0.5  # Starting height: 50cm above ground
        
        # Create sphere body with exact mass (density=0 so mass only from add_body)
        sphere_cfg = builder.default_shape_cfg.copy()
        sphere_cfg.density = 0.0  # Mass comes from add_body parameter only
        sphere_cfg.ke = 1.0e4  # Match ground contact stiffness
        sphere_cfg.kd = 1.0e2  # Match ground contact damping
        
        body_sphere = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, sphere_height), q=wp.quat_identity()),
            mass=sphere_mass
        )
        
        builder.add_shape_sphere(
            body=body_sphere,
            radius=sphere_radius,
            cfg=sphere_cfg,
        )

        # Finalize model
        self.model = builder.finalize()
        
        # Contact properties
        # STABILITY: Higher contact stiffness and damping prevent interpenetration
        self.model.soft_contact_ke = 1.0e4  # Increased from 1.0e2 (stiffer contacts)
        self.model.soft_contact_kd = 1.0e2  # Increased from 1.0e0 (more contact damping)
        self.model.soft_contact_mu = 0.5    # Reduced from 1.0 (lower friction can help)
       
        # Select the solver
        self.solver = newton.solvers.SolverSemiImplicit(model=self.model)

        # Allocate sim states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        
        # Rendering
        self.viewer.set_model(self.model)

        # Capture forward passes
        self.capture()
        
        # Frame counter and sphere tracking
        self.frame_counter = 0
        self.sphere_mass = sphere_mass
        self.sphere_body_id = 0  # First rigid body
        
        # Print info
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
            
            # Apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
           
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0 

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Print forces every 10 frames (only during non-graph simulation)
        if self.frame_counter % 10 == 0 and not self.graph:
            self.print_forces()
    
    def save_log(self, filename="force_log.csv"):
        """Generate and save force data summary at end of simulation"""
        import csv
        
        print(f"\nGenerating force summary from final state...")
        
        # Get final state data
        body_forces = self.state_0.body_f.numpy()
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_force_z = sphere_force[2]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        body_q = self.state_0.body_q.numpy()
        sphere_z = body_q[self.sphere_body_id, 2]
        
        body_qd = self.state_0.body_qd.numpy()
        sphere_vz = body_qd[self.sphere_body_id, 2]
        
        particle_forces = self.state_0.particle_f.numpy()
        total_particle_force = np.sum(particle_forces, axis=0)
        cube_force_z = total_particle_force[2]
        cube_force_mag = np.linalg.norm(total_particle_force)
        
        particle_q = self.state_0.particle_q.numpy()
        cube_avg_z = np.mean(particle_q[:, 2])
        
        # Create a simple summary file with final values
        header = ['time', 'sphere_force_z', 'sphere_force_mag', 'cube_force_z', 'cube_force_mag', 
                  'sphere_z', 'sphere_vz', 'cube_avg_z']
        
        # Write single row with final state
        data_row = [
            self.sim_time,
            sphere_force_z,
            sphere_force_mag,
            cube_force_z,
            cube_force_mag,
            sphere_z,
            sphere_vz,
            cube_avg_z
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data_row)
        
        print(f"\nFinal state saved to {filename}")
        print(f"  Simulation time: {self.sim_time:.2f}s")
        print(f"  Total frames: {self.frame_counter}")
        print(f"\nFinal Forces:")
        print(f"  Sphere: Fz={sphere_force_z:.2f}N, |F|={sphere_force_mag:.2f}N")
        print(f"  Cube:   Fz={cube_force_z:.2f}N, |F|={cube_force_mag:.2f}N")
        print(f"  Sphere position: z={sphere_z*100:.2f}cm")
        print(f"  Cube avg height: {cube_avg_z*100:.2f}cm")
    
    def plot_results(self, filename="simulation_results.png"):
        """Create visualization of final simulation state and save as PNG"""
        
        print(f"\nGenerating visualization...")
        
        # Get final state data
        body_forces = self.state_0.body_f.numpy()
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        body_q = self.state_0.body_q.numpy()
        sphere_pos = body_q[self.sphere_body_id, 0:3]
        
        particle_forces = self.state_0.particle_f.numpy()
        total_particle_force = np.sum(particle_forces, axis=0)
        cube_force_mag = np.linalg.norm(total_particle_force)
        
        particle_q = self.state_0.particle_q.numpy()
        cube_avg_z = np.mean(particle_q[:, 2])
        cube_min_z = np.min(particle_q[:, 2])
        cube_max_z = np.max(particle_q[:, 2])
        
        # Expected values
        sphere_weight = self.sphere_mass * 9.81
        cube_weight = 1.0 * 9.81  # 1kg cube
        total_weight = sphere_weight + cube_weight
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        
        # --- Plot 1: Force Comparison Bar Chart ---
        ax1 = plt.subplot(2, 3, 1)
        forces = [sphere_force_mag, sphere_weight, cube_force_mag, total_weight]
        labels = ['Sphere\nMeasured', 'Sphere\nExpected', 'Cube\nMeasured', 'Total\nExpected']
        colors = ['#3498db', '#95a5a6', '#2ecc71', '#95a5a6']
        
        bars = ax1.bar(labels, forces, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
        ax1.set_title('Final Forces: Measured vs Expected', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, force in zip(bars, forces):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{force:.2f}N',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # --- Plot 2: Force Components (3D) ---
        ax2 = plt.subplot(2, 3, 2)
        components = ['Fx', 'Fy', 'Fz']
        sphere_components = sphere_force
        cube_components = total_particle_force
        
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, sphere_components, width, label='Sphere', 
                       color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x + width/2, cube_components, width, label='Cube', 
                       color='#2ecc71', alpha=0.7, edgecolor='black')
        
        ax2.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
        ax2.set_title('Force Components', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(components)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # --- Plot 3: Vertical Positions ---
        ax3 = plt.subplot(2, 3, 3)
        positions = [sphere_pos[2]*100, cube_avg_z*100, cube_max_z*100, cube_min_z*100]
        pos_labels = ['Sphere\nZ', 'Cube\nAvg Z', 'Cube\nMax Z', 'Cube\nMin Z']
        pos_colors = ['#e74c3c', '#2ecc71', '#27ae60', '#16a085']
        
        bars = ax3.bar(pos_labels, positions, color=pos_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Height (cm)', fontsize=12, fontweight='bold')
        ax3.set_title('Vertical Positions', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=2, label='Ground')
        
        # Add value labels
        for bar, pos in zip(bars, positions):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pos:.2f}cm',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # --- Plot 4: Particle Force Distribution ---
        ax4 = plt.subplot(2, 3, 4)
        particle_force_mags = np.linalg.norm(particle_forces, axis=1)
        
        ax4.hist(particle_force_mags, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Force Magnitude (N)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Particles', fontsize=12, fontweight='bold')
        ax4.set_title('Particle Force Distribution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        ax4.axvline(np.mean(particle_force_mags), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(particle_force_mags):.2f}N')
        ax4.legend()
        
        # --- Plot 5: Cube Geometry (Top View) ---
        ax5 = plt.subplot(2, 3, 5)
        
        # Plot particle positions (top view: X-Y)
        scatter = ax5.scatter(particle_q[:, 0]*100, particle_q[:, 1]*100, 
                            c=particle_q[:, 2]*100, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot sphere position
        ax5.plot(sphere_pos[0]*100, sphere_pos[1]*100, 'r*', 
                markersize=20, label=f'Sphere (z={sphere_pos[2]*100:.1f}cm)')
        
        ax5.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax5.set_title('Top View (colored by Z height)', fontsize=13, fontweight='bold')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Height (cm)', fontsize=10)
        
        # --- Plot 6: Summary Statistics Table ---
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
SIMULATION SUMMARY
{'='*40}

Time: {self.sim_time:.2f}s ({self.frame_counter} frames)

SPHERE (Mass: {self.sphere_mass:.2f} kg)
  Position: ({sphere_pos[0]:.3f}, {sphere_pos[1]:.3f}, {sphere_pos[2]:.3f}) m
  Force: {sphere_force_mag:.2f} N
  Expected: {sphere_weight:.2f} N
  Error: {abs(sphere_force_mag - sphere_weight)/sphere_weight * 100:.1f}%

SOFT CUBE (Mass: 1.00 kg)
  Avg Height: {cube_avg_z*100:.2f} cm
  Height Range: {cube_min_z*100:.2f} - {cube_max_z*100:.2f} cm
  Total Force: {cube_force_mag:.2f} N
  Expected: {total_weight:.2f} N
  Error: {abs(cube_force_mag - total_weight)/total_weight * 100:.1f}%

PARTICLES
  Count: {self.model.particle_count}
  Avg Force: {np.mean(particle_force_mags):.2f} N
  Max Force: {np.max(particle_force_mags):.2f} N
  On Ground: {np.sum(particle_q[:, 2] < 0.01)}
        """
        
        ax6.text(0.1, 0.5, summary_text, 
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        fig.suptitle('Sphere on Soft Cube - Final State Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        
        # Show plot
        # plt.show()  # Uncomment to display interactively
        plt.close()
    
    def print_forces(self):
        """Print contact information for sphere and soft cube"""
        # Sphere forces
        body_forces = self.state_0.body_f.numpy()
        sphere_force = body_forces[self.sphere_body_id, 0:3]
        sphere_force_mag = np.linalg.norm(sphere_force)
        
        # Sphere position and velocity
        body_q = self.state_0.body_q.numpy()
        sphere_z = body_q[self.sphere_body_id, 2]
        
        body_qd = self.state_0.body_qd.numpy()
        sphere_vz = body_qd[self.sphere_body_id, 2]
        
        # Get soft cube particle forces (contact + internal)
        particle_forces = self.state_0.particle_f.numpy()
        
        # Count particles in contact with ground (z close to 0)
        particle_q = self.state_0.particle_q.numpy()
        particles_on_ground = np.sum(particle_q[:, 2] < 0.01)  # Within 1cm of ground
        
        # Count particles with significant forces
        particle_force_mags = np.linalg.norm(particle_forces, axis=1)
        particles_with_contact = np.sum(particle_force_mags > 1.0)  # Force > 1N threshold
        
        # Total force on soft cube particles
        total_particle_force = np.sum(particle_forces, axis=0)
        total_particle_force_mag = np.linalg.norm(total_particle_force)
        
        # Average position
        avg_z = np.mean(particle_q[:, 2])
        min_z = np.min(particle_q[:, 2])
        max_z = np.max(particle_q[:, 2])
        
        # Expected weights
        sphere_weight = self.sphere_mass * 9.81
        
        print(f"Frame {self.frame_counter:4d} | t={self.sim_time:.2f}s")
        print(f"  Sphere: z={sphere_z*100:.2f}cm, vz={sphere_vz:.3f}m/s")
        print(f"    Contact force: F=[{sphere_force[0]:7.2f}, {sphere_force[1]:7.2f}, {sphere_force[2]:7.2f}] N  |F|={sphere_force_mag:7.2f} N")
        if sphere_force_mag > 0.1:
            print(f"    Force/Weight: {sphere_force_mag/sphere_weight:.2f}x")
        print(f"  Soft cube:")
        print(f"    Avg z={avg_z*100:.2f}cm, Min z={min_z*100:.2f}cm, Max z={max_z*100:.2f}cm")
        print(f"    Particles on ground: {particles_on_ground}/{self.model.particle_count}")
        print(f"    Particles with forces >1N: {particles_with_contact}/{self.model.particle_count}")
        print(f"    Total force: F=[{total_particle_force[0]:7.2f}, {total_particle_force[1]:7.2f}, {total_particle_force[2]:7.2f}] N  |F|={total_particle_force_mag:.2f} N")
        print() 

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
    
    # Save force log to CSV
    example.save_log("force_log.csv")
    
    # Generate and save visualization
    example.plot_results("simulation_results.png")

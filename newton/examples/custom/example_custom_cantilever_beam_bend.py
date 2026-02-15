# Copyright (c) 2# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cantilever Beam Validation - Newton Physics
#
# Based on: "Sim-to-Real for Soft Robots using Differentiable FEM"
# Material: Smooth-On Dragon Skin 10 (Shore Hardness 10A)
#
# Simple gravity drop test (matching Isaac Sim validation)
# Records deflection at 5 tracking points along beam
# Compares to theoretical beam deflection
#
# Usage:
#   python cantilever_newton_validation.py --solver semi_implicit --plot
#   python cantilever_newton_validation.py --solver vbd --plot
#   python cantilever_newton_validation.py --resolution 20
#
###########################################################################

import argparse
import csv
import os
import numpy as np
import warp as wp

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

import newton
import newton.examples


class CantileverValidation:
    """
    Cantilever beam validation matching Isaac Sim parameters:
    
    Material: Dragon Skin 10
    - Young's Modulus: 263,824 Pa (from paper)
    - Poisson's Ratio: 0.4999 (nearly incompressible)
    - Density: 1,070 kg/m³
    
    Geometry: 10cm × 3cm × 3cm beam
    Time window: 0 to 1.5 seconds
    """
    
    def __init__(self, viewer, solver_type='semi_implicit', resolution=10, 
                 max_time=1.5, verbose=False):
        # Simulation parameters
        self.time_step = 0.01  # 100 Hz (matching Isaac Sim h=0.01s)
        self.fps = int(1.0 / self.time_step)
        self.frame = 0
        self.frame_dt = self.time_step
        self.sim_time = 0.0
        self.max_simulation_time = max_time
        
        # Solver-specific substeps
        if solver_type == 'semi_implicit':
            self.sim_substeps = 32
        else:
            self.sim_substeps = 10
        
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        self.solver_type = solver_type
        self.verbose = verbose
        self.resolution = resolution
        
        # Beam geometry (matching Isaac Sim)
        self.beam_length = 0.10  # 10cm
        self.beam_width = 0.03   # 3cm
        self.beam_height = 0.03  # 3cm
        
        # Material properties (Dragon Skin 10 from paper)
        self.youngs_modulus = 263824.0  # Pa
        self.poissons_ratio = 0.4999    # Nearly incompressible
        self.density = 1070.0           # kg/m³
        self.mass_damping = 9.00        # s⁻¹
        
        # Data recording (5 tracking points like Isaac Sim)
        self.time_history = []
        self.deflection_history = {
            'fixed_end': [],
            'quarter': [],
            'midpoint': [],
            'three_quarter': [],
            'free_end': []
        }
        self.position_history = {
            'fixed_end': [],
            'quarter': [],
            'midpoint': [],
            'three_quarter': [],
            'free_end': []
        }
        
        # Viewer
        self.viewer = viewer
        
        # Create model and solver
        self.model = self.create_model()
        self.solver = self.create_solver()
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Store initial positions
        self.initial_positions = self.state_0.particle_q.numpy().copy()
        
        # Track particles at key locations
        self.tracked_particle_indices = {}
        self._find_tracking_particles()
        
        self.viewer.set_model(self.model)
        
        # Calculate theoretical deflection
        self.calculate_theoretical()
        
        self.print_configuration()

    def calculate_theoretical(self):
        """Calculate theoretical tip deflection using beam theory"""
        E = self.youngs_modulus
        rho = self.density
        g = 9.81  # m/s²
        
        L = self.beam_length
        w = self.beam_width
        h = self.beam_height
        
        # Second moment of area (rectangular cross-section)
        I = (w * h**3) / 12
        
        # Weight per unit length
        weight_per_length = rho * w * h * g
        
        # Theoretical tip deflection: δ = (w*L^4)/(8*E*I)
        self.theoretical_tip_deflection = (weight_per_length * L**4) / (8 * E * I)
        
        if self.verbose:
            print(f"\nTheoretical Analysis:")
            print(f"  Second moment of area: I = {I:.6e} m⁴")
            print(f"  Weight per length: w = {weight_per_length:.4f} N/m")
            print(f"  Expected tip deflection: δ = {self.theoretical_tip_deflection*1000:.4f} mm")

    def create_model(self):
        """Create cantilever beam model"""
        scene = newton.ModelBuilder()
        
        # Convert Young's modulus to Lame parameters
        E = self.youngs_modulus
        nu = self.poissons_ratio
        k_mu = 0.5 * E / (1.0 + nu)
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        if self.verbose:
            print(f"Lame parameters:")
            print(f"  μ (shear): {k_mu:.2f} Pa")
            print(f"  λ (bulk): {k_lambda:.2f} Pa")
        
        # Create beam as soft grid
        # For 10cm x 3cm x 3cm beam with given resolution
        # resolution controls number of cells along the LENGTH (longest dimension)
        dim_x = self.resolution  # Along beam length (10cm)
        dim_y = max(int(self.resolution * self.beam_width / self.beam_length), 2)  # Along width (3cm)
        dim_z = max(int(self.resolution * self.beam_height / self.beam_length), 2)  # Along height (3cm)
        
        # Cell sizes
        cell_x = self.beam_length / dim_x
        cell_y = self.beam_width / dim_y
        cell_z = self.beam_height / dim_z
        
        if self.verbose:
            print(f"\nGrid configuration:")
            print(f"  Dimensions: {dim_x} × {dim_y} × {dim_z} cells")
            print(f"  Cell size: {cell_x*1000:.2f}mm × {cell_y*1000:.2f}mm × {cell_z*1000:.2f}mm")
            print(f"  Total particles: {(dim_x+1) * (dim_y+1) * (dim_z+1)}")
        
        # Position beam horizontally at z=2m, centered at origin in x-y
        # The grid creates particles starting from the origin, so we offset to center it
        beam_pos = wp.vec3(
            -self.beam_length / 2.0,  # Center in X
            -self.beam_width / 2.0,   # Center in Y  
            2.0                        # Height at z=2m
        )
        
        # Add soft grid (horizontal beam)
        scene.add_soft_grid(
            pos=beam_pos,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_x,
            cell_y=cell_y,
            cell_z=cell_z,
            density=self.density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=self.mass_damping,
            fix_left=True,   # Fix left end (X=0 after offset) - this is the anchor
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=1.0e-3,   # Surface triangle stiffness
            tri_ka=1.0e-3,
            tri_kd=1.0e-3,
            tri_drag=0.0,
            tri_lift=0.0,
        )
        
        # Add ground plane (for visualization, beam shouldn't hit it)
        scene.add_ground_plane()
        
        # Finalize
        model = scene.finalize(requires_grad=False)
        
        if self.verbose:
            print(f"\nModel created:")
            print(f"  Particles: {model.particle_count}")
            print(f"  Tets: {model.tet_count}")
            print(f"  Triangles: {model.tri_count}")
        
        return model

    def create_solver(self):
        """Create solver based on type"""
        if self.solver_type == 'semi_implicit':
            solver = newton.solvers.SolverSemiImplicit(self.model)
        elif self.solver_type == 'xpbd':
            solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        elif self.solver_type == 'vbd':
            solver = newton.solvers.SolverVBD(self.model, iterations=10)
        else:
            raise ValueError(f"Unknown solver: {self.solver_type}")
        
        return solver

    def _find_tracking_particles(self):
        """Find particles at 5 key positions along beam"""
        positions = self.initial_positions
        
        # Sort particles by x coordinate (beam runs along X axis)
        x_coords = [(idx, pos[0]) for idx, pos in enumerate(positions)]
        x_coords.sort(key=lambda x: x[1])
        
        num_particles = len(x_coords)
        
        # Find particles at key positions along the beam
        # fixed_end = leftmost (minimum x)
        # free_end = rightmost (maximum x)
        tracking_positions = {
            'fixed_end': 0,                      # Leftmost particle (anchor)
            'quarter': num_particles // 4,       # 1/4 along beam
            'midpoint': num_particles // 2,      # Middle of beam
            'three_quarter': 3 * num_particles // 4,  # 3/4 along beam
            'free_end': num_particles - 1        # Rightmost particle (tip)
        }
        
        for label, pos_idx in tracking_positions.items():
            idx, x_val = x_coords[pos_idx]
            self.tracked_particle_indices[label] = idx
        
        if self.verbose:
            print(f"\nTracking particles:")
            for label, idx in self.tracked_particle_indices.items():
                pos = positions[idx]
                print(f"  {label:15s}: particle {idx:4d}, x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
            print(f"  Beam extent: x=[{x_coords[0][1]:.4f}, {x_coords[-1][1]:.4f}]")

    def print_configuration(self):
        """Print configuration summary"""
        print(f"\n{'='*70}")
        print(f"Newton Cantilever Validation")
        print(f"Solver: {self.solver_type.upper()}")
        print(f"{'='*70}")
        print(f"Material: Dragon Skin 10")
        print(f"  Young's Modulus: {self.youngs_modulus:,.0f} Pa")
        print(f"  Poisson's Ratio: {self.poissons_ratio}")
        print(f"  Density: {self.density:,.0f} kg/m³")
        print(f"  Mass Damping: {self.mass_damping} s⁻¹")
        print(f"\nGeometry: {self.beam_length*100:.1f}cm × {self.beam_width*100:.1f}cm × {self.beam_height*100:.1f}cm")
        print(f"Resolution: {self.resolution} elements")
        print(f"\nSimulation:")
        print(f"  Time step: {self.time_step}s")
        print(f"  Max time: {self.max_simulation_time}s")
        print(f"  Substeps: {self.sim_substeps}")
        print(f"\nTheoretical tip deflection: {self.theoretical_tip_deflection*1000:.4f} mm")
        print(f"{'='*70}\n")

    def record_data(self):
        """Record deflection at tracking points"""
        current_positions = self.state_0.particle_q.numpy()
        
        self.time_history.append(self.sim_time)
        
        for label, idx in self.tracked_particle_indices.items():
            current_z = current_positions[idx][2]
            initial_z = self.initial_positions[idx][2]
            deflection = initial_z - current_z  # Positive = downward
            
            self.deflection_history[label].append(deflection * 1000)  # Convert to mm
            self.position_history[label].append(current_positions[idx].copy())

    def simulate(self):
        """Run substeps for one frame"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Step solver (no contacts needed for pure gravity drop)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None,
                self.sim_dt
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Run one frame"""
        if self.sim_time > self.max_simulation_time:
            return False
        
        self.simulate()
        self.record_data()
        
        # Print progress
        if self.frame % 10 == 0 and self.verbose:
            tip_def = self.deflection_history['free_end'][-1]
            print(f"t = {self.sim_time:.3f}s | Tip deflection: {tip_def:.4f} mm")
        
        self.sim_time += self.frame_dt
        self.frame += 1
        
        return True

    def render(self):
        """Render current state"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def save_csv(self, output_dir='./'):
        """Save data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Main data CSV
        csv_filename = os.path.join(output_dir, f'cantilever_{self.solver_type}_data.csv')
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['time_s']
            for label in ['fixed_end', 'quarter', 'midpoint', 'three_quarter', 'free_end']:
                fieldnames.append(f'{label}_deflection_mm')
                fieldnames.extend([f'{label}_x_m', f'{label}_y_m', f'{label}_z_m'])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, time in enumerate(self.time_history):
                row = {'time_s': time}
                
                for label in ['fixed_end', 'quarter', 'midpoint', 'three_quarter', 'free_end']:
                    row[f'{label}_deflection_mm'] = self.deflection_history[label][i]
                    pos = self.position_history[label][i]
                    row[f'{label}_x_m'] = pos[0]
                    row[f'{label}_y_m'] = pos[1]
                    row[f'{label}_z_m'] = pos[2]
                
                writer.writerow(row)
        
        print(f"✓ Data saved: {csv_filename}")
        
        # Summary CSV
        summary_filename = os.path.join(output_dir, f'cantilever_{self.solver_type}_summary.csv')
        
        final_tip_deflection = self.deflection_history['free_end'][-1]
        theoretical_mm = self.theoretical_tip_deflection * 1000
        error_percent = abs(final_tip_deflection - theoretical_mm) / theoretical_mm * 100
        
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Solver', self.solver_type])
            writer.writerow(['Beam Length (m)', self.beam_length])
            writer.writerow(['Resolution (elements)', self.resolution])
            writer.writerow(['Young\'s Modulus (Pa)', self.youngs_modulus])
            writer.writerow(['Poisson\'s Ratio', self.poissons_ratio])
            writer.writerow(['Density (kg/m³)', self.density])
            writer.writerow(['Theoretical Tip Deflection (mm)', f'{theoretical_mm:.6f}'])
            writer.writerow(['Simulated Tip Deflection (mm)', f'{final_tip_deflection:.6f}'])
            writer.writerow(['Error (%)', f'{error_percent:.2f}'])
            writer.writerow(['Status', 'PASS' if error_percent < 15 else 'FAIL'])
        
        print(f"✓ Summary saved: {summary_filename}")

    def plot_results(self, save_path=None):
        """Generate validation plots"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return
        
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Newton Cantilever Validation - {self.solver_type.upper()} Solver', 
                    fontsize=14, fontweight='bold')
        
        times = np.array(self.time_history)
        
        # Plot 1: Tip deflection vs time
        ax1 = axes[0, 0]
        tip_deflections = np.array(self.deflection_history['free_end'])
        ax1.plot(times, tip_deflections, 'b-', linewidth=2, label='Newton')
        ax1.axhline(y=self.theoretical_tip_deflection * 1000, color='r', linestyle='--', 
                    linewidth=2, label=f'Theory ({self.theoretical_tip_deflection*1000:.3f} mm)')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Tip Deflection (mm)', fontsize=11)
        ax1.set_title('Free End Deflection vs Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Deflection profile at final time
        ax2 = axes[0, 1]
        labels_list = ['fixed_end', 'quarter', 'midpoint', 'three_quarter', 'free_end']
        x_positions = [self.position_history[label][-1][0] for label in labels_list]
        deflections_final = [self.deflection_history[label][-1] for label in labels_list]
        
        # Theoretical curve
        x_theory = np.linspace(min(x_positions), max(x_positions), 100)
        L = self.beam_length
        x_from_fixed = x_theory - min(x_positions)
        
        E = self.youngs_modulus
        w_beam = self.beam_width
        h_beam = self.beam_height
        I = (w_beam * h_beam**3) / 12
        w = self.density * w_beam * h_beam * 9.81
        
        deflection_theory = (w * x_from_fixed**2 / (24 * E * I)) * (6 * L**2 - 4 * L * x_from_fixed + x_from_fixed**2)
        
        ax2.plot(x_theory, deflection_theory * 1000, 'r--', linewidth=2, label='Theory')
        ax2.plot(x_positions, deflections_final, 'bo-', markersize=8, linewidth=2, label='Newton')
        ax2.set_xlabel('X Position (m)', fontsize=11)
        ax2.set_ylabel('Deflection (mm)', fontsize=11)
        ax2.set_title(f'Deflection Profile at t={times[-1]:.3f}s', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.invert_yaxis()
        
        # Plot 3: All tracking points
        ax3 = axes[1, 0]
        colors = ['green', 'blue', 'orange', 'purple', 'red']
        for (label, color) in zip(labels_list, colors):
            deflections = np.array(self.deflection_history[label])
            ax3.plot(times, deflections, color=color, linewidth=1.5, 
                    label=label.replace('_', ' ').title())
        
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Deflection (mm)', fontsize=11)
        ax3.set_title('Deflection at Multiple Points', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Plot 4: Summary
        ax4 = axes[1, 1]
        final_deflection = tip_deflections[-1]
        theoretical = self.theoretical_tip_deflection * 1000
        error_percent = abs(final_deflection - theoretical) / theoretical * 100
        
        summary_text = f"""
VALIDATION SUMMARY

Solver: {self.solver_type.upper()}

Material: Dragon Skin 10
E = {self.youngs_modulus:,.0f} Pa
ν = {self.poissons_ratio}
ρ = {self.density:,.0f} kg/m³

Beam: {self.beam_length*100:.1f} × {self.beam_width*100:.1f} × {self.beam_height*100:.1f} cm
Resolution: {self.resolution} elements

RESULTS (at t={times[-1]:.3f}s):
Theoretical:  {theoretical:.4f} mm
Simulated:    {final_deflection:.4f} mm
Error:        {error_percent:.2f}%

Status: {"✓ PASS" if error_percent < 15 else "✗ FAIL"}
"""
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()


def run_validation(viewer, args):
    """Run validation for single solver"""
    example = CantileverValidation(
        viewer, 
        solver_type=args.solver,
        resolution=args.resolution,
        max_time=args.max_time,
        verbose=args.verbose
    )
    
    # Run simulation
    while example.step():
        example.render()
    
    # Save results
    example.save_csv(output_dir=args.output_dir)
    
    # Plot if requested
    if args.plot:
        save_path = os.path.join(args.output_dir, f'cantilever_{args.solver}.png')
        example.plot_results(save_path=save_path)
    
    return example


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", type=str, default="semi_implicit",
                       choices=["semi_implicit", "xpbd", "vbd"],
                       help="Solver type")
    parser.add_argument("--resolution", type=int, default=10,
                       help="Mesh resolution (elements along length)")
    parser.add_argument("--max-time", type=float, default=1.5,
                       help="Simulation time (seconds)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--output-dir", type=str, default="./cantilever_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    viewer, args = newton.examples.init(parser)
    
    run_validation(viewer, args)

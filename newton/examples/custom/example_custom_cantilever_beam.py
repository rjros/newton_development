# Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cantilever Beam Benchmark - Newton Multi-Solver Comparison
#
# Replicates the Isaac Sim cantilever validation for Newton physics.
# Tests multiple solvers (Semi-Implicit, XPBD, VBD) with Dragon Skin 10 material.
#
# Three-phase test:
# 1. SETTLING: Beam drops under gravity (3s)
# 2. RAISING: Kinematically raise to horizontal (1s)  
# 3. RELEASE: Release and observe oscillation (3s)
#
# Usage:
#   python cantilever_newton_benchmark.py --solver semi_implicit
#   python cantilever_newton_benchmark.py --solver xpbd
#   python cantilever_newton_benchmark.py --solver vbd
#   python cantilever_newton_benchmark.py --compare-all
#
###########################################################################

import argparse
import numpy as np
import warp as wp

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: uv pip install matplotlib")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Frequency analysis will be limited.")

import newton
import newton.examples


class CantileverBenchmark:
    """
    Cantilever beam benchmark matching Isaac Sim configuration
    
    Material: Dragon Skin 10
    - Young's Modulus: 234,900 Pa
    - Poisson's Ratio: 0.439
    - Density: 1,210 kg/m³
    - Mass Damping: 9.11 s⁻¹
    
    Geometry: 10cm × 3cm × 3cm beam
    """
    
    # Phase constants
    PHASE_SETTLING = 0
    PHASE_RAISING = 1
    PHASE_OBSERVING = 2
    
    def __init__(self, viewer, solver_type='semi_implicit', verbose=False):
        # Simulation parameters
        self.fps = 100  # 100 Hz like Isaac Sim
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        
        # Solver-specific substeps
        if solver_type == 'semi_implicit':
            self.sim_substeps = 32
        else:
            self.sim_substeps = 10
        
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        self.solver_type = solver_type
        self.verbose = verbose
        
        # Phase timing (matching Isaac Sim)
        self.settling_time = 3.0
        self.raise_duration = 1.0
        self.observation_time = 3.0
        
        self.raise_start_time = self.settling_time
        self.release_time = self.settling_time + self.raise_duration
        self.total_time = self.release_time + self.observation_time
        
        self.current_phase = self.PHASE_SETTLING
        
        # Beam geometry (matching Isaac Sim)
        self.beam_length = 0.10  # 10cm
        self.beam_width = 0.03   # 3cm
        self.beam_height = 0.03  # 3cm
        
        # Data recording
        self.time_history = []
        self.phase_history = []
        self.tip_z_history = []
        self.tip_y_history = []
        self.tip_x_history = []
        
        # Viewer
        self.viewer = viewer
        
        # Create model and solver
        self.model = self.create_model()
        self.solver = self.create_solver()
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Store initial and kinematic positions
        self.initial_positions = None
        self.settled_positions = None
        self.horizontal_positions = None
        
        # Track tip particle index
        self.tip_particle_idx = None
        
        self.viewer.set_model(self.model)
        
        self._find_tip_particle()
        
        # Store initial horizontal position
        self.initial_positions = self.state_0.particle_q.numpy().copy()
        self.horizontal_positions = self.initial_positions.copy()
        
        print(f"\n{'='*70}")
        print(f"Newton Cantilever Benchmark")
        print(f"Solver: {solver_type.upper()}")
        print(f"{'='*70}")
        print(f"Material: Dragon Skin 10")
        print(f"  Young's Modulus: 234,900 Pa")
        print(f"  Poisson's Ratio: 0.439")
        print(f"  Density: 1,210 kg/m³")
        print(f"  Mass Damping: 9.11 s⁻¹")
        print(f"\nGeometry: {self.beam_length*100:.1f}cm × {self.beam_width*100:.1f}cm × {self.beam_height*100:.1f}cm")
        print(f"\nPhase Timing:")
        print(f"  Settling:  0 to {self.settling_time}s")
        print(f"  Raising:   {self.settling_time}s to {self.release_time}s")  
        print(f"  Observing: {self.release_time}s to {self.total_time}s")
        print(f"{'='*70}\n")

    def create_model(self):
        """Create cantilever beam model"""
        scene = newton.ModelBuilder()
        
        # Dragon Skin 10 material properties
        young_mod = 234900.0  # Pa
        poisson_ratio = 0.439
        density = 1210.0  # kg/m³
        mass_damping = 9.11  # s⁻¹
        
        # Convert to Lame parameters
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        
        if self.verbose:
            print(f"Lame parameters:")
            print(f"  μ (shear): {k_mu:.2f} Pa")
            print(f"  λ (bulk):  {k_lambda:.2f} Pa")
        
        # Create beam as soft grid
        # Resolution: 10 cells along length (10cm / 1cm = 10)
        dim_x = 10  # Length direction
        dim_y = 3   # Width direction  
        dim_z = 3   # Height direction
        
        cell_size = 0.01  # 1cm cells
        
        # Add soft grid
        scene.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),  # Center at z=2m (horizontal)
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=mass_damping,  # Mass-proportional damping
            fix_left=True,  # Fix left end (anchor)
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
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

    def _find_tip_particle(self):
        """Find the rightmost particle (tip of cantilever)"""
        positions = self.state_0.particle_q.numpy()
        
        # Find particle with maximum x coordinate
        x_coords = positions[:, 0]
        self.tip_particle_idx = np.argmax(x_coords)
        
        tip_pos = positions[self.tip_particle_idx]
        
        if self.verbose:
            print(f"\nTip particle:")
            print(f"  Index: {self.tip_particle_idx}")
            print(f"  Position: ({tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f})")

    def raise_beam_kinematically(self, current_time):
        """Kinematically interpolate beam from settled to horizontal"""
        if self.settled_positions is None:
            return
        
        # Interpolation factor (0 at start, 1 at end)
        t = (current_time - self.raise_start_time) / self.raise_duration
        t = np.clip(t, 0.0, 1.0)
        
        # Smoothstep interpolation
        t_smooth = 3*t**2 - 2*t**3
        
        # Interpolate
        current_pos = self.settled_positions + t_smooth * (self.horizontal_positions - self.settled_positions)
        
        # Set positions
        self.state_0.particle_q = wp.array(current_pos, dtype=wp.vec3)

    def record_data(self):
        """Record tip position"""
        positions = self.state_0.particle_q.numpy()
        tip_pos = positions[self.tip_particle_idx]
        
        # Convert to mm for plotting (matching Isaac Sim)
        self.time_history.append(self.sim_time)
        self.phase_history.append(self.current_phase)
        self.tip_x_history.append(tip_pos[0] * 1000)  # mm
        self.tip_y_history.append(tip_pos[1] * 1000)  # mm
        self.tip_z_history.append(tip_pos[2] * 1000)  # mm

    def simulate(self):
        """Run substeps for one frame"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            
            # Step solver
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                None,  # No contacts needed
                self.sim_dt
            )
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Run one frame"""
        # Handle phase transitions and kinematic control
        if self.sim_time < self.raise_start_time:
            # Phase 1: Settling
            if self.current_phase != self.PHASE_SETTLING:
                print(f"\nPHASE 1: SETTLING")
                self.current_phase = self.PHASE_SETTLING
            
            self.simulate()
            
        elif self.sim_time < self.release_time:
            # Phase 2: Raising
            if self.current_phase != self.PHASE_RAISING:
                print(f"\nPHASE 2: RAISING (kinematic)")
                # Store settled positions
                self.settled_positions = self.state_0.particle_q.numpy().copy()
                self.current_phase = self.PHASE_RAISING
            
            # Kinematic interpolation (no physics)
            self.raise_beam_kinematically(self.sim_time)
            
        elif self.sim_time < self.total_time:
            # Phase 3: Observing
            if self.current_phase != self.PHASE_OBSERVING:
                print(f"\nPHASE 3: RELEASE (observing)")
                self.current_phase = self.PHASE_OBSERVING
            
            self.simulate()
            
        else:
            # Simulation complete
            return False
        
        # Record data
        self.record_data()
        
        # Print progress
        if self.frame % 50 == 0 and self.verbose:
            print(f"[{['SETTLING', 'RAISING', 'OBSERVING'][self.current_phase]}] "
                  f"t={self.sim_time:.3f}s | Tip z: {self.tip_z_history[-1]:.3f}mm")
        
        self.sim_time += self.frame_dt
        self.frame += 1
        
        return True

    def render(self):
        """Render current state"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def get_results(self):
        """Return results dictionary"""
        times = np.array(self.time_history)
        phases = np.array(self.phase_history)
        tip_z = np.array(self.tip_z_history)
        
        # Find horizontal reference (at release)
        release_idx = np.where(times >= self.release_time)[0][0]
        horizontal_z = tip_z[release_idx]
        
        # Release phase data
        release_mask = phases == self.PHASE_OBSERVING
        release_times = times[release_mask] - self.release_time
        release_tip_z = tip_z[release_mask]
        
        results = {
            'solver': self.solver_type,
            'times': times,
            'phases': phases,
            'tip_z': tip_z,
            'tip_x': np.array(self.tip_x_history),
            'tip_y': np.array(self.tip_y_history),
            'horizontal_z': horizontal_z,
            'release_times': release_times,
            'release_tip_z': release_tip_z,
        }
        
        # Calculate statistics
        if len(release_tip_z) > 0:
            results['max_displacement'] = np.max(horizontal_z - release_tip_z)
            results['final_z'] = release_tip_z[-1]
            
            # Find oscillation frequency
            if SCIPY_AVAILABLE and len(release_tip_z) > 20:
                peaks, _ = signal.find_peaks(release_tip_z, distance=10)
                if len(peaks) >= 2:
                    peak_times = release_times[peaks]
                    periods = np.diff(peak_times)
                    results['frequency_hz'] = 1.0 / np.mean(periods)
                    results['period_s'] = np.mean(periods)
        
        return results


def plot_single_solver(results, save_path=None):
    """Plot results for a single solver"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    
    times = results['times']
    tip_z = results['tip_z']
    solver_name = results['solver'].upper().replace('_', '-')
    
    # Get phase boundaries
    settling_time = 3.0
    release_time = 4.0
    
    # Plot 1: Complete timeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, tip_z, 'b-', linewidth=2, label='Tip Z')
    ax1.axvline(x=settling_time, color='orange', linestyle='--', linewidth=2)
    ax1.axvline(x=release_time, color='red', linestyle='--', linewidth=2)
    ax1.axvspan(0, settling_time, alpha=0.2, color='gray', label='Settling')
    ax1.axvspan(settling_time, release_time, alpha=0.2, color='orange', label='Raising')
    ax1.axvspan(release_time, times[-1], alpha=0.2, color='green', label='Observing')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Tip Z Position (mm)', fontsize=11)
    ax1.set_title(f'Complete Timeline - {solver_name} Solver', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Release phase detail
    ax2 = fig.add_subplot(gs[1, :])
    release_times = results['release_times']
    release_tip_z = results['release_tip_z']
    horizontal_z = results['horizontal_z']
    displacement = horizontal_z - release_tip_z
    
    ax2.plot(release_times, displacement, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time Since Release (s)', fontsize=11)
    ax2.set_ylabel('Displacement from Horizontal (mm)', fontsize=11)
    ax2.set_title(f'Motion After Release - {solver_name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Summary text
    ax3 = fig.add_subplot(gs[2, :])
    
    summary = f"""
NEWTON CANTILEVER BENCHMARK - {solver_name} SOLVER

Material: Dragon Skin 10
  E = 234,900 Pa, ν = 0.439, ρ = 1,210 kg/m³, α = 9.11 s⁻¹

Geometry: 10.0 × 3.0 × 3.0 cm

RESULTS:
  Horizontal (release): {horizontal_z:.3f} mm
"""
    
    if 'max_displacement' in results:
        summary += f"  Max displacement: {results['max_displacement']:.3f} mm\n"
        summary += f"  Final z: {results['final_z']:.3f} mm\n"
    
    if 'frequency_hz' in results:
        summary += f"  Oscillation freq: {results['frequency_hz']:.3f} Hz\n"
        summary += f"  Period: {results['period_s']:.4f} s\n"
    
    ax3.text(0.1, 0.5, summary, transform=ax3.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.axis('off')
    
    fig.suptitle(f'Newton Cantilever Benchmark - {solver_name} Solver', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    plt.show()


def plot_solver_comparison(all_results, save_path=None):
    """Plot comparison of multiple solvers"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = {'semi_implicit': 'blue', 'xpbd': 'red', 'vbd': 'green'}
    
    # Plot 1: Complete timeline comparison
    ax1 = axes[0]
    for results in all_results:
        solver = results['solver']
        label = solver.upper().replace('_', '-')
        ax1.plot(results['times'], results['tip_z'], 
                color=colors.get(solver, 'black'), linewidth=2, label=label)
    
    ax1.axvline(x=3.0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=4.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Tip Z Position (mm)', fontsize=12)
    ax1.set_title('Solver Comparison - Complete Timeline', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Release phase comparison
    ax2 = axes[1]
    for results in all_results:
        solver = results['solver']
        label = solver.upper().replace('_', '-')
        displacement = results['horizontal_z'] - results['release_tip_z']
        ax2.plot(results['release_times'], displacement,
                color=colors.get(solver, 'black'), linewidth=2, label=label)
    
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time Since Release (s)', fontsize=12)
    ax2.set_ylabel('Displacement from Horizontal (mm)', fontsize=12)
    ax2.set_title('Solver Comparison - Motion After Release', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {save_path}")
    
    plt.show()


def run_single_solver(solver_type, viewer, args):
    """Run benchmark for a single solver"""
    example = CantileverBenchmark(viewer, solver_type=solver_type, verbose=args.verbose)
    
    # Run simulation
    while example.step():
        example.render()
    
    # Get results
    results = example.get_results()
    
    # Plot
    if args.plot:
        plot_single_solver(results, save_path=f'cantilever_{solver_type}.png')
    
    return results


def run_comparison(viewer, args):
    """Run all solvers and compare"""
    all_results = []
    
    for solver_type in ['semi_implicit', 'xpbd', 'vbd']:
        print(f"\n{'='*70}")
        print(f"Running {solver_type.upper()} solver...")
        print(f"{'='*70}")
        
        results = run_single_solver(solver_type, viewer, args)
        all_results.append(results)
    
    # Plot comparison
    if args.plot:
        plot_solver_comparison(all_results, save_path='cantilever_comparison.png')
    
    return all_results


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", type=str, default="semi_implicit",
                       choices=["semi_implicit", "xpbd", "vbd"],
                       help="Solver type to use")
    parser.add_argument("--compare-all", action="store_true",
                       help="Run all solvers and compare")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    viewer, args = newton.examples.init(parser)
    
    if args.compare_all:
        run_comparison(viewer, args)
    else:
        run_single_solver(args.solver, viewer, args)

# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Controlled Vertical Indenter Test
#
# A 10mm diameter sphere moves vertically using a prismatic joint
# This will be used as an indenter for compression testing
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, verbose=False):
        
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
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
        
        # Ground plane for reference
        builder.add_ground_plane()
        
        # Indenter parameters
        indenter_diameter = 0.01  # 10mm = 0.01m
        indenter_radius = indenter_diameter / 2.0
        indenter_mass = 0.1  # 100g
        
        # Starting position (above ground)
        start_height = 0.3  # 30cm above ground
        
        # Create indenter sphere with prismatic joint (vertical motion only)
        # First, create the indenter body (link)
        indenter_cfg = builder.default_shape_cfg.copy()
        indenter_cfg.density = 0.0  # Mass from add_link only
        
        body_indenter = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, start_height), q=wp.quat_identity()),
            mass=indenter_mass,
            key="indenter"
        )
        
        builder.add_shape_sphere(
            body=body_indenter,
            radius=indenter_radius,
            cfg=indenter_cfg,
        )
        
        # Add prismatic joint for vertical motion (Z-axis only)
        # Joint connects indenter to world (parent=-1)
        joint_indenter = builder.add_joint_prismatic(
            parent=-1,  # World frame (like in basic_joints example)
            child=body_indenter,
            axis=wp.vec3(0.0, 0.0, 1.0),  # Z-axis (vertical)
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, start_height), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            limit_lower=-0.5,  # Can go down to -50cm from start
            limit_upper=0.5,   # Can go up to 50cm from start
            target_ke=1e4,     # Position control stiffness
            target_kd=1e3,     # Position control damping
            limit_ke=1e5,      # Limit stiffness
            limit_kd=1e3,      # Limit damping
            key="indenter_joint"
        )
        
        # Create articulation from the joint (like in basic_joints example)
        builder.add_articulation([joint_indenter], key="indenter_articulation")
        
        # Finalize model
        self.model = builder.finalize()
        
        # Solver
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        #self.solver = newton.solvers.SolverSemiImplicit(self.model)
        #self.solver = newton.solvers.SolverSemiImplicit(self.model, joint_attach_ke=1600.0, joint_attach_kd=20.0)
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Evaluate forward kinematics (required for proper joint initialization)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Store indenter info
        self.indenter_body_id = body_indenter
        self.indenter_joint_id = 0  # First joint
        self.indenter_mass = indenter_mass
        self.indenter_radius = indenter_radius
        
        # Get joint q start index BEFORE capture (can't call .numpy() during capture)
        self.joint_q_start_idx = self.model.joint_q_start.numpy()[self.indenter_joint_id]
        
        # Motion profile parameters
        self.motion_mode = "sinusoidal"  # Options: "sinusoidal", "linear_down", "step"
        self.amplitude = 0.015  # 20cm amplitude for sinusoidal motion
        self.frequency = 0.05  # 0.5 Hz
        self.start_height = start_height
        
        # Rendering
        self.viewer.set_model(self.model)
        
        # Set camera close to indenter sphere
        # Position camera looking at the starting position of the sphere
        cam_distance = 0.5  # 50cm away
        cam_height = start_height  # Same height as sphere
        self.viewer.set_camera(
            pos=wp.vec3(cam_distance, 0.0, cam_height),  # 50cm in front, at sphere height
            pitch=0.0,  # Looking straight ahead
            yaw=180.0   # Looking back at origin where sphere is
        )
        
        # Capture
        self.capture()
        
        # Frame counter
        self.frame_counter = 0
        
        # Print info
        print(f"\n{'='*70}")
        print(f"Controlled Vertical Indenter")
        print(f"{'='*70}")
        print(f"Indenter:")
        print(f"  Diameter: {indenter_diameter*1000:.1f}mm")
        print(f"  Radius: {indenter_radius*1000:.1f}mm")
        print(f"  Mass: {indenter_mass*1000:.1f}g")
        print(f"  Start height: {start_height*100:.1f}cm")
        print(f"Motion control:")
        print(f"  Mode: {self.motion_mode}")
        print(f"  Amplitude: {self.amplitude*100:.1f}cm")
        print(f"  Frequency: {self.frequency:.2f} Hz")
        print(f"{'='*70}\n")

    def capture(self):
        if wp.get_device().is_cuda:
            # Create a warp array for joint control that we can update during capture
            # Following the anymal example pattern
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
            
            # Apply viewer forces
            self.viewer.apply_forces(self.state_0)
            
            # Collide
            contacts = self.model.collide(self.state_0)
            
            # Step solver
            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            
            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_control(self):
        """Update joint target position based on motion profile"""
        t = self.sim_time
        
        if self.motion_mode == "sinusoidal":
            # Sinusoidal motion: z = z0 + A*sin(2πft)
            target_z = self.start_height + self.amplitude * np.sin(2.0 * np.pi * self.frequency * t)
        
        elif self.motion_mode == "linear_down":
            # Linear motion downward at constant speed
            speed = 0.0005  # 5cm/s
            target_z = self.start_height - speed * t
            target_z = max(0.25, 0.0)  # Don't go below ground
        
        elif self.motion_mode == "step":
            # Step down at specific times
            if t < 1.0:
                target_z = self.start_height
            elif t < 3.0:
                target_z = 0.1  # Step to 10cm
            else:
                target_z = 0.05  # Step to 5cm
        
        else:
            target_z = self.start_height
        
        # Set joint target position
        # Joint position is relative displacement along axis
        joint_target = float(target_z - self.start_height)
        
        # Update control - create full array with zeros and target at correct index
        target_array = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        target_array[self.joint_q_start_idx] = joint_target
        
        # Convert to warp and copy (like anymal example)
        target_wp = wp.array(target_array, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos, target_wp)

    def step(self):
        # Update control BEFORE simulating (outside of CUDA graph)
        self.update_control()
        
        # Run simulation
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        # Update time after simulation
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Print status every 10 frames
        if self.frame_counter % 10 == 0:
            self.print_status()
    
    def print_status(self):
        """Print indenter position and control status"""
        # Get joint state (use pre-stored index)
        joint_pos = self.state_0.joint_q.numpy()[self.joint_q_start_idx]
        joint_vel = self.state_0.joint_qd.numpy()[self.joint_q_start_idx]
        
        # Get target
        joint_target = self.control.joint_target_pos.numpy()[self.joint_q_start_idx]
        
        # Get body position
        body_q = self.state_0.body_q.numpy()
        indenter_z = body_q[self.indenter_body_id, 2]
        
        # Get body velocity
        body_qd = self.state_0.body_qd.numpy()
        indenter_vz = body_qd[self.indenter_body_id, 2]
        
        # Position error
        pos_error = joint_pos - joint_target
        
        print(f"Frame {self.frame_counter:4d} | t={self.sim_time:.2f}s")
        print(f"  Indenter body: z={indenter_z*100:.2f}cm, vz={indenter_vz:.3f}m/s")
        print(f"  Joint: pos={joint_pos*100:.2f}cm, vel={joint_vel:.3f}m/s, target={joint_target*100:.2f}cm")
        print(f"  Tracking error: {abs(pos_error)*1000:.2f}mm")
        print()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages.")
    parser.add_argument("--motion", type=str, default="sinusoidal", 
                       choices=["sinusoidal", "linear_down", "step"],
                       help="Motion profile for indenter")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)
    
    # Override motion mode if specified
    if hasattr(args, 'motion'):
        example.motion_mode = args.motion

    # Run example
    newton.examples.run(example, args)

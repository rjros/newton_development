# 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Simple Sphere Movement Test - Debug Version
#
# Just a 10mm sphere moving vertically with prismatic joint
# No soft cube - for debugging joint control
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
        print(f"  Physics substeps: {self.sim_substeps}")
        print(f"  Physics dt: {self.sim_dt:.6f} s")
        print()
        
        # Build model
        builder = newton.ModelBuilder()
        
        # Ground plane for reference
        builder.add_ground_plane()
        
        # Indenter parameters
        indenter_diameter = 0.01  # 10mm
        indenter_radius = indenter_diameter / 2.0
        indenter_mass = 0.1  # 100g
        
        # Starting position
        start_height = 0.1  # 10cm above ground
        
        # Create indenter sphere with prismatic joint
        indenter_cfg = builder.default_shape_cfg.copy()
        indenter_cfg.density = 0.0
        
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
        
        # Prismatic joint for vertical motion
        joint_indenter = builder.add_joint_prismatic(
            parent=-1,
            child=body_indenter,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, start_height), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            limit_lower=-0.15,
            limit_upper=0.15,
            target_ke=1e4,
            target_kd=1e3,
            limit_ke=1e5,
            limit_kd=1e3,
            key="indenter_joint"
        )
        
        builder.add_articulation([joint_indenter], key="indenter_articulation")
        
        # Finalize
        self.model = builder.finalize()
        
        # Solver
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            joint_attach_ke=1600.0,
            joint_attach_kd=20.0
        )
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        
        # Evaluate FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # Store info FIRST
        self.indenter_body_id = body_indenter
        self.indenter_joint_id = 0
        self.start_height = start_height
        self.joint_q_start_idx = self.model.joint_q_start.numpy()[self.indenter_joint_id]
        
        # NOW initialize control to starting position (after joint_q_start_idx is defined)
        initial_target = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        initial_target[self.joint_q_start_idx] = 0.0  # Start at initial height
        self.control.joint_target_pos = wp.array(initial_target, dtype=wp.float32)
        
        # Motion parameters
        self.target_depth = 0.05  # Move down to 5cm above ground
        self.move_speed = 0.01    # 10mm/s (slow)
        
        # Rendering
        self.viewer.set_model(self.model)
        
        # Camera positioned to see the sphere clearly
        # Look from the side, slightly above
        self.viewer.set_camera(
            pos=wp.vec3(0.2, -0.1, 0.08),  # 20cm in front, 10cm to side, 8cm high
            pitch=-10.0,  # Look slightly down
            yaw=160.0     # Look toward origin
        )
        
        # Print initial state
        print(f"Initial sphere position: {self.state_0.body_q.numpy()[body_indenter, 0:3]}")
        print(f"Joint q_start index: {self.joint_q_start_idx}")
        print(f"Initial joint_q: {self.state_0.joint_q.numpy()}")
        print()
        
        # Capture
        self.capture()
        
        self.frame_counter = 0
        
        print(f"\n{'='*70}")
        print(f"Simple Sphere Movement Test")
        print(f"{'='*70}")
        print(f"Sphere:")
        print(f"  Diameter: {indenter_diameter*1000:.1f}mm")
        print(f"  Mass: {indenter_mass*1000:.1f}g")
        print(f"  Start height: {start_height*100:.1f}cm")
        print(f"Motion:")
        print(f"  Target: {self.target_depth*100:.1f}cm above ground")
        print(f"  Speed: {self.move_speed*1000:.1f}mm/s")
        print(f"{'='*70}\n")

    def capture(self):
        if wp.get_device().is_cuda:
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
        """Move sphere down slowly"""
        # Get current position
        body_q = self.state_0.body_q.numpy()
        current_z = body_q[self.indenter_body_id, 2]
        
        # Calculate target
        if current_z > self.target_depth:
            target_z = current_z - self.move_speed * self.frame_dt
            target_z = max(target_z, self.target_depth)  # Don't overshoot
        else:
            target_z = self.target_depth  # Hold at target
        
        # Convert to joint coordinate
        joint_target = float(target_z - self.start_height)
        
        # Update control
        target_array = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        target_array[self.joint_q_start_idx] = joint_target
        target_wp = wp.array(target_array, dtype=wp.float32)
        wp.copy(self.control.joint_target_pos, target_wp)
        
        # Debug: print first time
        if self.frame_counter == 0:
            print(f"First control update:")
            print(f"  current_z: {current_z*100:.2f}cm")
            print(f"  target_z: {target_z*100:.2f}cm")
            print(f"  joint_target: {joint_target*100:.2f}cm")
            print()

    def step(self):
        # Update control
        self.update_control()
        
        # Simulate
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
        self.frame_counter += 1
        
        # Print every 30 frames (0.5s)
        if self.frame_counter % 30 == 0:
            self.print_status()
    
    def print_status(self):
        """Print sphere status"""
        body_q = self.state_0.body_q.numpy()
        sphere_z = body_q[self.indenter_body_id, 2]
        
        body_qd = self.state_0.body_qd.numpy()
        sphere_vz = body_qd[self.indenter_body_id, 2]
        
        joint_pos = self.state_0.joint_q.numpy()[self.joint_q_start_idx]
        joint_target = self.control.joint_target_pos.numpy()[self.joint_q_start_idx]
        
        print(f"[t={self.sim_time:.2f}s] Sphere z={sphere_z*100:.2f}cm, vz={sphere_vz*1000:.1f}mm/s | "
              f"Joint: pos={joint_pos*100:.2f}cm, target={joint_target*100:.2f}cm")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true")
    
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, verbose=args.verbose)
    
    newton.examples.run(example, args)

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Franka Robot Arm Indenting Soft Cube
#
# Based on cloth_franka example pattern. The Franka arm's end-effector
# approaches and indents a soft cube, demonstrating:
# - Featherstone solver for robot articulation
# - VBD solver for soft cube deformation
# - Two-phase simulation (robot → soft body)
# - Jacobian-based end-effector control
#
# Command: python franka_indent_cube.py
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD
from newton.utils import transform_twist


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    # compute pose difference between end effector and target
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 15
        self.iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        
        self.viewer = viewer
        
        # Contact parameters
        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3
        self.soft_contact_mu = 0.5
        
        # Particle parameters
        self.particle_radius = 0.005
        self.particle_self_contact_radius = 0.002
        self.particle_self_contact_margin = 0.003
        
        print(f"Franka Arm Indenting Soft Cube")
        print(f"{'='*70}")
        print(f"  Frame rate: {self.fps} Hz")
        print(f"  Substeps: {self.sim_substeps}")
        print(f"  VBD iterations: {self.iterations}")
        print(f"  dt: {self.sim_dt:.6f} s")
        print()
        
        # === Build Scene ===
        scene = newton.ModelBuilder()
        
        # Add ground plane
        scene.add_ground_plane()
        
        # === Add Soft Cube ===
        cell_size = 0.05  # 5cm cells
        cell_dim = 3      # 3x3x3 grid
        total_mass = 1.0
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
       
        
        # === Add Franka Robot ===
        franka = newton.ModelBuilder()
        self.create_articulation(franka)
        scene.add_world(franka)
        
        self.bodies_per_world = franka.body_count
        self.dof_q_per_world = franka.joint_coord_count
        self.dof_qd_per_world = franka.joint_dof_count
        


        # Material properties
        young_mod = 1.5e4  # 15 kPa
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        k_damp = 100.0
        
        self.cube_height = cell_dim * cell_size  # 0.15m
        
        # Grid Dimensions (soft cube)
        dim_x = 6
        dim_y = 6
        dim_z = 3
        cell_size = 0.1 

        # Create 4 grid with different damping values
        # Unit of damping value
        k_damp= 1e-1
        
        scene.add_soft_grid(
                pos=wp.vec3(0.0, -0.5, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_y,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
                particle_radius=0.008
        )
        
        # Color mesh (required for VBD)
        scene.color()
        
        # === Finalize Model ===
        self.model = scene.finalize(requires_grad=False)
        
        # Set contact parameters
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.soft_contact_mu
        
        # Update shape materials (cloth_franka pattern)
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        
        shape_ke[...] = self.soft_contact_ke
        shape_kd[...] = self.soft_contact_kd
        shape_mu[...] = self.soft_contact_mu
        
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype,
            device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype,
            device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype,
            device=self.model.shape_material_mu.device
        )
        
        # === Initialize States ===
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()
        
        # === Create Collision Pipeline ===
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            soft_contact_margin=0.01
        )
        self.contacts = self.collision_pipeline.contacts()
        
        # === Initialize Solvers ===
        # Featherstone for robot
        self.robot_solver = SolverFeatherstone(
            self.model,
            update_mass_matrix_interval=self.sim_substeps
        )
        
        # Setup robot control
        self.set_up_control()
        
        # VBD for soft cube
        self.model.edge_rest_angle.zero_()  # VBD workaround
        self.soft_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,  # Key flag!
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.soft_contact_ke,
        )
        
        # === Gravity Arrays ===
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)
        
        # === Set Viewer ===
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 0.5), -42.0, -58.0)
        
        print(f"Soft Cube:")
        print(f"  Size: {self.cube_height*100:.0f}cm cube")
        print(f"  Particles: {num_particles}")
        print(f"  Young's modulus: {young_mod/1000:.0f} kPa")
        print(f"\nFranka Robot:")
        print(f"  End-effector control via Jacobian")
        print(f"  Motion: Approach → Indent → Hold → Retract")
        print(f"\nSolvers:")
        print(f"  Robot: Featherstone")
        print(f"  Soft cube: VBD ({self.iterations} iterations)")
        print(f"{'='*70}\n")
        
        # Initialize FK
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        
        # CUDA graph capture
        self.capture()
    
    def create_articulation(self, builder):
        """Load Franka robot from URDF"""
        asset_path = newton.utils.download_asset("franka_emika_panda")
        
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-0.3, -0.3, 0.0),  # Position robot to side
                wp.quat_identity(),
            ),
            floating=False,
            scale=1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        
        # Initial joint configuration (arm up)
        builder.joint_q[:6] = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0]
        
        # Define indentation keyframes
        # Format: [duration, x, y, z, qw, qx, qy, qz, gripper]
        self.robot_key_poses = np.array([
            # Approach cube top (above center)
            [3.0, 0.0, 0.0, 0.25, 1, 0, 0, 0, 0.04],
            
            # Indent (press down 3cm)
            [3.0, 0.0, 0.0, 0.12, 1, 0, 0, 0, 0.04],
            
            # Hold (maintain indent)
            [2.0, 0.0, 0.0, 0.12, 1, 0, 0, 0, 0.04],
            
            # Retract slowly
            [3.0, 0.0, 0.0, 0.20, 1, 0, 0, 0, 0.04],
            
            # Return to start
            [3.0, 0.0, 0.0, 0.25, 1, 0, 0, 0, 0.04],
        ], dtype=np.float32)
        
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform(
            [0.0, 0.0, 0.22],
            wp.quat_identity(),
        )
    
    def set_up_control(self):
        """Setup Jacobian-based end-effector control"""
        out_dim = 6
        in_dim = self.model.joint_dof_count
        
        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x
        
        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]
        
        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]
        
        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()
    
    def compute_body_jacobian(self, model: Model, joint_q: wp.array, joint_qd: wp.array, include_rotation: bool = False):
        """Compute end-effector Jacobian"""
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        
        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3
        
        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1,
                inputs=[self.temp_state_for_jacobian.body_qd],
                outputs=[self.body_out]
            )
        
        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()
    
    def generate_control_joint_qd(self, state_in: State):
        """Generate joint velocities to reach target end-effector pose"""
        t_mod = (
            self.sim_time
            if self.sim_time < self.robot_key_poses_time[-1]
            else self.sim_time % self.robot_key_poses_time[-1]
        )
        include_rotation = True
        current_interval = np.searchsorted(self.robot_key_poses_time, t_mod)
        self.target = self.targets[current_interval]
        
        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )
        
        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)
        
        # Null-space control
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J
        
        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        
        delta_q = J_inv @ delta_target + N @ delta_q_null
        
        # Gripper control
        delta_q[-2] = self.target[-1] - q[-2]
        delta_q[-1] = self.target[-1] - q[-1]
        
        self.target_joint_qd.assign(delta_q)
    
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
    
    def simulate(self):
        """Two-phase simulation (cloth_franka pattern)"""
        self.soft_solver.rebuild_bvh(self.state_0)
        
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            
            self.viewer.apply_forces(self.state_0)
            
            # === PHASE 1: Robot Solver ===
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            
            self.model.shape_contact_pair_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            
            self.state_0.particle_f.zero_()
            
            # === PHASE 2: Soft Body Solver ===
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)
            
            self.collision_pipeline.collide(self.state_0, self.contacts)
            
            self.soft_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            
            self.state_0, self.state_1 = self.state_1, self.state_0
            
            self.sim_time += self.sim_dt
    
    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        
        self.sim_time += self.frame_dt
    
    def render(self):
        if self.viewer is None:
            return
        
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=840)  # 14 seconds (full cycle)
    viewer, args = newton.examples.init(parser)
    
    example = Example(viewer, args)
    
    newton.examples.run(example, args)

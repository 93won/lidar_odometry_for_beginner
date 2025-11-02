#!/usr/bin/env python3
"""
@file      gauss_newton_solver.py
@brief     Gauss-Newton Solver for non-linear optimization
@author    Seungwon Choi
@date      2025-09-29
@copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.

@par License
This project is released under the MIT License.

Chapter 2: Gauss-Newton Solver
Simplified version for ICP pose optimization using Gauss-Newton method.
Includes robust loss functions and SE3 pose optimization for point-to-plane
correspondences in LiDAR odometry.
"""

import numpy as np
from se3_utils import SE3, skew_symmetric
from typing import List, Callable, Tuple


class GaussNewtonSolver:
    """Gauss-Newton solver for SE3 pose optimization"""
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6):
        """
        Initialize Gauss-Newton solver
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def solve_point_to_plane_icp(self, correspondences: List[dict], 
                                initial_pose: SE3) -> Tuple[SE3, dict]:
        """
        Solve point-to-plane ICP using Gauss-Newton
        
        Args:
            correspondences: List of point correspondences
            initial_pose: Initial SE3 pose estimate
            
        Returns:
            Tuple of (optimized_pose, statistics)
        """
        current_pose = initial_pose
        
        stats = {
            'iterations': 0,
            'initial_cost': 0.0,
            'final_cost': 0.0,
            'converged': False
        }
        
        # Calculate initial cost
        initial_cost = self._calculate_cost(correspondences, current_pose)
        stats['initial_cost'] = initial_cost
        
        prev_cost = initial_cost
        
        for iteration in range(self.max_iterations):
            # Build linear system: J^T * J * delta = -J^T * r
            try:
                H, b = self._build_linear_system(correspondences, current_pose)
                
                # Solve for update: H * delta = -b
                if np.linalg.det(H) < 1e-12:
                    print(f"Singular Hessian at iteration {iteration}")
                    break
                    
                delta = np.linalg.solve(H, -b)
                
                # Check for convergence
                if np.linalg.norm(delta) < self.tolerance:
                    stats['converged'] = True
                    break
                
                # Apply update using SE3 exponential map
                # Right multiplication: T_new = T_current * exp(delta)
                delta_se3 = SE3.exp(delta)
                current_pose = current_pose * delta_se3
                
                # Calculate new cost
                current_cost = self._calculate_cost(correspondences, current_pose)
                
                # Check cost convergence
                cost_change = abs(prev_cost - current_cost)
                if cost_change < 1e-6:
                    stats['converged'] = True
                    break
                
                prev_cost = current_cost
                stats['iterations'] = iteration + 1
                
                print(f"GN Iteration {iteration}: cost={current_cost:.6f}, delta_norm={np.linalg.norm(delta):.6f}")
                
            except np.linalg.LinAlgError as e:
                print(f"Linear algebra error at iteration {iteration}: {e}")
                break
        
        stats['final_cost'] = prev_cost
        
        return current_pose, stats
    
    def _build_linear_system(self, correspondences: List[dict], 
                           pose: SE3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the linear system J^T * J * delta = -J^T * r
        
        Args:
            correspondences: List of correspondences
            pose: Current pose estimate
            
        Returns:
            Tuple of (H=J^T*J, b=J^T*r)
        """
        H = np.zeros((6, 6))  # Hessian approximation
        b = np.zeros(6)       # Gradient
        
        for corr in correspondences:
            if not corr['valid']:
                continue
                
            source_point = corr['source_point']
            target_point = corr['target_point']
            plane_normal = corr['plane_normal']
            weight = corr['weight']
            
            # Transform source point with current pose
            transformed_point = pose * source_point
            
            # Calculate residual: r = n^T * (R*p + t - q)
            residual = np.dot(plane_normal, transformed_point - target_point)
            
            # Calculate Jacobian for right multiplication: T' = T * exp(xi)
            # Residual: r = n^T * (T * exp(xi) * p - q)
            # where n is the plane normal in target frame, p is source point
            
            # Translation part: ∂r/∂t = n^T * R
            # (rotation is applied to the translation direction)
            jac_translation = plane_normal.T @ pose.rotation
            
            # Rotation part: ∂r/∂ω = -n^T * R * [p]_×
            # (negative sign for right perturbation with skew of original point)
            p_skew = skew_symmetric(source_point)
            jac_rotation = -plane_normal.T @ pose.rotation @ p_skew
            
            # Combine jacobians: J = [∂r/∂t, ∂r/∂ω]
            jacobian = np.concatenate([jac_translation, jac_rotation])
            
            # Apply weight
            weighted_jacobian = weight * jacobian
            weighted_residual = weight * residual
            
            # Accumulate to linear system
            H += np.outer(weighted_jacobian, weighted_jacobian)
            b += weighted_jacobian * weighted_residual
        
        return H, b
    
    def _calculate_cost(self, correspondences: List[dict], pose: SE3) -> float:
        """
        Calculate total cost for given pose
        
        Args:
            correspondences: List of correspondences
            pose: SE3 pose
            
        Returns:
            Total weighted cost
        """
        total_cost = 0.0
        valid_count = 0
        
        for corr in correspondences:
            if not corr['valid']:
                continue
                
            source_point = corr['source_point']
            target_point = corr['target_point']
            plane_normal = corr['plane_normal']
            weight = corr['weight']
            
            # Transform source point
            transformed_point = pose * source_point
            
            # Calculate point-to-plane residual
            residual = np.dot(plane_normal, transformed_point - target_point)
            
            # Add to cost
            total_cost += weight * residual * residual
            valid_count += 1
        
        return total_cost / max(valid_count, 1)


class HuberLoss:
    """Huber loss function for robust optimization"""
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss
        
        Args:
            delta: Huber threshold
        """
        self.delta = delta
    
    def compute_loss_and_weight(self, residual: float) -> Tuple[float, float]:
        """
        Compute Huber loss and reweighting factor
        
        Args:
            residual: Input residual
            
        Returns:
            Tuple of (loss_value, weight)
        """
        abs_residual = abs(residual)
        
        if abs_residual <= self.delta:
            # Quadratic region
            loss = 0.5 * residual * residual
            weight = 1.0
        else:
            # Linear region
            loss = self.delta * abs_residual - 0.5 * self.delta * self.delta
            weight = self.delta / abs_residual
            
        return loss, weight


class RobustGaussNewtonSolver(GaussNewtonSolver):
    """Robust Gauss-Newton solver with M-estimator"""
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6, 
                 loss_function: HuberLoss = None):
        """
        Initialize robust solver
        
        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            loss_function: Robust loss function
        """
        super().__init__(max_iterations, tolerance)
        self.loss_function = loss_function or HuberLoss(delta=1.0)
    
    def solve_point_to_plane_icp(self, correspondences: List[dict], 
                                initial_pose: SE3) -> Tuple[SE3, dict]:
        """
        Solve robust point-to-plane ICP
        
        Args:
            correspondences: List of correspondences
            initial_pose: Initial pose
            
        Returns:
            Tuple of (optimized_pose, statistics)
        """
        current_pose = initial_pose
        
        stats = {
            'iterations': 0,
            'initial_cost': 0.0,
            'final_cost': 0.0,
            'converged': False
        }
        
        # Calculate initial cost
        initial_cost = self._calculate_robust_cost(correspondences, current_pose)
        stats['initial_cost'] = initial_cost
        
        prev_cost = initial_cost
        
        for iteration in range(self.max_iterations):
            # Update weights based on current residuals
            self._update_robust_weights(correspondences, current_pose)
            
            # Build linear system with robust weights
            try:
                H, b = self._build_linear_system(correspondences, current_pose)
                
                if np.linalg.det(H) < 1e-12:
                    print(f"Singular Hessian at iteration {iteration}")
                    break
                    
                delta = np.linalg.solve(H, -b)
                
                if np.linalg.norm(delta) < self.tolerance:
                    stats['converged'] = True
                    break
                
                # Apply update
                delta_se3 = SE3.exp(delta)
                current_pose = current_pose * delta_se3
                
                # Calculate new cost
                current_cost = self._calculate_robust_cost(correspondences, current_pose)
                
                cost_change = abs(prev_cost - current_cost)
                if cost_change < 1e-6:
                    stats['converged'] = True
                    break
                
                prev_cost = current_cost
                stats['iterations'] = iteration + 1
                
                print(f"Robust GN Iteration {iteration}: cost={current_cost:.6f}, delta_norm={np.linalg.norm(delta):.6f}")
                
            except np.linalg.LinAlgError as e:
                print(f"Linear algebra error at iteration {iteration}: {e}")
                break
        
        stats['final_cost'] = prev_cost
        return current_pose, stats
    
    def _update_robust_weights(self, correspondences: List[dict], pose: SE3):
        """Update robust weights based on current residuals"""
        for corr in correspondences:
            if not corr['valid']:
                continue
                
            source_point = corr['source_point']
            target_point = corr['target_point']
            plane_normal = corr['plane_normal']
            
            # Transform and calculate residual
            transformed_point = pose * source_point
            residual = np.dot(plane_normal, transformed_point - target_point)
            
            # Compute robust weight
            _, weight = self.loss_function.compute_loss_and_weight(residual)
            corr['weight'] = weight
    
    def _calculate_robust_cost(self, correspondences: List[dict], pose: SE3) -> float:
        """Calculate total robust cost"""
        total_cost = 0.0
        valid_count = 0
        
        for corr in correspondences:
            if not corr['valid']:
                continue
                
            source_point = corr['source_point']
            target_point = corr['target_point']
            plane_normal = corr['plane_normal']
            
            transformed_point = pose * source_point
            residual = np.dot(plane_normal, transformed_point - target_point)
            
            # Use robust loss
            loss_value, _ = self.loss_function.compute_loss_and_weight(residual)
            total_cost += loss_value
            valid_count += 1
        
        return total_cost / max(valid_count, 1)

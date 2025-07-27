import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import threading
import time
from collections import defaultdict

# Import your modules
from assets.FrontEnd_Module import Pipeline, KITTIDataset
from assets.LoopDetection import BOVW
from assets.Mapping import LocalMapper, MapPoint, KeyFrame
from assets.Optimizer import ScipyOptimizer

class VSLAMPipeline:
    """
    Complete Visual SLAM Pipeline integrating all components:
    - Frontend: Feature detection, matching, pose estimation
    - Loop Detection: Bag of Visual Words
    - Mapping: Local mapping and triangulation
    - Optimization: Bundle adjustment and pose graph optimization
    """
    
    def __init__(self, 
                 feature_detector='FD_SIFT',
                 feature_matcher='FM_BF_NORM_L2',
                 enable_loop_detection=True,
                 enable_optimization=True,
                 optimization_interval=5):
        """
        Initialize VSLAM Pipeline
        
        Args:
            feature_detector: Feature detection method ('FD_SIFT', 'FD_ORB', 'FD_BRISK')
            feature_matcher: Feature matching method ('FM_BF_NORM_L2', 'FM_BF_NORM_Hamming')
            enable_loop_detection: Enable loop closure detection
            enable_optimization: Enable bundle adjustment optimization
            optimization_interval: Interval for optimization (in keyframes)
        """
        # Core components
        self.frontend = Pipeline()
        self.loop_detector = BOVW() if enable_loop_detection else None
        self.optimizer = None  # Will be initialized when camera matrix is set
        
        # Configuration
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.enable_loop_detection = enable_loop_detection
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        
        # State variables
        self.camera_matrix = None
        self.ground_truth_poses = None
        self.current_frame_id = 0
        self.keyframe_counter = 0
        self.is_initialized = False
        
        # Results storage
        self.estimated_trajectory = []
        self.ground_truth_trajectory = []
        self.loop_closures = []
        self.optimization_times = []
        self.processing_times = []
        
        # Visualization data
        self.trajectory_plot_data = {
            'estimated_x': [],
            'estimated_y': [],
            'estimated_z': [],
            'gt_x': [],
            'gt_y': [],
            'gt_z': []
        }
        
        print("✓ VSLAM Pipeline initialized")
        print(f"  Feature Detector: {feature_detector}")
        print(f"  Feature Matcher: {feature_matcher}")
        print(f"  Loop Detection: {'Enabled' if enable_loop_detection else 'Disabled'}")
        print(f"  Optimization: {'Enabled' if enable_optimization else 'Disabled'}")
    
    def set_camera_calibration(self, camera_matrix: np.ndarray):
        """Set camera intrinsic parameters"""
        self.camera_matrix = camera_matrix
        self.frontend.set_k_intrinsic(camera_matrix)
        
        if self.enable_optimization:
            self.optimizer = ScipyOptimizer(
                camera_matrix=camera_matrix,
                local_window_size=5,
                global_optimization_interval=self.optimization_interval
            )
        
        print(f"✓ Camera calibration set:")
        print(f"  fx: {camera_matrix[0,0]:.2f}, fy: {camera_matrix[1,1]:.2f}")
        print(f"  cx: {camera_matrix[0,2]:.2f}, cy: {camera_matrix[1,2]:.2f}")
    
    def set_ground_truth_poses(self, poses: np.ndarray):
        """Set ground truth poses for evaluation"""
        self.ground_truth_poses = poses
        print(f"✓ Ground truth poses loaded: {len(poses)} poses")
    
    def load_kitti_dataset(self, 
                          images_path: str, 
                          calib_file: str, 
                          poses_file: str = None,
                          sequence_length: int = None) -> List[str]:
        """
        Load KITTI dataset
        
        Args:
            images_path: Path to image directory
            calib_file: Path to calibration file
            poses_file: Path to ground truth poses file (optional)
            sequence_length: Limit number of images to process
            
        Returns:
            List of image file paths
        """
        # Load calibration
        try:
            with open(calib_file, 'r') as f:
                K, P = KITTIDataset.load_calib(f)
            self.set_camera_calibration(K)
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return []
        
        # Load ground truth poses if available
        if poses_file and os.path.exists(poses_file):
            try:
                with open(poses_file, 'r') as f:
                    poses = KITTIDataset.load_poses(f, sequence_length or 1000)
                self.set_ground_truth_poses(poses)
            except Exception as e:
                print(f"⚠️ Failed to load ground truth poses: {e}")
        
        # Load image paths
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_path, ext)))
        
        image_files.sort()
        
        if sequence_length:
            image_files = image_files[:sequence_length]
        
        print(f"✓ KITTI dataset loaded:")
        print(f"  Images: {len(image_files)}")
        print(f"  Calibration: {calib_file}")
        print(f"  Ground truth: {'Available' if poses_file else 'Not provided'}")
        
        return image_files
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the VSLAM pipeline
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        # Initialize result dictionary
        result = {
            'frame_id': self.current_frame_id,
            'processed_frame': frame.copy(),
            'pose': None,
            'trajectory_point': None,
            'loop_closure': None,
            'num_features': 0,
            'num_matches': 0,
            'processing_time': 0
        }
        
        try:
            # Frontend processing: feature detection, matching, pose estimation
            frontend_result = self.frontend.VisualOdometry(
                frame, 
                FeatureDetector=self.feature_detector,
                FeatureMatcher=self.feature_matcher
            )
            
            processed_frame, essential_matrix, fundamental_matrix, trajectory_path, detector_used, map_points, descriptors, keypoints = frontend_result
            
            if processed_frame is not None:
                result['processed_frame'] = processed_frame
            
            # Update trajectory
            if trajectory_path and len(trajectory_path) > 0:
                latest_point = trajectory_path[-1]
                result['trajectory_point'] = latest_point
                self.trajectory_plot_data['estimated_x'].append(latest_point[0])
                self.trajectory_plot_data['estimated_y'].append(latest_point[1])
                self.trajectory_plot_data['estimated_z'].append(latest_point[2])
                
                # Store current pose
                result['pose'] = self.frontend.current_pose.copy()
                self.estimated_trajectory.append(result['pose'])
            
            # Add ground truth trajectory point if available
            if (self.ground_truth_poses is not None and 
                self.current_frame_id < len(self.ground_truth_poses)):
                
                gt_pose = self.ground_truth_poses[self.current_frame_id]
                gt_point = (-gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3])
                self.trajectory_plot_data['gt_x'].append(gt_point[0])
                self.trajectory_plot_data['gt_y'].append(gt_point[1])
                self.trajectory_plot_data['gt_z'].append(gt_point[2])
                self.ground_truth_trajectory.append(gt_pose)
            
            # Count features and matches
            if keypoints:
                result['num_features'] = len(keypoints)
            
            # Loop closure detection
            if (self.enable_loop_detection and 
                self.loop_detector is not None and 
                descriptors is not None and 
                keypoints is not None):
                
                try:
                    # Compute visual words histogram
                    visual_words = self.loop_detector.Histogram(descriptors)
                    
                    # Add to history
                    self.loop_detector.historyOfBOVW(visual_words, descriptors, keypoints)
                    
                    # Check for loop closures
                    loop_closures = self.loop_detector.LoopChecks()
                    
                    if loop_closures:
                        result['loop_closure'] = loop_closures[-1]  # Latest loop closure
                        self.loop_closures.extend(loop_closures)
                        print(f"🔄 Loop closure detected at frame {self.current_frame_id}")
                        
                except Exception as e:
                    print(f"⚠️ Loop detection failed: {e}")
            
            # Optimization (Bundle Adjustment)
            if (self.enable_optimization and 
                self.optimizer is not None and 
                self.current_frame_id > 0 and
                keypoints is not None and
                descriptors is not None):
                
                try:
                    # Add keyframe to optimizer
                    if result['pose'] is not None:
                        kf_id = self.optimizer.add_keyframe(
                            result['pose'], 
                            keypoints, 
                            descriptors
                        )
                        
                        # Add map points if available
                        if map_points is not None and len(map_points) > 0:
                            for i, point_3d in enumerate(map_points):
                                if point_3d is not None and len(point_3d) >= 3:
                                    # Create synthetic observations for the point
                                    observations = [(kf_id, keypoints[min(i, len(keypoints)-1)].pt)]
                                    self.optimizer.add_map_point(point_3d[:3], observations)
                        
                        # Perform optimization at regular intervals
                        if self.current_frame_id % self.optimization_interval == 0:
                            opt_start = time.time()
                            self.optimizer.optimize_step(kf_id, result.get('loop_closure'))
                            opt_time = time.time() - opt_start
                            self.optimization_times.append(opt_time)
                            print(f"🔧 Optimization completed in {opt_time:.3f}s")
                            
                except Exception as e:
                    print(f"⚠️ Optimization failed: {e}")
            
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            result['error'] = str(e)
        
        # Record processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        self.processing_times.append(processing_time)
        
        self.current_frame_id += 1
        return result
    
    def process_sequence(self, image_files: List[str], 
                        progress_callback=None) -> List[Dict]:
        """
        Process a complete image sequence
        
        Args:
            image_files: List of image file paths
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processing results for each frame
        """
        results = []
        total_frames = len(image_files)
        
        print(f"🚀 Starting VSLAM processing on {total_frames} frames...")
        
        for i, image_path in enumerate(image_files):
            try:
                # Load image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"⚠️ Failed to load image: {image_path}")
                    continue
                
                # Process frame
                result = self.process_frame(frame)
                results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / total_frames
                    progress_callback(progress, i + 1, total_frames, result)
                
                # Print progress
                if (i + 1) % 10 == 0 or i == 0:
                    avg_time = np.mean(self.processing_times[-10:]) if self.processing_times else 0
                    print(f"📊 Processed {i+1}/{total_frames} frames "
                          f"(avg: {avg_time:.3f}s/frame)")
                
            except Exception as e:
                print(f"❌ Error processing frame {i}: {e}")
                continue
        
        print(f"✅ VSLAM processing completed!")
        print(f"   Total frames processed: {len(results)}")
        print(f"   Average processing time: {np.mean(self.processing_times):.3f}s")
        print(f"   Loop closures detected: {len(self.loop_closures)}")
        
        return results
    
    def compute_trajectory_error(self) -> Dict:
        """
        Compute trajectory error metrics (ATE, RPE) if ground truth is available
        
        Returns:
            Dictionary containing error metrics
        """
        if (not self.ground_truth_trajectory or 
            not self.estimated_trajectory or 
            len(self.ground_truth_trajectory) != len(self.estimated_trajectory)):
            return {'error': 'Ground truth or estimated trajectory not available or length mismatch'}
        
        # Absolute Trajectory Error (ATE)
        ate_errors = []
        for gt_pose, est_pose in zip(self.ground_truth_trajectory, self.estimated_trajectory):
            # Extract positions
            gt_pos = gt_pose[:3, 3]
            est_pos = est_pose[:3, 3]
            
            # Compute Euclidean distance
            error = np.linalg.norm(gt_pos - est_pos)
            ate_errors.append(error)
        
        ate_rmse = np.sqrt(np.mean(np.array(ate_errors) ** 2))
        ate_mean = np.mean(ate_errors)
        ate_std = np.std(ate_errors)
        ate_max = np.max(ate_errors)
        
        # Relative Pose Error (RPE) - simplified version
        rpe_errors = []
        for i in range(1, len(self.ground_truth_trajectory)):
            # Ground truth relative pose
            gt_rel = np.linalg.inv(self.ground_truth_trajectory[i-1]) @ self.ground_truth_trajectory[i]
            
            # Estimated relative pose
            est_rel = np.linalg.inv(self.estimated_trajectory[i-1]) @ self.estimated_trajectory[i]
            
            # Error in relative pose
            rel_error = np.linalg.inv(gt_rel) @ est_rel
            
            # Translation error
            trans_error = np.linalg.norm(rel_error[:3, 3])
            rpe_errors.append(trans_error)
        
        rpe_rmse = np.sqrt(np.mean(np.array(rpe_errors) ** 2)) if rpe_errors else 0
        rpe_mean = np.mean(rpe_errors) if rpe_errors else 0
        
        return {
            'ate_rmse': ate_rmse,
            'ate_mean': ate_mean,
            'ate_std': ate_std,
            'ate_max': ate_max,
            'rpe_rmse': rpe_rmse,
            'rpe_mean': rpe_mean,
            'num_poses': len(self.estimated_trajectory)
        }
    
    def save_trajectory(self, filename: str):
        """Save estimated trajectory to file"""
        if not self.estimated_trajectory:
            print("⚠️ No trajectory to save")
            return
        
        try:
            with open(filename, 'w') as f:
                for pose in self.estimated_trajectory:
                    # Save in KITTI format (3x4 transformation matrix)
                    pose_3x4 = pose[:3, :].flatten()
                    f.write(' '.join(map(str, pose_3x4)) + '\n')
            
            print(f"✅ Trajectory saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save trajectory: {e}")
    
    def create_trajectory_plot(self, save_path: str = None) -> plt.Figure:
        """
        Create trajectory visualization plot
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(12, 8))
        
        # 2D trajectory plot (X-Z plane)
        ax1 = plt.subplot(2, 2, 1)
        if self.trajectory_plot_data['estimated_x']:
            ax1.plot(self.trajectory_plot_data['estimated_z'], 
                    self.trajectory_plot_data['estimated_x'], 
                    'b-', label='Estimated', linewidth=2)
        
        if self.trajectory_plot_data['gt_x']:
            ax1.plot(self.trajectory_plot_data['gt_z'], 
                    self.trajectory_plot_data['gt_x'], 
                    'r--', label='Ground Truth', linewidth=2)
        
        ax1.set_xlabel('Z (forward)')
        ax1.set_ylabel('X (right)')
        ax1.set_title('Trajectory (Top View)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # 3D trajectory plot
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        if self.trajectory_plot_data['estimated_x']:
            ax2.plot(self.trajectory_plot_data['estimated_x'],
                    self.trajectory_plot_data['estimated_y'],
                    self.trajectory_plot_data['estimated_z'],
                    'b-', label='Estimated', linewidth=2)
        
        if self.trajectory_plot_data['gt_x']:
            ax2.plot(self.trajectory_plot_data['gt_x'],
                    self.trajectory_plot_data['gt_y'],
                    self.trajectory_plot_data['gt_z'],
                    'r--', label='Ground Truth', linewidth=2)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('3D Trajectory')
        ax2.legend()
        
        # Error plot (if ground truth available)
        if (self.trajectory_plot_data['estimated_x'] and 
            self.trajectory_plot_data['gt_x'] and
            len(self.trajectory_plot_data['estimated_x']) == len(self.trajectory_plot_data['gt_x'])):
            
            ax3 = plt.subplot(2, 2, 3)
            errors = []
            for i in range(len(self.trajectory_plot_data['estimated_x'])):
                est_pos = np.array([self.trajectory_plot_data['estimated_x'][i],
                                  self.trajectory_plot_data['estimated_y'][i],
                                  self.trajectory_plot_data['estimated_z'][i]])
                gt_pos = np.array([self.trajectory_plot_data['gt_x'][i],
                                 self.trajectory_plot_data['gt_y'][i],
                                 self.trajectory_plot_data['gt_z'][i]])
                error = np.linalg.norm(est_pos - gt_pos)
                errors.append(error)
            
            ax3.plot(errors, 'g-', linewidth=2)
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Position Error (m)')
            ax3.set_title('Trajectory Error Over Time')
            ax3.grid(True)
        
        # Processing time plot
        ax4 = plt.subplot(2, 2, 4)
        if self.processing_times:
            ax4.plot(self.processing_times, 'orange', linewidth=2)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Processing Time (s)')
            ax4.set_title('Processing Time Per Frame')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Trajectory plot saved to {save_path}")
        
        return fig
    
    def reset(self):
        """Reset the pipeline state"""
        self.frontend.reset()
        if self.loop_detector:
            self.loop_detector.reset()
        
        self.current_frame_id = 0
        self.keyframe_counter = 0
        self.is_initialized = False
        
        # Clear results
        self.estimated_trajectory = []
        self.ground_truth_trajectory = []
        self.loop_closures = []
        self.optimization_times = []
        self.processing_times = []
        
        # Clear plot data
        for key in self.trajectory_plot_data:
            self.trajectory_plot_data[key] = []
        
        print("✅ Pipeline reset completed")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'frames_processed': self.current_frame_id,
            'total_processing_time': sum(self.processing_times),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'min_processing_time': np.min(self.processing_times) if self.processing_times else 0,
            'loop_closures_detected': len(self.loop_closures),
            'optimization_calls': len(self.optimization_times),
            'avg_optimization_time': np.mean(self.optimization_times) if self.optimization_times else 0,
            'trajectory_length': len(self.estimated_trajectory),
            'feature_detector': self.feature_detector,
            'feature_matcher': self.feature_matcher,
            'loop_detection_enabled': self.enable_loop_detection,
            'optimization_enabled': self.enable_optimization
        }
        
        # Add trajectory error metrics if available
        error_metrics = self.compute_trajectory_error()
        if 'error' not in error_metrics:
            stats.update(error_metrics)
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing VSLAM Pipeline...")
    
    # Initialize pipeline
    pipeline = VSLAMPipeline(
        feature_detector='FD_SIFT',
        feature_matcher='FM_BF_NORM_L2',
        enable_loop_detection=True,
        enable_optimization=True
    )
    
    # Test with synthetic data
    print("\n" + "="*50)
    print("VSLAM Pipeline initialized and ready!")
    print("Use with Streamlit interface for complete functionality.")
    print("="*50)
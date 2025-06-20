class VisualSLAM:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.local_mapper = LocalMapper(camera_matrix)
        self.loop_detector = LoopClosureDetector()
        self.pose_optimizer = PoseGraphOptimizer()
        self.current_pose = np.eye(4)
        
    def process_frame(self, image):
        """Main processing pipeline"""
        # Extract features
        features, descriptors = self.extract_features(image)
        
        # Estimate pose (from your existing VO)
        new_pose = self.estimate_pose(features, descriptors)
        
        # Create keyframe if necessary
        if self.should_create_keyframe():
            kf = KeyFrame(new_pose, features, descriptors, len(self.keyframes))
            self.keyframes.append(kf)
            
            # Local mapping
            self.local_mapper.add_keyframe(kf)
            
            # Loop closure detection
            loop_id, matches = self.loop_detector.detect_loop_closure(kf)
            if loop_id is not None:
                self.handle_loop_closure(len(self.keyframes)-1, loop_id)
                
        return new_pose

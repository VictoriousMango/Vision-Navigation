import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle
from collections import defaultdict

class BagOfVisualWords:
    def __init__(self, vocab_size=500):
        """
        Initialize BoVW with specified vocabulary size
        
        Args:
            vocab_size (int): Number of visual words in vocabulary (clusters)
        """
        self.vocab_size = vocab_size
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.scaler = StandardScaler()
        self.vocabulary = None
        
    def extract_sift_features(self, image_path):
        """
        Extract SIFT keypoints and descriptors from an image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            tuple: (keypoints, descriptors) or (None, None) if no features found
        """
        # Read image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, None
        
        # Detect SIFT keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def build_vocabulary(self, image_paths):
        """
        Build visual vocabulary by clustering SIFT descriptors
        
        Args:
            image_paths (list): List of paths to training images
        """
        print("Extracting SIFT features from all images...")
        all_descriptors = []
        
        # Extract descriptors from all images
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")
                
            keypoints, descriptors = self.extract_sift_features(image_path)
            
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # Concatenate all descriptors
        if not all_descriptors:
            raise ValueError("No SIFT descriptors found in any image")
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors extracted: {len(all_descriptors)}")
        
        # Normalize descriptors
        all_descriptors = self.scaler.fit_transform(all_descriptors)
        
        # Cluster descriptors to create visual vocabulary
        print(f"Clustering descriptors into {self.vocab_size} visual words...")
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=42, n_init=10)
        self.kmeans.fit(all_descriptors)
        
        # Store vocabulary (cluster centers)
        self.vocabulary = self.kmeans.cluster_centers_
        print("Vocabulary built successfully!")
    
    def image_to_bow_histogram(self, image_path):
        """
        Convert an image to a bag-of-visual-words histogram
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            numpy.ndarray: Histogram of visual words (normalized)
        """
        if self.kmeans is None:
            raise ValueError("Vocabulary not built yet. Call build_vocabulary() first.")
        
        # Extract SIFT features
        keypoints, descriptors = self.extract_sift_features(image_path)
        
        if descriptors is None:
            # Return zero histogram if no features found
            return np.zeros(self.vocab_size)
        
        # Normalize descriptors using the same scaler
        descriptors = self.scaler.transform(descriptors)
        
        # Assign each descriptor to nearest cluster (visual word)
        visual_words = self.kmeans.predict(descriptors)
        
        # Create histogram
        histogram = np.zeros(self.vocab_size)
        for word in visual_words:
            histogram[word] += 1
        
        # Normalize histogram (L2 normalization)
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
        
        return histogram
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        model_data = {
            'vocab_size': self.vocab_size,
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'vocabulary': self.vocabulary
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.vocabulary = model_data['vocabulary']
        print(f"Model loaded from {filepath}")


# Example usage and demonstration
def demonstrate_bovw():
    """
    Demonstrate how to use the Bag of Visual Words implementation
    """
    
    # Sample image paths (replace with your actual image paths)
    # These should be organized in folders by class, e.g.:
    # dataset/
    #   ├── cats/
    #   ├── dogs/
    #   └── birds/
    
    def get_image_paths_and_labels(dataset_dir):
        """
        Get image paths and their corresponding labels from a dataset directory
        
        Args:
            dataset_dir (str): Path to dataset directory
            
        Returns:
            tuple: (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, image_name))
                        labels.append(class_name)
        
        return image_paths, labels
    
    # Initialize BoVW
    bovw = BagOfVisualWords(vocab_size=300)
    
    # Example with dummy paths (replace with actual paths)
    print("=== Bag of Visual Words Demo ===")
    print("Note: Replace image paths with your actual dataset")
    
    # Simulated training process
    print("\n1. Building vocabulary...")
    # train_images, train_labels = get_image_paths_and_labels("path/to/training/dataset")
    # bovw.build_vocabulary(train_images)
    
    print("\n2. Converting images to BoVW histograms...")
    # Example of converting single image
    # histogram = bovw.image_to_bow_histogram("path/to/image.jpg")
    # print(f"Histogram shape: {histogram.shape}")
    # print(f"Histogram sum: {np.sum(histogram)}")
    
    print("\n3. Training classifier...")
    # Convert all training images to histograms
    # train_histograms = []
    # for image_path in train_images:
    #     hist = bovw.image_to_bow_histogram(image_path)
    #     train_histograms.append(hist)
    
    # train_histograms = np.array(train_histograms)
    
    # Train SVM classifier
    # classifier = SVC(kernel='rbf', C=1.0)
    # classifier.fit(train_histograms, train_labels)
    
    print("\n4. Testing...")
    # test_images, test_labels = get_image_paths_and_labels("path/to/test/dataset")
    # test_histograms = []
    # for image_path in test_images:
    #     hist = bovw.image_to_bow_histogram(image_path)
    #     test_histograms.append(hist)
    
    # test_histograms = np.array(test_histograms)
    # predictions = classifier.predict(test_histograms)
    # accuracy = accuracy_score(test_labels, predictions)
    # print(f"Classification accuracy: {accuracy:.2f}")
    
    print("\n5. Saving/Loading model...")
    # bovw.save_model("bovw_model.pkl")
    # bovw.load_model("bovw_model.pkl")


# Utility functions for visualization and analysis
def visualize_sift_features(image_path, bovw):
    """
    Visualize SIFT keypoints on an image
    
    Args:
        image_path (str): Path to the image
        bovw (BagOfVisualWords): BoVW instance
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    keypoints, descriptors = bovw.extract_sift_features(image_path)
    
    if keypoints:
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, 
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Display or save the image
        cv2.imshow('SIFT Keypoints', img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"Found {len(keypoints)} SIFT keypoints")
        print(f"Descriptor shape: {descriptors.shape}")
    else:
        print("No SIFT keypoints found")


def analyze_vocabulary(bovw):
    """
    Analyze the learned visual vocabulary
    
    Args:
        bovw (BagOfVisualWords): Trained BoVW instance
    """
    if bovw.vocabulary is None:
        print("Vocabulary not built yet")
        return
    
    print(f"Vocabulary size: {bovw.vocab_size}")
    print(f"Descriptor dimensionality: {bovw.vocabulary.shape[1]}")
    
    # Analyze cluster centers
    distances = []
    for i in range(len(bovw.vocabulary)):
        for j in range(i+1, len(bovw.vocabulary)):
            dist = np.linalg.norm(bovw.vocabulary[i] - bovw.vocabulary[j])
            distances.append(dist)
    
    print(f"Average distance between visual words: {np.mean(distances):.4f}")
    print(f"Min distance: {np.min(distances):.4f}")
    print(f"Max distance: {np.max(distances):.4f}")


if __name__ == "__main__":
    demonstrate_bovw()
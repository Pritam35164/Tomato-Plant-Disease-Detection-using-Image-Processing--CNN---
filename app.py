import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import io

# ===============================
# IMAGE VALIDATOR CLASS
# ===============================

class ImageValidator:
    def __init__(self):
        # Enhanced color profiles with more precise ranges for tomato plants
        self.tomato_color_profiles = [
            {'r_range': (50, 150), 'g_range': (80, 180), 'b_range': (30, 100), 'weight': 1.0},  # Standard tomato green
            {'r_range': (100, 190), 'g_range': (90, 170), 'b_range': (35, 105), 'weight': 0.9},  # Slightly yellowing  
            {'r_range': (40, 130), 'g_range': (70, 160), 'b_range': (40, 110), 'weight': 0.8},  # Darker green
            {'r_range': (60, 160), 'g_range': (90, 190), 'b_range': (30, 95), 'weight': 0.85},  # Bright healthy
            {'r_range': (110, 200), 'g_range': (100, 180), 'b_range': (25, 90), 'weight': 0.75},  # Yellowish
        ]
        
        # Refined non-tomato profiles with more specific criteria
        self.non_tomato_profiles = [
            {'r_range': (20, 100), 'g_range': (150, 220), 'b_range': (20, 80), 'g_ratio_min': 0.65, 'weight': 0.95},  # Very green (non-tomato plants)
            {'r_range': (170, 255), 'g_range': (30, 150), 'b_range': (30, 150), 'r_ratio_min': 0.6, 'weight': 0.9},   # Very red objects
            {'r_range': (40, 150), 'g_range': (40, 150), 'b_range': (140, 255), 'b_ratio_min': 0.5, 'weight': 0.9},   # Blue/purple objects
            {'r_range': (180, 255), 'g_range': (180, 255), 'b_range': (180, 255), 'v_min': 200, 'weight': 0.95},      # Very bright/white objects
            {'r_range': (0, 60), 'g_range': (0, 60), 'b_range': (0, 60), 'v_max': 60, 'weight': 0.95},                # Very dark/black objects
        ]
        
    def _extract_texture_features(self, image_array):
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM (Gray Level Co-occurrence Matrix) features
        distances = [1, 3]  # Distance between pixel pairs
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles to consider
        
        # Calculate GLCM
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                          symmetric=True, normed=True, levels=256)
        
        # Calculate GLCM properties
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Additional statistical features
        std_dev = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())
        
        # Calculate edge density using Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.sum(edge_magnitude > 30) / (gray.shape[0] * gray.shape[1])
        
        # LBP for additional texture information
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                 range=(0, n_points + 2), density=True)
        
        # Return feature dictionary
        return {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy, 
            'correlation': correlation,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurt,
            'edge_density': edge_density,
            'lbp_hist': lbp_hist
        }
    
    def _color_histogram_analysis(self, image_array):
        # Extract HSV color space features for better color analysis
        hsv_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # Extract RGB features
        avg_r = np.mean(image_array[:, :, 0])
        avg_g = np.mean(image_array[:, :, 1])
        avg_b = np.mean(image_array[:, :, 2])
        
        std_r = np.std(image_array[:, :, 0])
        std_g = np.std(image_array[:, :, 1])
        std_b = np.std(image_array[:, :, 2])
        
        # HSV statistics
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)
        
        std_h = np.std(h)
        std_s = np.std(s)
        std_v = np.std(v)
        
        # Color ratios
        total = avg_r + avg_g + avg_b
        if total == 0:
            return 0.1
            
        r_ratio = avg_r / total
        g_ratio = avg_g / total
        b_ratio = avg_b / total
        
        # Extract texture features
        texture_features = self._extract_texture_features(image_array)
        
        # Check against non-tomato profiles with enhanced criteria
        for profile in self.non_tomato_profiles:
            r_match = (avg_r >= profile['r_range'][0] and avg_r <= profile['r_range'][1])
            g_match = (avg_g >= profile['g_range'][0] and avg_g <= profile['g_range'][1])
            b_match = (avg_b >= profile['b_range'][0] and avg_b <= profile['b_range'][1])
            
            # Additional ratio checks
            ratio_match = True
            if 'g_ratio_min' in profile and g_ratio < profile['g_ratio_min']:
                ratio_match = False
            if 'r_ratio_min' in profile and r_ratio < profile['r_ratio_min']:
                ratio_match = False
            if 'b_ratio_min' in profile and b_ratio < profile['b_ratio_min']:
                ratio_match = False
            
            # Brightness checks
            brightness_match = True
            if 'v_min' in profile and avg_v < profile['v_min']:
                brightness_match = False
            if 'v_max' in profile and avg_v > profile['v_max']:
                brightness_match = False
                
            # Combined texture and color check for non-tomato objects
            if (r_match and g_match and b_match) and ratio_match and brightness_match:
                # Further validate with texture
                if texture_features['edge_density'] < 0.05 or texture_features['energy'] > 0.6:
                    # Very smooth or very uniform texture often indicates non-plant
                    return 0.1  # Not a tomato plant
                
                # Check texture homogeneity - most plants have some texture variation
                if texture_features['homogeneity'] > 0.95:
                    return 0.15  # Likely not a plant texture
        
        # Check for tomato characteristics with enhanced scoring
        profile_scores = []
        for profile in self.tomato_color_profiles:
            r_match = (avg_r >= profile['r_range'][0] and avg_r <= profile['r_range'][1])
            g_match = (avg_g >= profile['g_range'][0] and avg_g <= profile['g_range'][1])
            b_match = (avg_b >= profile['b_range'][0] and avg_b <= profile['b_range'][1])
            
            # Calculate match score with more gradation
            if r_match and g_match and b_match:
                profile_scores.append(1.0 * profile['weight'])
            elif (r_match and g_match) or (g_match and b_match):
                profile_scores.append(0.75 * profile['weight'])
            elif r_match or g_match:  # At least one channel matches
                profile_scores.append(0.4 * profile['weight'])
            else:
                profile_scores.append(0.0)
        
        best_profile_score = max(profile_scores) if profile_scores else 0
        
        # Additional checks for tomato plant characteristics based on texture and color
        plant_indicators = 0.0
        
        # Check for appropriate green and red ratios for tomato plants
        if 0.32 < g_ratio < 0.55 and 0.25 < r_ratio < 0.45:
            plant_indicators += 0.5
        elif 0.25 < g_ratio < 0.6:  # Relaxed green range
            plant_indicators += 0.3
            
        # Check HSV color space for characteristic tomato plant colors
        if 35 < avg_h < 85 and 50 < avg_s < 200:  # Green-yellow HSV range
            plant_indicators += 0.2
            
        # Texture analysis for leaf characteristics
        # Moderate texture variation is characteristic of leaves
        if 0.1 < texture_features['edge_density'] < 0.4 and texture_features['homogeneity'] < 0.9:
            plant_indicators += 0.2
            
        # Leaf veins create specific contrast patterns
        if 0.3 < texture_features['contrast'] < 8.0:
            plant_indicators += 0.2
            
        # Penalize extreme homogeneity or heterogeneity
        if texture_features['homogeneity'] > 0.95 or texture_features['homogeneity'] < 0.3:
            plant_indicators -= 0.3
            
        # Combine scores with weighted approach
        combined_score = (best_profile_score * 0.6) + (plant_indicators * 0.4)
        
        # Apply penalties for unusual color distributions
        if g_ratio < 0.2 or g_ratio > 0.7:  # Very non-green
            combined_score *= 0.4
        if r_ratio > 0.6:  # Too red for most tomato leaves
            combined_score *= 0.5
        if b_ratio > 0.5:  # Too blue for tomato leaves
            combined_score *= 0.3
            
        # Check color standard deviations - leaves typically have some variation
        color_std_avg = (std_r + std_g + std_b) / 3
        if color_std_avg < 10:  # Too uniform
            combined_score *= 0.6
            
        # Normalize score to [0,1] range
        final_score = max(0.0, min(1.0, combined_score))
        
        # Scale to percentage for display
        return final_score * 100


# ===============================
# TOMATO DISEASE CLASSIFIER CLASS
# ===============================

class TomatoDiseaseClassifier:
    def __init__(self):
        self.disease_categories = [
            "Healthy",
            "Early Blight",
            "Late Blight",
            "Leaf Mold",
            "Septoria Leaf Spot",
            "Bacterial Spot",
            "Target Spot",
            "Spider Mites",
            "Yellow Leaf Curl Virus",
            "Mosaic Virus"
        ]
        
        self.disease_descriptions = {
            "Healthy": "No disease detected. The plant appears to be in good condition with normal coloration and texture.",
            "Early Blight": "Fungal disease characterized by dark spots with concentric rings that form a target-like pattern. Often starts on older, lower leaves and can cause significant defoliation if untreated.",
            "Late Blight": "Fungal disease causing dark, water-soaked spots that quickly enlarge and develop white fungal growth on leaf undersides. Can spread very rapidly in humid conditions.",
            "Leaf Mold": "Fungal disease common in high humidity with yellow spots on upper leaf surfaces and olive-green to grayish-purple mold on the undersides. Typically begins on older leaves.",
            "Septoria Leaf Spot": "Fungal disease causing small circular spots with dark borders and lighter centers. Spots often have yellow halos and can cause severe defoliation.",
            "Bacterial Spot": "Bacterial disease creating small, water-soaked spots on leaves and fruits. Spots become angular and have a greasy appearance, eventually turning dark brown with yellow halos.",
            "Target Spot": "Fungal disease characterized by circular brown spots with concentric rings and yellow halos. Often appears during periods of high humidity and leaf wetness.",
            "Spider Mites": "Tiny pests causing fine stippling or speckling on leaf surfaces. Severe infestations lead to bronzing or yellowing and fine webbing between leaves and stems.",
            "Yellow Leaf Curl Virus": "Viral disease causing severe leaf curling, yellowing, and stunting. Leaves curl upward and inward, plants become stunted, and fruit production is significantly reduced.",
            "Mosaic Virus": "Viral disease creating mottled patterns of yellow and green on leaves. Can cause leaf distortion, stunted growth, and reduced fruit production."
        }
        
    def _extract_disease_features(self, image_array):
        """Extract comprehensive features for disease classification"""
        
        # === Color Features ===
        # RGB statistics
        avg_r = np.mean(image_array[:, :, 0])
        avg_g = np.mean(image_array[:, :, 1])
        avg_b = np.mean(image_array[:, :, 2])
        
        std_r = np.std(image_array[:, :, 0])
        std_g = np.std(image_array[:, :, 1])
        std_b = np.std(image_array[:, :, 2])
        
        # HSV color space
        hsv_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_img)
        
        avg_h = np.mean(h)
        avg_s = np.mean(s)
        avg_v = np.mean(v)
        
        std_h = np.std(h)
        std_s = np.std(s)
        std_v = np.std(v)
        
        # LAB color space (better for leaf color analysis)
        lab_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_img)
        
        avg_l = np.mean(l)
        avg_a = np.mean(a)
        avg_b_lab = np.mean(b)
        
        std_l = np.std(l)
        std_a = np.std(a)
        std_b_lab = np.std(b)
        
        # === Texture Features ===
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # GLCM texture features
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True, levels=256)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Edge features
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Edge density in different thresholds for better disease pattern detection
        edge_low = np.sum(edge_magnitude > 20) / (gray.shape[0] * gray.shape[1])
        edge_med = np.sum(edge_magnitude > 50) / (gray.shape[0] * gray.shape[1])
        edge_high = np.sum(edge_magnitude > 100) / (gray.shape[0] * gray.shape[1])
        
        # Spot detection (important for many diseases)
        # Use adaptive thresholding to identify potential spots
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Count potential spots and get their stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter spots by size to avoid noise
        spot_sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        medium_spots = [s for s in spot_sizes if 10 < s < 200]
        large_spots = [s for s in spot_sizes if s >= 200]
        
        spot_density = len(medium_spots) / (gray.shape[0] * gray.shape[1])
        large_spot_density = len(large_spots) / (gray.shape[0] * gray.shape[1])
        
        # Return all features as a dictionary
        return {
            # RGB statistics
            'avg_r': avg_r, 'avg_g': avg_g, 'avg_b': avg_b,
            'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
            'r_g_ratio': avg_r / (avg_g + 1e-6),
            'g_b_ratio': avg_g / (avg_b + 1e-6),
            
            # HSV statistics
            'avg_h': avg_h, 'avg_s': avg_s, 'avg_v': avg_v,
            'std_h': std_h, 'std_s': std_s, 'std_v': std_v,
            
            # LAB statistics
            'avg_l': avg_l, 'avg_a': avg_a, 'avg_b_lab': avg_b_lab,
            'std_l': std_l, 'std_a': std_a, 'std_b_lab': std_b_lab,
            
            # GLCM texture
            'contrast': contrast, 'dissimilarity': dissimilarity,
            'homogeneity': homogeneity, 'energy': energy, 'correlation': correlation,
            
            # Edge features
            'edge_low': edge_low, 'edge_med': edge_med, 'edge_high': edge_high,
            
            # Spot features
            'spot_density': spot_density, 'large_spot_density': large_spot_density,
            'num_medium_spots': len(medium_spots), 'num_large_spots': len(large_spots)
        }
    
    def _analyze_disease_patterns(self, features):
        """Analyze extracted features to identify disease patterns"""
        
        # Initialize scores for each disease
        scores = {disease: 0.0 for disease in self.disease_categories}
        
        # Analyze for Healthy - make this more restrictive
        if (90 < features['avg_g'] < 180 and 
            features['avg_g'] > features['avg_r'] * 1.3 and  # Stronger green dominance
            features['avg_g'] > features['avg_b'] * 1.5 and
            features['std_g'] < 35 and
            features['edge_med'] < 0.15 and  # Less edge density expected in healthy leaves
            features['spot_density'] < 0.008 and  # Stricter spot threshold
            features['energy'] > 0.25):
            
            scores["Healthy"] += 40  # Reduced from 50
            
            # Additional check for very healthy leaves - make stricter
            if (features['avg_g'] > 1.7 * features['avg_r'] and 
                features['std_h'] < 18 and
                features['spot_density'] < 0.003):
                scores["Healthy"] += 25  # Reduced from 30
        
        # Early Blight - make more sensitive
        if (features['contrast'] > 2.5 and  # Reduced threshold to catch more cases
            features['edge_med'] > 0.12 and
            features['spot_density'] > 0.01 and
            features['avg_r'] > 55 and
            features['std_h'] > 12):
            
            scores["Early Blight"] += 45  # Increased from 40
            
            # Stronger indicators
            if (features['num_large_spots'] > 2 and  # Reduced from 3
                features['r_g_ratio'] > 0.65 and  # Reduced from 0.7
                features['std_v'] > 35):  # Reduced from 40
                scores["Early Blight"] += 35  # Increased from 30
        
        # Late Blight - enhance sensitivity
        if (features['avg_v'] < 130 and  # Raised threshold
            features['edge_high'] > 0.08 and  # Lowered threshold
            features['std_v'] > 30 and
            (features['avg_r'] < features['avg_g'] * 1.1) and  # Added buffer
            features['homogeneity'] < 0.45):  # Raised threshold
            
            scores["Late Blight"] += 45  # Increased from 40
            
            # Water-soaked appearance indicators
            if (features['avg_l'] < 130 and  # Raised threshold
                features['std_l'] > 35 and  # Lowered threshold
                features['large_spot_density'] > 0.015):  # Lowered threshold
                scores["Late Blight"] += 35  # Increased from 30
        
        # Leaf Mold - adjusted for better detection
        if (features['avg_h'] < 45 and  # Expanded range (was 20-40)
            features['avg_h'] > 15 and
            features['avg_s'] > 90 and  # Lowered from 100
            features['std_s'] > 35 and  # Lowered from 40
            features['homogeneity'] < 0.55):  # Raised from 0.5
            
            scores["Leaf Mold"] += 45  # Increased from 40
            
            # Yellow-green pattern indicators
            if (features['avg_b'] < 110 and  # Raised threshold
                features['spot_density'] > 0.015 and  # Lowered threshold
                features['num_medium_spots'] > 8):  # Lowered from 10
                scores["Leaf Mold"] += 35  # Increased from 30
        
        # Septoria Leaf Spot - more sensitive detection
        if (features['num_medium_spots'] > 12 and  # Lowered from 15
            features['spot_density'] > 0.025 and  # Lowered from 0.03
            features['edge_med'] > 0.18 and  # Lowered from 0.2
            features['dissimilarity'] > 1.8):  # Lowered from 2.0
            
            scores["Septoria Leaf Spot"] += 45  # Increased from 40
            
            # Distinctive spot pattern - more sensitive
            if (features['contrast'] > 3.5 and  # Lowered from 4.0
                features['std_l'] > 28 and  # Lowered from 30
                features['edge_high'] > 0.09):  # Lowered from 0.1
                scores["Septoria Leaf Spot"] += 35  # Increased from 30
        
        # Bacterial Spot - increased sensitivity
        if (features['num_medium_spots'] > 18 and  # Lowered from 20
            features['edge_med'] > 0.14 and  # Lowered from 0.15
            features['spot_density'] > 0.02 and  # Lowered from 0.025
            features['std_v'] > 22 and  # Lowered from 25
            features['homogeneity'] < 0.55):  # Raised from 0.5
            
            scores["Bacterial Spot"] += 45  # Increased from 40
            
            # Water-soaked appearance - more sensitive
            if (features['avg_v'] < 140 and  # Raised from 130
                features['avg_s'] < 130 and  # Raised from 120
                features['contrast'] > 2.7):  # Lowered from 3.0
                scores["Bacterial Spot"] += 35  # Increased from 30
        
        # Target Spot - enhanced detection
        if (features['large_spot_density'] > 0.012 and  # Lowered from 0.015
            features['num_large_spots'] > 1 and  # Lowered from 2
            features['contrast'] > 3.2 and  # Lowered from 3.5
            features['avg_r'] > 75 and  # Lowered from 80
            features['edge_high'] > 0.1):  # Lowered from 0.12
            
            scores["Target Spot"] += 45  # Increased from 40
            
            # Target-like pattern - more sensitive
            if (features['avg_r'] > features['avg_g'] * 0.9 and  # Added buffer
                features['std_h'] > 22 and  # Lowered from 25
                features['edge_med'] > 0.18):  # Lowered from 0.2
                scores["Target Spot"] += 35  # Increased from 30
        
        # Spider Mites - increased sensitivity
        if (features['edge_low'] > 0.28 and  # Lowered from 0.3
            features['edge_med'] < 0.18 and  # Raised from 0.15
            features['spot_density'] > 0.035 and  # Lowered from 0.04
            features['num_medium_spots'] > 25 and  # Lowered from 30
            features['avg_s'] < 110):  # Raised from 100
            
            scores["Spider Mites"] += 45  # Increased from 40
            
            # Bronzing/yellowing - more sensitive
            if (features['avg_h'] < 55 and  # Raised upper bound
                features['avg_h'] > 22 and  # Lowered lower bound
                features['avg_s'] < 90 and  # Raised from 80
                features['homogeneity'] > 0.45):  # Lowered from 0.5
                scores["Spider Mites"] += 35  # Increased from 30
        
        # Yellow Leaf Curl Virus - enhanced detection
        if (features['avg_h'] < 45 and  # Raised from 40
            features['avg_h'] > 18 and  # Lowered from 20
            features['avg_s'] > 75 and  # Lowered from 80
            features['avg_s'] < 160 and  # Raised from 150
            features['edge_high'] > 0.13 and  # Lowered from 0.15
            features['homogeneity'] < 0.45):  # Raised from 0.4
            
            scores["Yellow Leaf Curl Virus"] += 45  # Increased from 40
            
            # Curling indicators - more sensitive
            if (features['edge_med'] > 0.22 and  # Lowered from 0.25
                features['std_h'] < 22 and  # Raised from 20
                features['avg_v'] > 140):  # Lowered from 150
                scores["Yellow Leaf Curl Virus"] += 35  # Increased from 30
        
        # Mosaic Virus - increased sensitivity
        if (features['std_h'] > 22 and  # Lowered from 25
            features['std_g'] > 35 and  # Lowered from 40
            features['homogeneity'] < 0.45 and  # Raised from 0.4
            features['avg_h'] < 65 and  # Raised from 60
            features['edge_low'] > 0.22):  # Lowered from 0.25
            
            scores["Mosaic Virus"] += 45  # Increased from 40
            
            # Mottled pattern - more sensitive
            if (features['energy'] < 0.25 and  # Raised from 0.2
                features['std_s'] > 30 and  # Lowered from 35
                features['std_h'] > 25):  # Lowered from 30
                scores["Mosaic Virus"] += 35  # Increased from 30
        
        # Find the top 2 diseases with highest scores
        sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_disease = sorted_diseases[0][0]
        secondary_disease = sorted_diseases[1][0]
        
        # Calculate confidence with more accurate scaling
        max_score = sorted_diseases[0][1]
        second_score = sorted_diseases[1][1]
        
        # Base confidence on primary disease score and gap between top two
        score_gap = max_score - second_score
        
        # Formula for confidence: higher base score and larger gap = higher confidence
        confidence = min(98.0, max_score * 0.7 + score_gap * 0.6)
        
        # If scores are very close, reduce confidence
        if score_gap < 10:
            confidence = min(confidence, 85.0)
            
        # If all scores are low, default to lower confidence
        if max_score < 30:
            confidence = min(confidence, 70.0)

        # If max_score is very low, default to healthy but with low confidence
        if max_score < 20:
            primary_disease = "Healthy"
            confidence = max(50.0, confidence * 0.7)
            
        # Ensure confidence has reasonable bounds
        confidence = max(50.0, min(98.0, confidence))
            
        return {
            "primary_disease": primary_disease,
            "confidence": confidence,
            "secondary_disease": secondary_disease,
            "secondary_confidence": max(0, second_score * 0.7),
            "scores": scores
        }
    
    def classify_disease(self, image_array):
        """Classify tomato plant disease from an image"""
        # Extract features from the image
        features = self._extract_disease_features(image_array)
        
        # Analyze features to determine disease
        result = self._analyze_disease_patterns(features)
        
        # Add disease description
        result["description"] = self.disease_descriptions[result["primary_disease"]]
        
        return result


# ===============================
# STREAMLIT UI
# ===============================

def main():
    st.set_page_config(
        page_title="Tomato Disease Detector",
        page_icon="🍅",
        layout="wide"
    )
    
    st.title("🍅 Tomato Plant Disease Detector")
    st.subheader("Upload an image of a tomato plant leaf to detect diseases")
    
    # Initialize the classifier and validator
    validator = ImageValidator()
    classifier = TomatoDiseaseClassifier()
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, width=400)
            
            # Convert PIL image to numpy array for processing
            img_array = np.array(image)
            
            # Validate that the image is of a tomato plant
            tomato_score = validator._color_histogram_analysis(img_array)
            
            st.write("---")
            
            if tomato_score < 60:
                st.error(f"⚠️ This doesn't appear to be a tomato plant leaf (Confidence: {tomato_score:.1f}%)")
                st.write("Please upload a clear image of a tomato plant leaf.")
                st.stop()
            else:
                st.success(f"✅ Tomato plant leaf detected (Confidence: {tomato_score:.1f}%)")
        
        with col2:
            st.subheader("Disease Analysis")
            
            # Process image and get results
            with st.spinner("Analyzing image..."):
                # Classify the disease
                result = classifier.classify_disease(img_array)
                
                # Display primary disease result
                st.markdown(f"### Detected Disease: **{result['primary_disease']}**")
                st.progress(result['confidence'] / 100)
                st.write(f"Confidence: {result['confidence']:.1f}%")
                
                # Display description
                st.markdown("### Description:")
                st.write(result['description'])
                
                # Show secondary possibility if confidence is lower
                if result['confidence'] < 85 and result['secondary_disease'] != "Healthy":
                    st.markdown("### Alternative possibility:")
                    st.write(f"**{result['secondary_disease']}** ({result['secondary_confidence']:.1f}% confidence)")
                
                # Show all scores as a bar chart
                st.markdown("### Disease Likelihood Scores:")
                chart_data = {k: v for k, v in result['scores'].items() if v > 0}
                st.bar_chart(chart_data)
    
    # Additional information section
    st.markdown("---")
    st.markdown("""
    ### How to use this tool
    
    1. Upload a clear image of a tomato plant leaf
    2. The system will verify if it's a tomato plant
    3. If verified, it will analyze and identify potential diseases
    
    ### Tips for best results
    
    - Use well-lit, clear images of the leaf
    - Include both healthy and affected areas in the image
    - For better accuracy, capture the leaf against a neutral background
    - If possible, include multiple leaves in the image
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ for Tomato Farmers | © 2024 TomatoCare")


if __name__ == "__main__":
    main()
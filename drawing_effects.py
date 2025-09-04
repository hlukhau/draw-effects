import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import imageio
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology, measure, segmentation
from sklearn.cluster import KMeans
import random
import math

class DrawingEffectGenerator:
    def __init__(self):
        self.styles = {
            'oil': self._oil_painting_style,
            'pastel': self._pastel_style,
            'pencil': self._pencil_style
        }
    
    def create_drawing_video(self, input_path, output_path, style='oil', detail_level=3, video_duration=10, progress_callback=None):
        """
        Create a drawing effect video from an input image
        
        Args:
            input_path: Path to input image
            output_path: Path for output video
            style: Drawing style ('oil', 'pastel', 'pencil')
            detail_level: Level of detail (1-5, higher = more detailed)
            video_duration: Duration of video in seconds
            progress_callback: Function to call with progress updates
        """
        if progress_callback:
            progress_callback(5, "Loading and preprocessing image...")
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(input_path)
        
        if progress_callback:
            progress_callback(15, "Segmenting image by detail levels...")
        
        # Segment image into different detail levels
        detail_segments = self._segment_by_detail_levels(image, detail_level)
        
        if progress_callback:
            progress_callback(30, f"Generating {style} style stroke patterns...")
        
        # Generate stroke patterns for each detail level
        stroke_layers = self._generate_progressive_strokes(image, detail_segments, style)
        
        if progress_callback:
            progress_callback(60, "Creating animation frames...")
        
        # Create animation frames with progressive drawing
        frames = self._create_progressive_animation(image, stroke_layers, video_duration)
        
        if progress_callback:
            progress_callback(85, "Rendering video...")
        
        # Save as video
        self._save_video(frames, output_path, fps=30)
        
        if progress_callback:
            progress_callback(100, "Video generation completed!")
    
    def _load_and_preprocess_image(self, input_path):
        """Load and preprocess the input image"""
        # Load image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (max 800px on longest side)
        h, w = img.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def _segment_by_detail_levels(self, image, detail_level):
        """
        Segment image into different detail levels using various techniques
        Returns segments from coarse to fine detail
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        segments = []
        
        # Level 1: Large regions (color-based segmentation)
        # Use K-means clustering on colors to find large uniform regions
        resized_for_kmeans = cv2.resize(image, (w//4, h//4))
        pixels = resized_for_kmeans.reshape(-1, 3)
        n_clusters = min(8, detail_level * 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        color_segments = labels.reshape(h//4, w//4)
        color_segments = cv2.resize(color_segments.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        segments.append(('color_regions', color_segments, kmeans.cluster_centers_))
        
        # Level 2: Medium structures (edge-based regions)
        # Find medium-scale edges and create regions
        blur_size = self._ensure_odd_kernel_size(7)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 2)
        edges_medium = cv2.Canny(blurred, 30, 80)
        
        # Create regions between edges
        distance = cv2.distanceTransform(255 - edges_medium, cv2.DIST_L2, 5)
        medium_regions = (distance > 10).astype(np.uint8)
        segments.append(('medium_structures', medium_regions, None))
        
        # Level 3: Fine details (high-frequency edges)
        blur_size = self._ensure_odd_kernel_size(3)
        fine_blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 1)
        edges_fine = cv2.Canny(fine_blurred, 50, 150)
        segments.append(('fine_details', edges_fine, None))
        
        # Level 4: Very fine details (texture)
        if detail_level >= 4:
            # Use Laplacian for texture detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.abs(laplacian)
            texture = (texture > np.percentile(texture, 80)).astype(np.uint8)
            segments.append(('texture', texture, None))
        
        # Level 5: Ultra fine details (noise-like details)
        if detail_level >= 5:
            # High-pass filter for very fine details
            blur_size = self._ensure_odd_kernel_size(1)
            very_blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0.5)
            high_freq = cv2.absdiff(gray, very_blurred)
            ultra_fine = (high_freq > 15).astype(np.uint8)
            segments.append(('ultra_fine', ultra_fine, None))
        
        return segments[:detail_level + 1]
    
    def _generate_progressive_strokes(self, image, detail_segments, style):
        """
        Generate stroke patterns for each detail level
        Each level avoids areas already covered by previous levels
        """
        h, w = image.shape[:2]
        stroke_layers = []
        covered_mask = np.zeros((h, w), dtype=np.uint8)  # Track what's already drawn
        
        style_func = self.styles.get(style, self.styles['oil'])
        
        for i, (segment_type, segment_data, extra_data) in enumerate(detail_segments):
            if segment_type == 'color_regions':
                # Large brush strokes for color regions
                strokes = self._create_color_region_strokes(image, segment_data, extra_data, covered_mask)
                stroke_size = max(15 - i * 3, 3)
            elif segment_type == 'medium_structures':
                # Medium strokes for structural elements
                strokes = self._create_structural_strokes(image, segment_data, covered_mask)
                stroke_size = max(10 - i * 2, 2)
            elif segment_type == 'fine_details':
                # Fine strokes for edges and details
                strokes = self._create_detail_strokes(image, segment_data, covered_mask)
                stroke_size = max(5 - i, 1)
            else:  # texture, ultra_fine
                # Very fine strokes for texture
                strokes = self._create_texture_strokes(image, segment_data, covered_mask)
                stroke_size = 1
            
            # Apply style-specific effects
            styled_strokes = style_func(strokes, image, stroke_size)
            stroke_layers.append(styled_strokes)
            
            # Update covered mask
            stroke_mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
            covered_mask = np.maximum(covered_mask, stroke_mask.astype(np.uint8) * 255)
        
        return stroke_layers
    
    def _create_color_region_strokes(self, image, segments, cluster_centers, covered_mask):
        """Create large brush strokes for color regions"""
        h, w = image.shape[:2]
        strokes = np.zeros_like(image)
        
        for cluster_id in range(len(cluster_centers)):
            # Find pixels belonging to this cluster
            cluster_mask = (segments == cluster_id)
            
            # Skip if area is too small or already mostly covered
            if np.sum(cluster_mask) < 100:
                continue
                
            # Reduce overlap with already covered areas
            cluster_mask = cluster_mask & (covered_mask < 128)
            if np.sum(cluster_mask) < 50:
                continue
            
            # Create brush strokes in this region
            color = cluster_centers[cluster_id].astype(np.uint8)
            
            # Find contours of the region
            cluster_mask_uint8 = cluster_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(cluster_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    # Create brush strokes along the contour and fill
                    cv2.fillPoly(strokes, [contour], color.tolist())
                    
                    # Add some random brush strokes within the region
                    for _ in range(max(1, int(cv2.contourArea(contour) / 1000))):
                        # Random point within contour
                        x, y, w_rect, h_rect = cv2.boundingRect(contour)
                        rand_x = random.randint(x, min(x + w_rect, w - 1))
                        rand_y = random.randint(y, min(y + h_rect, h - 1))
                        
                        if cv2.pointPolygonTest(contour, (rand_x, rand_y), False) >= 0:
                            # Draw brush stroke
                            stroke_length = random.randint(10, 30)
                            angle = random.uniform(0, 2 * math.pi)
                            end_x = int(rand_x + stroke_length * math.cos(angle))
                            end_y = int(rand_y + stroke_length * math.sin(angle))
                            cv2.line(strokes, (rand_x, rand_y), (end_x, end_y), color.tolist(), thickness=8)
        
        return strokes
    
    def _create_structural_strokes(self, image, structure_mask, covered_mask):
        """Create medium-sized strokes for structural elements"""
        h, w = image.shape[:2]
        strokes = np.zeros_like(image)
        
        # Avoid already covered areas
        structure_mask = structure_mask & (covered_mask < 128)
        
        # Find contours in structure mask
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Get average color in this region
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                avg_color = cv2.mean(image, mask=mask)[:3]
                
                # Create directional strokes
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                num_strokes = max(1, int(cv2.contourArea(contour) / 100))
                
                for _ in range(num_strokes):
                    start_x = random.randint(x, min(x + w_rect, w - 1))
                    start_y = random.randint(y, min(y + h_rect, h - 1))
                    
                    if cv2.pointPolygonTest(contour, (start_x, start_y), False) >= 0:
                        stroke_length = random.randint(8, 20)
                        angle = random.uniform(0, 2 * math.pi)
                        end_x = int(start_x + stroke_length * math.cos(angle))
                        end_y = int(start_y + stroke_length * math.sin(angle))
                        cv2.line(strokes, (start_x, start_y), (end_x, end_y), avg_color, thickness=4)
        
        return strokes
    
    def _create_detail_strokes(self, image, edge_mask, covered_mask):
        """Create fine strokes for edges and details"""
        h, w = image.shape[:2]
        strokes = np.zeros_like(image)
        
        # Avoid already covered areas
        edge_mask = edge_mask & (covered_mask < 128)
        
        # Find edge pixels
        edge_points = np.where(edge_mask > 0)
        
        for i in range(0, len(edge_points[0]), 3):  # Sample every 3rd point
            y, x = edge_points[0][i], edge_points[1][i]
            
            # Get local color
            color = image[y, x]
            
            # Create small directional stroke
            stroke_length = random.randint(3, 8)
            angle = random.uniform(0, 2 * math.pi)
            end_x = int(x + stroke_length * math.cos(angle))
            end_y = int(y + stroke_length * math.sin(angle))
            
            # Ensure stroke stays within bounds
            end_x = max(0, min(end_x, w - 1))
            end_y = max(0, min(end_y, h - 1))
            
            cv2.line(strokes, (x, y), (end_x, end_y), color.tolist(), thickness=2)
        
        return strokes
    
    def _create_texture_strokes(self, image, texture_mask, covered_mask):
        """Create very fine strokes for texture details"""
        h, w = image.shape[:2]
        strokes = np.zeros_like(image)
        
        # Avoid already covered areas
        texture_mask = texture_mask & (covered_mask < 128)
        
        # Find texture pixels
        texture_points = np.where(texture_mask > 0)
        
        for i in range(0, len(texture_points[0]), 5):  # Sample every 5th point
            y, x = texture_points[0][i], texture_points[1][i]
            
            # Get local color
            color = image[y, x]
            
            # Create tiny stroke or dot
            if random.random() < 0.5:
                # Dot
                cv2.circle(strokes, (x, y), 1, color.tolist(), -1)
            else:
                # Tiny line
                stroke_length = random.randint(1, 3)
                angle = random.uniform(0, 2 * math.pi)
                end_x = int(x + stroke_length * math.cos(angle))
                end_y = int(y + stroke_length * math.sin(angle))
                
                end_x = max(0, min(end_x, w - 1))
                end_y = max(0, min(end_y, h - 1))
                
                cv2.line(strokes, (x, y), (end_x, end_y), color.tolist(), thickness=1)
        
        return strokes
    
    def _oil_painting_style(self, strokes, image, stroke_size):
        """Apply oil painting style effects"""
        if not np.any(strokes):
            return strokes
        
        # Convert to PIL for oil painting effect
        pil_strokes = Image.fromarray(strokes)
        
        # Apply oil painting effect (blur + enhance)
        blurred = pil_strokes.filter(ImageFilter.GaussianBlur(radius=1.5))
        enhanced = ImageEnhance.Color(blurred).enhance(1.3)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
        
        # Add texture
        textured = self._add_brush_texture(np.array(enhanced), strokes)
        
        return textured
    
    def _pastel_style(self, strokes, image, stroke_size):
        """Apply pastel style effects"""
        if not np.any(strokes):
            return strokes
        
        # Soften colors
        softened = cv2.bilateralFilter(strokes, 15, 50, 50)
        
        # Reduce saturation slightly
        hsv = cv2.cvtColor(softened, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8  # Reduce saturation
        softened = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add soft texture
        textured = self._add_soft_texture(softened, strokes)
        
        return textured
    
    def _pencil_style(self, strokes, image, stroke_size):
        """Apply pencil/hatching style effects"""
        if not np.any(strokes):
            return strokes
        
        # Convert to grayscale for pencil effect
        gray_strokes = cv2.cvtColor(strokes, cv2.COLOR_RGB2GRAY)
        
        # Create hatching pattern
        hatched = self._create_hatching_pattern(gray_strokes, strokes)
        
        # Convert back to RGB
        pencil_strokes = cv2.cvtColor(hatched, cv2.COLOR_GRAY2RGB)
        
        return pencil_strokes
    
    def _add_brush_texture(self, image, strokes):
        """Add brush texture to oil painting"""
        # Create random brush strokes
        textured = image.copy()
        h, w = image.shape[:2]
        
        # Create mask from strokes
        mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
        
        # Add random variations to simulate brush texture
        for _ in range(50):
            if np.any(mask):
                y, x = np.where(mask)
                if len(y) > 0:
                    idx = random.randint(0, len(y) - 1)
                    center_y, center_x = y[idx], x[idx]
                    
                    # Create small brush stroke
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = center_y + dy, center_x + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                                # Add slight color variation
                                variation = np.random.normal(0, 5, 3)
                                textured[ny, nx] = np.clip(textured[ny, nx] + variation, 0, 255)
        
        return textured.astype(np.uint8)
    
    def _add_soft_texture(self, image, strokes):
        """Add soft texture for pastel effect"""
        textured = image.copy().astype(np.float32)
        
        # Create mask from strokes
        mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
        
        # Add subtle noise for pastel texture
        noise = np.random.normal(0, 3, image.shape)
        textured[mask] += noise[mask]
        
        return np.clip(textured, 0, 255).astype(np.uint8)
    
    def _create_hatching_pattern(self, gray_image, strokes):
        """Create pencil hatching pattern"""
        hatched = gray_image.copy()
        h, w = gray_image.shape
        
        # Create mask from strokes
        mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
        
        # Create diagonal hatching lines
        for i in range(0, h + w, 4):  # Every 4 pixels
            for j in range(max(0, i - w), min(i + 1, h)):
                x, y = i - j, j
                if 0 <= x < w and 0 <= y < h and mask[y, x]:
                    # Create hatching effect
                    if (x + y) % 8 < 2:  # Create hatching pattern
                        hatched[y, x] = max(0, hatched[y, x] - 30)
        
        # Add cross-hatching for darker areas
        for i in range(0, h + w, 6):  # Every 6 pixels, opposite direction
            for j in range(max(0, i - h), min(i + 1, w)):
                x, y = j, i - j
                if 0 <= x < w and 0 <= y < h and mask[y, x] and gray_image[y, x] < 128:
                    if (x - y) % 8 < 2:  # Cross-hatching pattern
                        hatched[y, x] = max(0, hatched[y, x] - 20)
        
        return hatched
    
    def _create_progressive_animation(self, image, stroke_layers, video_duration):
        """Create animation frames showing progressive drawing"""
        frames = []
        h, w = image.shape[:2]
        
        # Start with blank canvas
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        # Number of frames per layer
        frames_per_layer = max(10, int(video_duration * 30 / len(stroke_layers)))  # 30 FPS, minimum 10 frames per layer
        
        for layer_idx, strokes in enumerate(stroke_layers):
            # Get stroke positions
            stroke_mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
            stroke_positions = np.where(stroke_mask)
            if len(stroke_positions[0]) == 0:
                continue
            
            # Create indices for random stroke order
            indices = list(range(len(stroke_positions[0])))
            random.shuffle(indices)
            
            strokes_per_frame = max(1, len(indices) // frames_per_layer)
            
            for frame_idx in range(frames_per_layer):
                frame_canvas = canvas.copy()
                
                # Add all previous layers completely
                for prev_idx in range(layer_idx):
                    prev_strokes = stroke_layers[prev_idx]
                    prev_mask = np.any(prev_strokes > 0, axis=2) if len(prev_strokes.shape) == 3 else prev_strokes > 0
                    frame_canvas[prev_mask] = prev_strokes[prev_mask]
                
                # Add partial current layer
                start_idx = frame_idx * strokes_per_frame
                end_idx = min(start_idx + strokes_per_frame, len(indices))
                
                for i in range(start_idx, end_idx):
                    if i < len(indices):
                        idx = indices[i]
                        y, x = stroke_positions[0][idx], stroke_positions[1][idx]
                        if y < h and x < w:  # Bounds check
                            frame_canvas[y, x] = strokes[y, x]
                
                frames.append(frame_canvas.copy())
        
        # Create final composite frame
        final_canvas = canvas.copy()
        for strokes in stroke_layers:
            stroke_mask = np.any(strokes > 0, axis=2) if len(strokes.shape) == 3 else strokes > 0
            final_canvas[stroke_mask] = strokes[stroke_mask]
        
        # Hold final frame for a bit longer
        for _ in range(30):  # 1 second at 30 FPS
            frames.append(final_canvas.copy())
        
        return frames
    
    def _save_video(self, frames, output_path, fps=30):
        """Save frames as video"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Use imageio to create video
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)
    
    def _ensure_odd_kernel_size(self, size):
        """Ensure kernel size is positive and odd"""
        size = max(1, int(size))  # Ensure positive
        if size % 2 == 0:  # If even, make it odd
            size += 1
        return size
    
    def create_segmentation_images(self, input_path, output_dir, file_id, color_threshold=30, progress_callback=None):
        """
        Create segmentation images based on color similarity with adjustable threshold
        
        Args:
            input_path: Path to input image
            output_dir: Directory to save segmentation images
            file_id: Unique file identifier
            color_threshold: Color similarity threshold (10-100, lower = more segments)
            progress_callback: Function to call with progress updates
        
        Returns:
            List of output filenames
        """
        if progress_callback:
            progress_callback(5, "Loading and preprocessing image...")
        
        # Clean up old segmentation files for this file_id
        self._cleanup_old_segmentation_files(output_dir, file_id)
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(input_path)
        
        if progress_callback:
            progress_callback(20, "Creating color-based segmentation...")
        
        # Create segmentation with the specified threshold
        segments = self._segment_by_color_similarity(image, color_threshold)
        
        if progress_callback:
            progress_callback(50, "Computing average colors for segments...")
        
        # Compute average colors for each segment
        segment_info = self._compute_segment_average_colors(image, segments)
        
        if progress_callback:
            progress_callback(70, "Creating visualization images...")
        
        output_files = []
        
        # Save original image for reference
        original_filename = f"{file_id}_00_original_t{color_threshold}.png"
        original_path = os.path.join(output_dir, original_filename)
        cv2.imwrite(original_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        output_files.append(original_filename)
        
        # Create and save segmentation visualization (colored segments)
        vis_image = self._visualize_color_segments_simple(image, segments)
        segments_filename = f"{file_id}_01_segments_t{color_threshold}.png"
        segments_path = os.path.join(output_dir, segments_filename)
        cv2.imwrite(segments_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        output_files.append(segments_filename)
        
        # Create and save mean color image
        mean_color_image = self._create_mean_color_image(image, segments, segment_info)
        mean_filename = f"{file_id}_02_mean_colors_t{color_threshold}.png"
        mean_path = os.path.join(output_dir, mean_filename)
        cv2.imwrite(mean_path, cv2.cvtColor(mean_color_image, cv2.COLOR_RGB2BGR))
        output_files.append(mean_filename)
        
        # Save segment information as JSON
        import json
        segment_data = {
            'threshold': color_threshold,
            'total_segments': len(segment_info),
            'segments': [
                {
                    'id': int(seg_id),
                    'pixel_count': int(info['pixel_count']),
                    'average_color': {
                        'r': int(info['avg_color'][0]),
                        'g': int(info['avg_color'][1]),
                        'b': int(info['avg_color'][2])
                    },
                    'average_color_hex': f"#{int(info['avg_color'][0]):02x}{int(info['avg_color'][1]):02x}{int(info['avg_color'][2]):02x}"
                }
                for seg_id, info in segment_info.items()
            ]
        }
        
        json_filename = f"{file_id}_03_segment_info_t{color_threshold}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(segment_data, f, indent=2)
        output_files.append(json_filename)
        
        if progress_callback:
            progress_callback(100, f"Segmentation completed! Found {len(segment_info)} segments.")
        
        return output_files
    
    def _cleanup_old_segmentation_files(self, output_dir, file_id):
        """Remove old segmentation files for this file_id to prevent confusion"""
        import glob
        
        # Patterns to match old segmentation files
        patterns = [
            f"{file_id}_*_original_t*.png",
            f"{file_id}_*_segments_t*.png", 
            f"{file_id}_*_mean_colors_t*.png",
            f"{file_id}_*_segment_info_t*.json"
        ]
        
        for pattern in patterns:
            files_to_remove = glob.glob(os.path.join(output_dir, pattern))
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be removed
    
    def _compute_segment_average_colors(self, image, segments):
        """
        Compute average color for each segment
        
        Args:
            image: Original RGB image
            segments: Segmented image with segment IDs
            
        Returns:
            Dictionary with segment info: {segment_id: {'avg_color': [r,g,b], 'pixel_count': count}}
        """
        unique_segments = np.unique(segments)
        unique_segments = unique_segments[unique_segments > 0]  # Exclude background
        
        segment_info = {}
        
        for segment_id in unique_segments:
            mask = (segments == segment_id)
            segment_pixels = image[mask]
            
            if len(segment_pixels) > 0:
                avg_color = np.mean(segment_pixels, axis=0)
                segment_info[segment_id] = {
                    'avg_color': avg_color,
                    'pixel_count': len(segment_pixels)
                }
        
        return segment_info
    
    def _create_mean_color_image(self, image, segments, segment_info):
        """
        Create image where each segment is filled with its average color
        
        Args:
            image: Original image
            segments: Segmented image
            segment_info: Dictionary with segment average colors
            
        Returns:
            Image with segments filled with average colors
        """
        mean_image = np.zeros_like(image)
        
        for segment_id, info in segment_info.items():
            mask = (segments == segment_id)
            mean_image[mask] = info['avg_color']
        
        return mean_image.astype(np.uint8)
    
    def _segment_by_color_similarity(self, image, threshold):
        """
        Segment image based on color similarity using region growing
        
        Args:
            image: Input image (RGB)
            threshold: Color similarity threshold
            
        Returns:
            Segmented image with different segment IDs
        """
        h, w = image.shape[:2]
        segments = np.zeros((h, w), dtype=np.int32)
        visited = np.zeros((h, w), dtype=bool)
        segment_id = 1
        
        # Convert threshold to a reasonable range for color distance
        color_threshold = threshold * 2.55  # Scale to 0-255 range
        
        for y in range(0, h, 2):  # Skip every other pixel for speed
            for x in range(0, w, 2):
                if not visited[y, x]:
                    # Start new segment from this pixel
                    segment_pixels = self._region_growing(image, (x, y), color_threshold, visited)
                    
                    # Only create segment if it has enough pixels
                    if len(segment_pixels) > 20:  # Minimum segment size
                        for px, py in segment_pixels:
                            segments[py, px] = segment_id
                        segment_id += 1
        
        return segments
    
    def _region_growing(self, image, seed, threshold, visited):
        """
        Region growing algorithm for color-based segmentation
        """
        h, w = image.shape[:2]
        seed_x, seed_y = seed
        
        if visited[seed_y, seed_x]:
            return []
        
        seed_color = image[seed_y, seed_x].astype(np.float32)
        region_pixels = []
        stack = [(seed_x, seed_y)]
        
        while stack:
            x, y = stack.pop()
            
            if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
                continue
            
            pixel_color = image[y, x].astype(np.float32)
            color_distance = np.sqrt(np.sum((pixel_color - seed_color) ** 2))
            
            if color_distance <= threshold:
                visited[y, x] = True
                region_pixels.append((x, y))
                
                # Add neighbors to stack
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        stack.append((nx, ny))
        
        return region_pixels
    
    def _visualize_color_segments_simple(self, image, segments):
        """Visualize color-based segmentation with random colors for each segment"""
        h, w = image.shape[:2]
        vis_image = np.zeros_like(image)
        
        # Get unique segment IDs
        unique_segments = np.unique(segments)
        unique_segments = unique_segments[unique_segments > 0]  # Exclude background
        
        # Generate random colors for each segment
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 256, (len(unique_segments), 3))
        
        for i, segment_id in enumerate(unique_segments):
            mask = (segments == segment_id)
            if np.any(mask):
                vis_image[mask] = colors[i]
        
        # Keep background black
        vis_image[segments == 0] = [0, 0, 0]
        
        return vis_image
    
    def generate_segment_brush_strokes(self, output_dir, file_id, segment_id):
        """
        Generate brush strokes for a specific segment based on its geometry
        
        Args:
            output_dir: Directory containing segment data
            file_id: File identifier
            segment_id: ID of the segment to draw
            
        Returns:
            List of brush stroke data for frontend rendering
        """
        import json
        import math
        from scipy import ndimage
        from skimage import measure
        
        # Load segment information - look in the outputs directory
        outputs_dir = os.path.join(os.path.dirname(output_dir), 'outputs')
        if not os.path.exists(outputs_dir):
            outputs_dir = output_dir  # fallback to original directory
        
        print(f"Looking for segment files in: {outputs_dir}")
        print(f"Files in directory: {os.listdir(outputs_dir) if os.path.exists(outputs_dir) else 'Directory not found'}")
        
        json_files = [f for f in os.listdir(outputs_dir) if f.startswith(f"{file_id}_") and f.endswith('.json')]
        if not json_files:
            # Try alternative pattern matching
            json_files = [f for f in os.listdir(outputs_dir) if file_id in f and f.endswith('.json')]
            
        if not json_files:
            raise Exception(f"No segment data found for file_id: {file_id} in directory: {outputs_dir}")
        
        json_path = os.path.join(outputs_dir, json_files[0])
        print(f"Loading segment data from: {json_path}")
        
        with open(json_path, 'r') as f:
            segment_data = json.load(f)
        
        # Find the target segment
        target_segment = None
        for segment in segment_data['segments']:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            raise Exception(f"Segment {segment_id} not found in {len(segment_data['segments'])} segments")
        
        # Load the mean color image to get segment mask
        mean_color_files = [f for f in os.listdir(outputs_dir) 
                           if f.startswith(f"{file_id}_") and 'mean_colors' in f and f.endswith('.png')]
        if not mean_color_files:
            # Try alternative pattern matching
            mean_color_files = [f for f in os.listdir(outputs_dir) 
                               if file_id in f and 'mean_colors' in f and f.endswith('.png')]
            
        if not mean_color_files:
            raise Exception(f"Mean color image not found for file_id: {file_id}")
        
        mean_color_path = os.path.join(outputs_dir, mean_color_files[0])
        print(f"Loading mean color image from: {mean_color_path}")
        
        mean_image = cv2.imread(mean_color_path)
        mean_image = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)
        
        # Create mask for the target segment
        target_color = np.array([
            target_segment['average_color']['r'],
            target_segment['average_color']['g'], 
            target_segment['average_color']['b']
        ])
        
        # Find pixels matching the segment color
        mask = np.all(mean_image == target_color, axis=2)
        
        print(f"Found {np.sum(mask)} pixels for segment {segment_id}")
        
        # Analyze segment geometry and generate brush strokes
        brush_strokes = self._analyze_segment_and_create_strokes(mask, target_segment)
        
        print(f"Generated {len(brush_strokes)} brush strokes for segment {segment_id}")
        
        return brush_strokes
    
    def _analyze_segment_and_create_strokes(self, mask, segment_info):
        """
        Analyze segment geometry and create appropriate brush strokes
        
        Args:
            mask: Boolean mask of the segment
            segment_info: Segment information including color
            
        Returns:
            List of brush stroke data
        """
        from scipy import ndimage
        from skimage import measure, morphology
        import math
        
        # Get segment properties
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if not regions:
            return []
        
        # Use the largest region if multiple exist
        main_region = max(regions, key=lambda r: r.area)
        
        # Extract geometric properties
        centroid = main_region.centroid
        bbox = main_region.bbox  # (min_row, min_col, max_row, max_col)
        area = main_region.area
        perimeter = main_region.perimeter
        
        # Calculate dimensions
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 1
        
        # Determine brush strategy based on geometry
        brush_strokes = []
        segment_color = f"#{segment_info['average_color']['r']:02x}{segment_info['average_color']['g']:02x}{segment_info['average_color']['b']:02x}"
        
        # Calculate appropriate brush width based on area
        brush_width = max(2, min(15, int(math.sqrt(area) / 10)))
        
        if area < 100:
            # Very small segments: single dot or short stroke
            brush_strokes.append({
                'color': segment_color,
                'width': max(2, brush_width // 2),
                'points': [
                    {'x': int(centroid[1]), 'y': int(centroid[0])}
                ]
            })
            
        elif aspect_ratio > 3:
            # Long horizontal segments: horizontal strokes
            brush_strokes.extend(self._create_horizontal_strokes(mask, segment_color, brush_width))
            
        elif aspect_ratio < 0.33:
            # Long vertical segments: vertical strokes  
            brush_strokes.extend(self._create_vertical_strokes(mask, segment_color, brush_width))
            
        elif area < 500:
            # Small to medium segments: radial strokes from center
            brush_strokes.extend(self._create_radial_strokes(mask, segment_color, brush_width))
            
        else:
            # Large segments: combination of techniques
            if width > height:
                brush_strokes.extend(self._create_horizontal_strokes(mask, segment_color, brush_width))
            else:
                brush_strokes.extend(self._create_vertical_strokes(mask, segment_color, brush_width))
            
            # Add some cross-hatching for texture
            brush_strokes.extend(self._create_cross_hatch_strokes(mask, segment_color, brush_width // 2))
        
        return brush_strokes
    
    def _create_horizontal_strokes(self, mask, color, brush_width):
        """Create horizontal brush strokes across the segment"""
        strokes = []
        h, w = mask.shape
        
        # Find y coordinates where the segment exists
        y_coords = np.where(np.any(mask, axis=1))[0]
        
        # Create strokes every few pixels vertically
        step = max(2, brush_width // 2)
        for y in y_coords[::step]:
            # Find x range for this y coordinate
            x_coords = np.where(mask[y, :])[0]
            if len(x_coords) > 0:
                x_start, x_end = x_coords[0], x_coords[-1]
                
                # Create stroke points
                points = []
                num_points = max(2, (x_end - x_start) // 5)
                for i in range(num_points):
                    x = x_start + (x_end - x_start) * i / (num_points - 1)
                    points.append({'x': int(x), 'y': int(y)})
                
                if len(points) > 1:
                    strokes.append({
                        'color': color,
                        'width': brush_width,
                        'points': points
                    })
        
        return strokes
    
    def _create_vertical_strokes(self, mask, color, brush_width):
        """Create vertical brush strokes across the segment"""
        strokes = []
        h, w = mask.shape
        
        # Find x coordinates where the segment exists
        x_coords = np.where(np.any(mask, axis=0))[0]
        
        # Create strokes every few pixels horizontally
        step = max(2, brush_width // 2)
        for x in x_coords[::step]:
            # Find y range for this x coordinate
            y_coords = np.where(mask[:, x])[0]
            if len(y_coords) > 0:
                y_start, y_end = y_coords[0], y_coords[-1]
                
                # Create stroke points
                points = []
                num_points = max(2, (y_end - y_start) // 5)
                for i in range(num_points):
                    y = y_start + (y_end - y_start) * i / (num_points - 1)
                    points.append({'x': int(x), 'y': int(y)})
                
                if len(points) > 1:
                    strokes.append({
                        'color': color,
                        'width': brush_width,
                        'points': points
                    })
        
        return strokes
    
    def _create_radial_strokes(self, mask, color, brush_width):
        """Create radial brush strokes from the center of the segment"""
        from skimage import measure
        
        strokes = []
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if not regions:
            return strokes
        
        main_region = max(regions, key=lambda r: r.area)
        centroid = main_region.centroid
        
        # Create strokes in different directions from center
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in angles:
            radians = math.radians(angle)
            dx = math.cos(radians)
            dy = math.sin(radians)
            
            points = []
            # Start from center and move outward
            for distance in range(0, 30, 3):
                x = int(centroid[1] + dx * distance)
                y = int(centroid[0] + dy * distance)
                
                # Check if point is within image bounds and segment
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    points.append({'x': x, 'y': y})
                else:
                    break
            
            if len(points) > 1:
                strokes.append({
                    'color': color,
                    'width': brush_width,
                    'points': points
                })
        
        return strokes
    
    def _create_cross_hatch_strokes(self, mask, color, brush_width):
        """Create cross-hatching strokes for texture"""
        strokes = []
        h, w = mask.shape
        
        # Diagonal strokes (45 degrees)
        for offset in range(-max(h, w), max(h, w), brush_width * 3):
            points = []
            for x in range(w):
                y = x + offset
                if 0 <= y < h and mask[y, x]:
                    points.append({'x': x, 'y': y})
            
            if len(points) > 2:
                strokes.append({
                    'color': color,
                    'width': brush_width,
                    'points': points
                })
        
        return strokes

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
import json

class DrawingEffectGenerator:
    def __init__(self):
        self.styles = {
            'pencil': self._pencil_style
        }
    
    def create_drawing_video(self, input_path, output_path, style='pencil', detail_level=3, video_duration=10, color_threshold=30, progress_callback=None):
        """
        Create a drawing effect video from an input image following the same logic as interactive drawing
        
        Args:
            input_path: Path to input image
            output_path: Path for output video
            style: Drawing style ('pencil' only now)
            detail_level: Level of detail (1-5, higher = more detailed)
            video_duration: Duration of video in seconds
            color_threshold: Color similarity threshold (10-100, lower = more segments)
            progress_callback: Function to call with progress updates
        """
        if progress_callback:
            progress_callback(5, "Loading and preprocessing image...")
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(input_path)
        
        if progress_callback:
            progress_callback(15, "Generating image segmentation...")
        
        # Generate segmentation (same as interactive mode)
        segments_data = self._generate_video_segments(image, color_threshold, detail_level)
        
        if progress_callback:
            progress_callback(30, f"Sorting segments by size...")
        
        # Sort segments by pixel count (largest first) - same as interactive drawing
        sorted_segments = sorted(segments_data, key=lambda x: x['pixel_count'], reverse=True)
        
        if progress_callback:
            progress_callback(40, f"Calculating video timing for {len(sorted_segments)} segments...")
        
        # Calculate timing for video duration
        total_frames = int(video_duration * 30)  # 30 FPS
        frames_per_segment = max(1, total_frames // len(sorted_segments)) if sorted_segments else 1
        
        if progress_callback:
            progress_callback(50, "Creating animation frames...")
        
        # Create animation frames following the same logic as drawAllSegments
        frames = self._create_segment_based_animation(
            image, sorted_segments, frames_per_segment, total_frames, progress_callback
        )
        
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
    
    def _segment_by_detail_levels(self, image, detail_level, color_threshold):
        """
        Segment image into different detail levels using various techniques
        Returns segments from coarse to fine detail
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        segments = []
        
        # Level 1: Large regions (color-based segmentation)
        # Use K-means clustering on colors to find large uniform regions
        # Adjust number of clusters based on color_threshold
        resized_for_kmeans = cv2.resize(image, (w//4, h//4))
        pixels = resized_for_kmeans.reshape(-1, 3)
        
        # Calculate number of clusters based on color_threshold and detail_level
        # Lower color_threshold = more segments (more clusters)
        # Higher color_threshold = fewer segments (fewer clusters)
        base_clusters = max(4, min(16, int(100 / color_threshold * detail_level)))
        n_clusters = min(base_clusters, detail_level * 2)
        
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
        
        style_func = self.styles.get(style, self.styles['pencil'])
        
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
        
        # Post-process cleanup: now that new files are created, we can safely clean up old ones
        self._post_process_cleanup(output_dir, file_id, color_threshold)
        
        return output_files
    
    def _cleanup_old_segmentation_files(self, output_dir, file_id):
        """Remove old segmentation files for this file_id to prevent confusion"""
        import glob
        
        # CONSERVATIVE CLEANUP STRATEGY:
        # Don't remove any files during segmentation process to prevent UI from losing content
        # Only clean up when we have excessive files (more than 20 files = 5 complete sets)
        # This ensures the UI always has something to display during re-segmentation
        
        # Get all existing segmentation files for this file_id
        all_files = []
        patterns = [
            f"{file_id}_*_original_t*.png",
            f"{file_id}_*_segments_t*.png", 
            f"{file_id}_*_mean_colors_t*.png",
            f"{file_id}_*_segment_info_t*.json"
        ]
        
        for pattern in patterns:
            files_found = glob.glob(os.path.join(output_dir, pattern))
            all_files.extend(files_found)
        
        # Only cleanup if we have excessive files (more than 20 files = 5 complete sets)
        # This is much more conservative to prevent UI flickering
        if len(all_files) > 20:
            # Sort by modification time and keep only the 12 most recent files (3 complete sets)
            all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            files_to_remove = all_files[12:]  # Remove older files beyond the 3 most recent sets
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be removed
    
    def _post_process_cleanup(self, output_dir, file_id, current_threshold):
        """
        Clean up old segmentation files AFTER new ones are successfully created
        This prevents UI from losing content during re-segmentation
        """
        import glob
        
        # Get all existing segmentation files for this file_id
        all_files = []
        patterns = [
            f"{file_id}_*_original_t*.png",
            f"{file_id}_*_segments_t*.png", 
            f"{file_id}_*_mean_colors_t*.png",
            f"{file_id}_*_segment_info_t*.json"
        ]
        
        for pattern in patterns:
            files_found = glob.glob(os.path.join(output_dir, pattern))
            all_files.extend(files_found)
        
        # Group files by threshold value to keep the most recent ones
        threshold_groups = {}
        current_files = []  # Files with current threshold
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Extract threshold from filename (format: *_t{threshold}.*)
            import re
            threshold_match = re.search(r'_t(\d+)\.', filename)
            if threshold_match:
                threshold = int(threshold_match.group(1))
                if threshold == current_threshold:
                    current_files.append(file_path)
                else:
                    if threshold not in threshold_groups:
                        threshold_groups[threshold] = []
                    threshold_groups[threshold].append(file_path)
        
        # Keep current threshold files and the 2 most recent other threshold sets
        files_to_keep = current_files.copy()
        
        # Sort threshold groups by modification time and keep the 2 most recent
        if threshold_groups:
            # Get the most recent file from each threshold group to determine recency
            threshold_recency = []
            for threshold, files in threshold_groups.items():
                most_recent_time = max(os.path.getmtime(f) for f in files)
                threshold_recency.append((most_recent_time, threshold, files))
            
            # Sort by recency and keep the 2 most recent threshold sets
            threshold_recency.sort(reverse=True)
            for _, threshold, files in threshold_recency[:2]:
                files_to_keep.extend(files)
        
        # Remove files that are not in the keep list
        files_to_remove = [f for f in all_files if f not in files_to_keep]
        
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
    
    def generate_segment_brush_strokes(self, output_dir, file_id, segment_id, brush_type='pencil', stroke_density=1.0):
        """
        Generate brush strokes for a specific segment based on its geometry and brush type
        
        Args:
            output_dir: Directory containing segment data
            file_id: File identifier
            segment_id: ID of the segment to draw
            brush_type: Type of brush ('pencil')
            stroke_density: Density multiplier for stroke count (0.5-2.0)
            
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
        
        # CRITICAL FIX: Exclude boundary JSON files when looking for segment data
        json_files = [f for f in os.listdir(outputs_dir) 
                     if f.startswith(f"{file_id}_") and f.endswith('.json') 
                     and 'boundaries' not in f]  # Exclude boundary files
        if not json_files:
            # Try alternative pattern matching, still excluding boundary files
            json_files = [f for f in os.listdir(outputs_dir) 
                         if file_id in f and f.endswith('.json') 
                         and 'boundaries' not in f]
            
        if not json_files:
            raise Exception(f"No segment data found for file_id: {file_id} in directory: {outputs_dir}")
        
        # CRITICAL FIX: Always select the NEWEST JSON file by modification time
        json_files_with_time = []
        for json_file in json_files:
            file_path = os.path.join(outputs_dir, json_file)
            mod_time = os.path.getmtime(file_path)
            json_files_with_time.append((json_file, mod_time))
        
        # Sort by modification time (newest first)
        json_files_with_time.sort(key=lambda x: x[1], reverse=True)
        newest_json_file = json_files_with_time[0][0]
        
        json_path = os.path.join(outputs_dir, newest_json_file)
        # print(f"DEBUG: Found {len(json_files)} JSON files for file_id: {file_id}")
        # print(f"DEBUG: Available JSON files: {[f for f, _ in json_files_with_time]}")
        # print(f"DEBUG: Selected NEWEST JSON file: {newest_json_file}")
        print(f"Loading segment data from: {json_path}")
        
        with open(json_path, 'r') as f:
            segment_data = json.load(f)
        
        # Debug: Print the structure of segment_data
        # print(f"DEBUG: segment_data keys: {list(segment_data.keys()) if isinstance(segment_data, dict) else 'Not a dict'}")
        # print(f"DEBUG: segment_data type: {type(segment_data)}")
        
        # Handle different possible data structures
        segments_list = None
        if isinstance(segment_data, dict):
            if 'segments' in segment_data:
                segments_list = segment_data['segments']
            elif 'segment_data' in segment_data:
                segments_list = segment_data['segment_data']
            else:
                # Maybe the data is directly a list or has other keys
                for key in segment_data.keys():
                    if isinstance(segment_data[key], list):
                        segments_list = segment_data[key]
                        # print(f"DEBUG: Using key '{key}' as segments list")
                        break
        elif isinstance(segment_data, list):
            segments_list = segment_data
            
        if segments_list is None:
            raise Exception(f"Could not find segments data in JSON structure. Available keys: {list(segment_data.keys()) if isinstance(segment_data, dict) else 'Data is not a dict'}")
        
        # Debug: Print available segment IDs
        available_ids = [segment.get('id', 'NO_ID') for segment in segments_list]
        # print(f"DEBUG: Available segment IDs: {sorted(available_ids)[:10]}... (showing first 10)")
        # print(f"DEBUG: Looking for segment_id: {segment_id} (type: {type(segment_id)})")
        # print(f"DEBUG: Total segments: {len(segments_list)}")
        
        # Find the target segment
        target_segment = None
        for segment in segments_list:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            # Try alternative matching strategies
            # print(f"DEBUG: Direct ID match failed. Trying alternative strategies...")
            
            # Strategy 1: Try converting segment_id to int if it's a string
            if isinstance(segment_id, str) and segment_id.isdigit():
                segment_id_int = int(segment_id)
                for segment in segments_list:
                    if segment['id'] == segment_id_int:
                        target_segment = segment
                        # print(f"DEBUG: Found segment using int conversion: {segment_id_int}")
                        break
            
            # Strategy 2: Try converting segment IDs to string if segment_id is string
            if not target_segment and isinstance(segment_id, str):
                for segment in segments_list:
                    if str(segment['id']) == segment_id:
                        target_segment = segment
                        # print(f"DEBUG: Found segment using string conversion: {segment_id}")
                        break
            
            # Strategy 3: Try using segment_id as index if it's within range
            if not target_segment and isinstance(segment_id, int) and 0 <= segment_id < len(segments_list):
                target_segment = segments_list[segment_id]
                # print(f"DEBUG: Found segment using index: {segment_id}")
        
        if not target_segment:
            raise Exception(f"Segment {segment_id} not found in {len(segments_list)} segments. Available IDs range: {min(available_ids) if available_ids else 'None'} to {max(available_ids) if available_ids else 'None'}")
        
        # Debug: Print target segment structure
        # print(f"🔍 DEBUG: Target segment keys: {list(target_segment.keys()) if isinstance(target_segment, dict) else 'Not a dict'}")
        # print(f"🔍 DEBUG: Target segment type: {type(target_segment)}")
        # if isinstance(target_segment, dict):
        #     print(f"🔍 DEBUG: Target segment sample data: {dict(list(target_segment.items())[:3])}")  # Show first 3 items
        
        # Check if average_color exists and handle different structures
        if 'average_color' not in target_segment:
            # print(f"❌ ERROR: 'average_color' field missing from segment {segment_id}")
            # print(f"❌ ERROR: Available fields: {list(target_segment.keys()) if isinstance(target_segment, dict) else 'N/A'}")
            raise Exception(f"Segment {segment_id} missing 'average_color' field. Available fields: {list(target_segment.keys()) if isinstance(target_segment, dict) else 'N/A'}")
        
        # Load the mean color image to get segment mask
        mean_color_files = [f for f in os.listdir(outputs_dir) 
                           if f.startswith(f"{file_id}_") and 'mean_colors' in f and f.endswith('.png')]
        if not mean_color_files:
            # Try alternative pattern matching
            mean_color_files = [f for f in os.listdir(outputs_dir) 
                               if file_id in f and 'mean_colors' in f and f.endswith('.png')]
            
        if not mean_color_files:
            raise Exception(f"Mean color image not found for file_id: {file_id}")
        
        # CRITICAL FIX: Always select the NEWEST mean color image by modification time
        mean_color_files_with_time = []
        for mean_color_file in mean_color_files:
            file_path = os.path.join(outputs_dir, mean_color_file)
            mod_time = os.path.getmtime(file_path)
            mean_color_files_with_time.append((mean_color_file, mod_time))
        
        # Sort by modification time (newest first)
        mean_color_files_with_time.sort(key=lambda x: x[1], reverse=True)
        newest_mean_color_file = mean_color_files_with_time[0][0]
        
        mean_color_path = os.path.join(outputs_dir, newest_mean_color_file)
        # print(f"DEBUG: Found {len(mean_color_files)} mean color files for file_id: {file_id}")
        # print(f"DEBUG: Available mean color files: {[f for f, _ in mean_color_files_with_time]}")
        # print(f"DEBUG: Selected NEWEST mean color file: {newest_mean_color_file}")
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
        
        # print(f"🔍 DEBUG: Found {np.sum(mask)} pixels for segment {segment_id}")
        # print(f"🔍 DEBUG: Target color: {target_color}")
        # print(f"🔍 DEBUG: Mean image shape: {mean_image.shape}")
        # print(f"🔍 DEBUG: Mask shape: {mask.shape}")
        # print(f"🔍 DEBUG: Mask has any True values: {np.any(mask)}")
        
        if np.sum(mask) == 0:
            print(f"[ERROR] No pixels found for segment {segment_id} with color {target_color}")
            return []
        
        # Analyze segment geometry and generate brush strokes with brush type
        brush_strokes = self._analyze_segment_and_create_strokes(mask, target_segment, brush_type, stroke_density)
        
        print(f"[SUCCESS] Generated {len(brush_strokes)} {brush_type} brush strokes for segment {segment_id}")
        
        return brush_strokes
    
    def _analyze_segment_and_create_strokes(self, mask, segment_info, brush_type, stroke_density):
        """
        Analyze segment geometry and create appropriate brush strokes
        
        Args:
            mask: Boolean mask of the segment
            segment_info: Segment information including color
            brush_type: Type of brush ('pencil')
            stroke_density: Density multiplier for stroke count (0.5-2.0)
            
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
        
        # Determine brush strategy based on geometry and brush type
        brush_strokes = []
        segment_color = f"#{segment_info['average_color']['r']:02x}{segment_info['average_color']['g']:02x}{segment_info['average_color']['b']:02x}"
        
        # Calculate appropriate brush width based on area with minimum width of 6
        base_stroke_width = max(6, int(math.sqrt(area) / 10))
        
        # Special handling for pencil and brush types (both use same stroke generation)
        if brush_type == 'pencil' or brush_type == 'brush':
            return self._generate_pencil_hatching_strokes(mask, segment_info['average_color'], base_stroke_width, stroke_density, brush_type)
        
        if area < 100:
            # Very small segments: single dot or short stroke with minimum width 6
            brush_strokes.append({
                'color': segment_color,
                'width': base_stroke_width,  # Use full width, not reduced
                'points': [
                    {'x': int(centroid[1]), 'y': int(centroid[0])}
                ],
                'type': brush_type
            })
            
        elif aspect_ratio > 3:
            # Long horizontal segments: horizontal strokes
            brush_strokes.extend(self._create_horizontal_strokes(mask, segment_color, base_stroke_width, brush_type))
            
        elif aspect_ratio < 0.33:
            # Long vertical segments: vertical strokes  
            brush_strokes.extend(self._create_vertical_strokes(mask, segment_color, base_stroke_width, brush_type))
            
        elif area < 500:
            # Small to medium segments: radial strokes from center
            brush_strokes.extend(self._create_radial_strokes(mask, segment_color, base_stroke_width, brush_type))
            
        else:
            # Large segments: combination of techniques
            if width > height:
                brush_strokes.extend(self._create_horizontal_strokes(mask, segment_color, base_stroke_width, brush_type))
            else:
                brush_strokes.extend(self._create_vertical_strokes(mask, segment_color, base_stroke_width, brush_type))
            
            # Add cross-hatching for texture
            brush_strokes.extend(self._create_cross_hatch_strokes(mask, segment_color, base_stroke_width // 2, brush_type))
        
        return brush_strokes
    
    def _generate_pencil_hatching_strokes(self, segment_mask, avg_color, base_stroke_width, stroke_density, brush_type='pencil'):
        """Generate pencil hatching strokes aligned with segment's longest axis"""
        strokes = []
        
        # print(f"🎨 DEBUG: Starting pencil stroke generation")
        # print(f"🎨 DEBUG: Segment mask shape: {segment_mask.shape}")
        # print(f"🎨 DEBUG: Mask has True values: {np.any(segment_mask)}")
        # print(f"🎨 DEBUG: Total True pixels: {np.sum(segment_mask)}")
        # print(f"🎨 DEBUG: Stroke density: {stroke_density}")
        
        # Get segment properties
        labeled_mask = measure.label(segment_mask.astype(int))
        regions = measure.regionprops(labeled_mask)
        
        # print(f"🎨 DEBUG: Found {len(regions)} regions in mask")
        
        if not regions:
            print(f"[ERROR] No regions found in segment mask!")
            return []
        
        props = regions[0]
        centroid = props.centroid
        orientation = props.orientation
        bbox = props.bbox
        
        # Calculate stroke parameters
        segment_height = bbox[2] - bbox[0]
        segment_width = bbox[3] - bbox[1]
        
        # Determine stroke spacing and count based on segment size
        area = props.area
        if area < 50:
            base_spacing = max(1, int(base_stroke_width * 0.5))
            base_strokes = max(8, int(area / 4))
        elif area < 200:
            base_spacing = max(1, int(base_stroke_width * 0.6))
            base_strokes = max(15, int(area / 8))
        else:
            base_spacing = max(1, int(base_stroke_width * 0.7))
            base_strokes = max(25, int(area / 10))
        
        # Apply density multiplier to increase stroke count
        # Allow more strokes for higher density values
        if stroke_density <= 2.0:
            max_strokes = 100
        elif stroke_density <= 5.0:
            max_strokes = 250
        else:  # stroke_density up to 10.0
            max_strokes = 500
        
        num_strokes = int(base_strokes * stroke_density)
        stroke_spacing = max(1, int(base_spacing / stroke_density))
        
        num_strokes = min(num_strokes, max_strokes)
        
        # Make pencil color slightly darker for visibility
        pencil_color = [
            max(0, int(float(avg_color['r']) * 0.9)),
            max(0, int(float(avg_color['g']) * 0.9)),
            max(0, int(float(avg_color['b']) * 0.9))
        ]
        
        # Calculate appropriate brush width based on area and brush type
        if brush_type == 'brush':
            stroke_width = base_stroke_width  # Full width for brush (minimum 6)
        else:  # pencil
            stroke_width = max(1, int(base_stroke_width * 0.8))  # Thinner for pencil effect
        
        # Generate primary hatching strokes along the major axis
        for i in range(num_strokes):
            # Add angle variation (±10 degrees)
            angle_variation = (np.random.random() - 0.5) * 20 * np.pi / 180
            stroke_angle = orientation + angle_variation
            
            dx = np.cos(stroke_angle)
            dy = np.sin(stroke_angle)
            
            # Find a starting point within the segment
            attempts = 0
            start_point = None
            
            while attempts < 10 and start_point is None:
                # Try different starting positions
                if segment_width > segment_height:
                    # For wide segments, start from left/right edges
                    start_y = bbox[0] + np.random.random() * segment_height
                    start_x = bbox[1] + (i / num_strokes) * segment_width + np.random.random() * stroke_spacing - stroke_spacing/2
                else:
                    # For tall segments, start from top/bottom edges
                    start_x = bbox[1] + np.random.random() * segment_width
                    start_y = bbox[0] + (i / num_strokes) * segment_height + np.random.random() * stroke_spacing - stroke_spacing/2
                
                # Ensure starting point is within bounds and in segment
                start_x = max(bbox[1], min(bbox[3]-1, start_x))
                start_y = max(bbox[0], min(bbox[2]-1, start_y))
                
                if (0 <= int(start_y) < segment_mask.shape[0] and 
                    0 <= int(start_x) < segment_mask.shape[1] and
                    segment_mask[int(start_y), int(start_x)]):
                    start_point = (start_x, start_y)
                
                attempts += 1
            
            if start_point is None:
                continue
            
            # Generate stroke points (2-5 points as requested)
            num_points = np.random.randint(2, 6)
            stroke_points = []
            
            current_x, current_y = start_point
            
            # Calculate step size to create stroke of appropriate length
            max_stroke_length = min(segment_width, segment_height) * 0.8
            step_size = max_stroke_length / (num_points - 1) if num_points > 1 else 0
            
            for point_idx in range(num_points):
                # Add the current point if it's within segment boundaries
                if (0 <= int(current_y) < segment_mask.shape[0] and 
                    0 <= int(current_x) < segment_mask.shape[1] and
                    segment_mask[int(current_y), int(current_x)]):
                    stroke_points.append({'x': current_x, 'y': current_y})
                
                # Move to next point along the stroke direction
                if point_idx < num_points - 1:
                    current_x += dx * step_size
                    current_y += dy * step_size
                    
                    # Ensure we don't go outside the bounding box
                    if (current_x < bbox[1] or current_x >= bbox[3] or
                        current_y < bbox[0] or current_y >= bbox[2]):
                        break
                        
                    if (int(current_y) >= segment_mask.shape[0] or 
                        int(current_x) >= segment_mask.shape[1] or
                        not segment_mask[int(current_y), int(current_x)]):
                        break
            
            # Only add stroke if we have at least 2 points
            if len(stroke_points) >= 2:
                strokes.append({
                    'color': f'rgb({pencil_color[0]}, {pencil_color[1]}, {pencil_color[2]})',
                    'width': stroke_width,
                    'points': stroke_points,
                    'type': brush_type
                })
        
        # Add multiple layers of fine detail strokes for maximum pencil density and realism
        if area > 50:
            # Layer 1: Fine detail strokes (very thin)
            fine_stroke_count = min(num_strokes, 25)
            
            for i in range(fine_stroke_count):
                # Fine stroke angle with more variation
                angle_variation = (np.random.random() - 0.5) * 40 * np.pi / 180
                stroke_angle = orientation + angle_variation
                
                dx = np.cos(stroke_angle)
                dy = np.sin(stroke_angle)
                
                # Random starting points for fine details
                attempts = 0
                start_point = None
                
                while attempts < 6 and start_point is None:
                    start_x = bbox[1] + np.random.random() * segment_width
                    start_y = bbox[0] + np.random.random() * segment_height
                    
                    if (0 <= int(start_y) < segment_mask.shape[0] and 
                        0 <= int(start_x) < segment_mask.shape[1] and
                        segment_mask[int(start_y), int(start_x)]):
                        start_point = (start_x, start_y)
                    
                    attempts += 1
                
                if start_point is None:
                    continue
                
                # Generate very short fine strokes (2-3 points)
                num_points = np.random.randint(2, 4)
                stroke_points = []
                current_x, current_y = start_point
                
                # Very short fine strokes
                fine_stroke_length = min(segment_width, segment_height) * 0.2
                step_size = fine_stroke_length / (num_points - 1) if num_points > 1 else 0
                
                for point_idx in range(num_points):
                    if (0 <= int(current_y) < segment_mask.shape[0] and 
                        0 <= int(current_x) < segment_mask.shape[1] and
                        segment_mask[int(current_y), int(current_x)]):
                        stroke_points.append({'x': current_x, 'y': current_y})
                    
                    if point_idx < num_points - 1:
                        current_x += dx * step_size
                        current_y += dy * step_size
                        
                        if (current_x < bbox[1] or current_x >= bbox[3] or
                            current_y < bbox[0] or current_y >= bbox[2]):
                            break
                        
                        if (int(current_y) >= segment_mask.shape[0] or 
                            int(current_x) >= segment_mask.shape[1] or
                            not segment_mask[int(current_y), int(current_x)]):
                            break
                
                if len(stroke_points) >= 2:
                    # Use different width based on brush type
                    if brush_type == 'brush':
                        fine_stroke_width = base_stroke_width  # Full width for brush
                    else:  # pencil
                        fine_stroke_width = max(1, int(base_stroke_width * 0.3))  # Very thin strokes for pencil
                    strokes.append({
                        'color': f'rgb({pencil_color[0]}, {pencil_color[1]}, {pencil_color[2]})',
                        'width': fine_stroke_width,
                        'points': stroke_points,
                        'type': brush_type
                    })
            
            # Layer 2: Micro detail strokes (extremely thin) for large segments
            if area > 200:
                micro_stroke_count = min(int(area / 15), 30)
                
                for i in range(micro_stroke_count):
                    # Micro stroke with high angle variation
                    angle_variation = (np.random.random() - 0.5) * 60 * np.pi / 180
                    stroke_angle = orientation + angle_variation
                    
                    dx = np.cos(stroke_angle)
                    dy = np.sin(stroke_angle)
                    
                    # Random micro positions
                    attempts = 0
                    start_point = None
                    
                    while attempts < 4 and start_point is None:
                        start_x = bbox[1] + np.random.random() * segment_width
                        start_y = bbox[0] + np.random.random() * segment_height
                        
                        if (0 <= int(start_y) < segment_mask.shape[0] and 
                            0 <= int(start_x) < segment_mask.shape[1] and
                            segment_mask[int(start_y), int(start_x)]):
                            start_point = (start_x, start_y)
                        
                        attempts += 1
                    
                    if start_point is None:
                        continue
                    
                    # Generate tiny micro strokes (2 points only)
                    num_points = 2
                    stroke_points = []
                    current_x, current_y = start_point
                    
                    # Tiny micro strokes
                    micro_stroke_length = min(segment_width, segment_height) * 0.1
                    step_size = micro_stroke_length
                    
                    # First point
                    stroke_points.append({'x': current_x, 'y': current_y})
                    
                    # Second point
                    current_x += dx * step_size
                    current_y += dy * step_size
                    
                    if (0 <= int(current_y) < segment_mask.shape[0] and 
                        0 <= int(current_x) < segment_mask.shape[1] and
                        segment_mask[int(current_y), int(current_x)] and
                        current_x >= bbox[1] and current_x < bbox[3] and
                        current_y >= bbox[0] and current_y < bbox[2]):
                        stroke_points.append({'x': current_x, 'y': current_y})
                    
                    if len(stroke_points) >= 2:
                        micro_stroke_width = 1  # Minimal thickness
                        strokes.append({
                            'color': f'rgb({pencil_color[0]}, {pencil_color[1]}, {pencil_color[2]})',
                            'width': micro_stroke_width,
                            'points': stroke_points,
                            'type': brush_type
                        })
        
        # Add cross-hatching for larger segments (more visible)
        if area > 60 and len(strokes) > 0:  # Even lower threshold for maximum cross-hatching
            cross_hatch_count = min(len(strokes) // 1.2, 30)  # Even more cross-hatching strokes
            
            # Add multiple layers of cross-hatching at different angles
            cross_angles = [
                np.pi/2,      # Perpendicular (90°)
                np.pi/3,      # 60 degrees
                2*np.pi/3,    # 120 degrees
                np.pi/4,      # 45 degrees
                3*np.pi/4     # 135 degrees
            ]
            
            for angle_offset in cross_angles:
                layer_count = max(1, int(cross_hatch_count / len(cross_angles)))
                
                for i in range(layer_count):
                    # Cross-hatch angle with variation
                    cross_angle = orientation + angle_offset + (np.random.random() - 0.5) * 10 * np.pi / 180
                    
                    dx = np.cos(cross_angle)
                    dy = np.sin(cross_angle)
                    
                    # Find starting point for cross-hatch
                    attempts = 0
                    start_point = None
                    
                    while attempts < 8 and start_point is None:
                        start_x = bbox[1] + np.random.random() * segment_width
                        start_y = bbox[0] + np.random.random() * segment_height
                        
                        if (0 <= int(start_y) < segment_mask.shape[0] and 
                            0 <= int(start_x) < segment_mask.shape[1] and
                            segment_mask[int(start_y), int(start_x)]):
                            start_point = (start_x, start_y)
                        
                        attempts += 1
                    
                    if start_point is None:
                        continue
                    
                    # Generate cross-hatch stroke (2-4 points for cross-hatching)
                    num_points = np.random.randint(2, 5)
                    stroke_points = []
                    current_x, current_y = start_point
                    
                    cross_stroke_length = min(segment_width, segment_height) * 0.6  # Longer cross-hatching
                    step_size = cross_stroke_length / (num_points - 1) if num_points > 1 else 0
                    
                    for point_idx in range(num_points):
                        if (0 <= int(current_y) < segment_mask.shape[0] and 
                            0 <= int(current_x) < segment_mask.shape[1] and
                            segment_mask[int(current_y), int(current_x)]):
                            stroke_points.append({'x': current_x, 'y': current_y})
                        
                        if point_idx < num_points - 1:
                            current_x += dx * step_size
                            current_y += dy * step_size
                            
                            # Check boundaries
                            if (current_x < bbox[1] or current_x >= bbox[3] or
                                current_y < bbox[0] or current_y >= bbox[2]):
                                break
                            
                            if (int(current_y) >= segment_mask.shape[0] or 
                                int(current_x) >= segment_mask.shape[1] or
                                not segment_mask[int(current_y), int(current_x)]):
                                break
                    
                    if len(stroke_points) >= 2:
                        # Use different width based on brush type
                        if brush_type == 'brush':
                            stroke_width = base_stroke_width  # Full width for brush
                        else:  # pencil
                            stroke_width = max(1, int(base_stroke_width * 0.7))  # Cross-hatch thickness for pencil
                        strokes.append({
                            'color': f'rgb({pencil_color[0]}, {pencil_color[1]}, {pencil_color[2]})',
                            'width': stroke_width,
                            'points': stroke_points,
                            'type': brush_type
                        })
        
        # print(f"🎨 DEBUG: Pencil stroke generation completed")
        # print(f"🎨 DEBUG: Total strokes generated: {len(strokes)}")
        # print(f"🎨 DEBUG: Stroke types: {[s.get('type', 'unknown') for s in strokes[:5]]}")  # Show first 5
        
        return strokes
    
    def _create_horizontal_strokes(self, mask, color, base_stroke_width, brush_type):
        """Create horizontal brush strokes across the segment"""
        strokes = []
        h, w = mask.shape
        
        # Find y coordinates where the segment exists
        y_coords = np.where(np.any(mask, axis=1))[0]
        
        # Create strokes with step calculated for width 6
        step = max(1, 6 // 3)
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
                        'width': base_stroke_width,
                        'points': points,
                        'type': brush_type
                    })
        
        return strokes
    
    def _create_vertical_strokes(self, mask, color, base_stroke_width, brush_type):
        """Create vertical brush strokes across the segment"""
        strokes = []
        h, w = mask.shape
        
        # Find x coordinates where the segment exists
        x_coords = np.where(np.any(mask, axis=0))[0]
        
        # Create strokes with step calculated for width 6
        step = max(1, 6 // 3)
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
                        'width': base_stroke_width,
                        'points': points,
                        'type': brush_type
                    })
        
        return strokes
    
    def _create_radial_strokes(self, mask, color, base_stroke_width, brush_type):
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
                    'width': base_stroke_width,
                    'points': points,
                    'type': brush_type
                })
        
        return strokes
    
    def _create_cross_hatch_strokes(self, mask, color, base_stroke_width, brush_type):
        """Create cross-hatching strokes for texture"""
        strokes = []
        h, w = mask.shape
        
        # Diagonal strokes (45 degrees) - spacing calculated for width 6
        for offset in range(-max(h, w), max(h, w), max(1, 6)):
            points = []
            for x in range(w):
                y = x + offset
                if 0 <= y < h and mask[y, x]:
                    points.append({'x': x, 'y': y})
            
            if len(points) > 2:
                strokes.append({
                    'color': color,
                    'width': base_stroke_width,
                    'points': points,
                    'type': brush_type
                })
        
        # Diagonal strokes (-45 degrees) for cross-hatching effect - spacing calculated for width 6
        for offset in range(-max(h, w), max(h, w), max(1, 6)):
            points = []
            for x in range(w):
                y = -x + offset + h
                if 0 <= y < h and mask[y, x]:
                    points.append({'x': x, 'y': y})
            
            if len(points) > 2:
                strokes.append({
                    'color': color,
                    'width': base_stroke_width,
                    'points': points,
                    'type': brush_type
                })
        
        return strokes

    def _generate_video_segments(self, image, color_threshold, detail_level):
        """
        Generate segments for video creation using the same logic as interactive mode
        Returns list of segment data with pixel counts and colors
        """
        # Use K-means clustering to create segments (same as interactive segmentation)
        h, w = image.shape[:2]
        
        # Calculate number of clusters based on color_threshold and detail_level
        base_clusters = max(4, min(16, int(100 / color_threshold * detail_level)))
        n_clusters = min(base_clusters, detail_level * 3)
        
        # Reshape image for K-means
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        segmented_image = labels.reshape(h, w)
        
        # Create segment data similar to interactive mode
        segments_data = []
        for segment_id in range(n_clusters):
            # Find pixels belonging to this segment
            mask = (segmented_image == segment_id)
            pixel_count = np.sum(mask)
            
            if pixel_count > 10:  # Only include segments with reasonable size
                # Calculate mean color for this segment
                segment_pixels = image[mask]
                mean_color = np.mean(segment_pixels, axis=0).astype(int)
                
                segments_data.append({
                    'id': segment_id,
                    'mask': mask,
                    'pixel_count': pixel_count,
                    'mean_color': mean_color,
                    'cluster_center': kmeans.cluster_centers_[segment_id]
                })
        
        return segments_data
    
    def _create_segment_based_animation(self, image, sorted_segments, frames_per_segment, total_frames, progress_callback):
        """
        Create animation frames by drawing segments one by one, following interactive drawing logic
        """
        h, w = image.shape[:2]
        frames = []
        
        # Start with cream background (same as interactive mode)
        canvas = np.full((h, w, 3), [255, 243, 205], dtype=np.uint8)  # Cream color #fff3cd
        
        # Add initial preview frames (empty canvas)
        preview_frames = min(15, total_frames // 10)  # Show empty canvas for ~0.5 seconds
        for _ in range(preview_frames):
            frames.append(canvas.copy())
        
        # Calculate remaining frames for actual drawing
        drawing_frames = total_frames - preview_frames - 30  # Reserve 30 frames for final hold
        frames_per_segment = max(1, drawing_frames // len(sorted_segments)) if sorted_segments else 1
        
        current_frame = 0
        total_segments = len(sorted_segments)
        
        for segment_idx, segment_data in enumerate(sorted_segments):
            if progress_callback and segment_idx % 5 == 0:  # Update progress every 5 segments
                progress = 50 + int(35 * segment_idx / total_segments)
                progress_callback(progress, f"Drawing segment {segment_idx + 1}/{total_segments}...")
            
            # Generate brush strokes for this segment
            segment_strokes = self._generate_segment_strokes_for_video(image, segment_data)
            
            # Create frames showing progressive drawing of this segment
            segment_frames = self._create_segment_drawing_frames(
                canvas, segment_strokes, frames_per_segment
            )
            
            # Add segment frames to total frames
            frames.extend(segment_frames)
            
            # Update canvas with completed segment
            if segment_strokes:
                canvas = segment_frames[-1].copy()
            
            current_frame += len(segment_frames)
        
        # Add final hold frames (show completed drawing for 1 second)
        for _ in range(30):
            frames.append(canvas.copy())
        
        return frames
    
    def _generate_segment_strokes_for_video(self, image, segment_data):
        """
        Generate brush strokes for a single segment (for video generation)
        Uses the same logic as the interactive drawing but works directly with image data
        """
        mask = segment_data['mask']
        mean_color = segment_data['mean_color']
        
        # Use the same stroke generation logic as interactive mode
        # Default stroke density of 1.0 for video
        stroke_density = 1.0
        
        # Create a mock segment object similar to what _analyze_segment_and_create_strokes expects
        # Ensure color values are properly formatted as integers
        try:
            # Convert numpy values to Python integers safely
            r_val = int(np.round(float(mean_color[0])))
            g_val = int(np.round(float(mean_color[1])))
            b_val = int(np.round(float(mean_color[2])))
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Error parsing color values {mean_color}: {e}")
            # Use default gray color if parsing fails
            r_val, g_val, b_val = 128, 128, 128
        
        mock_segment = {
            'id': segment_data['id'],
            'pixel_count': segment_data['pixel_count'],
            'average_color': {
                'r': r_val,
                'g': g_val,
                'b': b_val
            }
        }
        
        # Generate strokes using the existing method that works with masks directly
        strokes = self._analyze_segment_and_create_strokes(mask, mock_segment, 'pencil', stroke_density)
        
        return strokes
    
    def _create_segment_drawing_frames(self, canvas, strokes, num_frames):
        """
        Create frames showing progressive drawing of a single segment
        """
        frames = []
        
        if not strokes or num_frames <= 0:
            frames.append(canvas.copy())
            return frames
        
        # Group strokes by type for proper layering
        stroke_groups = {}
        for stroke in strokes:
            stroke_type = stroke.get('type', 'pencil')
            if stroke_type not in stroke_groups:
                stroke_groups[stroke_type] = []
            stroke_groups[stroke_type].append(stroke)
        
        # Calculate strokes per frame
        total_strokes = len(strokes)
        strokes_per_frame = max(1, total_strokes // num_frames)
        
        current_canvas = canvas.copy()
        stroke_index = 0
        
        for frame_idx in range(num_frames):
            frame_canvas = current_canvas.copy()
            
            # Add strokes for this frame
            strokes_to_add = min(strokes_per_frame, total_strokes - stroke_index)
            
            for i in range(strokes_to_add):
                if stroke_index < total_strokes:
                    stroke = strokes[stroke_index]
                    self._apply_stroke_to_canvas(frame_canvas, stroke)
                    stroke_index += 1
            
            frames.append(frame_canvas.copy())
            current_canvas = frame_canvas.copy()
        
        return frames
    
    def _apply_stroke_to_canvas(self, canvas, stroke):
        """
        Apply a single brush stroke to the canvas
        """
        if 'points' not in stroke or len(stroke['points']) == 0:
            return
        
        color = stroke.get('color', [0, 0, 0])
        width = max(1, int(stroke.get('width', 2)))
        points = stroke['points']
        
        # Convert color to BGR for OpenCV
        bgr_color = tuple(map(int, color[::-1]))
        
        if len(points) == 1:
            # Single point - draw a dot
            x, y = int(points[0][0]), int(points[0][1])
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                cv2.circle(canvas, (x, y), width, bgr_color, -1)
        else:
            # Multiple points - draw lines
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
                
                # Check bounds
                if (0 <= pt1[0] < canvas.shape[1] and 0 <= pt1[1] < canvas.shape[0] and
                    0 <= pt2[0] < canvas.shape[1] and 0 <= pt2[1] < canvas.shape[0]):
                    cv2.line(canvas, pt1, pt2, bgr_color, width)

    def create_video_from_canvas_frames(self, frames, file_id, duration, frame_rate):
        """
        Create video from captured canvas frames
        
        Args:
            frames: List of frame data with dataURL and timestamp
            file_id: File identifier for naming
            duration: Target video duration in seconds
            frame_rate: Target frame rate
            
        Returns:
            Path to created video file
        """
        import base64
        from io import BytesIO
        from PIL import Image
        
        try:
            output_path = os.path.join(self.output_dir, f"{file_id}_canvas_video.mp4")
            
            # Convert frame data URLs to PIL Images
            pil_frames = []
            for frame_data in frames:
                try:
                    # Extract base64 data from data URL
                    data_url = frame_data['dataURL']
                    if data_url.startswith('data:image/jpeg;base64,'):
                        base64_data = data_url.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        
                        # Convert to PIL Image
                        pil_image = Image.open(BytesIO(image_data))
                        
                        # Convert to RGB if necessary
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        pil_frames.append(pil_image)
                        
                except Exception as e:
                    print(f"Warning: Failed to process frame: {e}")
                    continue
            
            if not pil_frames:
                print("Error: No valid frames to create video")
                return None
            
            print(f"Creating video from {len(pil_frames)} canvas frames...")
            
            # Get frame dimensions from first frame
            width, height = pil_frames[0].size
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
            
            if not video_writer.isOpened():
                print("Error: Could not open video writer")
                return None
            
            # Calculate how many times to repeat each frame to match target duration
            target_total_frames = int(duration * frame_rate)
            frame_repeat_count = max(1, target_total_frames // len(pil_frames))
            
            # Write frames to video
            frames_written = 0
            for pil_frame in pil_frames:
                # Convert PIL image to OpenCV format (BGR)
                opencv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                
                # Write frame multiple times to achieve target duration
                for _ in range(frame_repeat_count):
                    video_writer.write(opencv_frame)
                    frames_written += 1
                    
                    # Stop if we've reached target duration
                    if frames_written >= target_total_frames:
                        break
                
                if frames_written >= target_total_frames:
                    break
            
            # Fill remaining time if needed
            if frames_written < target_total_frames and pil_frames:
                last_frame = cv2.cvtColor(np.array(pil_frames[-1]), cv2.COLOR_RGB2BGR)
                for _ in range(target_total_frames - frames_written):
                    video_writer.write(last_frame)
            
            video_writer.release()
            
            print(f"Video created successfully: {output_path}")
            print(f"Total frames written: {frames_written}, Target: {target_total_frames}")
            
            return output_path
            
        except Exception as e:
            print(f"Error creating video from canvas frames: {e}")
            import traceback
            traceback.print_exc()
            return None

    def detect_contrast_boundaries(self, input_path, output_dir, file_id, sensitivity=50, fragmentation=50, progress_callback=None):
        """
        Detect contrast boundaries based on segmentation results with adjustable sensitivity and fragmentation
        
        Args:
            input_path: Path to input image
            output_dir: Directory to save boundary data
            file_id: Unique file identifier
            sensitivity: Sensitivity parameter (1-100, higher = more sensitive to weak boundaries)
            fragmentation: Boundary fragmentation level (1-100, higher = more fragments)
            progress_callback: Function to call with progress updates
            
        Returns:
            List of boundary fragment data with coordinates and individual contrast values
        """
        if progress_callback:
            progress_callback(5, "Loading image for boundary detection...")
        
        # CRITICAL FIX: Use the same image dimensions as stroke generation
        # Load the mean color image to ensure coordinate system consistency
        mean_color_files = [f for f in os.listdir(output_dir) 
                           if f.startswith(f"{file_id}_") and 'mean_colors' in f and f.endswith('.png')]
        if not mean_color_files:
            # Fallback to original image preprocessing if mean color image not found
            image = self._load_and_preprocess_image(input_path)
        else:
            # Use the newest mean color image for consistent coordinate system
            mean_color_files_with_time = []
            for mean_color_file in mean_color_files:
                file_path = os.path.join(output_dir, mean_color_file)
                mod_time = os.path.getmtime(file_path)
                mean_color_files_with_time.append((mean_color_file, mod_time))
            
            # Sort by modification time (newest first)
            mean_color_files_with_time.sort(key=lambda x: x[1], reverse=True)
            newest_mean_color_file = mean_color_files_with_time[0][0]
            
            mean_color_path = os.path.join(output_dir, newest_mean_color_file)
            print(f"Using mean color image for boundary detection: {mean_color_path}")
            
            # Load mean color image to ensure same coordinate system as strokes
            image = cv2.imread(mean_color_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"🔍 DEBUG: Boundary detection using mean color image size: {image.shape[:2]} (H x W)")
        
        if progress_callback:
            progress_callback(15, "Generating segmentation for boundary detection...")
        
        # Generate segmentation to get fragments as base for boundary detection
        segment_mask = self._segment_by_color_similarity(image, threshold=20)
        
        # Convert segment mask to list of segments with coordinates
        segments = self._convert_mask_to_segments(segment_mask)
        
        if progress_callback:
            progress_callback(40, f"Generated {len(segments)} segments, detecting boundaries...")
        
        # Detect boundaries between segments with fragmentation
        # print(f"DEBUG: detect_contrast_boundaries called with fragmentation={fragmentation}, type={type(fragmentation)}")
        boundaries = self._detect_segment_boundaries(image, segments, sensitivity, fragmentation)
        
        if progress_callback:
            progress_callback(60, f"Found {len(boundaries)} boundary fragments, analyzing colors...")
        
        # print(f"DEBUG: Fragmentation={fragmentation}, Total boundaries after fragmentation: {len(boundaries)}")
        
        # Analyze contrast for each boundary fragment
        boundary_data = self._analyze_fragment_contrast(image, boundaries, segments, sensitivity)
        
        if progress_callback:
            progress_callback(80, "Saving boundary data...")
        
        # Save boundary information
        boundary_filename = f"{file_id}_boundaries_s{sensitivity}.json"
        boundary_path = os.path.join(output_dir, boundary_filename)
        
        boundary_info = {
            'sensitivity': sensitivity,
            'total_boundaries': len(boundary_data),
            'boundaries': boundary_data
        }
        
        with open(boundary_path, 'w') as f:
            json.dump(boundary_info, f, indent=2)
        
        if progress_callback:
            progress_callback(100, f"Found {len(boundary_data)} contrast boundaries!")
        
        return boundary_data
    
    def _convert_mask_to_segments(self, segment_mask):
        """
        Convert segment mask to list of segments with coordinates
        
        Args:
            segment_mask: 2D array with segment IDs
            
        Returns:
            List of segments with coordinates
        """
        segments = []
        unique_ids = np.unique(segment_mask)
        
        for segment_id in unique_ids:
            if segment_id == 0:  # Skip background
                continue
                
            # Find all coordinates for this segment
            y_coords, x_coords = np.where(segment_mask == segment_id)
            coordinates = [(int(y), int(x)) for y, x in zip(y_coords, x_coords)]
            
            if len(coordinates) > 10:  # Minimum segment size
                segments.append({
                    'id': int(segment_id),
                    'coordinates': coordinates,
                    'pixel_count': len(coordinates)
                })
        
        return segments
    
    def _detect_segment_boundaries(self, image, segments, sensitivity, fragmentation):
        """
        Detect boundaries between image segments with adjustable sensitivity and fragmentation
        
        Args:
            image: Input image (RGB format)
            segments: List of image segments from segmentation
            sensitivity: Detection sensitivity (1-100, higher = detect weaker boundaries)
            fragmentation: Fragmentation level (1-100, higher = more fragments)
            
        Returns:
            List of boundary contours
        """
        height, width = image.shape[:2]
        
        # Create segment mask for each segment
        segment_masks = {}
        for i, segment in enumerate(segments):
            mask = np.zeros((height, width), dtype=np.uint8)
            for y, x in segment['coordinates']:
                if 0 <= y < height and 0 <= x < width:
                    mask[y, x] = 255
            segment_masks[i] = mask
        
        boundaries = []
        boundary_id_counter = 0  # Global counter for unique boundary IDs
        original_boundaries_count = 0  # Track original boundaries before fragmentation
        
        # print(f"DEBUG: Starting boundary detection with fragmentation={fragmentation}")
        
        # Find boundaries between adjacent segments
        for i, segment_i in enumerate(segments):
            # Find contours of current segment
            contours_i, _ = cv2.findContours(segment_masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours_i:
                contour_length = cv2.arcLength(contour, False)
                
                # Filter by minimum length based on sensitivity
                min_length = max(15, int(40 * (1 - sensitivity / 100.0)))  # Lower min for high sensitivity
                if contour_length < min_length:
                    continue
                
                # Smooth contour based on sensitivity
                # Keep contours detailed regardless of sensitivity - sensitivity should only affect contrast filtering
                # Use minimal smoothing to preserve detail
                smoothing_factor = 0.002  # Very minimal smoothing for all sensitivity levels
                epsilon = smoothing_factor * contour_length
                smoothed = cv2.approxPolyDP(contour, epsilon, False)
                
                # Count original boundary before fragmentation
                original_boundaries_count += 1
                
                # NEW LOGIC: Find the most contrasted fragment of each boundary
                if fragmentation > 1:
                    # Generate all possible fragments for contrast analysis
                    all_fragments = self._fragment_boundary_into_pieces(smoothed, fragmentation)
                    # print(f"DEBUG: Generated {len(all_fragments)} fragments for contrast analysis")
                    
                    # Find the fragment with highest contrast
                    best_fragment = self._find_highest_contrast_fragment(image, all_fragments)
                    if best_fragment is not None:
                        boundary_with_id = (best_fragment, boundary_id_counter)
                        boundaries.append(boundary_with_id)
                        boundary_id_counter += 1
                        # print(f"DEBUG: Selected best fragment from {len(all_fragments)} candidates")
                    # else:
                        # print(f"DEBUG: No suitable fragment found, skipping boundary")
                else:
                    # Single boundary gets one ID (fragmentation = 1)
                    boundary_with_id = (smoothed, boundary_id_counter)
                    boundaries.append(boundary_with_id)
                    boundary_id_counter += 1
        
        # print(f"DEBUG: Fragmentation complete. Original boundaries: {original_boundaries_count}, Final fragments: {len(boundaries)}")
        # print(f"DEBUG: Expected fragments with fragmentation {fragmentation}: {original_boundaries_count * fragmentation}")
        
        return boundaries
    
    def _fragment_boundary_into_pieces(self, contour, num_fragments):
        """
        Split a boundary contour into exactly num_fragments pieces based on arc length
        
        Args:
            contour: OpenCV contour to fragment
            num_fragments: Number of fragments to create (2-6)
            
        Returns:
            List of contour fragments
        """
        if num_fragments <= 1:
            return [contour]
            
        # Convert contour to simple point list
        contour_points = []
        for point in contour:
            contour_points.append([int(point[0][0]), int(point[0][1])])
        
        if len(contour_points) < 2:
            return [contour]
        
        # Calculate cumulative distances along the contour
        cumulative_distances = [0.0]
        total_length = 0.0
        
        for i in range(1, len(contour_points)):
            p1 = contour_points[i-1]
            p2 = contour_points[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
            cumulative_distances.append(total_length)
        
        if total_length == 0:
            return [contour]
        
        # Calculate fragment boundaries based on equal arc length
        fragment_length = total_length / num_fragments
        fragments = []
        
        for i in range(num_fragments):
            start_distance = i * fragment_length
            end_distance = (i + 1) * fragment_length if i < num_fragments - 1 else total_length
            
            # Find start and end indices
            start_idx = 0
            end_idx = len(contour_points) - 1
            
            # Find start index
            for j in range(len(cumulative_distances)):
                if cumulative_distances[j] >= start_distance:
                    start_idx = j
                    break
            
            # Find end index
            for j in range(start_idx, len(cumulative_distances)):
                if cumulative_distances[j] >= end_distance:
                    end_idx = j
                    break
            
            # Ensure we have at least 2 points per fragment
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, len(contour_points) - 1)
            
            # Extract fragment points
            fragment_points = contour_points[start_idx:end_idx + 1]
            
            if len(fragment_points) >= 2:
                # Convert back to OpenCV contour format
                fragment_contour = np.array(fragment_points).reshape(-1, 1, 2).astype(np.int32)
                fragments.append(fragment_contour)
        
        return fragments if fragments else [contour]
    
    def _find_highest_contrast_fragment(self, image, fragments):
        """
        Find the fragment with the highest contrast from a list of fragments
        
        Args:
            image: Original image for contrast analysis
            fragments: List of contour fragments to analyze
            
        Returns:
            The fragment with highest contrast, or None if no suitable fragment found
        """
        if not fragments:
            return None
        
        best_fragment = None
        highest_contrast = 0.0
        
        for fragment in fragments:
            try:
                # Calculate contrast for this fragment
                contrast_ratio = self._calculate_fragment_contrast(image, fragment)
                
                if contrast_ratio > highest_contrast:
                    highest_contrast = contrast_ratio
                    best_fragment = fragment
                    
            except Exception as e:
                print(f"DEBUG: Error analyzing fragment contrast: {e}")
                continue
        
        # print(f"DEBUG: Best fragment has contrast ratio: {highest_contrast:.3f}")
        return best_fragment
    
    def _calculate_fragment_contrast(self, image, contour):
        """
        Calculate contrast ratio for a single contour fragment
        
        Args:
            image: Original image
            contour: Single contour fragment
            
        Returns:
            Contrast ratio (0.0 to 1.0+)
        """
        if len(contour) < 2:
            return 0.0
        
        # Sample colors along the contour
        left_colors = []
        right_colors = []
        
        # Sample every few points along the contour
        sample_step = max(1, len(contour) // 10)  # Sample ~10 points
        
        for i in range(0, len(contour), sample_step):
            point = contour[i][0] if len(contour[i].shape) > 1 else contour[i]
            x, y = int(point[0]), int(point[1])
            
            # Skip points near image edges
            if x < 3 or y < 3 or x >= image.shape[1] - 3 or y >= image.shape[0] - 3:
                continue
            
            # Calculate perpendicular direction for sampling
            if i < len(contour) - 1:
                next_point = contour[i + 1][0] if len(contour[i + 1].shape) > 1 else contour[i + 1]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                
                # Perpendicular direction (rotate 90 degrees)
                perp_dx = -dy
                perp_dy = dx
                
                # Normalize
                length = np.sqrt(perp_dx**2 + perp_dy**2)
                if length > 0:
                    perp_dx /= length
                    perp_dy /= length
                    
                    # Sample on both sides of the boundary
                    offset = 3
                    left_x = int(x + perp_dx * offset)
                    left_y = int(y + perp_dy * offset)
                    right_x = int(x - perp_dx * offset)
                    right_y = int(y - perp_dy * offset)
                    
                    # Check bounds and sample colors
                    if (0 <= left_x < image.shape[1] and 0 <= left_y < image.shape[0] and
                        0 <= right_x < image.shape[1] and 0 <= right_y < image.shape[0]):
                        left_colors.append(image[left_y, left_x])
                        right_colors.append(image[right_y, right_x])
        
        if len(left_colors) == 0 or len(right_colors) == 0:
            return 0.0
        
        # Calculate average colors
        left_avg = np.mean(left_colors, axis=0)
        right_avg = np.mean(right_colors, axis=0)
        
        # Calculate contrast ratio
        left_brightness = np.mean(left_avg)
        right_brightness = np.mean(right_avg)
        
        if min(left_brightness, right_brightness) == 0:
            return 0.0
        
        contrast_ratio = abs(left_brightness - right_brightness) / max(left_brightness, right_brightness)
        return contrast_ratio
    
    def _analyze_fragment_contrast(self, image, boundaries, segments, sensitivity):
        """
        Analyze contrast for each boundary fragment based on local background/object colors
        
        Args:
            image: Input image (RGB format)
            boundaries: List of (contour, boundary_id) tuples
            segments: List of image segments
            sensitivity: Detection sensitivity
            
        Returns:
            List of boundary data with individual contrast values
        """
        boundary_data = []
        
        for contour, boundary_id in boundaries:
            # Convert contour to list of points
            contour_points = []
            for point in contour:
                x, y = int(point[0][0]), int(point[0][1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    contour_points.append([x, y])
            
            if len(contour_points) < 3:
                continue
            
            # Sample points along the boundary for color analysis
            sample_indices = np.linspace(0, len(contour_points) - 1, min(20, len(contour_points)), dtype=int)
            sampled_points = [contour_points[i] for i in sample_indices]
            
            # Analyze colors on both sides of the boundary
            left_colors = []
            right_colors = []
            
            for i in range(len(sampled_points) - 1):
                p1 = sampled_points[i]
                p2 = sampled_points[i + 1]
                
                # Calculate perpendicular direction for sampling
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Normalize and get perpendicular
                    nx = -dy / length  # perpendicular x
                    ny = dx / length   # perpendicular y
                    
                    # Sample colors on both sides
                    offset = 3  # pixels to offset from boundary
                    
                    # Left side
                    left_x = int(p1[0] + nx * offset)
                    left_y = int(p1[1] + ny * offset)
                    if 0 <= left_x < image.shape[1] and 0 <= left_y < image.shape[0]:
                        left_colors.append(image[left_y, left_x])
                    
                    # Right side
                    right_x = int(p1[0] - nx * offset)
                    right_y = int(p1[1] - ny * offset)
                    if 0 <= right_x < image.shape[1] and 0 <= right_y < image.shape[0]:
                        right_colors.append(image[right_y, right_x])
            
            if not left_colors or not right_colors:
                continue
            
            # Calculate average colors on each side
            left_avg = np.mean(left_colors, axis=0)
            right_avg = np.mean(right_colors, axis=0)
            
            # Calculate local contrast ratio
            left_brightness = np.mean(left_avg)
            right_brightness = np.mean(right_avg)
            contrast_ratio = abs(left_brightness - right_brightness) / 255.0
            
            # Filter by contrast threshold based on sensitivity
            if sensitivity >= 80:
                min_contrast = 0.02  # Very low threshold - detect very subtle boundaries
            elif sensitivity >= 60:
                min_contrast = 0.05  # Low threshold - detect weak boundaries
            elif sensitivity >= 40:
                min_contrast = 0.1   # Medium threshold - moderate boundaries
            elif sensitivity >= 20:
                min_contrast = 0.2   # Higher threshold - stronger boundaries
            else:
                min_contrast = 0.4   # High threshold - only very strong boundaries
            
            if contrast_ratio >= min_contrast:
                # Determine brightest and darkest colors
                if left_brightness > right_brightness:
                    brightest_color = left_avg
                    darkest_color = right_avg
                else:
                    brightest_color = right_avg
                    darkest_color = left_avg
                
                # ENHANCEMENT: Enhance contrast by making bright colors brighter and dark colors darker
                # Make brightest color even brighter
                brightest_color = np.array([
                    min(255, int(brightest_color[0] * 1.2)),  # Increase brightness by 20%
                    min(255, int(brightest_color[1] * 1.2)),
                    min(255, int(brightest_color[2] * 1.2))
                ])
                
                # Make darkest color even darker
                darkest_color = np.array([
                    max(0, int(darkest_color[0] * 0.8)),  # Decrease brightness by 20%
                    max(0, int(darkest_color[1] * 0.8)),
                    max(0, int(darkest_color[2] * 0.8))
                ])
                
                # Calculate contour length
                contour_length = cv2.arcLength(contour, False)
                
                boundary_data.append({
                    'id': boundary_id,  # Use the unique boundary ID
                    'contour_points': contour_points,
                    'brightest_color': {
                        'r': int(brightest_color[0]),
                        'g': int(brightest_color[1]),
                        'b': int(brightest_color[2]),
                        'hex': f"#{int(brightest_color[0]):02x}{int(brightest_color[1]):02x}{int(brightest_color[2]):02x}"
                    },
                    'darkest_color': {
                        'r': int(darkest_color[0]),
                        'g': int(darkest_color[1]),
                        'b': int(darkest_color[2]),
                        'hex': f"#{int(darkest_color[0]):02x}{int(darkest_color[1]):02x}{int(darkest_color[2]):02x}"
                    },
                    'contrast_ratio': float(contrast_ratio),
                    'length': float(contour_length),
                    'point_count': len(contour_points),
                    'left_avg_color': {
                        'r': int(left_avg[0]),
                        'g': int(left_avg[1]),
                        'b': int(left_avg[2])
                    },
                    'right_avg_color': {
                        'r': int(right_avg[0]),
                        'g': int(right_avg[1]),
                        'b': int(right_avg[2])
                    }
                })
        
        # Sort boundaries by length (longest first) as requested
        boundary_data.sort(key=lambda x: x['length'], reverse=True)
        
        # With fragmentation, we want to keep ALL fragments that pass contrast filtering
        # Don't limit the number of boundaries when fragmentation is used
        # print(f"DEBUG: _analyze_fragment_contrast processed {len(boundary_data)} boundaries after contrast filtering")
        
        return boundary_data
    
    def _detect_image_boundaries(self, image, sensitivity):
        """
        Detect boundaries using edge detection algorithms with improved sensitivity for multiple boundaries
        
        Args:
            image: Input RGB image
            sensitivity: Sensitivity parameter (1-100)
            
        Returns:
            List of boundary contours
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise - less blur for higher sensitivity to preserve detail
        if sensitivity >= 80:
            blur_size = 1  # Minimal blur for very high sensitivity - preserve fine details
        elif sensitivity >= 60:
            blur_size = 3  # Light blur for high sensitivity
        elif sensitivity >= 40:
            blur_size = 5  # Medium blur for medium sensitivity
        else:
            blur_size = 7  # More blur for low sensitivity - remove noise
        
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Calculate Canny thresholds based on sensitivity
        # Higher sensitivity = lower thresholds = detect weaker/subtle boundaries
        if sensitivity >= 80:
            # Very high sensitivity - detect even very weak boundaries
            low_threshold = max(5, 15 - int((sensitivity - 80) / 4))
            high_threshold = max(10, 30 - int((sensitivity - 80) / 2))
        elif sensitivity >= 60:
            # High sensitivity - detect weak boundaries
            low_threshold = max(10, 25 - int((sensitivity - 60) / 4))
            high_threshold = max(20, 50 - int((sensitivity - 60) / 2))
        elif sensitivity >= 40:
            # Medium sensitivity - balanced detection
            low_threshold = max(20, 40 - int((sensitivity - 40) / 2))
            high_threshold = max(40, 80 - (sensitivity - 40))
        else:
            # Low sensitivity - only strong boundaries
            low_threshold = max(30, 60 - sensitivity / 2)
            high_threshold = max(60, 120 - sensitivity)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Apply morphological operations to connect nearby edges
        # Smaller kernel for higher sensitivity to preserve more boundaries
        kernel_size = 3 if sensitivity >= 60 else 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (boundaries) - use RETR_LIST to get more boundaries
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and smooth contours based on sensitivity
        # Higher sensitivity = accept shorter boundaries (find more detail)
        min_contour_length = max(10, 50 - int(sensitivity / 2))  # Shorter boundaries for higher sensitivity
        max_boundaries = min(50, max(5, int(sensitivity / 2)))  # More boundaries for higher sensitivity
        
        filtered_contours = []
        
        for contour in contours:
            contour_length = cv2.arcLength(contour, False)
            if contour_length >= min_contour_length:
                # Smooth the contour to reduce jagged polyline appearance
                # Higher sensitivity = less smoothing to preserve detail
                epsilon = 0.005 * contour_length  # Base approximation accuracy
                if sensitivity >= 70:
                    epsilon *= 0.3  # Much less smoothing for high sensitivity - preserve detail
                elif sensitivity >= 50:
                    epsilon *= 0.6  # Less smoothing for medium-high sensitivity
                elif sensitivity <= 30:
                    epsilon *= 1.5  # More smoothing for low sensitivity
                
                # Apply Douglas-Peucker algorithm to smooth the contour
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, False)
                
                # Only keep contours with reasonable number of points after smoothing
                if len(smoothed_contour) >= 3:
                    # Split long boundaries into fragments for easier individual drawing
                    fragments = self._split_boundary_into_fragments(smoothed_contour, contour_length)
                    filtered_contours.extend(fragments)
        
        # Sort by contour length (longest first) and limit number of boundaries
        filtered_contours.sort(key=lambda c: cv2.arcLength(c, False), reverse=True)
        return filtered_contours[:max_boundaries]
    
    def _split_boundary_into_fragments(self, contour, contour_length):
        """
        Split long boundaries into smaller fragments for easier individual drawing
        
        Args:
            contour: OpenCV contour points
            contour_length: Length of the contour
            
        Returns:
            List of contour fragments
        """
        # Define maximum fragment length based on contour length
        if contour_length > 1000:
            max_fragment_length = 300  # Split very long boundaries into ~300px fragments
        elif contour_length > 500:
            max_fragment_length = 200  # Split long boundaries into ~200px fragments
        else:
            return [contour]  # Keep shorter boundaries as single fragments
        
        fragments = []
        contour_points = contour.reshape(-1, 2)  # Convert to (N, 2) array
        
        if len(contour_points) < 4:
            return [contour]  # Too few points to split
        
        # Calculate cumulative distances along the contour
        distances = [0]
        for i in range(1, len(contour_points)):
            dist = np.linalg.norm(contour_points[i] - contour_points[i-1])
            distances.append(distances[-1] + dist)
        
        total_length = distances[-1]
        if total_length <= max_fragment_length:
            return [contour]  # No need to split
        
        # Calculate split points
        num_fragments = int(np.ceil(total_length / max_fragment_length))
        fragment_length = total_length / num_fragments
        
        current_fragment_start = 0
        
        for fragment_idx in range(num_fragments):
            # Find start and end points for this fragment
            start_distance = fragment_idx * fragment_length
            end_distance = min((fragment_idx + 1) * fragment_length, total_length)
            
            # Find indices corresponding to these distances
            start_idx = current_fragment_start
            end_idx = len(contour_points) - 1
            
            # Find the end index for this fragment
            for i in range(current_fragment_start, len(distances)):
                if distances[i] >= end_distance:
                    end_idx = i
                    break
            
            # Ensure we have at least 3 points for a valid fragment
            if end_idx - start_idx >= 2:
                fragment_points = contour_points[start_idx:end_idx + 1]
                
                # Add some overlap between fragments for continuity (except for the last fragment)
                if fragment_idx < num_fragments - 1 and end_idx < len(contour_points) - 1:
                    # Add 1-2 extra points for overlap
                    overlap_end = min(end_idx + 2, len(contour_points))
                    fragment_points = contour_points[start_idx:overlap_end]
                
                # Convert back to OpenCV contour format
                fragment_contour = fragment_points.reshape(-1, 1, 2).astype(np.int32)
                fragments.append(fragment_contour)
                
                # Update start for next fragment (with some overlap)
                current_fragment_start = max(start_idx, end_idx - 1)
            
        return fragments if fragments else [contour]
    
    
    def draw_single_boundary(self, output_dir, file_id, boundary_id, color_type='brightest', sensitivity=50):
        """
        Draw a single boundary line using the specified color
        
        Args:
            output_dir: Directory containing boundary data
            file_id: File identifier
            boundary_id: ID of the boundary to draw
            color_type: 'brightest' or 'darkest' color to use
            sensitivity: Sensitivity parameter used for boundary detection
            
        Returns:
            Stroke data for frontend rendering
        """
        # Load boundary data
        boundary_filename = f"{file_id}_boundaries_s{sensitivity}.json"
        boundary_path = os.path.join(output_dir, boundary_filename)
        
        if not os.path.exists(boundary_path):
            raise Exception(f"Boundary data not found: {boundary_filename}")
        
        with open(boundary_path, 'r') as f:
            boundary_info = json.load(f)
        
        # Find the target boundary
        target_boundary = None
        for boundary in boundary_info['boundaries']:
            if boundary['id'] == boundary_id:
                target_boundary = boundary
                break
        
        if not target_boundary:
            raise Exception(f"Boundary {boundary_id} not found")
        
        # Select color based on type
        if color_type == 'brightest':
            color_data = target_boundary['brightest_color']
        else:  # darkest
            color_data = target_boundary['darkest_color']
        
        # Create stroke data
        stroke_points = []
        for x, y in target_boundary['contour_points']:
            stroke_points.append({'x': float(x), 'y': float(y)})
        
        # Calculate appropriate line width based on boundary length
        base_width = max(1, min(4, int(target_boundary['length'] / 100)))
        
        stroke_data = {
            'color': f"rgb({color_data['r']}, {color_data['g']}, {color_data['b']})",
            'width': base_width,
            'points': stroke_points,
            'type': 'boundary_line',
            'boundary_id': boundary_id,
            'color_type': color_type,
            'contrast_ratio': target_boundary['contrast_ratio']
        }
        
        return [stroke_data]  # Return as list for consistency with other stroke methods

    def highlight_contrast_boundaries(self, output_dir, file_id, sensitivity=50):
        """
        Highlight boundaries using contrast-based sorting and dynamic line thickness
        1. Sort fragments by contrast and draw only the most contrasted half
        2. Draw with dominant color (enhanced dark or light)
        3. Variable thickness based on contrast (1-5px)
        4. Tapering thickness from center to edges
        
        Args:
            output_dir: Directory containing boundary data
            file_id: File identifier
            sensitivity: Sensitivity parameter used for boundary detection
            
        Returns:
            List of stroke data for frontend rendering
        """
        # Load boundary data
        boundary_filename = f"{file_id}_boundaries_s{sensitivity}.json"
        boundary_path = os.path.join(output_dir, boundary_filename)
        
        if not os.path.exists(boundary_path):
            raise Exception(f"Boundary data not found: {boundary_filename}")
        
        with open(boundary_path, 'r') as f:
            boundary_info = json.load(f)
        
        boundaries = boundary_info['boundaries']
        
        # Sort boundaries by length (longest first) as requested
        sorted_boundaries = sorted(boundaries, 
                                 key=lambda x: x['length'], reverse=True)
        
        # Take only the most contrasted half
        half_count = max(1, len(sorted_boundaries) // 2)
        top_contrast_boundaries = sorted_boundaries[:half_count]
        
        print(f"Drawing {half_count} most contrasted boundaries out of {len(sorted_boundaries)} total")
        
        stroke_data_list = []
        
        for boundary in top_contrast_boundaries:
            # Get boundary data
            contrast = boundary['contrast_ratio']
            brightest_color = boundary['brightest_color']
            darkest_color = boundary['darkest_color']
            
            # Determine dominant color (dark vs light)
            dominant_color = self._get_dominant_color(brightest_color, darkest_color)
            
            # Calculate line thickness based on contrast (1-5px)
            # Use more aggressive scaling and proper rounding
            thickness_scale = contrast * 8  # Scale contrast more aggressively (0-8 instead of 0-4)
            max_thickness = max(1, min(5, round(1 + thickness_scale)))  # Round instead of int()
            
            # Debug: print thickness calculation
            print(f"Boundary {boundary['id']}: contrast={contrast:.3f}, max_thickness={max_thickness}")
            
            # Create tapered stroke points with variable thickness
            stroke_points = self._create_tapered_stroke_points(
                boundary['contour_points'], max_thickness)
            
            stroke_data = {
                'color': f"rgb({dominant_color['r']}, {dominant_color['g']}, {dominant_color['b']})",
                'width': max_thickness,
                'points': stroke_points,
                'type': 'highlight_boundary',
                'boundary_id': boundary['id'],
                'contrast_ratio': contrast,
                'tapered': True
            }
            
            stroke_data_list.append(stroke_data)
        
        return stroke_data_list
    
    def _get_dominant_color(self, brightest_color, darkest_color):
        """
        Determine dominant color and enhance it
        If dark is closer to black than light to white, use dark, otherwise light
        """
        # Calculate distances to black and white
        bright_avg = (brightest_color['r'] + brightest_color['g'] + brightest_color['b']) / 3
        dark_avg = (darkest_color['r'] + darkest_color['g'] + darkest_color['b']) / 3
        
        # Distance from extremes
        bright_to_white = 255 - bright_avg  # Distance to white
        dark_to_black = dark_avg  # Distance to black
        
        # Choose the color that's closer to its extreme
        if dark_to_black < bright_to_white:
            # Dark color is closer to black than bright to white
            base_color = darkest_color
            is_dark = True
        else:
            # Bright color is closer to white than dark to black
            base_color = brightest_color
            is_dark = False
        
        # Enhance the color (make it more extreme)
        # ENHANCEMENT: Stronger contrast - make light colors lighter and dark colors darker
        enhanced_color = {}
        for channel in ['r', 'g', 'b']:
            value = base_color[channel]
            if is_dark:  # Make darker (stronger darkening)
                enhanced_color[channel] = max(0, int(value * 0.5))  # Changed from 0.7 to 0.5
            else:  # Make lighter (stronger lightening)
                enhanced_color[channel] = min(255, int(value * 1.5))  # Changed from 1.3 to 1.5
        
        return enhanced_color
    
    def _create_tapered_stroke_points(self, contour_points, max_thickness):
        """
        Create stroke points with variable thickness that tapers from center to edges
        """
        stroke_points = []
        num_points = len(contour_points)
        
        if num_points < 2:
            return stroke_points
        
        for i, (x, y) in enumerate(contour_points):
            # Calculate thickness for this point
            # Thickness starts at 1, grows to max at center, then back to 1
            progress = i / (num_points - 1) if num_points > 1 else 0
            
            # Create a bell curve for thickness
            if progress <= 0.5:
                # Growing phase (0 to 0.5)
                thickness_factor = progress * 2  # 0 to 1
            else:
                # Shrinking phase (0.5 to 1)
                thickness_factor = (1 - progress) * 2  # 1 to 0
            
            # Calculate actual thickness (1 to max_thickness)
            # Use round() instead of int() for better thickness distribution
            thickness = max(1, round(1 + thickness_factor * (max_thickness - 1)))
            
            # Debug: print thickness for first few points
            if i < 3:
                print(f"  Point {i}: progress={progress:.2f}, thickness_factor={thickness_factor:.2f}, thickness={thickness}")
            
            stroke_points.append({
                'x': float(x), 
                'y': float(y),
                'thickness': thickness
            })
        
        return stroke_points

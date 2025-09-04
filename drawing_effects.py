import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
import imageio
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology, measure
import random
import math

class DrawingEffectGenerator:
    def __init__(self):
        self.styles = {
            'oil': self._oil_painting_style,
            'pastel': self._pastel_style,
            'pencil': self._pencil_style
        }
    
    def create_drawing_video(self, input_path, output_path, style='oil', detail_level=3, progress_callback=None):
        """
        Create a drawing effect video from an input image
        
        Args:
            input_path: Path to input image
            output_path: Path for output video
            style: Drawing style ('oil', 'pastel', 'pencil')
            detail_level: Level of detail (1-5, higher = more detailed)
            progress_callback: Function to call with progress updates
        """
        if progress_callback:
            progress_callback(5, "Loading and preprocessing image...")
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(input_path)
        
        if progress_callback:
            progress_callback(15, "Analyzing image structure...")
        
        # Generate stroke layers based on detail level
        stroke_layers = self._generate_stroke_layers(image, detail_level)
        
        if progress_callback:
            progress_callback(30, f"Applying {style} style effects...")
        
        # Apply style-specific effects
        style_func = self.styles.get(style, self.styles['oil'])
        styled_layers = [style_func(layer, image) for layer in stroke_layers]
        
        if progress_callback:
            progress_callback(60, "Generating animation frames...")
        
        # Create animation frames
        frames = self._create_animation_frames(styled_layers, image.shape)
        
        if progress_callback:
            progress_callback(85, "Rendering video...")
        
        # Save as video
        self._save_video(frames, output_path)
        
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
    
    def _generate_stroke_layers(self, image, detail_level):
        """Generate different stroke layers from coarse to fine"""
        layers = []
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Generate layers with different levels of detail
        layer_configs = [
            {'blur': 15, 'threshold': 50, 'stroke_size': 20},  # Very coarse
            {'blur': 10, 'threshold': 40, 'stroke_size': 15},  # Coarse
            {'blur': 5, 'threshold': 30, 'stroke_size': 10},   # Medium
            {'blur': 3, 'threshold': 20, 'stroke_size': 5},    # Fine
            {'blur': 1, 'threshold': 10, 'stroke_size': 2}     # Very fine
        ]
        
        # Use only the number of layers based on detail_level
        num_layers = min(detail_level + 1, len(layer_configs))
        
        for i in range(num_layers):
            config = layer_configs[i]
            layer = self._create_stroke_layer(image, gray, config)
            layers.append(layer)
        
        return layers
    
    def _create_stroke_layer(self, image, gray, config):
        """Create a single stroke layer"""
        # Apply blur
        blurred = cv2.GaussianBlur(gray, (config['blur'], config['blur']), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, config['threshold'], config['threshold'] * 2)
        
        # Dilate edges to create stroke paths
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (config['stroke_size'], config['stroke_size']))
        strokes = cv2.dilate(edges, kernel, iterations=1)
        
        # Create colored strokes
        stroke_mask = strokes > 0
        layer = np.zeros_like(image)
        layer[stroke_mask] = image[stroke_mask]
        
        return layer, stroke_mask
    
    def _oil_painting_style(self, layer_data, original_image):
        """Apply oil painting style effects"""
        layer, mask = layer_data
        
        if not np.any(mask):
            return layer, mask
        
        # Convert to PIL for oil painting effect
        pil_layer = Image.fromarray(layer)
        
        # Apply oil painting effect (blur + enhance)
        blurred = pil_layer.filter(ImageFilter.GaussianBlur(radius=1.5))
        enhanced = ImageEnhance.Color(blurred).enhance(1.3)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
        
        # Add texture
        textured = self._add_brush_texture(np.array(enhanced), mask)
        
        return textured, mask
    
    def _pastel_style(self, layer_data, original_image):
        """Apply pastel style effects"""
        layer, mask = layer_data
        
        if not np.any(mask):
            return layer, mask
        
        # Soften colors
        softened = cv2.bilateralFilter(layer, 15, 50, 50)
        
        # Reduce saturation slightly
        hsv = cv2.cvtColor(softened, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8  # Reduce saturation
        softened = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Add soft texture
        textured = self._add_soft_texture(softened, mask)
        
        return textured, mask
    
    def _pencil_style(self, layer_data, original_image):
        """Apply pencil/hatching style effects"""
        layer, mask = layer_data
        
        if not np.any(mask):
            return layer, mask
        
        # Convert to grayscale for pencil effect
        gray_layer = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
        
        # Create hatching pattern
        hatched = self._create_hatching_pattern(gray_layer, mask)
        
        # Convert back to RGB
        pencil_layer = cv2.cvtColor(hatched, cv2.COLOR_GRAY2RGB)
        
        return pencil_layer, mask
    
    def _add_brush_texture(self, image, mask):
        """Add brush texture to oil painting"""
        # Create random brush strokes
        textured = image.copy()
        h, w = image.shape[:2]
        
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
    
    def _add_soft_texture(self, image, mask):
        """Add soft texture for pastel effect"""
        textured = image.copy().astype(np.float32)
        
        # Add subtle noise for pastel texture
        noise = np.random.normal(0, 3, image.shape)
        textured[mask] += noise[mask]
        
        return np.clip(textured, 0, 255).astype(np.uint8)
    
    def _create_hatching_pattern(self, gray_image, mask):
        """Create pencil hatching pattern"""
        hatched = gray_image.copy()
        h, w = gray_image.shape
        
        # Create diagonal hatching lines
        for i in range(0, h + w, 4):  # Every 4 pixels
            for j in range(max(0, i - w), min(i + 1, h)):
                x, y = i - j, j
                if 0 <= x < w and 0 <= y < h and mask[y, x]:
                    # Darken the pixel to create hatching effect
                    hatched[y, x] = max(0, hatched[y, x] - 30)
        
        return hatched
    
    def _create_animation_frames(self, styled_layers, image_shape):
        """Create animation frames showing progressive drawing"""
        frames = []
        h, w = image_shape[:2]
        
        # Start with blank canvas
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        # Number of frames per layer
        frames_per_layer = 15
        
        for layer_idx, (layer, mask) in enumerate(styled_layers):
            # Get stroke positions
            stroke_positions = np.where(mask)
            if len(stroke_positions[0]) == 0:
                continue
            
            # Randomly shuffle stroke positions for more natural drawing
            indices = list(range(len(stroke_positions[0])))
            random.shuffle(indices)
            
            # Animate this layer
            strokes_per_frame = max(1, len(indices) // frames_per_layer)
            
            for frame_idx in range(frames_per_layer):
                frame_canvas = canvas.copy()
                
                # Add all previous layers
                for prev_idx in range(layer_idx):
                    prev_layer, prev_mask = styled_layers[prev_idx]
                    frame_canvas[prev_mask] = prev_layer[prev_mask]
                
                # Add partial current layer
                start_idx = frame_idx * strokes_per_frame
                end_idx = min((frame_idx + 1) * strokes_per_frame, len(indices))
                
                for i in range(start_idx, end_idx):
                    idx = indices[i]
                    y, x = stroke_positions[0][idx], stroke_positions[1][idx]
                    frame_canvas[y, x] = layer[y, x]
                
                frames.append(frame_canvas)
        
        # Add final frames to show completed drawing
        final_canvas = canvas.copy()
        for layer, mask in styled_layers:
            final_canvas[mask] = layer[mask]
        
        # Hold final frame for a bit longer
        for _ in range(30):
            frames.append(final_canvas)
        
        return frames
    
    def _save_video(self, frames, output_path, fps=12):
        """Save frames as video"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Use imageio to create video
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)

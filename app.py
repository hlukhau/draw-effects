import os
import uuid
import shutil
import glob
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

import flash_effect
from drawing_effects import DrawingEffectGenerator
import threading
import json
import numpy as np
import cv2
from PIL import Image
import io
import base64
import sys
sys.path.append('tools')
from tools.flash import preprocess_image, render_lighting

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global cache for individual segments data
individual_segments_cache = {}

# Global cache for individual segment masks to avoid repeated segmentation
individual_segment_masks_cache = {}

def cleanup_folders():
    """Clean up uploads and outputs folders on application startup"""
    try:
        # Clean uploads folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed upload file: {filename}")
        
        # Clean outputs folder
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for filename in os.listdir(app.config['OUTPUT_FOLDER']):
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed output file: {filename}")
        
        print("Startup cleanup completed successfully")
        
    except Exception as e:
        print(f"Error during startup cleanup: {str(e)}")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Clean up folders on startup
cleanup_folders()

# Store processing status
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(exclude_file_id=None):
    """Clean up old files from uploads and outputs folders, optionally excluding a specific file_id"""
    try:
        deleted_count = 0
        
        # Clean uploads folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if exclude_file_id and filename.startswith(exclude_file_id):
                    continue  # Skip files belonging to the current upload
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Removed old upload file: {filename}")
        
        # Clean outputs folder
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for filename in os.listdir(app.config['OUTPUT_FOLDER']):
                if exclude_file_id and filename.startswith(exclude_file_id):
                    continue  # Skip files belonging to the current upload
                
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Removed old output file: {filename}")
        
        # Clear processing status for old files
        if exclude_file_id:
            keys_to_remove = []
            for key in processing_status.keys():
                if not key.startswith(exclude_file_id):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del processing_status[key]
        
        # Clear caches for old files
        if exclude_file_id:
            cache_keys_to_remove = []
            for cache_key in individual_segments_cache.keys():
                if cache_key != exclude_file_id:
                    cache_keys_to_remove.append(cache_key)
            
            for cache_key in cache_keys_to_remove:
                del individual_segments_cache[cache_key]
                if cache_key in individual_segment_masks_cache:
                    del individual_segment_masks_cache[cache_key]
        
        print(f"Cleanup completed: removed {deleted_count} old files")
        return deleted_count
        
    except Exception as e:
        print(f"Error during file cleanup: {str(e)}")
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)

        print(filename)
        
        # Clean up old files before saving new one
        cleanup_old_files(exclude_file_id=file_id)
        
        # Safely extract file extension with proper error handling
        if '.' in filename:
            file_extension = filename.rsplit('.', 1)[1].lower()
        else:
            # Default to jpg if no extension found
            file_extension = 'jpg'
            
        unique_filename = f"{file_id}.{file_extension}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': unique_filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    file_id = data.get('file_id')
    style = data.get('style', 'oil')
    color_threshold = data.get('color_threshold', 30)
    video_duration = data.get('video_duration', 10)
    mode = data.get('mode', 'segmentation')  # 'segmentation' or 'video'
    
    if not file_id:
        return jsonify({'error': 'No file ID provided'}), 400
    
    # Find the uploaded file
    uploaded_file = None
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith(file_id):
            uploaded_file = filename
            break
    
    if not uploaded_file:
        return jsonify({'error': 'File not found'}), 404
    
    # Initialize processing status
    processing_status[file_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting processing...'
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_image_background,
        args=(file_id, uploaded_file, style, color_threshold, video_duration, mode)
    )
    thread.start()
    
    return jsonify({'success': True, 'file_id': file_id})

def process_image_background(file_id, filename, style, color_threshold, video_duration, mode):
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create drawing effect generator
        generator = DrawingEffectGenerator()
        
        # Update status callback
        def update_status(progress, message):
            processing_status[file_id] = {
                'status': 'processing',
                'progress': progress,
                'message': message
            }
        
        if mode == 'segmentation':
            # Generate segmentation images
            output_files = generator.create_segmentation_images(
                input_path, 
                app.config['OUTPUT_FOLDER'],
                file_id,
                color_threshold=color_threshold,
                progress_callback=update_status
            )
            
            processing_status[file_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Segmentation completed!',
                'output_files': output_files,
                'mode': 'segmentation'
            }
        else:
            # Generate drawing effect video
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_drawing.mp4")
            generator.create_drawing_video(
                input_path, 
                output_path, 
                style=style, 
                color_threshold=color_threshold,
                video_duration=video_duration,
                progress_callback=update_status
            )
            
            processing_status[file_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Video generation completed!',
                'output_file': f"{file_id}_drawing.mp4",
                'mode': 'video'
            }
        
    except Exception as e:
        processing_status[file_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/status/<file_id>')
def get_status(file_id):
    status = processing_status.get(file_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/download/<file_id>')
def download_video(file_id):
    output_filename = f"{file_id}_drawing.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True, download_name=f"drawing_effect_{file_id}.mp4")
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<file_id>')
def preview_video(file_id):
    output_filename = f"{file_id}_drawing.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    if os.path.exists(output_path):
        return send_file(output_path)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/segmentation/<file_id>')
def get_segmentation(file_id):
    status = processing_status.get(file_id, {'status': 'not_found'})
    if status.get('mode') == 'segmentation' and status.get('status') == 'completed':
        output_files = status.get('output_files', [])
        
        # Verify files actually exist
        existing_files = []
        for filename in output_files:
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.exists(file_path):
                existing_files.append(filename)
        
        return jsonify({
            'output_files': existing_files,
            'total_files': len(existing_files),
            'status': 'completed'
        })
    else:
        return jsonify({'error': 'Segmentation not available', 'status': status.get('status', 'unknown')}), 404

@app.route('/outputs/<filename>')
def serve_output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/draw_segment', methods=['POST'])
def draw_segment():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        segment_id = data.get('segment_id')
        brush_type = data.get('brush_type', 'pencil')  # Default to pencil brush
        stroke_density = data.get('stroke_density', 1.0)  # Default density
        
        if not file_id or segment_id is None:
            return jsonify({'success': False, 'error': 'Missing file_id or segment_id'})
        
        # Generate brush strokes for the segment
        generator = DrawingEffectGenerator()
        brush_strokes = generator.generate_segment_brush_strokes(
            output_dir=app.config['OUTPUT_FOLDER'],
            file_id=file_id,
            segment_id=segment_id,
            brush_type=brush_type,
            stroke_density=stroke_density
        )
        
        return jsonify({
            'success': True,
            'brush_strokes': brush_strokes,
            'brush_type': brush_type
        })
        
    except Exception as e:
        print(f"Error in draw_segment: {str(e)}")  # Add logging
        return jsonify({'success': False, 'error': str(e)})

@app.route('/draw_individual_segment', methods=['POST'])
def draw_individual_segment():
    """Draw individual segment without color merging - each segment is processed separately"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        segment_id = data.get('segment_id')
        brush_type = data.get('brush_type', 'pencil')  # Default to pencil brush
        stroke_density = data.get('stroke_density', 1.0)  # Default density
        
        if not file_id or segment_id is None:
            return jsonify({'success': False, 'error': 'Missing file_id or segment_id'})
        
        # Generate fill data for the individual segment (without color merging)
        generator = DrawingEffectGenerator()
        fill_data = generator.generate_individual_segment_strokes(
            output_dir=app.config['OUTPUT_FOLDER'],
            file_id=file_id,
            segment_id=segment_id,
            brush_type=brush_type,
            external_cache=individual_segment_masks_cache,
            stroke_density=stroke_density
        )
        
        print(f"[INDIVIDUAL API] Generated data for segment {segment_id}: {len(fill_data)} items")
        if fill_data and len(fill_data) > 0:
            print(f"[INDIVIDUAL API] First item type: {fill_data[0].get('type', 'unknown')}")
            print(f"[INDIVIDUAL API] First item keys: {list(fill_data[0].keys())}")
        
        return jsonify({
            'success': True,
            'brush_strokes': fill_data,  # Still called brush_strokes for compatibility
            'brush_type': brush_type,
            'mode': 'individual'  # Indicate this is individual segment mode
        })
        
    except Exception as e:
        print(f"Error in draw_individual_segment: {str(e)}")  # Add logging
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_individual_segments', methods=['POST'])
def get_individual_segments():
    """Get list of individual segments for a file"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({'success': False, 'error': 'Missing file_id'})
        
        # Check cache first
        if file_id in individual_segments_cache:
            print(f"[CACHE HIT] Using cached individual segments for file_id: {file_id}")
            cached_data = individual_segments_cache[file_id]
            return jsonify({
                'success': True,
                'segments': cached_data['segments'],
                'source_file': cached_data['source_file']
            })
        
        print(f"[CACHE MISS] Loading individual segments for file_id: {file_id}")
        generator = DrawingEffectGenerator()
        
        # Load individual segmentation data
        outputs_dir = app.config['OUTPUT_FOLDER']
        
        # Look for individual segmentation files first
        json_files = [f for f in os.listdir(outputs_dir) 
                     if f.startswith(f"{file_id}_") and f.endswith('.json') 
                     and 'individual' in f and 'boundaries' not in f]
        
        if not json_files:
            # Fallback to regular segmentation files if individual not found
            json_files = [f for f in os.listdir(outputs_dir) 
                         if f.startswith(f"{file_id}_") and f.endswith('.json') 
                         and 'boundaries' not in f and 'individual' not in f]
        
        if not json_files:
            return jsonify({'success': False, 'error': 'No segmentation data found'})
        
        # Get the newest file
        json_files_with_time = []
        for json_file in json_files:
            file_path = os.path.join(outputs_dir, json_file)
            mod_time = os.path.getmtime(file_path)
            json_files_with_time.append((json_file, mod_time))
        
        json_files_with_time.sort(key=lambda x: x[1], reverse=True)
        newest_json_file = json_files_with_time[0][0]
        
        json_path = os.path.join(outputs_dir, newest_json_file)
        
        with open(json_path, 'r') as f:
            segment_data = json.load(f)
        
        # Extract segments list
        segments_list = None
        if isinstance(segment_data, dict):
            if 'segments' in segment_data:
                segments_list = segment_data['segments']
            elif 'segment_data' in segment_data:
                segments_list = segment_data['segment_data']
            else:
                for key in segment_data.keys():
                    if isinstance(segment_data[key], list):
                        segments_list = segment_data[key]
                        break
        elif isinstance(segment_data, list):
            segments_list = segment_data
        
        if segments_list is None:
            return jsonify({'success': False, 'error': 'Could not find segments in data'})
        
        print(f"[GET_INDIVIDUAL_SEGMENTS] Loaded {len(segments_list)} segments from {newest_json_file}")
        
        # Cache the result for future requests
        individual_segments_cache[file_id] = {
            'segments': segments_list,
            'source_file': newest_json_file
        }
        print(f"[CACHE STORE] Cached individual segments for file_id: {file_id}")
        
        return jsonify({
            'success': True,
            'segments': segments_list,
            'source_file': newest_json_file
        })
        
    except Exception as e:
        print(f"Error in get_individual_segments: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_individual_segments_cache', methods=['POST'])
def clear_individual_segments_cache():
    """Clear the individual segments cache"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if file_id and file_id in individual_segments_cache:
            del individual_segments_cache[file_id]
            # Also clear the masks cache
            if file_id in individual_segment_masks_cache:
                del individual_segment_masks_cache[file_id]
            print(f"[CACHE CLEAR] Cleared individual segments cache for file_id: {file_id}")
            return jsonify({'success': True, 'message': f'Cache cleared for file_id: {file_id}'})
        elif file_id:
            print(f"[CACHE CLEAR] No cache found for file_id: {file_id}")
            return jsonify({'success': True, 'message': f'No cache found for file_id: {file_id}'})
        else:
            # Clear all cache if no file_id specified
            cache_count = len(individual_segments_cache)
            masks_cache_count = len(individual_segment_masks_cache)
            individual_segments_cache.clear()
            individual_segment_masks_cache.clear()
            print(f"[CACHE CLEAR] Cleared all individual segments cache ({cache_count} entries) and masks cache ({masks_cache_count} entries)")
            return jsonify({'success': True, 'message': f'Cleared all cache ({cache_count + masks_cache_count} entries)'})
            
    except Exception as e:
        print(f"Error in clear_individual_segments_cache: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/detect_boundaries', methods=['POST'])
def detect_boundaries():
    """Detect contrast boundaries in an uploaded image"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        sensitivity = data.get('sensitivity', 50)  # Default sensitivity
        fragmentation = data.get('fragmentation', 50)  # Default fragmentation
        
        # print(f"DEBUG: API received fragmentation={fragmentation}, type={type(fragmentation)}")
        
        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400
        
        # Find the uploaded file
        uploaded_file = None
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(file_id):
                uploaded_file = filename
                break
        
        if not uploaded_file:
            return jsonify({'error': 'File not found'}), 404
        
        # Initialize processing status
        processing_status[file_id + '_boundaries'] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting boundary detection...'
        }
        
        # Start boundary detection in background thread
        thread = threading.Thread(
            target=detect_boundaries_background,
            args=(file_id, uploaded_file, sensitivity, fragmentation)
        )
        thread.start()
        
        return jsonify({'success': True, 'file_id': file_id, 'mode': 'boundaries'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def detect_boundaries_background(file_id, filename, sensitivity, fragmentation):
    """Background task for boundary detection"""
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Create drawing effect generator
        generator = DrawingEffectGenerator()
        
        # Update status callback
        def update_status(progress, message):
            processing_status[file_id + '_boundaries'] = {
                'status': 'processing',
                'progress': progress,
                'message': message
            }
        
        # Detect boundaries
        boundary_data = generator.detect_contrast_boundaries(
            input_path,
            app.config['OUTPUT_FOLDER'],
            file_id,
            sensitivity=sensitivity,
            fragmentation=fragmentation,
            progress_callback=update_status
        )
        
        processing_status[file_id + '_boundaries'] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Found {len(boundary_data)} boundaries!',
            'boundary_data': boundary_data,
            'sensitivity': sensitivity,
            'fragmentation': fragmentation,
            'mode': 'boundaries'
        }
        
        # print(f"DEBUG: Saved {len(boundary_data)} boundaries to processing_status with fragmentation={fragmentation}")
        
    except Exception as e:
        processing_status[file_id + '_boundaries'] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }

@app.route('/boundaries/<file_id>')
def get_boundaries(file_id):
    """Get detected boundaries for a file"""
    # print(f"DEBUG: get_boundaries called for file_id={file_id}")
    # print(f"DEBUG: processing_status keys: {list(processing_status.keys())}")
    
    status = processing_status.get(file_id + '_boundaries', {'status': 'not_found'})
    # print(f"DEBUG: status for {file_id + '_boundaries'}: {status.get('status', 'unknown')}")
    
    if status.get('mode') == 'boundaries' and status.get('status') == 'completed':
        boundary_data = status.get('boundary_data', [])
        # print(f"DEBUG: API returning {len(boundary_data)} boundaries for file_id={file_id}")
        # print(f"DEBUG: First few boundary IDs: {[b.get('id') for b in boundary_data[:5]] if boundary_data else 'None'}")
        return jsonify({
            'boundary_data': boundary_data,
            'total_boundaries': len(boundary_data),
            'sensitivity': status.get('sensitivity', 50),
            'fragmentation': status.get('fragmentation', 1),
            'status': 'completed'
        })
    else:
        # print(f"DEBUG: Boundaries not available for file_id={file_id}, status={status.get('status', 'unknown')}")
        return jsonify({'error': 'Boundaries not available', 'status': status.get('status', 'unknown')}), 404

@app.route('/draw_boundary', methods=['POST'])
def draw_boundary():
    """Draw a single boundary line"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        boundary_id = data.get('boundary_id')
        color_type = data.get('color_type', 'brightest')  # 'brightest' or 'darkest'
        sensitivity = data.get('sensitivity', 50)
        
        if not file_id or boundary_id is None:
            return jsonify({'success': False, 'error': 'Missing file_id or boundary_id'})
        
        # Generate boundary stroke
        generator = DrawingEffectGenerator()
        boundary_strokes = generator.draw_single_boundary(
            output_dir=app.config['OUTPUT_FOLDER'],
            file_id=file_id,
            boundary_id=boundary_id,
            color_type=color_type,
            sensitivity=sensitivity
        )
        
        return jsonify({
            'success': True,
            'boundary_strokes': boundary_strokes,
            'boundary_id': boundary_id,
            'color_type': color_type
        })
        
    except Exception as e:
        print(f"Error in draw_boundary: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/highlight_boundaries', methods=['POST'])
def highlight_boundaries():
    """Highlight boundaries using contrast-based sorting and dynamic line thickness"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        sensitivity = data.get('sensitivity', 50)
        
        if not file_id:
            return jsonify({'success': False, 'error': 'Missing file_id'})
        
        # Generate highlight boundary strokes
        generator = DrawingEffectGenerator()
        highlight_strokes = generator.highlight_contrast_boundaries(
            output_dir=app.config['OUTPUT_FOLDER'],
            file_id=file_id,
            sensitivity=sensitivity
        )
        
        return jsonify({
            'success': True,
            'highlight_strokes': highlight_strokes,
            'total_boundaries': len(highlight_strokes)
        })
        
    except Exception as e:
        print(f"Error in highlight_boundaries: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/boundary_status/<file_id>')
def get_boundary_status(file_id):
    """Get boundary detection status"""
    status = processing_status.get(file_id + '_boundaries', {'status': 'not_found'})
    return jsonify(status)

@app.route('/flash_animation', methods=['POST'])
def flash_animation():
    """
    Создает анимацию flash эффекта для текущего состояния канваса
    """
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        frame_identifier = data.get('frame_identifier', 'latest')  # Идентификатор кадра
        relief_strength = data.get('relief_strength', 0.05)
        
        if not file_id:
            return jsonify({'success': False, 'error': 'Missing file_id'})
        
        # Определяем путь к изображению на основе frame_identifier
        image_path = None
        possible_paths = []
        
        if frame_identifier == 'latest':
            # Пробуем найти последний сохраненный кадр канваса
            possible_paths = [
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_canvas_state.png"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_mean_color.png"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.png"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.jpg"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.jpeg")
            ]
        else:
            # Используем конкретный кадр
            possible_paths = [
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_{frame_identifier}.png"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_mean_color.png"),
                os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.png")
            ]
        
        # Ищем первый существующий файл
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                print(f"Found image file: {image_path}")
                break
        
        if not image_path:
            print(f"No image file found for file_id: {file_id}")
            print(f"Checked paths: {possible_paths}")
            return jsonify({'success': False, 'error': f'Image file not found for file_id: {file_id}'})
        
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        
        # Предварительная обработка изображения
        session_id = flash_effect.preprocess_image(image, relief_strength)
        
        # Генерируем все кадры анимации
        frames, positions = flash_effect.get_animation_frames(session_id, frame_step=1)
        
        # Конвертируем кадры в base64
        animation_frames = []
        for i, frame in enumerate(frames):
            frame_base64 = flash_effect.frame_to_base64(frame)
            animation_frames.append({
                'frame_index': i,
                'light_position': positions[i],
                'image_data': frame_base64
            })
        
        # Очищаем предрасчитанные данные после использования
        flash_effect.cleanup_session(session_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_frames': len(animation_frames),
            'frames': animation_frames,
            'animation_duration': len(animation_frames) * 100,  # миллисекунды
            'relief_strength': relief_strength
        })
        
    except Exception as e:
        print(f"Error in flash_animation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """
    Удаляет загруженный файл и все связанные с ним результаты
    """
    try:
        deleted_files = []
        
        # Удаляем исходный загруженный файл
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(file_id):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(f"uploads/{filename}")
        
        # Удаляем все файлы результатов
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            if filename.startswith(file_id):
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(f"outputs/{filename}")
        
        # Очищаем статус обработки
        keys_to_remove = []
        for key in processing_status.keys():
            if key.startswith(file_id):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del processing_status[key]
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted {len(deleted_files)} files',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        print(f"Error in delete_file: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_canvas_frame', methods=['POST'])
def save_canvas_frame():
    """
    Сохраняет кадр канваса на сервер для использования в flash анимации
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        file_id = request.form.get('file_id')
        frame_type = request.form.get('frame_type', 'canvas_state')
        
        if not file_id:
            return jsonify({'success': False, 'error': 'Missing file_id'})
        
        # Сохраняем файл
        filename = f"{file_id}_{frame_type}.png"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
        
    except Exception as e:
        print(f"Error in save_canvas_frame: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/light_effect', methods=['POST'])
def light_effect():
    """
    Применяет эффект освещения к изображению канваса
    """
    try:
        data = request.get_json()
        canvas_data = data.get('canvas_data')  # base64 image data
        relief_strength = float(data.get('relief_strength', 0.05))
        
        if not canvas_data:
            return jsonify({'success': False, 'error': 'No canvas data provided'})
        
        # Декодируем base64 изображение
        if canvas_data.startswith('data:image'):
            canvas_data = canvas_data.split(',')[1]
        
        image_bytes = base64.b64decode(canvas_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Предварительная обработка изображения
        session_id = preprocess_image(image, relief_strength)
        
        # Генерируем кадры анимации освещения
        frames = []
        # Основные кадры с освещением: 0-200 с шагом 2.5 (81 кадр)
        # Дополнительные кадры для полного исчезания: 201-220 с шагом 2.5 (8 кадров)
        positions = [i * 2.5 for i in range(89)]  # 0-220 с шагом 2.5 (89 кадров)
        
        for light_position in positions:
            # Получаем preprocessed_data из flash модуля
            from tools.flash import preprocessed_data
            processed_image = render_lighting(preprocessed_data[session_id], light_position)
            
            # Конвертируем в base64
            img_io = io.BytesIO()
            processed_image.save(img_io, 'JPEG', quality=80)
            img_io.seek(0)
            img_str = base64.b64encode(img_io.getvalue()).decode('ascii')
            frames.append(img_str)
        
        # Очищаем данные сессии
        if session_id in preprocessed_data:
            del preprocessed_data[session_id]
        
        return jsonify({
            'success': True,
            'frames': frames,
            'total_frames': len(frames)
        })
        
    except Exception as e:
        print(f"Error in light_effect: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

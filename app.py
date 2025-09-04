import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from drawing_effects import DrawingEffectGenerator
import threading
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Store processing status
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        file_extension = filename.rsplit('.', 1)[1].lower()
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
    detail_level = data.get('detail_level', 3)
    video_duration = data.get('video_duration', 10)
    
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
        args=(file_id, uploaded_file, style, detail_level, video_duration)
    )
    thread.start()
    
    return jsonify({'success': True, 'file_id': file_id})

def process_image_background(file_id, filename, style, detail_level, video_duration):
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_drawing.mp4")
        
        # Create drawing effect generator
        generator = DrawingEffectGenerator()
        
        # Update status callback
        def update_status(progress, message):
            processing_status[file_id] = {
                'status': 'processing',
                'progress': progress,
                'message': message
            }
        
        # Generate drawing effect video
        generator.create_drawing_video(
            input_path, 
            output_path, 
            style=style, 
            detail_level=detail_level,
            video_duration=video_duration,
            progress_callback=update_status
        )
        
        processing_status[file_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Video generation completed!',
            'output_file': f"{file_id}_drawing.mp4"
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

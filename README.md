# Drawing Effects - Photo to Drawing Video Generator

Transform your photos into beautiful animated drawing videos! This Flask web application simulates the process of drawing an image with various artistic styles, starting with broad strokes and progressing to fine details.

## Features

- **Photo Upload**: Easy drag-and-drop or click-to-browse file upload
- **Multiple Artistic Styles**:
  - **Oil Painting**: Thick brush strokes with rich colors
  - **Pastel**: Soft, dreamy colors with smooth transitions  
  - **Pencil Sketching**: Hatching and cross-hatching techniques
- **Adjustable Detail Levels**: Control the complexity and processing time (1-5 scale)
- **Animated Drawing Process**: Watch your photo come to life stroke by stroke
- **Video Download**: Download your drawing animation as MP4

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd draw-effects
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

## Usage

1. **Upload a Photo**: Drag and drop an image or click to browse (supports JPEG, PNG, GIF, BMP up to 16MB)
2. **Choose Settings**:
   - Select your preferred drawing style (Oil, Pastel, or Pencil)
   - Adjust detail level (higher = more detailed but longer processing time)
3. **Generate Video**: Click "Create Drawing Video" and wait for processing
4. **Download**: Preview your video and download it when ready

## Technical Details

- **Backend**: Flask web framework with background processing
- **Image Processing**: OpenCV, PIL, scikit-image for artistic effects
- **Video Generation**: imageio with H.264 encoding
- **Frontend**: Bootstrap 5 with responsive design

## File Structure

```
draw-effects/
├── app.py                 # Main Flask application
├── drawing_effects.py     # Core image processing and video generation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── app.js        # Frontend JavaScript
├── uploads/              # Temporary uploaded images
└── outputs/              # Generated videos
```

## How It Works

1. **Image Analysis**: The uploaded image is analyzed to detect edges and structures
2. **Layer Generation**: Multiple stroke layers are created from coarse to fine detail
3. **Style Application**: Artistic effects are applied based on the selected style
4. **Animation**: Frames are generated showing progressive drawing simulation
5. **Video Encoding**: All frames are compiled into an MP4 video

## Requirements

- Python 3.7+
- OpenCV
- PIL/Pillow
- NumPy
- SciPy
- scikit-image
- imageio
- Flask

## Browser Support

Modern browsers with HTML5 video support:
- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

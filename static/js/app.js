// Drawing Effects App JavaScript

let currentFileId = null;
let processingInterval = null;

// Style descriptions
const styleDescriptions = {
    pencil: "Pencil sketching with hatching and cross-hatching techniques, building up from light outlines to detailed shading."
};

// Global variables for canvas interaction
let canvasData = {
    segments: null,
    segmentInfo: null,
    selectedSegmentId: null,
    meanColorImage: null,
    originalSegments: null,
    currentBrushType: 'pencil',
    backgroundColor: '#fff3cd',
    isDrawingAll: false,
    drawingInterrupted: false,
    canvasInitialized: false,
    preserveCanvasState: false,
    videoResults: [], // Store multiple video results
    cumulativeFrames: [], // Store all frames for cumulative video
    masterVideoRecorder: null // Single video recorder for cumulative recording
};

// Global variables for timing
let drawingStartTime = null;
let drawingTimer = null;

// Brush type configurations
const brushTypes = {
    pencil: {
        name: 'Pencil',
        opacity: 0.7,
        blendMode: 'source-over',
        textureIntensity: 0.8,
        strokeVariation: 0.3
    }
};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateStyleDescription();
});

function initializeEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const chooseFileBtn = document.getElementById('chooseFileBtn');
    const styleSelect = document.getElementById('styleSelect');
    const colorThreshold = document.getElementById('colorThreshold');
    const videoDuration = document.getElementById('videoDuration');
    const strokeDensity = document.getElementById('strokeDensity');
    const videoFps = document.getElementById('videoFps');

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Click on upload area (but not on the button)
    uploadArea.addEventListener('click', function(e) {
        // Only trigger file input if the click wasn't on the choose file button
        if (e.target !== chooseFileBtn && !chooseFileBtn.contains(e.target)) {
            fileInput.click();
        }
    });
    
    // Choose file button click
    chooseFileBtn.addEventListener('click', function(e) {
        e.stopPropagation(); // Prevent event bubbling to uploadArea
        fileInput.click();
    });

    // Style selection change
    styleSelect.addEventListener('change', updateStyleDescription);

    // Range sliders
    colorThreshold.addEventListener('input', function() {
        document.getElementById('thresholdValue').textContent = this.value;
    });

    videoDuration.addEventListener('input', function() {
        document.getElementById('durationValue').textContent = this.value;
    });

    videoFps.addEventListener('change', function() {
        document.getElementById('fpsValue').textContent = this.value;
    });

    // New range sliders for stroke density
    strokeDensity.addEventListener('input', function() {
        document.getElementById('densityValue').textContent = parseFloat(this.value).toFixed(1);
    });
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a valid image file (JPEG, PNG, GIF, or BMP)', 'danger');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'danger');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        showImagePreview(e.target.result, file.name);
    };
    reader.readAsDataURL(file);

    // Upload file
    uploadFile(file);
}

function showImagePreview(src, fileName) {
    const preview = document.getElementById('imagePreview');
    const img = document.getElementById('previewImg');
    const fileNameSpan = document.getElementById('fileName');
    
    img.src = src;
    fileNameSpan.textContent = fileName;
    preview.style.display = 'block';
    
    // Enable process buttons
    document.getElementById('segmentBtn').disabled = false;
    currentFileId = null;
    
    // Clean up existing video resources when loading new image
    canvasData.videoResults.forEach(videoInfo => {
        if (videoInfo.url.startsWith('blob:')) {
            URL.revokeObjectURL(videoInfo.url);
        }
    });
    
    // Reset canvas and video state when new image is loaded
    canvasData.canvasInitialized = false;
    canvasData.preserveCanvasState = false;
    canvasData.cumulativeFrames = [];
    canvasData.videoResults = [];
    canvasData.masterVideoRecorder = null;
    
    hideResults();
    hideSegmentationResults();
}

function clearImage() {
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('segmentBtn').disabled = true;
    currentFileId = null;
    hideResults();
    hideSegmentationResults();
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    showAlert('Uploading image...', 'info');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentFileId = data.file_id;
            document.getElementById('segmentBtn').disabled = false;
            showAlert('Image uploaded successfully!', 'success');
        } else {
            showAlert(data.error || 'Upload failed', 'danger');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showAlert('Upload failed. Please try again.', 'danger');
    });
}

function processImage() {
    if (!currentFileId) {
        showAlert('Please upload an image first', 'warning');
        return;
    }

    const style = document.getElementById('styleSelect').value;
    const colorThreshold = parseInt(document.getElementById('colorThreshold').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);

    // Update button state
    const segmentBtn = document.getElementById('segmentBtn');
    segmentBtn.disabled = true;
    segmentBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    // Hide previous results
    hideResults();
    hideSegmentationResults();

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            file_id: currentFileId,
            style: style,
            color_threshold: colorThreshold,
            video_duration: videoDuration,
            stroke_density: strokeDensity,
            mode: 'video'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            startStatusPolling(currentFileId, 'video');
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error processing image: ' + error.message, 'danger');
        
        // Reset button
        segmentBtn.disabled = false;
        segmentBtn.innerHTML = '<i class="fas fa-magic"></i> Generate Drawing Video';
    });
}

function processSegmentation() {
    if (!currentFileId) {
        showAlert('Please upload an image first', 'warning');
        return;
    }

    const style = document.getElementById('styleSelect').value;
    const colorThreshold = parseInt(document.getElementById('colorThreshold').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);

    const segmentBtn = document.getElementById('segmentBtn');
    const processingStatus = document.getElementById('processingStatus');
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');

    // Disable button and show progress
    segmentBtn.disabled = true;
    segmentBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    processingStatus.style.display = 'block';
    progressBar.style.width = '0%';
    statusMessage.textContent = 'Starting segmentation...';

    // Check if canvas exists and has been drawn on
    const canvas = document.getElementById('drawingCanvas');
    if (canvas && canvasData.canvasInitialized) {
        canvasData.preserveCanvasState = true;
        showAlert('Re-fragmenting while preserving drawing state...', 'info');
    } else {
        canvasData.preserveCanvasState = false;
        // Hide previous results only if no canvas state to preserve
        hideResults();
        hideSegmentationResults();
    }

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            style: style,
            color_threshold: colorThreshold,
            video_duration: videoDuration,
            stroke_density: strokeDensity,
            mode: 'segmentation'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            startStatusPolling(currentFileId, 'segmentation');
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error processing image: ' + error.message, 'danger');
        
        // Reset button
        segmentBtn.disabled = false;
        segmentBtn.innerHTML = '<i class="fas fa-eye"></i> Show Segmentation';
    });
}

function startStatusPolling(fileId, mode = 'video') {
    if (processingInterval) {
        clearInterval(processingInterval);
    }
    
    processingInterval = setInterval(() => {
        fetch(`/status/${fileId}`)
        .then(response => response.json())
        .then(data => {
            updateProcessingStatus(data);
            
            if (data.status === 'completed') {
                clearInterval(processingInterval);
                if (mode === 'segmentation') {
                    showSegmentationResults(fileId);
                } else {
                    showResults(fileId);
                }
            } else if (data.status === 'error') {
                clearInterval(processingInterval);
                showAlert('Processing failed: ' + data.message, 'danger');
                resetSegmentButton();
            }
        })
        .catch(error => {
            console.error('Status polling error:', error);
        });
    }, 1000);
}

function updateProcessingStatus(data) {
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');

    progressBar.style.width = `${data.progress || 0}%`;
    progressBar.textContent = `${data.progress || 0}%`;
    statusMessage.textContent = data.message || 'Processing...';

    // Add pulse animation during processing
    if (data.status === 'processing') {
        statusMessage.classList.add('processing-pulse');
    } else {
        statusMessage.classList.remove('processing-pulse');
    }
}

function showResults(fileId) {
    const resultsSection = document.getElementById('resultsSection');
    const resultVideo = document.getElementById('resultVideo');
    const downloadBtn = document.getElementById('downloadBtn');

    // Set video source
    resultVideo.src = `/preview/${fileId}`;
    
    // Set download link
    downloadBtn.onclick = () => {
        window.open(`/download/${fileId}`, '_blank');
    };

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    showAlert('Drawing video created successfully!', 'success');
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

function showSegmentationResults(fileId) {
    fetch(`/status/${fileId}`)
    .then(response => response.json())
    .then(data => {
        if (data.status === 'completed' && data.mode === 'segmentation') {
            const resultsSection = document.getElementById('segmentationResults');
            const imagesContainer = document.getElementById('segmentationImages');
            
            // Smart cleanup: preserve canvas container position when preserving state
            let canvasContainer = null;
            if (canvasData.preserveCanvasState) {
                // Save the canvas container before clearing
                canvasContainer = imagesContainer.querySelector('.canvas-container');
                if (canvasContainer) {
                    canvasContainer.remove(); // Temporarily remove it
                    console.log('Temporarily removed canvas container to preserve position');
                }
            }
            
            // Clear all content to ensure proper positioning with flexbox
            imagesContainer.innerHTML = '';
            imagesContainer.className = 'd-flex flex-nowrap justify-content-between';
            console.log('Cleared all content for new segmentation with proper positioning');
            
            // Add images
            const imageFiles = data.output_files.filter(f => f.endsWith('.png'));
            const jsonFile = data.output_files.find(f => f.endsWith('.json'));
            
            // Add strong cache-busting with multiple parameters
            const timestamp = new Date().getTime();
            const randomId = Math.random().toString(36).substring(2, 15);
            const sessionId = Math.random().toString(36).substring(2, 8);
            
            console.log(`Loading segmentation images with cache-busting: t=${timestamp}, r=${randomId}`);
            
            imageFiles.forEach((filename, index) => {
                const col = document.createElement('div');
                col.className = 'flex-fill mx-2';
                
                const card = document.createElement('div');
                card.className = 'card';
                
                const img = document.createElement('img');
                // Use multiple cache-busting parameters and force reload
                img.src = `/outputs/${filename}?t=${timestamp}&r=${randomId}&s=${sessionId}&v=${index}`;
                img.className = 'card-img-top';
                img.style.height = '200px';
                img.style.objectFit = 'cover';
                
                // Force browser to not use cache
                img.crossOrigin = 'anonymous';
                
                // Enhanced error handling with multiple retry attempts
                let retryCount = 0;
                const maxRetries = 3;
                
                img.onerror = function() {
                    retryCount++;
                    console.error(`Failed to load image (attempt ${retryCount}):`, filename);
                    
                    if (retryCount <= maxRetries) {
                        const newTimestamp = new Date().getTime();
                        const newRandom = Math.random().toString(36).substring(2, 15);
                        this.src = `/outputs/${filename}?t=${newTimestamp}&r=${newRandom}&retry=${retryCount}&nocache=1`;
                        console.log(`Retrying image load (${retryCount}/${maxRetries}):`, this.src);
                    } else {
                        console.error(`Failed to load image after ${maxRetries} attempts:`, filename);
                        // Show placeholder or error message
                        this.style.backgroundColor = '#f8f9fa';
                        this.style.border = '2px dashed #dee2e6';
                        this.alt = 'Failed to load image';
                    }
                };
                
                img.onload = function() {
                    console.log('Successfully loaded updated image:', filename);
                    // Force a repaint to ensure image is displayed
                    this.style.opacity = '0.99';
                    setTimeout(() => { this.style.opacity = '1'; }, 10);
                };
                
                const cardBody = document.createElement('div');
                cardBody.className = 'card-body p-2';
                
                const title = document.createElement('h6');
                title.className = 'card-title mb-0';
                
                // Set descriptive titles
                if (filename.includes('original')) {
                    title.textContent = 'Original Image';
                } else if (filename.includes('segments')) {
                    title.textContent = 'Colored Segments';
                } else if (filename.includes('mean_colors')) {
                    title.textContent = 'Average Colors';
                    // Store the mean color image for canvas with enhanced cache-busting
                    const newMeanColorImage = `/outputs/${filename}?t=${timestamp}&r=${randomId}&s=${sessionId}&canvas=1`;
                    
                    // Only update mean color image if we're not preserving canvas state
                    if (!canvasData.preserveCanvasState) {
                        canvasData.meanColorImage = newMeanColorImage;
                        console.log('Updated mean color image for canvas:', newMeanColorImage);
                    } else {
                        // Store new image for potential future use but don't replace current
                        canvasData.newMeanColorImage = newMeanColorImage;
                        console.log('Stored new mean color image for future use:', newMeanColorImage);
                    }
                }
                
                cardBody.appendChild(title);
                card.appendChild(img);
                card.appendChild(cardBody);
                col.appendChild(card);
                imagesContainer.appendChild(col);
                
                // Force DOM update and ensure visibility
                console.log('Appended image card to container:', filename);
                
                // Ensure the image card is visible
                col.style.display = 'block';
                col.style.opacity = '1';
                
                // Force a reflow to ensure the DOM is updated
                col.offsetHeight;
            });
            
            // Add segment information if JSON file exists
            if (jsonFile) {
                fetch(`/outputs/${jsonFile}?t=${timestamp}`)
                .then(response => response.json())
                .then(segmentData => {
                    // Store segment data for canvas interaction
                    canvasData.segmentInfo = segmentData;
                    
                    const infoCol = document.createElement('div');
                    infoCol.className = 'col-12 mt-3';
                    
                    const infoCard = document.createElement('div');
                    infoCard.className = 'card';
                    
                    const infoHeader = document.createElement('div');
                    infoHeader.className = 'card-header bg-secondary text-white';
                    infoHeader.innerHTML = '<h6 class="mb-0"><i class="fas fa-info-circle"></i> Segment Information</h6>';
                    
                    const infoBody = document.createElement('div');
                    infoBody.className = 'card-body';
                    
                    // Summary
                    const summary = document.createElement('div');
                    summary.className = 'mb-3';
                    summary.innerHTML = `
                        <strong>Threshold:</strong> ${segmentData.threshold}<br>
                        <strong>Total Segments:</strong> ${segmentData.total_segments}
                    `;
                    
                    infoBody.appendChild(summary);
                    
                    // Handle canvas container creation or preservation
                    let canvasContainerToUse;
                    
                    if (canvasData.preserveCanvasState && canvasContainer) {
                        // Use the saved canvas container from earlier
                        canvasContainerToUse = canvasContainer;
                        console.log('Using preserved canvas container');
                    } else {
                        // Create new interactive canvas section BEFORE the table
                        canvasContainerToUse = document.createElement('div');
                        canvasContainerToUse.className = 'mb-4 p-3 border rounded bg-light canvas-container';
                        canvasContainerToUse.innerHTML = `
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h6 class="mb-0"><i class="fas fa-paint-brush"></i> Interactive Drawing & Video Results</h6>
                                <div class="d-flex align-items-center">
                                    <div class="mr-3">
                                        <label for="brushTypeSelect" class="mb-0 mr-2"><small>Brush:</small></label>
                                        <select id="brushTypeSelect" class="form-control form-control-sm" style="width: auto; display: inline-block;">
                                            <option value="pencil">Pencil</option>
                                        </select>
                                    </div>
                                    <div class="mr-3">
                                        <label for="backgroundColorSelect" class="mb-0 mr-2"><small>Background:</small></label>
                                        <select id="backgroundColorSelect" class="form-control form-control-sm" style="width: auto; display: inline-block;">
                                            <option value="transparent">Original</option>
                                            <option value="#ffffff">White</option>
                                            <option value="#f8f9fa">Light Gray</option>
                                            <option value="#343a40">Dark Gray</option>
                                            <option value="#000000">Black</option>
                                            <option value="#fff3cd" selected>Cream</option>
                                        </select>
                                    </div>
                                    <button id="drawAllBtn" class="btn btn-warning btn-sm mr-2">
                                        <i class="fas fa-palette"></i> Draw All
                                    </button>
                                    <button id="drawLargeBtn" class="btn btn-success btn-sm mr-2">
                                        <i class="fas fa-expand"></i> Large Fragments
                                    </button>
                                    <button id="drawMediumBtn" class="btn btn-primary btn-sm mr-2">
                                        <i class="fas fa-circle"></i> Medium Fragments
                                    </button>
                                    <button id="drawSmallBtn" class="btn btn-info btn-sm mr-2">
                                        <i class="fas fa-brush"></i> Small Fragments
                                    </button>
                                    <button id="stopDrawingBtn" class="btn btn-danger btn-sm mr-2" style="display: none;">
                                        <i class="fas fa-stop"></i> Stop
                                    </button>
                                    <button id="clearCanvasBtn" class="btn btn-outline-secondary btn-sm mr-2">
                                        <i class="fas fa-eraser"></i> Clear Canvas
                                    </button>
                                    <button id="clearVideosBtn" class="btn btn-outline-danger btn-sm mr-2">
                                        <i class="fas fa-trash"></i> Clear Videos
                                    </button>
                                    <button id="generateVideoBtn" class="btn btn-success btn-sm">
                                        <i class="fas fa-video"></i> Generate Video
                                    </button>
                                </div>
                            </div>
                            <div id="drawingProgress" class="mb-3" style="display: none;">
                                <div class="alert alert-info">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <strong>Drawing:</strong> <span id="currentDrawingSegment">-</span> | 
                                            <strong>Progress:</strong> <span id="drawingProgressInfo">-</span> |
                                            <strong>Brush:</strong> <span id="currentBrushType">Pencil</span> |
                                            <strong>Time:</strong> <span id="drawingTimer">00:00</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="text-center">
                                        <h6 class="mb-2">Drawing Preview</h6>
                                        <div class="position-relative d-inline-block">
                                            <canvas id="drawingCanvas" style="border: 2px solid #ddd; cursor: crosshair; max-width: 100%; height: auto;"></canvas>
                                            <canvas id="segmentOverlay" style="position: absolute; top: 0; left: 0; pointer-events: none; border: 2px solid transparent;"></canvas>
                                        </div>
                                        
                                        <!-- Progress Bars under Canvas -->
                                        <div id="canvasProgressBars" class="mt-3" style="display: none;">
                                            <div class="mb-2">
                                                <div class="d-flex justify-content-between align-items-center mb-1">
                                                    <small class="text-muted fw-bold">Drawing Progress</small>
                                                    <small id="drawingProgressPercent" class="text-muted">0%</small>
                                                </div>
                                                <div class="progress" style="height: 12px;">
                                                    <div id="drawingProgressBar" class="progress-bar bg-warning" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                                                </div>
                                            </div>
                                            <div id="videoProgressContainer" style="display: none;">
                                                <div class="d-flex justify-content-between align-items-center mb-1">
                                                    <small class="text-muted fw-bold">Video Generation</small>
                                                    <small id="videoProgressText" class="text-muted">Processing...</small>
                                                </div>
                                                <div class="progress" style="height: 12px;">
                                                    <div id="videoProgressBar" class="progress-bar bg-success progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="mt-2">
                                            <small class="text-muted">Click "Draw All" or use individual "Draw" buttons in the table below</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="text-center">
                                        <h6 class="mb-2">Video Result</h6>
                                        <div id="videoResultContainer" style="display: none;">
                                            <video id="embeddedResultVideo" controls style="border: 2px solid #ddd; max-width: 100%; height: auto;">
                                                Your browser does not support the video tag.
                                            </video>
                                            <div class="mt-2">
                                                <button id="embeddedDownloadBtn" class="btn btn-success btn-sm">
                                                    <i class="fas fa-download"></i> Download Video
                                                </button>
                                            </div>
                                        </div>
                                        <div id="videoPlaceholder" class="d-flex align-items-center justify-content-center" style="border: 2px dashed #ddd; min-height: 200px; background-color: #f8f9fa;">
                                            <div class="text-center text-muted">
                                                <i class="fas fa-video fa-3x mb-2"></i>
                                                <p>Video will appear here after drawing</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    // Always append the canvas container (either preserved or new)
                    infoBody.appendChild(canvasContainerToUse);
                    
                    // Initialize interactive canvas AFTER creating the elements
                    // Only initialize if not preserving existing canvas state
                    if (!canvasData.preserveCanvasState) {
                        setTimeout(() => initializeInteractiveCanvas(fileId), 100);
                    } else {
                        // Update segment info but preserve canvas
                        setTimeout(() => updateSegmentInfoOnly(fileId), 100);
                    }
                    
                    // Segments table
                    const table = document.createElement('table');
                    table.className = 'table table-sm table-striped';
                    table.innerHTML = `
                        <thead>
                            <tr>
                                <th>Segment ID</th>
                                <th>Pixels</th>
                                <th>Average Color</th>
                                <th>RGB</th>
                                <th>Hex</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    `;
                    
                    const tbody = table.querySelector('tbody');
                    
                    // Sort segments by pixel count (largest first)
                    const sortedSegments = segmentData.segments.sort((a, b) => b.pixel_count - a.pixel_count);
                    
                    sortedSegments.forEach((segment, index) => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${segment.id}</td>
                            <td>${segment.pixel_count.toLocaleString()}</td>
                            <td>
                                <div style="width: 30px; height: 20px; background-color: ${segment.average_color_hex}; border: 1px solid #ccc; display: inline-block;"></div>
                            </td>
                            <td>rgb(${segment.average_color.r}, ${segment.average_color.g}, ${segment.average_color.b})</td>
                            <td><code>${segment.average_color_hex}</code></td>
                            <td>
                                <button class="btn btn-sm btn-success" onclick="drawSingleSegment(${segment.id})" id="drawBtn_${segment.id}">
                                    <i class="fas fa-brush"></i> Draw
                                </button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                    infoBody.appendChild(table);
                    infoCard.appendChild(infoHeader);
                    infoCard.appendChild(infoBody);
                    infoCol.appendChild(infoCard);
                    
                    // Add info section AFTER the images container, not inside it
                    const segmentationResults = document.getElementById('segmentationResults');
                    const cardBody = segmentationResults.querySelector('.card-body');
                    
                    // When preserving canvas state, handle the info section carefully
                    if (canvasData.preserveCanvasState) {
                        const existingInfoCol = cardBody.querySelector('.col-12.mt-3');
                        if (existingInfoCol) {
                            // Move the existing canvas container to the new info body before replacing
                            const existingCanvasContainer = existingInfoCol.querySelector('.canvas-container');
                            if (existingCanvasContainer) {
                                console.log('Moving existing canvas container to preserve state');
                                // Remove the placeholder canvas container from new infoBody
                                const newCanvasContainer = infoBody.querySelector('.canvas-container');
                                if (newCanvasContainer) {
                                    newCanvasContainer.remove();
                                }
                                // Insert the existing canvas container in the right position (after summary, before table)
                                const table = infoBody.querySelector('table');
                                infoBody.insertBefore(existingCanvasContainer, table);
                            }
                            existingInfoCol.replaceWith(infoCol);
                        } else {
                            cardBody.appendChild(infoCol);
                        }
                    } else {
                        cardBody.appendChild(infoCol);
                    }
                    
                })
                .catch(error => {
                    console.error('Error loading segment data:', error);
                });
            }
            
            // Ensure results section is visible
            resultsSection.style.display = 'block';
            
            // Force visibility of the images container
            if (imagesContainer) {
                imagesContainer.style.display = 'block';
                imagesContainer.style.visibility = 'visible';
                console.log('Images container made visible, children count:', imagesContainer.children.length);
                
                // Log all image cards for debugging
                const imageCards = imagesContainer.querySelectorAll('.col-md-4, .col-sm-6');
                console.log('Image cards found:', imageCards.length);
                imageCards.forEach((card, index) => {
                    console.log(`Card ${index}:`, card.style.display, card.style.opacity);
                    // Ensure each card is visible
                    card.style.display = 'block';
                    card.style.opacity = '1';
                });
            }
            
            // Scroll to results if not preserving canvas state
            if (!canvasData.preserveCanvasState) {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            resetSegmentButton();
            
            // Show success message
            if (canvasData.preserveCanvasState) {
                showAlert('Segmentation updated successfully while preserving your drawing!', 'success');
            } else {
                showAlert('Segmentation completed successfully!', 'success');
            }
            
            // Force a final DOM refresh
            setTimeout(() => {
                console.log('Final DOM check - images container children:', imagesContainer?.children.length);
                if (imagesContainer) {
                    // Force repaint
                    imagesContainer.style.transform = 'translateZ(0)';
                    setTimeout(() => {
                        imagesContainer.style.transform = '';
                    }, 10);
                }
            }, 100);
        }
    })
    .catch(error => {
        console.error('Error loading segmentation results:', error);
        showAlert('Error loading segmentation results', 'danger');
        resetSegmentButton();
    });
}

function initializeInteractiveCanvas(fileId) {
    const canvas = document.getElementById('drawingCanvas');
    const overlay = document.getElementById('segmentOverlay');
    
    if (!canvas || !canvasData.meanColorImage) return;
    
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const overlayCtx = overlay.getContext('2d');
    
    // Load the mean color image
    const img = new Image();
    img.onload = function() {
        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;
        overlay.width = img.width;
        overlay.height = img.height;
        
        // Apply background color and draw the mean color image
        applyCanvasBackground(ctx, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        
        // Store original image data for segment detection
        canvasData.originalSegments = ctx.getImageData(0, 0, canvas.width, canvas.height);
        
        // Add click event listener
        canvas.addEventListener('click', handleCanvasClick);
        
        // Mark canvas as initialized
        canvasData.canvasInitialized = true;
    };
    img.src = canvasData.meanColorImage;
    
    // Initialize buttons and selectors
    document.getElementById('drawAllBtn').addEventListener('click', drawAllSegments);
    document.getElementById('drawLargeBtn').addEventListener('click', drawLargeFragments);
    document.getElementById('drawMediumBtn').addEventListener('click', drawMediumFragments);
    document.getElementById('drawSmallBtn').addEventListener('click', drawSmallFragments);
    document.getElementById('stopDrawingBtn').addEventListener('click', stopDrawingAll);
    document.getElementById('clearCanvasBtn').addEventListener('click', clearCanvas);
    document.getElementById('clearVideosBtn').addEventListener('click', clearVideoResults);
    document.getElementById('generateVideoBtn').addEventListener('click', generateVideoFromFrames);
    
    // Initialize brush type selector
    const brushSelector = document.getElementById('brushTypeSelect');
    if (brushSelector) {
        brushSelector.addEventListener('change', function() {
            canvasData.currentBrushType = this.value;
            updateBrushTypeDisplay();
        });
        // Set initial brush type
        canvasData.currentBrushType = brushSelector.value;
        updateBrushTypeDisplay();
    }
    
    // Initialize background color selector
    const backgroundSelector = document.getElementById('backgroundColorSelect');
    if (backgroundSelector) {
        backgroundSelector.addEventListener('change', function() {
            canvasData.backgroundColor = this.value;
            // Apply new background immediately
            if (canvas.width > 0) {
                applyCanvasBackground(ctx, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            }
        });
        canvasData.backgroundColor = backgroundSelector.value;
    }
}

// New function to update only segment info without reinitializing canvas
function updateSegmentInfoOnly(fileId) {
    // Update segment info for new fragmentation while preserving canvas drawing
    const canvas = document.getElementById('drawingCanvas');
    const overlay = document.getElementById('segmentOverlay');
    
    if (!canvas) {
        console.warn('Canvas not found for segment info update');
        return;
    }
    
    console.log('Updating segment info while preserving canvas state');
    console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
    
    // Store current canvas state before any updates
    const ctx = canvas.getContext('2d');
    const currentImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Re-initialize button event listeners if needed
    const drawAllBtn = document.getElementById('drawAllBtn');
    const drawLargeBtn = document.getElementById('drawLargeBtn');
    const drawMediumBtn = document.getElementById('drawMediumBtn');
    const drawSmallBtn = document.getElementById('drawSmallBtn');
    const stopBtn = document.getElementById('stopDrawingBtn');
    const clearBtn = document.getElementById('clearCanvasBtn');
    
    // Remove existing event listeners to avoid duplicates
    if (drawAllBtn) {
        drawAllBtn.replaceWith(drawAllBtn.cloneNode(true));
        document.getElementById('drawAllBtn').addEventListener('click', drawAllSegments);
    }
    if (drawLargeBtn) {
        drawLargeBtn.replaceWith(drawLargeBtn.cloneNode(true));
        document.getElementById('drawLargeBtn').addEventListener('click', drawLargeFragments);
    }
    if (drawMediumBtn) {
        drawMediumBtn.replaceWith(drawMediumBtn.cloneNode(true));
        document.getElementById('drawMediumBtn').addEventListener('click', drawMediumFragments);
    }
    if (drawSmallBtn) {
        drawSmallBtn.replaceWith(drawSmallBtn.cloneNode(true));
        document.getElementById('drawSmallBtn').addEventListener('click', drawSmallFragments);
    }
    if (stopBtn) {
        stopBtn.replaceWith(stopBtn.cloneNode(true));
        document.getElementById('stopDrawingBtn').addEventListener('click', stopDrawingAll);
    }
    if (clearBtn) {
        clearBtn.replaceWith(clearBtn.cloneNode(true));
        document.getElementById('clearCanvasBtn').addEventListener('click', clearCanvas);
    }
    
    const clearVideosBtn = document.getElementById('clearVideosBtn');
    if (clearVideosBtn) {
        clearVideosBtn.replaceWith(clearVideosBtn.cloneNode(true));
        document.getElementById('clearVideosBtn').addEventListener('click', clearVideoResults);
    }
    
    const generateVideoBtn = document.getElementById('generateVideoBtn');
    if (generateVideoBtn) {
        generateVideoBtn.replaceWith(generateVideoBtn.cloneNode(true));
        document.getElementById('generateVideoBtn').addEventListener('click', generateVideoFromFrames);
    }
    
    // Update brush and background selectors
    const brushSelector = document.getElementById('brushTypeSelect');
    if (brushSelector) {
        brushSelector.replaceWith(brushSelector.cloneNode(true));
        const newBrushSelector = document.getElementById('brushTypeSelect');
        newBrushSelector.addEventListener('change', function() {
            canvasData.currentBrushType = this.value;
            updateBrushTypeDisplay();
        });
        newBrushSelector.value = canvasData.currentBrushType;
    }
    
    const backgroundSelector = document.getElementById('backgroundColorSelect');
    if (backgroundSelector) {
        backgroundSelector.replaceWith(backgroundSelector.cloneNode(true));
        const newBackgroundSelector = document.getElementById('backgroundColorSelect');
        newBackgroundSelector.addEventListener('change', function() {
            canvasData.backgroundColor = this.value;
            // Don't redraw the canvas background to preserve drawing
        });
        newBackgroundSelector.value = canvasData.backgroundColor;
    }
    
    // Restore the exact canvas state after any DOM manipulations
    setTimeout(() => {
        const finalCanvas = document.getElementById('drawingCanvas');
        if (finalCanvas && currentImageData) {
            const finalCtx = finalCanvas.getContext('2d');
            // Ensure canvas dimensions haven't changed
            if (finalCanvas.width !== currentImageData.width || finalCanvas.height !== currentImageData.height) {
                finalCanvas.width = currentImageData.width;
                finalCanvas.height = currentImageData.height;
            }
            // Restore the exact pixel data
            finalCtx.putImageData(currentImageData, 0, 0);
            console.log('Canvas state fully restored after segment update');
        }
    }, 50);
    
    showAlert('Segmentation updated while preserving your drawing!', 'success');
}

function applyCanvasBackground(ctx, width, height) {
    if (canvasData.backgroundColor && canvasData.backgroundColor !== 'transparent') {
        ctx.fillStyle = canvasData.backgroundColor;
        ctx.fillRect(0, 0, width, height);
    } else {
        ctx.clearRect(0, 0, width, height);
    }
}

function stopDrawingAll() {
    canvasData.drawingInterrupted = true;
    canvasData.isDrawingAll = false;
    
    // Hide stop button, show draw buttons
    const stopBtn = document.getElementById('stopDrawingBtn');
    const drawAllBtn = document.getElementById('drawAllBtn');
    const drawLargeBtn = document.getElementById('drawLargeBtn');
    const drawMediumBtn = document.getElementById('drawMediumBtn');
    const drawSmallBtn = document.getElementById('drawSmallBtn');
    const progressDiv = document.getElementById('drawingProgress');
    
    if (stopBtn) stopBtn.style.display = 'none';
    if (drawAllBtn) {
        drawAllBtn.disabled = false;
        drawAllBtn.innerHTML = '<i class="fas fa-palette"></i> Draw All';
        drawAllBtn.style.display = 'inline-block';
    }
    if (drawLargeBtn) {
        drawLargeBtn.disabled = false;
        drawLargeBtn.innerHTML = '<i class="fas fa-expand"></i> Large Fragments';
        drawLargeBtn.style.display = 'inline-block';
    }
    if (drawMediumBtn) {
        drawMediumBtn.disabled = false;
        drawMediumBtn.innerHTML = '<i class="fas fa-circle"></i> Medium Fragments';
        drawMediumBtn.style.display = 'inline-block';
    }
    if (drawSmallBtn) {
        drawSmallBtn.disabled = false;
        drawSmallBtn.innerHTML = '<i class="fas fa-brush"></i> Small Fragments';
        drawSmallBtn.style.display = 'inline-block';
    }
    if (progressDiv) progressDiv.style.display = 'none';
    
    // Reset progress bars
    resetProgressBars();
    
    // Hide video generation progress
    showVideoGenerationProgress(false);
    
    stopDrawingTimer();
    showAlert('Drawing stopped by user', 'warning');
}

function handleCanvasClick(event) {
    const canvas = document.getElementById('drawingCanvas');
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);
    
    // Get pixel color at click position
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(x, y, 1, 1);
    const pixel = imageData.data;
    const clickedColor = `rgb(${pixel[0]}, ${pixel[1]}, ${pixel[2]})`;
    
    // Find matching segment
    const matchingSegment = canvasData.segmentInfo.segments.find(segment => {
        const segmentColor = `rgb(${segment.average_color.r}, ${segment.average_color.g}, ${segment.average_color.b})`;
        return segmentColor === clickedColor;
    });
    
    if (matchingSegment) {
        selectSegment(matchingSegment.id);
    }
}

function selectSegment(segmentId) {
    canvasData.selectedSegmentId = segmentId;
    
    const segment = canvasData.segmentInfo.segments.find(s => s.id === segmentId);
    if (!segment) return;
    
    // Check if interactive canvas elements exist
    const selectedSegmentId = document.getElementById('selectedSegmentId');
    const selectedSegmentColor = document.getElementById('selectedSegmentColor');
    const selectedSegmentPixels = document.getElementById('selectedSegmentPixels');
    const selectedSegmentInfo = document.getElementById('selectedSegmentInfo');
    const drawSegmentBtn = document.getElementById('drawSegmentBtn');
    
    // Only update UI if elements exist
    if (selectedSegmentId) {
        selectedSegmentId.textContent = segmentId;
    }
    
    if (selectedSegmentColor) {
        selectedSegmentColor.innerHTML = `
            <div style="width: 20px; height: 15px; background-color: ${segment.average_color_hex}; border: 1px solid #ccc; display: inline-block; margin-right: 5px;"></div>
            ${segment.average_color_hex}
        `;
    }
    
    if (selectedSegmentPixels) {
        selectedSegmentPixels.textContent = segment.pixel_count.toLocaleString();
    }
    
    if (selectedSegmentInfo) {
        selectedSegmentInfo.style.display = 'block';
    }
    
    if (drawSegmentBtn) {
        drawSegmentBtn.disabled = false;
    }
    
    // Highlight segment on overlay if canvas exists
    const canvas = document.getElementById('drawingCanvas');
    if (canvas) {
        highlightSegment(segmentId);
    }
}

function highlightSegment(segmentId) {
    const overlay = document.getElementById('segmentOverlay');
    const canvas = document.getElementById('drawingCanvas');
    
    // Check if required elements exist
    if (!overlay || !canvas) {
        console.warn('Canvas elements not found for highlighting');
        return;
    }
    
    const overlayCtx = overlay.getContext('2d');
    const ctx = canvas.getContext('2d');
    
    // Clear previous highlight
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    
    const segment = canvasData.segmentInfo.segments.find(s => s.id === segmentId);
    if (!segment) return;
    
    // Find all pixels of this segment and draw outline
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    const targetColor = [segment.average_color.r, segment.average_color.g, segment.average_color.b];
    
    // Create highlight overlay
    overlayCtx.strokeStyle = '#ff0000';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([5, 5]);
    
    // Find bounding box of segment
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const index = (y * canvas.width + x) * 4;
            const r = imageData.data[index];
            const g = imageData.data[index + 1];
            const b = imageData.data[index + 2];
            
            if (r === targetColor[0] && g === targetColor[1] && b === targetColor[2]) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }
    
    // Draw bounding rectangle
    if (maxX > minX && maxY > minY) {
        overlayCtx.strokeRect(minX - 2, minY - 2, maxX - minX + 4, maxY - minY + 4);
    }
}

function drawSingleSegment(segmentId) {
    if (!currentFileId) {
        showAlert('No file loaded', 'danger');
        return;
    }
    
    // Get current parameter values
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);

    // Show loading state
    const drawBtn = document.getElementById(`drawBtn_${segmentId}`);
    if (drawBtn) {
        drawBtn.disabled = true;
        drawBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Drawing...';
    }
    
    // Send request to backend to generate brush strokes for this segment
    fetch('/draw_segment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            segment_id: segmentId,
            brush_type: canvasData.currentBrushType,
            stroke_density: strokeDensity
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Apply brush strokes with enhanced visualization
            applyBrushStrokesFast(data.brush_strokes, segmentId);
            showAlert(`Successfully drew segment ${segmentId} with ${data.brush_strokes.length} ${brushTypes[canvasData.currentBrushType].name.toLowerCase()} strokes`, 'success');
        } else {
            showAlert('Error generating brush strokes: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error generating brush strokes', 'danger');
    })
    .finally(() => {
        // Reset button state
        if (drawBtn) {
            drawBtn.disabled = false;
            drawBtn.innerHTML = '<i class="fas fa-brush"></i> Draw';
        }
        
        // Update single segment progress (for individual draws)
        updateDrawingProgress(1, 1);
    });
}

function drawSelectedSegment() {
    if (!canvasData.selectedSegmentId) {
        showAlert('Please select a segment first', 'warning');
        return;
    }
    
    if (!currentFileId) {
        showAlert('No file loaded', 'danger');
        return;
    }
    
    const segmentId = canvasData.selectedSegmentId;
    
    // Show loading state and progress
    const drawBtn = document.getElementById('drawSegmentBtn');
    const progressDiv = document.getElementById('drawingProgress');
    const currentSegmentSpan = document.getElementById('currentDrawingSegment');
    const progressTextSpan = document.getElementById('drawingProgressText');
    const brushTypeSpan = document.getElementById('currentBrushType');
    
    if (drawBtn) {
        drawBtn.disabled = true;
        drawBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Drawing...';
    }
    
    // Show progress info and initialize progress bars for single segment
    if (progressDiv) progressDiv.style.display = 'block';
    if (currentSegmentSpan) currentSegmentSpan.textContent = segmentId;
    if (progressTextSpan) progressTextSpan.textContent = '1 / 1';
    if (brushTypeSpan) brushTypeSpan.textContent = brushTypes[canvasData.currentBrushType].name;
    
    // Initialize progress bars for single segment
    resetProgressBars();
    updateDrawingProgress(0, 1);
    
    startDrawingTimer();
    
    // Send request to backend to generate brush strokes for this segment
    fetch('/draw_segment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            segment_id: segmentId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Apply brush strokes with enhanced visualization
            applyBrushStrokesFast(data.brush_strokes, segmentId);
            showAlert(`Successfully drew segment ${segmentId} with ${data.brush_strokes.length} brush strokes`, 'success');
        } else {
            showAlert('Error generating brush strokes: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error generating brush strokes', 'danger');
    })
    .finally(() => {
        // Reset button state
        if (drawBtn) {
            drawBtn.disabled = false;
            drawBtn.innerHTML = '<i class="fas fa-brush"></i> Draw Selected Segment';
        }
        
        // Complete progress and hide progress display
        updateDrawingProgress(1, 1);
        setTimeout(() => {
            const progressDiv = document.getElementById('drawingProgress');
            if (progressDiv) progressDiv.style.display = 'none';
            resetProgressBars();
        }, 1000);
        
        stopDrawingTimer();
    });
}

function applyBrushStrokesFast(brushStrokes) {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    brushStrokes.forEach((stroke) => {
        // Get brush type configuration
        const brushType = stroke.type || canvasData.currentBrushType;
        const brushConfig = brushTypes[brushType] || brushTypes.pencil;
        
        // Apply brush-specific effects
        ctx.globalAlpha = brushConfig.opacity;
        ctx.globalCompositeOperation = brushConfig.blendMode;
        
        // Draw the actual brush stroke
        ctx.strokeStyle = stroke.color;
        ctx.lineWidth = stroke.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Apply minimal brush-specific effects for speed
        if (brushType === 'pencil') {
            ctx.globalAlpha = 0.95;
            ctx.globalCompositeOperation = 'source-over';
        }
        
        ctx.beginPath();
        if (stroke.points.length === 1) {
            // Single point - draw a dot
            ctx.arc(stroke.points[0].x, stroke.points[0].y, stroke.width / 2, 0, 2 * Math.PI);
            ctx.fillStyle = stroke.color;
            ctx.fill();
        } else {
            // Multiple points - draw a stroke
            ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
            for (let i = 1; i < stroke.points.length; i++) {
                ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
            }
            ctx.stroke();
        }
        
        // Reset context
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;
    });
}

function drawAllSegments() {
    if (!canvasData.segmentInfo || !currentFileId) {
        showAlert('No segment data available', 'danger');
        return;
    }
    
    // Get current parameter values
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);

    // Reset interruption flag
    canvasData.drawingInterrupted = false;
    canvasData.isDrawingAll = true;
    
    // Sort segments by pixel count (largest first)
    const sortedSegments = canvasData.segmentInfo.segments.sort((a, b) => b.pixel_count - a.pixel_count);
    
    drawSegmentsWithVideo(sortedSegments, 'all segments', videoDuration, videoFps, strokeDensity);
}

function drawSmallFragments() {
    if (!canvasData.segmentInfo || !currentFileId) {
        showAlert('No segment data available', 'danger');
        return;
    }
    
    // Get current parameter values
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);

    // Calculate average pixel count
    const totalPixels = canvasData.segmentInfo.segments.reduce((sum, segment) => sum + segment.pixel_count, 0);
    const averagePixels = totalPixels / canvasData.segmentInfo.segments.length;
    const largeThreshold = averagePixels * 10;
    
    console.log(`Average pixels per segment: ${averagePixels.toFixed(0)}`);
    console.log(`Large segment threshold (10x average): ${largeThreshold.toFixed(0)}`);
    
    // Filter out large segments (those with more than 10x average pixels)
    const smallSegments = canvasData.segmentInfo.segments.filter(segment => segment.pixel_count <= largeThreshold);
    
    // Sort small segments by pixel count (largest first)
    const sortedSmallSegments = smallSegments.sort((a, b) => b.pixel_count - a.pixel_count);
    
    console.log(`Total segments: ${canvasData.segmentInfo.segments.length}`);
    console.log(`Small segments to draw: ${sortedSmallSegments.length}`);
    console.log(`Large segments excluded: ${canvasData.segmentInfo.segments.length - sortedSmallSegments.length}`);
    
    if (sortedSmallSegments.length === 0) {
        showAlert('No small fragments found to draw', 'warning');
        return;
    }
    
    showAlert(`Drawing ${sortedSmallSegments.length} small fragments (excluding ${canvasData.segmentInfo.segments.length - sortedSmallSegments.length} large segments)`, 'info');
    
    // Reset interruption flag
    canvasData.drawingInterrupted = false;
    canvasData.isDrawingAll = true;
    
    drawSegmentsWithVideo(sortedSmallSegments, 'small fragments', videoDuration, videoFps, strokeDensity);
}

function drawLargeFragments() {
    if (!canvasData.segmentInfo || !currentFileId) {
        showAlert('No segment data available', 'danger');
        return;
    }
    
    // Get current parameter values
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);

    // Calculate average pixel count
    const totalPixels = canvasData.segmentInfo.segments.reduce((sum, segment) => sum + segment.pixel_count, 0);
    const averagePixels = totalPixels / canvasData.segmentInfo.segments.length;
    const largeThreshold = averagePixels * 10;
    
    console.log(`Average pixels per segment: ${averagePixels.toFixed(0)}`);
    console.log(`Large segment threshold (10x average): ${largeThreshold.toFixed(0)}`);
    
    // Filter only large segments (those with more than 10x average pixels)
    const largeSegments = canvasData.segmentInfo.segments.filter(segment => segment.pixel_count > largeThreshold);
    
    // Sort large segments by pixel count (largest first)
    const sortedLargeSegments = largeSegments.sort((a, b) => b.pixel_count - a.pixel_count);
    
    console.log(`Total segments: ${canvasData.segmentInfo.segments.length}`);
    console.log(`Large segments to draw: ${sortedLargeSegments.length}`);
    console.log(`Small/medium segments excluded: ${canvasData.segmentInfo.segments.length - sortedLargeSegments.length}`);
    
    if (sortedLargeSegments.length === 0) {
        showAlert('No large fragments found to draw', 'warning');
        return;
    }
    
    showAlert(`Drawing ${sortedLargeSegments.length} large fragments (excluding ${canvasData.segmentInfo.segments.length - sortedLargeSegments.length} smaller segments)`, 'info');
    
    // Reset interruption flag
    canvasData.drawingInterrupted = false;
    canvasData.isDrawingAll = true;
    
    drawSegmentsWithVideo(sortedLargeSegments, 'large fragments', videoDuration, videoFps, strokeDensity);
}

function drawMediumFragments() {
    if (!canvasData.segmentInfo || !currentFileId) {
        showAlert('No segment data available', 'danger');
        return;
    }
    
    // Get current parameter values
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);

    // Calculate average pixel count and thresholds
    const totalPixels = canvasData.segmentInfo.segments.reduce((sum, segment) => sum + segment.pixel_count, 0);
    const averagePixels = totalPixels / canvasData.segmentInfo.segments.length;
    const largeThreshold = averagePixels * 10;  // 10x больше среднего
    const smallThreshold = averagePixels / 2;   // 2x меньше среднего
    
    console.log(`Average pixels per segment: ${averagePixels.toFixed(0)}`);
    console.log(`Large segment threshold (10x average): ${largeThreshold.toFixed(0)}`);
    console.log(`Small segment threshold (0.5x average): ${smallThreshold.toFixed(0)}`);
    
    // Filter medium segments (between small and large thresholds)
    const mediumSegments = canvasData.segmentInfo.segments.filter(segment => 
        segment.pixel_count > smallThreshold && segment.pixel_count <= largeThreshold
    );
    
    // Sort medium segments by pixel count (largest first)
    const sortedMediumSegments = mediumSegments.sort((a, b) => b.pixel_count - a.pixel_count);
    
    const largeCount = canvasData.segmentInfo.segments.filter(s => s.pixel_count > largeThreshold).length;
    const smallCount = canvasData.segmentInfo.segments.filter(s => s.pixel_count <= smallThreshold).length;
    
    console.log(`Total segments: ${canvasData.segmentInfo.segments.length}`);
    console.log(`Medium segments to draw: ${sortedMediumSegments.length}`);
    console.log(`Large segments excluded: ${largeCount}`);
    console.log(`Small segments excluded: ${smallCount}`);
    
    if (sortedMediumSegments.length === 0) {
        showAlert('No medium fragments found to draw', 'warning');
        return;
    }
    
    showAlert(`Drawing ${sortedMediumSegments.length} medium fragments (excluding ${largeCount} large and ${smallCount} small segments)`, 'info');
    
    // Reset interruption flag
    canvasData.drawingInterrupted = false;
    canvasData.isDrawingAll = true;
    
    drawSegmentsWithVideo(sortedMediumSegments, 'medium fragments', videoDuration, videoFps, strokeDensity);
}

function drawSegmentsWithVideo(sortedSegments, description, videoDuration, videoFps, strokeDensity) {
    
    // Show progress and update buttons
    const progressDiv = document.getElementById('drawingProgress');
    const currentSegmentSpan = document.getElementById('currentDrawingSegment');
    const progressTextSpan = document.getElementById('drawingProgressText');
    const drawAllBtn = document.getElementById('drawAllBtn');
    const drawLargeBtn = document.getElementById('drawLargeBtn');
    const drawMediumBtn = document.getElementById('drawMediumBtn');
    const drawSmallBtn = document.getElementById('drawSmallBtn');
    const stopBtn = document.getElementById('stopDrawingBtn');
    
    if (progressDiv) progressDiv.style.display = 'block';
    if (drawAllBtn) {
        drawAllBtn.disabled = true;
        drawAllBtn.style.display = 'none';
    }
    if (drawLargeBtn) {
        drawLargeBtn.disabled = true;
        drawLargeBtn.style.display = 'none';
    }
    if (drawMediumBtn) {
        drawMediumBtn.disabled = true;
        drawMediumBtn.style.display = 'none';
    }
    if (drawSmallBtn) {
        drawSmallBtn.disabled = true;
        drawSmallBtn.style.display = 'none';
    }
    if (stopBtn) {
        stopBtn.style.display = 'inline-block';
    }
    
    // Update brush type display
    updateBrushTypeDisplay();
    
    // Reset and initialize progress bars
    resetProgressBars();
    updateDrawingProgress(0, sortedSegments.length);
    
    startDrawingTimer();
    
    // Initialize cumulative video recording
    const videoRecorder = new VideoRecorder({
        videoDuration: videoDuration,
        segmentCount: sortedSegments.length,
        canvas: document.getElementById('drawingCanvas'),
        frameRate: videoFps,
        timeCompression: true,
        isCumulative: true // Enable cumulative mode
    });
    
    // Store reference for potential future use
    canvasData.masterVideoRecorder = videoRecorder;
    
    // Start video recording
    videoRecorder.start();
    
    // Draw segments sequentially with dynamic delay
    let currentIndex = 0;
    
    function drawNextSegment() {
        // Check if drawing was interrupted
        if (canvasData.drawingInterrupted || !canvasData.isDrawingAll) {
            stopDrawingTimer();
            if (videoRecorder) {
                videoRecorder.stop();
            }
            return;
        }
        
        if (currentIndex >= sortedSegments.length) {
            // All segments drawn
            canvasData.isDrawingAll = false;
            if (progressDiv) progressDiv.style.display = 'none';
            if (drawAllBtn) {
                drawAllBtn.disabled = false;
                drawAllBtn.innerHTML = '<i class="fas fa-palette"></i> Draw All';
                drawAllBtn.style.display = 'inline-block';
            }
            if (drawLargeBtn) {
                drawLargeBtn.disabled = false;
                drawLargeBtn.innerHTML = '<i class="fas fa-expand"></i> Large Fragments';
                drawLargeBtn.style.display = 'inline-block';
            }
            if (drawMediumBtn) {
                drawMediumBtn.disabled = false;
                drawMediumBtn.innerHTML = '<i class="fas fa-circle"></i> Medium Fragments';
                drawMediumBtn.style.display = 'inline-block';
            }
            if (drawSmallBtn) {
                drawSmallBtn.disabled = false;
                drawSmallBtn.innerHTML = '<i class="fas fa-brush"></i> Small Fragments';
                drawSmallBtn.style.display = 'inline-block';
            }
            if (stopBtn) stopBtn.style.display = 'none';
            stopDrawingTimer();
            
            // Just capture final frame, don't auto-generate video
            if (videoRecorder) {
                videoRecorder.captureFrame(); // Final frame
                videoRecorder.stop(); // Stop recording but don't finalize
                showAlert('Drawing completed! Click "Generate Video" to create video from captured frames.', 'success');
            }
            
            showAlert(`Successfully drew ${description} (${sortedSegments.length} segments) with ${brushTypes[canvasData.currentBrushType].name}!`, 'success');
            return;
        }
        
        const segment = sortedSegments[currentIndex];
        
        // Update progress
        if (currentSegmentSpan) currentSegmentSpan.textContent = segment.id;
        if (progressTextSpan) progressTextSpan.textContent = `${currentIndex + 1} / ${sortedSegments.length}`;
        
        // Update drawing progress bar
        updateDrawingProgress(currentIndex + 1, sortedSegments.length);
        
        // Capture frame before drawing this segment
        if (videoRecorder) {
            videoRecorder.captureFrame();
        }
        
        // Draw this segment
        fetch('/draw_segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: currentFileId,
                segment_id: segment.id,
                brush_type: canvasData.currentBrushType,
                stroke_density: strokeDensity
            })
        })
        .then(response => response.json())
        .then(data => {
            // Check again if drawing was interrupted
            if (canvasData.drawingInterrupted || !canvasData.isDrawingAll) {
                stopDrawingTimer();
                if (videoRecorder) {
                    videoRecorder.stop();
                }
                return;
            }
            
            if (data.success) {
                // Apply brush strokes with enhanced visualization
                applyBrushStrokesFast(data.brush_strokes, segment.id);
                
                // Capture multiple frames during segment drawing for smoother video
                if (videoRecorder) {
                    // Capture frame immediately after drawing
                    videoRecorder.captureFrame();
                    
                    // For larger segments, capture additional intermediate frames
                    if (segment.pixel_count > 500) {
                        setTimeout(() => {
                            if (videoRecorder.isRecording) {
                                videoRecorder.captureFrame();
                            }
                        }, 50);
                    }
                }
                
                // Calculate dynamic delay based on segment size
                let delay = calculateSegmentDelay(segment.pixel_count);
                
                // Move to next segment after calculated delay
                currentIndex++;
                setTimeout(drawNextSegment, delay);
            } else {
                console.error(`Error drawing segment ${segment.id}:`, data.error);
                currentIndex++;
                setTimeout(drawNextSegment, 50); // Very short delay on error
            }
        })
        .catch(error => {
            console.error('Error:', error);
            currentIndex++;
            setTimeout(drawNextSegment, 50); // Very short delay on error
        });
    }
    
    // Start drawing
    drawNextSegment();
}

function calculateSegmentDelay(pixelCount) {
    // Much faster delays - aim to complete all segments within target time
    const baseDelay = 50; // Reduced from 500ms to 50ms
    const maxDelay = 200; // Reduced from 2000ms to 200ms
    const minDelay = 10;  // Reduced from 100ms to 10ms
    
    // Calculate delay based on segment size (smaller segments = shorter delay)
    let delay;
    if (pixelCount < 100) {
        delay = minDelay;
    } else if (pixelCount < 1000) {
        delay = baseDelay;
    } else if (pixelCount < 5000) {
        delay = baseDelay * 1.5;
    } else {
        delay = Math.min(maxDelay, baseDelay * 2);
    }
    
    return Math.floor(delay);
}

function clearCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    if (!canvas) {
        console.warn('Drawing canvas not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Clear the entire canvas first
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply background color if selected (not transparent)
    if (canvasData.backgroundColor && canvasData.backgroundColor !== 'transparent') {
        ctx.fillStyle = canvasData.backgroundColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        showAlert('Canvas cleared with background color applied', 'info');
    } else if (canvasData.meanColorImage) {
        // If transparent, reload the original mean color image
        const img = new Image();
        img.onload = function() {
            ctx.drawImage(img, 0, 0);
            showAlert('Canvas cleared to original segmentation', 'info');
        };
        img.onerror = function() {
            console.error('Failed to load mean color image for canvas clearing');
            showAlert('Error clearing canvas', 'danger');
        };
        img.src = canvasData.meanColorImage;
        return; // Exit early since we're loading async
    } else {
        showAlert('Canvas cleared', 'info');
    }
}

function hideSegmentationResults() {
    document.getElementById('segmentationResults').style.display = 'none';
}

function resetSegmentButton() {
    const segmentBtn = document.getElementById('segmentBtn');
    segmentBtn.disabled = false;
    segmentBtn.innerHTML = '<i class="fas fa-eye"></i> Show Segmentation';
    document.getElementById('processingStatus').style.display = 'none';
}

function resetApp() {
    currentFileId = null;
    clearImage();
    hideResults();
    hideSegmentationResults();
    
    // Clean up video resources before reset
    canvasData.videoResults.forEach(videoInfo => {
        if (videoInfo.url.startsWith('blob:')) {
            URL.revokeObjectURL(videoInfo.url);
        }
    });
    
    // Reset canvas data completely
    canvasData = {
        segments: null,
        segmentInfo: null,
        selectedSegmentId: null,
        meanColorImage: null,
        originalSegments: null,
        currentBrushType: 'pencil',
        backgroundColor: '#fff3cd',
        isDrawingAll: false,
        drawingInterrupted: false,
        canvasInitialized: false,
        preserveCanvasState: false,
        videoResults: [],
        cumulativeFrames: [],
        masterVideoRecorder: null
    };
    
    // Reset form values
    document.getElementById('styleSelect').value = 'pencil';
    document.getElementById('colorThreshold').value = 10;
    document.getElementById('thresholdValue').textContent = '10';
    document.getElementById('videoDuration').value = 10;
    document.getElementById('durationValue').textContent = '10';
    updateStyleDescription();
    
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    resetSegmentButton();
}

function generateVideoFromFrames() {
    if (canvasData.cumulativeFrames.length === 0) {
        showAlert('No frames captured yet. Please draw something first!', 'warning');
        return;
    }
    
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);
    
    showAlert(`Generating video from ${canvasData.cumulativeFrames.length} captured frames...`, 'info');
    
    // Disable the generate video button during processing
    const generateBtn = document.getElementById('generateVideoBtn');
    if (generateBtn) {
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    }
    
    // Show progress with detailed tracking
    showVideoGenerationProgressWithPercent(0);
    
    // Create a video recorder for manual generation
    const videoRecorder = new VideoRecorder({
        videoDuration: videoDuration,
        segmentCount: canvasData.cumulativeFrames.length,
        canvas: document.getElementById('drawingCanvas'),
        frameRate: videoFps,
        timeCompression: true,
        isCumulative: true
    });
    
    // Set the captured frames directly
    videoRecorder.capturedFrames = [...canvasData.cumulativeFrames];
    
    // Generate video from existing frames with progress tracking
    videoRecorder.finalizeWithProgress((progress) => {
        showVideoGenerationProgressWithPercent(progress);
    }).then((videoUrl) => {
        showVideoGenerationProgressWithPercent(100);
        
        setTimeout(() => {
            showVideoGenerationProgress(false);
            
            if (videoUrl) {
                showVideoResults(videoUrl);
                showAlert('Video generated successfully!', 'success');
            } else {
                console.warn('Video generation failed');
                showAlert('Video generation failed', 'danger');
            }
            
            // Re-enable the generate video button
            if (generateBtn) {
                generateBtn.disabled = false;
                generateBtn.innerHTML = '<i class="fas fa-video"></i> Generate Video';
            }
        }, 500);
        
    }).catch((error) => {
        console.error('Video generation error:', error);
        showVideoGenerationProgress(false);
        showAlert('Video generation failed: ' + error.message, 'danger');
        
        // Re-enable the generate video button
        if (generateBtn) {
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-video"></i> Generate Video';
        }
    });
}

function clearVideoResults() {
    if (canvasData.videoResults.length === 0 && canvasData.cumulativeFrames.length === 0) {
        showAlert('No video to clear', 'info');
        return;
    }
    
    // Confirm before clearing
    const frameCount = canvasData.cumulativeFrames.length;
    if (confirm(`Are you sure you want to clear the cumulative video? This will delete ${frameCount} recorded frames and cannot be undone.`)) {
        // Revoke object URLs to free memory
        canvasData.videoResults.forEach(videoInfo => {
            if (videoInfo.url.startsWith('blob:')) {
                URL.revokeObjectURL(videoInfo.url);
            }
        });
        
        // Clear all video data
        canvasData.videoResults = [];
        canvasData.cumulativeFrames = [];
        canvasData.masterVideoRecorder = null;
        
        // Update display
        const videoContainer = document.getElementById('videoResultContainer');
        const videoPlaceholder = document.getElementById('videoPlaceholder');
        
        if (videoContainer) {
            videoContainer.style.display = 'none';
        }
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'flex';
        }
        
        // Hide original results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
        
        showAlert('Cumulative video cleared - starting fresh on next drawing session', 'success');
    }
}

function updateStyleDescription() {
    const style = document.getElementById('styleSelect').value;
    const description = document.getElementById('styleDescription');
    const stylePreview = document.querySelector('.style-preview');

    description.textContent = styleDescriptions[style];
    
    // Update style-specific styling
    stylePreview.className = `style-preview p-3 rounded bg-light style-${style}`;
}

function updateBrushTypeDisplay() {
    const currentBrushSpan = document.getElementById('currentBrushType');
    if (currentBrushSpan && brushTypes[canvasData.currentBrushType]) {
        currentBrushSpan.textContent = brushTypes[canvasData.currentBrushType].name;
    }
}

function startDrawingTimer() {
    drawingStartTime = Date.now();
    const timerDisplay = document.getElementById('drawingTimer');
    if (timerDisplay) {
        drawingTimer = setInterval(() => {
            const elapsed = Date.now() - drawingStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
}

function stopDrawingTimer() {
    if (drawingTimer) {
        clearInterval(drawingTimer);
        drawingTimer = null;
    }
    if (drawingStartTime) {
        const elapsed = Date.now() - drawingStartTime;
        const minutes = Math.floor(elapsed / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        const finalTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        const timerDisplay = document.getElementById('drawingTimer');
        if (timerDisplay) {
            timerDisplay.textContent = finalTime;
        }
        
        showAlert(`Drawing completed in ${finalTime}`, 'success');
        drawingStartTime = null;
    }
}

function showAlert(message, type) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());

    // Create new alert
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alert);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

class VideoRecorder {
    constructor(options) {
        this.videoDuration = options.videoDuration;
        this.segmentCount = options.segmentCount;
        this.canvas = options.canvas;
        this.frameRate = options.frameRate || 30;
        this.timeCompression = options.timeCompression || false;
        this.isCumulative = options.isCumulative || false;
        
        this.capturedFrames = [];
        this.isRecording = false;
        this.totalFramesNeeded = this.videoDuration * this.frameRate;
        this.framesPerSegment = Math.max(1, Math.floor(this.totalFramesNeeded / this.segmentCount));
        
        console.log(`Video setup: ${this.videoDuration}s @ ${this.frameRate}fps = ${this.totalFramesNeeded} frames`);
        console.log(`Frames per segment: ${this.framesPerSegment}`);
        console.log(`Cumulative mode: ${this.isCumulative}`);
    }
    
    start() {
        this.isRecording = true;
        
        if (this.isCumulative && canvasData.cumulativeFrames.length > 0) {
            // Continue from existing frames
            this.capturedFrames = [...canvasData.cumulativeFrames];
            console.log(`Resuming cumulative recording with ${this.capturedFrames.length} existing frames`);
        } else {
            // Start fresh
            this.capturedFrames = [];
            // Capture initial canvas frame
            this.captureFrame();
        }
        
        console.log(`Started ${this.isCumulative ? 'cumulative' : 'new'} video recording for ${this.videoDuration}s at ${this.frameRate}fps`);
        return true;
    }
    
    captureFrame() {
        if (!this.isRecording) return;
        
        try {
            // Capture canvas as data URL
            const dataURL = this.canvas.toDataURL('image/jpeg', 0.8);
            this.capturedFrames.push(dataURL);
            
            // Log progress occasionally
            if (this.capturedFrames.length % 10 === 0) {
                console.log(`Captured ${this.capturedFrames.length} frames`);
            }
        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    }
    
    stop() {
        this.isRecording = false;
        
        if (this.isCumulative) {
            // Store frames for future sessions
            canvasData.cumulativeFrames = [...this.capturedFrames];
            console.log(`Stopped cumulative recording. Total frames: ${this.capturedFrames.length}`);
        } else {
            console.log(`Stopped recording. Captured ${this.capturedFrames.length} frames`);
        }
    }
    
    async finalize() {
        this.stop();
        
        if (this.capturedFrames.length === 0) {
            console.warn('No frames captured for video');
            return null;
        }
        
        console.log(`Creating ${this.isCumulative ? 'cumulative' : 'time-compressed'} video from ${this.capturedFrames.length} frames`);
        
        try {
            // Create a temporary canvas for video generation
            const videoCanvas = document.createElement('canvas');
            const ctx = videoCanvas.getContext('2d');
            
            // Set canvas size to match original
            videoCanvas.width = this.canvas.width;
            videoCanvas.height = this.canvas.height;
            
            // Create video stream from temporary canvas
            const stream = videoCanvas.captureStream(this.frameRate);
            
            // Set up MediaRecorder
            const mimeType = MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4';
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 2500000
            });
            
            const recordedChunks = [];
            
            return new Promise((resolve) => {
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    const videoBlob = new Blob(recordedChunks, { type: mimeType });
                    const videoUrl = URL.createObjectURL(videoBlob);
                    console.log(`${this.isCumulative ? 'Cumulative' : 'Time-compressed'} video created: ${(videoBlob.size / 1024 / 1024).toFixed(2)} MB`);
                    resolve(videoUrl);
                };
                
                // Start recording
                mediaRecorder.start();
                
                // Play back frames at the correct timing
                this.playbackFrames(videoCanvas, ctx).then(() => {
                    mediaRecorder.stop();
                });
            });
            
        } catch (error) {
            console.error('Error creating time-compressed video:', error);
            return null;
        }
    }
    
    async finalizeWithProgress(progressCallback) {
        this.stop();
        
        if (this.capturedFrames.length === 0) {
            console.warn('No frames captured for video');
            return null;
        }
        
        console.log(`Creating ${this.isCumulative ? 'cumulative' : 'time-compressed'} video from ${this.capturedFrames.length} frames`);
        
        try {
            // Create a temporary canvas for video generation
            const videoCanvas = document.createElement('canvas');
            const ctx = videoCanvas.getContext('2d');
            
            // Set canvas size to match original
            videoCanvas.width = this.canvas.width;
            videoCanvas.height = this.canvas.height;
            
            // Create video stream from temporary canvas
            const stream = videoCanvas.captureStream(this.frameRate);
            
            // Set up MediaRecorder
            const mimeType = MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : 'video/mp4';
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 2500000
            });
            
            const recordedChunks = [];
            
            return new Promise((resolve) => {
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    const videoBlob = new Blob(recordedChunks, { type: mimeType });
                    const videoUrl = URL.createObjectURL(videoBlob);
                    console.log(`${this.isCumulative ? 'Cumulative' : 'Time-compressed'} video created: ${(videoBlob.size / 1024 / 1024).toFixed(2)} MB`);
                    resolve(videoUrl);
                };
                
                // Start recording
                mediaRecorder.start();
                
                // Play back frames at the correct timing with progress tracking
                this.playbackFramesWithProgress(videoCanvas, ctx, progressCallback).then(() => {
                    mediaRecorder.stop();
                });
            });
            
        } catch (error) {
            console.error('Error creating time-compressed video:', error);
            return null;
        }
    }
    
    async playbackFrames(videoCanvas, ctx) {
        const frameInterval = 1000 / this.frameRate; // ms between frames
        const totalFrames = this.totalFramesNeeded;
        
        // Calculate how to distribute captured frames across target duration
        const frameStep = this.capturedFrames.length / totalFrames;
        
        console.log(`Playing back ${totalFrames} frames over ${this.videoDuration}s`);
        
        for (let i = 0; i < totalFrames; i++) {
            // Calculate which captured frame to use
            const capturedFrameIndex = Math.floor(i * frameStep);
            const frameIndex = Math.min(capturedFrameIndex, this.capturedFrames.length - 1);
            
            // Load and draw the frame
            await this.drawFrameToCanvas(ctx, this.capturedFrames[frameIndex]);
            
            // Wait for next frame timing
            if (i < totalFrames - 1) {
                await new Promise(resolve => setTimeout(resolve, frameInterval));
            }
        }
        
        console.log('Frame playback completed');
    }
    
    async playbackFramesWithProgress(videoCanvas, ctx, progressCallback) {
        const frameInterval = 1000 / this.frameRate; // ms between frames
        const totalFrames = this.totalFramesNeeded;
        
        // Calculate how to distribute captured frames across target duration
        const frameStep = this.capturedFrames.length / totalFrames;
        
        console.log(`Playing back ${totalFrames} frames over ${this.videoDuration}s with progress tracking`);
        
        for (let i = 0; i < totalFrames; i++) {
            // Calculate which captured frame to use
            const capturedFrameIndex = Math.floor(i * frameStep);
            const frameIndex = Math.min(capturedFrameIndex, this.capturedFrames.length - 1);
            
            // Load and draw the frame
            await this.drawFrameToCanvas(ctx, this.capturedFrames[frameIndex]);
            
            // Update progress
            const progress = ((i + 1) / totalFrames) * 100;
            if (progressCallback) {
                progressCallback(progress);
            }
            
            // Wait for next frame timing
            if (i < totalFrames - 1) {
                await new Promise(resolve => setTimeout(resolve, frameInterval));
            }
        }
        
        console.log('Frame playback with progress completed');
    }
    
    drawFrameToCanvas(ctx, dataURL) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                ctx.drawImage(img, 0, 0);
                resolve();
            };
            img.onerror = () => {
                console.error('Failed to load frame image');
                resolve();
            };
            img.src = dataURL;
        });
    }
}

// Progress Bar Management
function updateDrawingProgress(current, total) {
    const drawingProgressBar = document.getElementById('drawingProgressBar');
    const drawingProgressPercent = document.getElementById('drawingProgressPercent');
    const drawingProgressInfo = document.getElementById('drawingProgressInfo');
    const canvasProgressBars = document.getElementById('canvasProgressBars');
    
    // Show progress bars container
    if (canvasProgressBars) {
        canvasProgressBars.style.display = 'block';
    }
    
    if (drawingProgressBar && total > 0) {
        const progressPercent = Math.round((current / total) * 100);
        drawingProgressBar.style.width = `${progressPercent}%`;
        drawingProgressBar.setAttribute('aria-valuenow', progressPercent);
        
        // Update progress text in canvas area
        if (drawingProgressPercent) {
            drawingProgressPercent.textContent = `${progressPercent}%`;
        }
        
        // Update progress info in the old progress area
        if (drawingProgressInfo) {
            drawingProgressInfo.textContent = `${current} / ${total}`;
        }
    }
}

function showVideoGenerationProgress(show) {
    const videoProgressContainer = document.getElementById('videoProgressContainer');
    const videoProgressBar = document.getElementById('videoProgressBar');
    const videoProgressText = document.getElementById('videoProgressText');
    const canvasProgressBars = document.getElementById('canvasProgressBars');
    
    // Show main progress bars container
    if (canvasProgressBars) {
        canvasProgressBars.style.display = 'block';
    }
    
    if (videoProgressContainer) {
        videoProgressContainer.style.display = show ? 'block' : 'none';
    }
    
    if (show && videoProgressBar) {
        // Start indeterminate progress animation
        videoProgressBar.style.width = '100%';
        videoProgressBar.setAttribute('aria-valuenow', 100);
        
        // Update progress text
        if (videoProgressText) {
            videoProgressText.textContent = 'Processing...';
        }
    } else if (videoProgressBar) {
        // Reset progress bar
        videoProgressBar.style.width = '0%';
        videoProgressBar.setAttribute('aria-valuenow', 0);
        
        if (videoProgressText) {
            videoProgressText.textContent = '';
        }
    }
}

function showVideoGenerationProgressWithPercent(progress) {
    const videoProgressContainer = document.getElementById('videoProgressContainer');
    const videoProgressBar = document.getElementById('videoProgressBar');
    const videoProgressText = document.getElementById('videoProgressText');
    const canvasProgressBars = document.getElementById('canvasProgressBars');
    
    // Show main progress bars container
    if (canvasProgressBars) {
        canvasProgressBars.style.display = 'block';
    }
    
    if (videoProgressContainer) {
        videoProgressContainer.style.display = 'block';
    }
    
    if (videoProgressBar) {
        videoProgressBar.style.width = `${progress}%`;
        videoProgressBar.setAttribute('aria-valuenow', progress);
        // Remove animation classes for precise progress
        videoProgressBar.classList.remove('progress-bar-animated');
    }
    
    if (videoProgressText) {
        videoProgressText.textContent = `${Math.round(progress)}%`;
    }
}

function resetProgressBars() {
    const drawingProgressBar = document.getElementById('drawingProgressBar');
    const videoProgressBar = document.getElementById('videoProgressBar');
    const drawingProgressPercent = document.getElementById('drawingProgressPercent');
    const videoProgressText = document.getElementById('videoProgressText');
    const canvasProgressBars = document.getElementById('canvasProgressBars');
    
    if (drawingProgressBar) {
        drawingProgressBar.style.width = '0%';
        drawingProgressBar.setAttribute('aria-valuenow', 0);
    }
    
    if (videoProgressBar) {
        videoProgressBar.style.width = '0%';
        videoProgressBar.setAttribute('aria-valuenow', 0);
    }
    
    if (drawingProgressPercent) {
        drawingProgressPercent.textContent = '0%';
    }
    
    if (videoProgressText) {
        videoProgressText.textContent = '';
    }
    
    showVideoGenerationProgress(false);
    
    // Hide progress bars container when reset
    if (canvasProgressBars) {
        canvasProgressBars.style.display = 'none';
    }
}

function showVideoResults(videoUrl) {
    if (!videoUrl) {
        console.warn('No video URL provided');
        return;
    }
    
    // For cumulative mode, replace the previous video instead of adding to array
    const videoInfo = {
        url: videoUrl,
        timestamp: new Date().toLocaleTimeString(),
        fileId: currentFileId,
        frameCount: canvasData.cumulativeFrames.length
    };
    
    // Replace or add the cumulative video (only keep the latest cumulative version)
    if (canvasData.videoResults.length > 0) {
        // Revoke the old video URL to free memory
        const oldVideo = canvasData.videoResults[canvasData.videoResults.length - 1];
        if (oldVideo.url.startsWith('blob:')) {
            URL.revokeObjectURL(oldVideo.url);
        }
        canvasData.videoResults[canvasData.videoResults.length - 1] = videoInfo;
    } else {
        canvasData.videoResults.push(videoInfo);
    }
    
    // Always hide the video placeholder and show the video container
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const videoContainer = document.getElementById('videoResultContainer');
    
    if (videoPlaceholder) {
        videoPlaceholder.style.display = 'none';
    }
    
    if (videoContainer) {
        videoContainer.style.display = 'block';
        
        // Update the video directly in the existing container
        const existingVideo = videoContainer.querySelector('video');
        if (existingVideo) {
            existingVideo.src = videoUrl;
            existingVideo.load();
        } else {
            // Create or update video results area to show the cumulative video
            updateVideoResultsDisplay();
        }
    }
    
    console.log(`Updated cumulative video with ${videoInfo.frameCount} total frames`);
}

function updateVideoResultsDisplay() {
    const videoContainer = document.getElementById('videoResultContainer');
    if (!videoContainer) return;
    
    // Clear existing content
    videoContainer.innerHTML = '';
    
    // Create header for cumulative video
    const header = document.createElement('h6');
    header.className = 'mb-3';
    header.innerHTML = `<i class="fas fa-video"></i> Cumulative Drawing Video`;
    videoContainer.appendChild(header);
    
    // Show the latest (cumulative) video
    if (canvasData.videoResults.length > 0) {
        const videoInfo = canvasData.videoResults[canvasData.videoResults.length - 1];
        
        const videoDiv = document.createElement('div');
        videoDiv.className = 'mb-3 p-2 border rounded';
        
        const videoTitle = document.createElement('div');
        videoTitle.className = 'mb-2';
        videoTitle.innerHTML = `<small class="text-muted">Last Updated: ${videoInfo.timestamp} | Frames: ${videoInfo.frameCount || 'N/A'}</small>`;
        
        const video = document.createElement('video');
        video.controls = true;
        video.style.cssText = 'border: 1px solid #ddd; max-width: 100%; height: auto;';
        video.src = videoInfo.url;
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-success btn-sm mt-2';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Cumulative Video';
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = videoInfo.url;
            a.download = `cumulative_drawing_video_${videoInfo.fileId || 'video'}.webm`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };
        
        videoDiv.appendChild(videoTitle);
        videoDiv.appendChild(video);
        videoDiv.appendChild(downloadBtn);
        videoContainer.appendChild(videoDiv);
    }
    
    // Also update the original results section if it exists (for backward compatibility)
    const resultsSection = document.getElementById('resultsSection');
    const resultVideo = document.getElementById('resultVideo');
    const downloadBtn = document.getElementById('downloadBtn');
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    // Show the latest video in the original results section
    const latestVideo = canvasData.videoResults[canvasData.videoResults.length - 1];
    if (resultVideo && latestVideo) {
        resultVideo.src = latestVideo.url;
        resultVideo.load();
    }
    
    if (downloadBtn && latestVideo) {
        downloadBtn.onclick = () => {
            const a = document.createElement('a');
            a.href = latestVideo.url;
            a.download = `drawing_video_latest_${latestVideo.fileId || 'video'}.webm`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };
    }
}

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
    preEffectCanvasState: null,
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
    masterVideoRecorder: null, // Single video recorder for cumulative recording
    brushSprites: {}, // Cache for loaded brush sprites
    appliedEffects: [] // Track applied effects with frame ranges: {type, startFrame, endFrame, timestamp}
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
    },
    brush: {
        name: 'Brush',
        opacity: 0.8,
        blendMode: 'source-over',
        textureIntensity: 1.0,
        strokeVariation: 0.4,
        spriteUrl: '/static/brushes/brush1.png',
        spriteRandomization: 0.2
    }
};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateStyleDescription();
    initializeCollapsibleTables();
    initializeAnimationControls();
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

    // Boundary sensitivity slider
    const boundarySensitivity = document.getElementById('boundarySensitivity');
    if (boundarySensitivity) {
        boundarySensitivity.addEventListener('input', function() {
            document.getElementById('sensitivityValue').textContent = this.value;
        });
    }
    
    // Boundary fragmentation slider
    const boundaryFragmentation = document.getElementById('boundaryFragmentation');
    if (boundaryFragmentation) {
        boundaryFragmentation.addEventListener('input', function() {
            const value = parseInt(this.value);
            document.getElementById('fragmentationValue').textContent = value;
            
            // Update description based on value
            const description = value === 1 ? 
                'No fragmentation - boundaries remain whole' :
                `Split each boundary into ${value} fragments`;
            
            const helpText = boundaryFragmentation.parentElement.querySelector('.form-text');
            if (helpText) {
                helpText.innerHTML = `${description} (1=no split, 6=maximum fragments)`;
            }
        });
    }
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
    document.getElementById('boundariesBtn').disabled = false;
    currentFileId = null;
    
    // Check if canvas already exists and has content
    const existingCanvas = document.getElementById('drawingCanvas');
    const hasExistingCanvas = existingCanvas && canvasData.canvasInitialized;
    
    if (hasExistingCanvas) {
        // PRESERVE ALL EXISTING DATA when loading new image
        console.log('Preserving existing canvas and all data when loading new image');
        
        // Keep canvas state and all data
        canvasData.preserveCanvasState = true;
        
        // DON'T clean up video resources - keep them
        // DON'T reset canvas state - preserve everything
        // DON'T clear frames, videos, or effects - keep all
        
        // Show alert about preserving state
        showAlert('New image loaded - preserving existing canvas and all data', 'info');
        
        // Ensure animation section stays visible
        ensureAnimationSectionVisible();
        
    } else {
        // No existing canvas - normal cleanup for first image
        console.log('No existing canvas - normal initialization');
        
        // Clean up existing video resources when loading first image
        canvasData.videoResults.forEach(videoInfo => {
            if (videoInfo.url && videoInfo.url.startsWith('blob:')) {
                URL.revokeObjectURL(videoInfo.url);
            }
        });
        
        // Reset state only for first image
        canvasData.canvasInitialized = false;
        canvasData.preserveCanvasState = false;
        canvasData.cumulativeFrames = [];
        canvasData.videoResults = [];
        canvasData.masterVideoRecorder = null;
        canvasData.appliedEffects = [];
    }
    
    // Only hide results if no existing canvas to preserve
    if (!hasExistingCanvas) {
        hideResults();
        hideSegmentationResults();
    }
}

function clearImage() {
    if (!currentFileId) {
        // Если нет загруженного файла, просто скрываем превью
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('segmentBtn').disabled = true;
        document.getElementById('boundariesBtn').disabled = true;
        hideResults();
        hideSegmentationResults();
        hideBoundaryResults();
        return;
    }
    
    // Показываем подтверждение удаления
    if (!confirm('Вы уверены, что хотите удалить загруженное изображение и все связанные с ним результаты?')) {
        return;
    }
    
    // Показываем индикатор загрузки
    showAlert('Удаление файлов...', 'info');
    
    // Вызываем backend для удаления файлов
    fetch(`/delete/${currentFileId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Успешное удаление - очищаем интерфейс
            document.getElementById('imagePreview').style.display = 'none';
            document.getElementById('segmentBtn').disabled = true;
            document.getElementById('boundariesBtn').disabled = true;
            currentFileId = null;
            
            // Очищаем все результаты
            hideResults();
            hideSegmentationResults();
            hideBoundaryResults();
            
            // Очищаем canvas и видео данные
            canvasData.canvasInitialized = false;
            canvasData.preserveCanvasState = false;
            canvasData.cumulativeFrames = [];
            canvasData.videoResults = [];
            canvasData.masterVideoRecorder = null;
            canvasData.appliedEffects = [];
            
            // Освобождаем blob URLs для видео
            canvasData.videoResults.forEach(videoInfo => {
                if (videoInfo.url && videoInfo.url.startsWith('blob:')) {
                    URL.revokeObjectURL(videoInfo.url);
                }
            });
            
            showAlert(`Успешно удалено ${data.deleted_files.length} файлов`, 'success');
        } else {
            showAlert('Ошибка при удалении файлов: ' + (data.error || 'Неизвестная ошибка'), 'danger');
        }
    })
    .catch(error => {
        console.error('Error deleting files:', error);
        showAlert('Ошибка при удалении файлов. Попробуйте еще раз.', 'danger');
    });
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
            document.getElementById('boundariesBtn').disabled = false;
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
        segmentBtn.innerHTML = '<i class="fas fa-eye"></i> Segmentation';
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
    // Try to show results in Animation section first
    const animationVideoArea = document.getElementById('videoPreviewArea');
    const animationVideo = document.getElementById('resultVideo');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    
    if (animationVideoArea && animationVideo) {
        // Show video in Animation section
        animationVideoArea.style.display = 'block';
        animationVideo.src = `/preview/${fileId}`;
        
        // Setup download button
        if (downloadVideoBtn) {
            downloadVideoBtn.style.display = 'inline-block';
            downloadVideoBtn.onclick = () => {
                window.open(`/download/${fileId}`, '_blank');
            };
        }
        
        // Show animation section
        showAnimationSection();
    } else {
        // Fallback to legacy results section if it exists
        const resultsSection = document.getElementById('resultsSection');
        const resultVideo = document.getElementById('resultVideo');
        const downloadBtn = document.getElementById('downloadBtn');

        if (resultsSection && resultVideo) {
            // Set video source
            resultVideo.src = `/preview/${fileId}`;
            
            // Set download link
            if (downloadBtn) {
                downloadBtn.onclick = () => {
                    window.open(`/download/${fileId}`, '_blank');
                };
            }

            // Show results
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    showAlert('Drawing video created successfully!', 'success');
}

function hideResults() {
    // Hide legacy results section if it exists
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    
    // DON'T hide animation section when loading new image - it should persist
    // Animation section should only be hidden when explicitly requested
    // const animationSection = document.getElementById('animationSection');
    // if (animationSection) {
    //     animationSection.style.display = 'none';
    // }
    
    // Hide segmentation results
    const segmentationResults = document.getElementById('segmentationResults');
    if (segmentationResults) {
        segmentationResults.style.display = 'none';
    }
    
    // Hide boundary results
    const boundaryResults = document.getElementById('boundaryResults');
    if (boundaryResults) {
        boundaryResults.style.display = 'none';
    }
}

// New function to hide animation section only when explicitly needed
function hideAnimationSection() {
    const animationSection = document.getElementById('animationSection');
    if (animationSection) {
        animationSection.style.display = 'none';
    }
}

// Function to ensure animation section is visible when needed
function ensureAnimationSectionVisible() {
    const animationSection = document.getElementById('animationSection');
    if (animationSection && canvasData.canvasInitialized) {
        // Only show animation section if canvas has been initialized
        animationSection.style.display = 'block';
        console.log('Animation section restored to visible state');
    }
}

// Function to ensure all UI elements are properly visible after segmentation
function ensureUIElementsVisible() {
    // Ensure animation section is visible if canvas exists
    ensureAnimationSectionVisible();
    
    // Ensure highlight boundaries button is properly connected
    const highlightBtn = document.getElementById('highlightBoundariesBtn');
    if (highlightBtn && !highlightBtn.onclick) {
        console.log('Reconnecting Highlight Boundaries button event listener');
        highlightBtn.addEventListener('click', highlightContrastBoundaries);
    }
}

// Function to calculate optimal scaling for new image to fit existing canvas
function calculateOptimalImageScale(newImageWidth, newImageHeight, existingCanvasWidth, existingCanvasHeight) {
    if (!existingCanvasWidth || !existingCanvasHeight) {
        return { scale: 1, offsetX: 0, offsetY: 0 };
    }
    
    // Calculate scale factors for both dimensions
    const scaleX = newImageWidth / existingCanvasWidth;
    const scaleY = newImageHeight / existingCanvasHeight;
    
    // Use the larger scale to ensure canvas fits completely within new image
    const scale = Math.nax(scaleX, scaleY);
    
    // Calculate centered positioning
    const scaledCanvasWidth = existingCanvasWidth * scale;
    const scaledCanvasHeight = existingCanvasHeight * scale;
    
    const offsetX = (newImageWidth - scaledCanvasWidth) / 2;
    const offsetY = (newImageHeight - scaledCanvasHeight) / 2;
    
    console.log(`Canvas scaling: ${existingCanvasWidth}x${existingCanvasHeight} -> ${scaledCanvasWidth.toFixed(1)}x${scaledCanvasHeight.toFixed(1)} in ${newImageWidth}x${newImageHeight} image`);
    console.log(`Scale factor: ${scale.toFixed(3)}, Offset: (${offsetX.toFixed(1)}, ${offsetY.toFixed(1)})`);
    
    return {
        scale: scale,
        offsetX: offsetX,
        offsetY: offsetY,
        scaledWidth: scaledCanvasWidth,
        scaledHeight: scaledCanvasHeight
    };
}

// Function to preserve canvas content when loading new image with scaling
function preserveCanvasWithScaling(newImageWidth, newImageHeight) {
    const canvas = document.getElementById('drawingCanvas');
    if (!canvas || !canvasData.canvasInitialized) {
        return false;
    }
    
    const existingWidth = canvas.width;
    const existingHeight = canvas.height;
    
    // Calculate optimal scaling
    const scaleInfo = calculateOptimalImageScale(newImageWidth, newImageHeight, existingWidth, existingHeight);
    
    // Store scaling information for later use
    canvasData.imageScaling = scaleInfo;
    canvasData.originalCanvasSize = { width: existingWidth, height: existingHeight };
    
    console.log('Preserved canvas with scaling info:', scaleInfo);
    return true;
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
                    
                    // CRITICAL: Clear any stale segment references after re-segmentation
                    canvasData.selectedSegmentId = null;
                    
                    
                    // Debug: Log segment ID range to verify data freshness
                    if (segmentData.segments && segmentData.segments.length > 0) {
                        const segmentIds = segmentData.segments.map(s => s.id);
                        const minId = Math.min(...segmentIds);
                        const maxId = Math.max(...segmentIds);
                        console.log(`✅ Updated segment data: ${segmentData.segments.length} segments, ID range: ${minId}-${maxId}`);
                        
                        // Verify no duplicate IDs
                        const uniqueIds = new Set(segmentIds);
                        if (uniqueIds.size !== segmentIds.length) {
                            console.warn(`⚠️ Warning: Found duplicate segment IDs! Unique: ${uniqueIds.size}, Total: ${segmentIds.length}`);
                        }
                    }
                    
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
                                            <option value="brush">Brush</option>
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
                                    <button id="highlightBoundariesBtn" class="btn btn-outline-primary btn-sm mr-2">
                                        <i class="fas fa-highlighter"></i> Highlight Boundaries
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
                    
                    // Only create and initialize canvas if it doesn't exist yet (first time)
                    const existingCanvas = document.getElementById('drawingCanvas');
                    
                    if (!existingCanvas) {
                        // First time - create and initialize canvas
                        console.log('First time loading - creating canvas');
                        createCanvasInAnimationSection(canvasContainerToUse);
                        setTimeout(() => initializeInteractiveCanvas(fileId), 100);
                    } else {
                        // Re-segmentation - don't touch canvas at all, just update button handlers
                        console.log('Re-segmentation - canvas already exists, not touching it');
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
            
            // Ensure all UI elements are properly visible after segmentation
            setTimeout(() => {
                ensureUIElementsVisible();
            }, 200);
            
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
    document.getElementById('highlightBoundariesBtn').addEventListener('click', highlightContrastBoundaries);
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

// New function to update only segment info without touching canvas at all
function updateSegmentInfoOnly(fileId) {
    console.log('Updating segment info only - canvas will not be touched');
    
    // First, fetch and update the segment data for drawing functions
    fetch(`/status/${fileId}`)
    .then(response => response.json())
    .then(data => {
        if (data.status === 'completed' && data.mode === 'segmentation') {
            // Update segment data for drawing functions
            const jsonFile = data.output_files.find(f => f.endsWith('.json'));
            if (jsonFile) {
                fetch(`/outputs/${jsonFile}`)
                .then(response => response.json())
                .then(segmentData => {
                    // Update canvasData with new segment information
                    canvasData.segmentInfo = segmentData;
                    console.log('Updated segment data:', segmentData.segments.length, 'segments');
                })
                .catch(error => {
                    console.error('Error loading segment data:', error);
                });
            }
        }
    })
    .catch(error => {
        console.error('Error fetching status for segment update:', error);
    });
    
    // Update button event handlers
    // Canvas remains completely untouched
    
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
    
    const highlightBtn = document.getElementById('highlightBoundariesBtn');
    if (highlightBtn) {
        highlightBtn.replaceWith(highlightBtn.cloneNode(true));
        document.getElementById('highlightBoundariesBtn').addEventListener('click', highlightContrastBoundaries);
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
    
    console.log('Segment info updated - canvas was not touched');
    showAlert('Segmentation updated while preserving your drawing!', 'success');
    
    // Ensure all UI elements remain visible after re-segmentation
    setTimeout(() => {
        ensureUIElementsVisible();
    }, 300);
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


function applyBrushStrokesFast(brushStrokes, segmentId) {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');

    // Apply scaling if available
    if (canvasData.imageScaling) {
        const { scale, offsetX, offsetY } = canvasData.imageScaling;
        ctx.save();
        ctx.translate(offsetX, offsetY);
        ctx.scale(scale, scale);
    }

    brushStrokes.forEach((stroke) => {
        // Get brush type configuration
        const brushType = stroke.type || canvasData.currentBrushType;
        const brushConfig = brushTypes[brushType] || brushTypes.pencil;
        
        // Apply brush-specific effects
        ctx.globalAlpha = brushConfig.opacity;
        ctx.globalCompositeOperation = brushConfig.blendMode;
        
        if (brushType === 'brush') {
            // Use brush sprite rendering
            drawBrushSprite(ctx, stroke, brushConfig);
        } else {
            // Use traditional line rendering for pencil
            drawPencilStroke(ctx, stroke);
        }
        
        // Reset context
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = 'source-over';
        ctx.shadowBlur = 0;
    });

    if (canvasData.imageScaling) {
        ctx.restore()
    }
}

function drawPencilStroke(ctx, stroke) {
    // Draw the actual brush stroke
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = stroke.width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Apply minimal brush-specific effects for speed
    ctx.globalAlpha = 0.95;
    ctx.globalCompositeOperation = 'source-over';
    
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
}

async function drawBrushSprite(ctx, stroke, brushConfig) {
    if (stroke.points.length === 0) return;
    
    // Calculate sprite size based on stroke width
    let originalSize = stroke.width || 4;
    let baseSize = originalSize;
    
    // If size is smaller than 3, use pencil rendering instead of sprites
    if (originalSize < 3) {
        console.log(`🎨 Stroke size ${originalSize}px (<3), using pencil rendering instead of sprites`);
        drawPencilStroke(ctx, stroke);
        return;
    }
    
    // Load brush sprite if not cached
    const spriteImage = await loadBrushSprite(brushConfig.spriteUrl);
    if (!spriteImage) {
        console.warn('Failed to load brush sprite, falling back to pencil');
        drawPencilStroke(ctx, stroke);
        return;
    }
    
    // Parse stroke color
    const color = parseColor(stroke.color);
    if (!color) {
        console.warn('Invalid stroke color:', stroke.color);
        return;
    }
    
    // Ensure minimum sprite size for visibility (but stay with sprites)
    if (baseSize < 6) {
        console.log(`🎨 Sprite size ${originalSize}px (<6), setting minimum to 6px`);
        baseSize = 6;
    }
    
    // Draw sprites along the stroke path
    if (stroke.points.length === 1) {
        // Single point - draw one sprite with random rotation
        const point = stroke.points[0];
        // Size randomization: 0% to +30% increase only
        const size = baseSize * (1 + Math.random() * 0.3);
        // Angle randomization: ±30% (±0.52 radians ≈ ±30 degrees)
        const rotation = (Math.random() - 0.5) * 1.05; // ±0.52 radians
        drawColorizedSprite(ctx, spriteImage, point.x, point.y, size, color, rotation);
    } else {
        // Multiple points - draw sprites along the path with direction-based rotation
        const totalDistance = calculateStrokeDistance(stroke.points);
        // Use original size for spacing calculation to maintain proper density
        const baseSpacing = originalSize * 1.2; // Base spacing between sprites
        
        let currentDistance = 0;
        let spriteIndex = 0;
        
        while (currentDistance < totalDistance) {
            const t = currentDistance / totalDistance; // Normalized position (0 to 1)
            const point = interpolateAlongStroke(stroke.points, t);
            
            if (point) {
                // Calculate stroke direction at this point
                const direction = getStrokeDirectionAt(stroke.points, t);
                
                // Enhanced randomization (30% for all parameters)
                // Size: 0% to +30% increase only
                const size = baseSize * (1 + Math.random() * 0.3);
                
                // Angle: ±30% variation from stroke direction
                const rotationVariation = (Math.random() - 0.5) * 1.05; // ±0.52 radians (±30°)
                const rotation = direction + rotationVariation;
                
                drawColorizedSprite(ctx, spriteImage, point.x, point.y, size, color, rotation);
                
                // Spacing: 0% to +30% increase only
                const spacingIncrease = Math.random() * 0.3; // 0% to +30%
                const nextSpacing = baseSpacing * (1 + spacingIncrease);
                currentDistance += nextSpacing;
                spriteIndex++;
            } else {
                break;
            }
        }
    }
}

async function loadBrushSprite(spriteUrl) {
    // Check cache first
    if (canvasData.brushSprites[spriteUrl]) {
        return canvasData.brushSprites[spriteUrl];
    }
    
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = function() {
            canvasData.brushSprites[spriteUrl] = img;
            console.log(`🎨 Sprite loaded and cached:`, img.width, 'x', img.height, 'pixels');
            resolve(img);
        };
        img.onerror = function() {
            console.error('Failed to load brush sprite:', spriteUrl);
            resolve(null);
        };
        img.src = spriteUrl;
    });
}

function drawColorizedSprite(ctx, spriteImage, x, y, size, color, rotation = 0) {
    // Save the current context state
    ctx.save();
    
    // Move to the sprite position and apply rotation
    ctx.translate(x, y);
    ctx.rotate(rotation);
    
    // Create a temporary canvas for colorization
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCanvas.width = spriteImage.width;
    tempCanvas.height = spriteImage.height;
    
    // First, fill with the target color
    tempCtx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Then use the sprite as a mask with globalCompositeOperation
    tempCtx.globalCompositeOperation = 'destination-in';
    tempCtx.drawImage(spriteImage, 0, 0);
    
    // Draw the colorized sprite to the main canvas (centered)
    const halfSize = size / 2;
    ctx.drawImage(tempCanvas, -halfSize, -halfSize, size, size);
    
    // Restore the context state
    ctx.restore();
    
    // Debug: Log that sprite was drawn
    console.log(`🎨 Drew colorized sprite at (${x}, ${y}) with size ${size}, rotation ${rotation.toFixed(2)}rad, and color RGB(${color.r}, ${color.g}, ${color.b})`);
}

function parseColor(colorString) {
    // Parse RGB color string like "rgb(255, 0, 0)" or hex like "#ff0000"
    if (colorString.startsWith('rgb(')) {
        const match = colorString.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (match) {
            return {
                r: parseInt(match[1]),
                g: parseInt(match[2]),
                b: parseInt(match[3])
            };
        }
    } else if (colorString.startsWith('#')) {
        const hex = colorString.substring(1);
        if (hex.length === 6) {
            return {
                r: parseInt(hex.substring(0, 2), 16),
                g: parseInt(hex.substring(2, 4), 16),
                b: parseInt(hex.substring(4, 6), 16)
            };
        }
    }
    return null;
}

function calculateStrokeDistance(points) {
    let totalDistance = 0;
    for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i-1].x;
        const dy = points[i].y - points[i-1].y;
        totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    return totalDistance;
}

function interpolateAlongStroke(points, t) {
    if (points.length < 2) return points[0] || null;
    if (t <= 0) return points[0];
    if (t >= 1) return points[points.length - 1];
    
    // Calculate cumulative distances
    const distances = [0];
    let totalDistance = 0;
    
    for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i-1].x;
        const dy = points[i].y - points[i-1].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        totalDistance += distance;
        distances.push(totalDistance);
    }
    
    // Find target distance
    const targetDistance = t * totalDistance;
    
    // Find the segment containing the target distance
    for (let i = 1; i < distances.length; i++) {
        if (distances[i] >= targetDistance) {
            const segmentStart = distances[i-1];
            const segmentEnd = distances[i];
            const segmentT = (targetDistance - segmentStart) / (segmentEnd - segmentStart);
            
            // Interpolate between points[i-1] and points[i]
            const p1 = points[i-1];
            const p2 = points[i];
            
            return {
                x: p1.x + (p2.x - p1.x) * segmentT,
                y: p1.y + (p2.y - p1.y) * segmentT
            };
        }
    }
    
    return points[points.length - 1];
}

function getStrokeDirectionAt(points, t) {
    if (points.length < 2) return 0;
    
    // Calculate cumulative distances
    const distances = [0];
    let totalDistance = 0;
    
    for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i-1].x;
        const dy = points[i].y - points[i-1].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        totalDistance += distance;
        distances.push(totalDistance);
    }
    
    // Find target distance
    const targetDistance = t * totalDistance;
    
    // Find the segment containing the target distance
    for (let i = 1; i < distances.length; i++) {
        if (distances[i] >= targetDistance) {
            // Calculate direction of this segment
            const p1 = points[i-1];
            const p2 = points[i];
            
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            
            // Return angle in radians (Math.atan2 returns angle from -π to π)
            return Math.atan2(dy, dx);
        }
    }
    
    // Fallback: use direction of last segment
    if (points.length >= 2) {
        const p1 = points[points.length - 2];
        const p2 = points[points.length - 1];
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return Math.atan2(dy, dx);
    }
    
    return 0;
}

async function saveCanvasStateBeforeEffect(canvas) {
    try {
        // Save the current canvas state as a data URL
        canvasData.preEffectCanvasState = canvas.toDataURL('image/png');
        console.log('Canvas state saved before effect application');
        
        // Also save canvas state to server for flash animation
        await saveCanvasStateToServer(canvas);
        return true;
    } catch (error) {
        console.error('Error saving canvas state:', error);
        showAlert('Ошибка при сохранении состояния канваса', 'warning');
        return false;
    }
}

function saveCanvasStateToServer(canvas) {
    return new Promise((resolve, reject) => {
        try {
            // Convert canvas to blob and send to server
            canvas.toBlob(function(blob) {
                if (!blob || !currentFileId) {
                    resolve(false);
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', blob, 'canvas_state.png');
                formData.append('file_id', currentFileId);
                formData.append('frame_type', 'canvas_state');
                
                fetch('/save_canvas_frame', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Canvas state saved to server for flash animation');
                        resolve(true);
                    } else {
                        console.error('Failed to save canvas state to server:', data.error);
                        resolve(false);
                    }
                })
                .catch(error => {
                    console.error('Error saving canvas state to server:', error);
                    reject(error);
                });
            }, 'image/png');
        } catch (error) {
            console.error('Error saving canvas state to server:', error);
            reject(error);
        }
    });
}


function drawTaperedStrokes(highlightStrokes) {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    if (!highlightStrokes || highlightStrokes.length === 0) {
        console.warn('No highlight strokes to draw');
        return;
    }
    
    console.log(`Drawing ${highlightStrokes.length} tapered strokes`);
    
    highlightStrokes.forEach((stroke, index) => {
        if (!stroke.points || stroke.points.length < 2) {
            console.warn(`Stroke ${index} has insufficient points:`, stroke.points?.length || 0);
            return;
        }
        
        // Set stroke color
        ctx.strokeStyle = stroke.color;
        ctx.fillStyle = stroke.color;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.8;
        
        // Draw tapered stroke using variable line width
        const points = stroke.points;
        
        // Debug: log stroke information
        console.log(`Drawing stroke with ${points.length} points, base width: ${stroke.width}`);
        if (points.length > 0) {
            console.log(`First point thickness: ${points[0].thickness}, Last point thickness: ${points[points.length-1].thickness}`);
        }
        
        for (let i = 0; i < points.length - 1; i++) {
            const currentPoint = points[i];
            const nextPoint = points[i + 1];
            
            // Use the thickness from the current point (already calculated with tapering)
            const thickness = currentPoint.thickness || stroke.width || 2;
            
            // Debug: log thickness for first few points
            if (i < 3) {
                console.log(`Point ${i}: thickness = ${thickness}`);
            }
            
            ctx.lineWidth = Math.max(1, thickness); // Ensure minimum thickness of 1
            ctx.beginPath();
            ctx.moveTo(currentPoint.x, currentPoint.y);
            ctx.lineTo(nextPoint.x, nextPoint.y);
            ctx.stroke();
        }
        
        // Alternative method: Draw as filled polygon for smoother tapering
        if (stroke.use_polygon && points.length > 2) {
            ctx.beginPath();
            
            // Create outline points for the tapered stroke
            const outlinePoints = [];
            
            for (let i = 0; i < points.length; i++) {
                const point = points[i];
                const thickness = point.thickness || stroke.width || 2;
                const halfThickness = thickness / 2;
                
                // Calculate perpendicular direction
                let perpX = 0, perpY = 0;
                
                if (i === 0 && points.length > 1) {
                    // First point - use direction to next point
                    const dx = points[1].x - point.x;
                    const dy = points[1].y - point.y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    if (length > 0) {
                        perpX = -dy / length;
                        perpY = dx / length;
                    }
                } else if (i === points.length - 1) {
                    // Last point - use direction from previous point
                    const dx = point.x - points[i - 1].x;
                    const dy = point.y - points[i - 1].y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    if (length > 0) {
                        perpX = -dy / length;
                        perpY = dx / length;
                    }
                } else {
                    // Middle point - average of directions
                    const dx1 = point.x - points[i - 1].x;
                    const dy1 = point.y - points[i - 1].y;
                    const dx2 = points[i + 1].x - point.x;
                    const dy2 = points[i + 1].y - point.y;
                    
                    const length1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
                    const length2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                    
                    if (length1 > 0 && length2 > 0) {
                        const perp1X = -dy1 / length1;
                        const perp1Y = dx1 / length1;
                        const perp2X = -dy2 / length2;
                        const perp2Y = dx2 / length2;
                        
                        perpX = (perp1X + perp2X) / 2;
                        perpY = (perp1Y + perp2Y) / 2;
                        
                        const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                        if (perpLength > 0) {
                            perpX /= perpLength;
                            perpY /= perpLength;
                        }
                    }
                }
                
                // Add points on both sides
                outlinePoints.push({
                    x: point.x + perpX * halfThickness,
                    y: point.y + perpY * halfThickness
                });
            }
            
            // Add points on the other side (in reverse order)
            for (let i = points.length - 1; i >= 0; i--) {
                const point = points[i];
                const thickness = point.thickness || stroke.width || 2;
                const halfThickness = thickness / 2;
                
                // Calculate perpendicular direction (same as above)
                let perpX = 0, perpY = 0;
                
                if (i === 0 && points.length > 1) {
                    const dx = points[1].x - point.x;
                    const dy = points[1].y - point.y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    if (length > 0) {
                        perpX = -dy / length;
                        perpY = dx / length;
                    }
                } else if (i === points.length - 1) {
                    const dx = point.x - points[i - 1].x;
                    const dy = point.y - points[i - 1].y;
                    const length = Math.sqrt(dx * dx + dy * dy);
                    if (length > 0) {
                        perpX = -dy / length;
                        perpY = dx / length;
                    }
                } else {
                    const dx1 = point.x - points[i - 1].x;
                    const dy1 = point.y - points[i - 1].y;
                    const dx2 = points[i + 1].x - point.x;
                    const dy2 = points[i + 1].y - point.y;
                    
                    const length1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
                    const length2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                    
                    if (length1 > 0 && length2 > 0) {
                        const perp1X = -dy1 / length1;
                        const perp1Y = dx1 / length1;
                        const perp2X = -dy2 / length2;
                        const perp2Y = dx2 / length2;
                        
                        perpX = (perp1X + perp2X) / 2;
                        perpY = (perp1Y + perp2Y) / 2;
                        
                        const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                        if (perpLength > 0) {
                            perpX /= perpLength;
                            perpY /= perpLength;
                        }
                    }
                }
                
                outlinePoints.push({
                    x: point.x - perpX * halfThickness,
                    y: point.y - perpY * halfThickness
                });
            }
            
            // Draw the filled polygon
            if (outlinePoints.length > 0) {
                ctx.moveTo(outlinePoints[0].x, outlinePoints[0].y);
                for (let i = 1; i < outlinePoints.length; i++) {
                    ctx.lineTo(outlinePoints[i].x, outlinePoints[i].y);
                }
                ctx.closePath();
                ctx.fill();
            }
        }
    });
    
    // Reset context
    ctx.globalAlpha = 1.0;
    ctx.globalCompositeOperation = 'source-over';
    
    console.log('Finished drawing tapered strokes');
}

function drawAllSegments() {
    console.log('🎨 drawAllSegments called');
    
    // First, try using existing segment data if available
    if (canvasData.segmentInfo && canvasData.segmentInfo.segments && canvasData.segmentInfo.segments.length > 0) {
        console.log('✅ Using existing segment data');
        proceedWithDrawing();
        return;
    }
    
    // If no existing data, force refresh
    if (!currentFileId) {
        showAlert('No file selected', 'danger');
        return;
    }
    
    console.log('🔄 No existing segment data, forcing fresh reload...');
    
    // Find the latest JSON file for current file_id
    fetch(`/segmentation/${currentFileId}`)
    .then(response => {
        console.log('📡 Segmentation API response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('📊 Segmentation data:', data);
        
        if (data.status !== 'completed') {
            showAlert('Segmentation data not available', 'danger');
            throw new Error('Segmentation not completed');
        }
        
        // Find the JSON file
        const jsonFile = data.output_files.find(f => f.endsWith('.json'));
        if (!jsonFile) {
            showAlert('Segment data file not found', 'danger');
            throw new Error('JSON file not found');
        }
        
        console.log('📄 Loading JSON file:', jsonFile);
        
        // Force fresh load of segment data with cache busting
        const timestamp = new Date().getTime();
        return fetch(`/outputs/${jsonFile}?t=${timestamp}&fresh=true`);
    })
    .then(response => {
        console.log('📡 JSON file response status:', response.status);
        if (!response.ok) {
            throw new Error(`Failed to load segment data: ${response.status}`);
        }
        return response.json();
    })
    .then(freshSegmentData => {
        console.log('📊 Fresh segment data loaded:', freshSegmentData);
        
        // FORCE UPDATE: Replace potentially stale data with fresh data
        canvasData.segmentInfo = freshSegmentData;
        
        console.log('✅ Fresh segment data loaded successfully');
        
        proceedWithDrawing();
    })
    .catch(error => {
        console.error('❌ Error loading fresh segment data:', error);
        showAlert('Error loading segment data. Please try re-segmenting the image.', 'danger');
    });
    
    function proceedWithDrawing() {
        console.log('🚀 proceedWithDrawing called');
        
        // Validate segment data
        if (!canvasData.segmentInfo || !canvasData.segmentInfo.segments) {
            showAlert('No segment data available', 'danger');
            return;
        }
        
        const segments = canvasData.segmentInfo.segments;
        if (segments.length === 0) {
            showAlert('No segments available to draw', 'warning');
            return;
        }
        
        // Debug: Log current segment data before drawing
        const segmentIds = segments.map(s => s.id);
        const minId = Math.min(...segmentIds);
        const maxId = Math.max(...segmentIds);
        console.log(`🎨 Starting to draw ${segments.length} segments, ID range: ${minId}-${maxId}`);
        
        // Validate segment ID consistency
        const uniqueIds = new Set(segmentIds);
        if (uniqueIds.size !== segmentIds.length) {
            console.error(`❌ Critical error: Duplicate segment IDs detected! Unique: ${uniqueIds.size}, Total: ${segmentIds.length}`);
            showAlert('Error: Duplicate segment IDs detected. Please re-segment the image.', 'danger');
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
        const sortedSegments = segments.sort((a, b) => b.pixel_count - a.pixel_count);
        
        console.log('🎬 Calling drawSegmentsWithVideo with', sortedSegments.length, 'segments');
        drawSegmentsWithVideo(sortedSegments, 'all segments', videoDuration, videoFps, strokeDensity);
    }
}

function highlightContrastBoundaries() {
    if (!currentFileId) {
        showAlert('No file selected', 'danger');
        return;
    }
    
    // Get current parameter values
    const sensitivity = parseInt(document.getElementById('boundarySensitivity').value);
    const strokeDensity = parseFloat(document.getElementById('strokeDensity').value);
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);

    // Reset interruption flag
    canvasData.drawingInterrupted = false;
    canvasData.isDrawingAll = true;
    
    // First, get the highlight boundaries data
    fetch('/highlight_boundaries', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            file_id: currentFileId,
            sensitivity: sensitivity
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.highlight_strokes) {
            // Convert highlight strokes to segments format for drawing
            const highlightSegments = data.highlight_strokes.map((stroke, index) => ({
                id: `highlight_${index}`,
                pixel_count: stroke.points ? stroke.points.length : 10, // Use points count as size
                highlight_stroke: stroke // Store the stroke data
            }));
            
            // Start drawing with video recording
            drawHighlightBoundariesWithVideo(highlightSegments, videoDuration, videoFps, strokeDensity);
        } else {
            showAlert(data.error || 'Failed to get highlight boundaries', 'danger');
        }
    })
    .catch(error => {
        console.error('Error getting highlight boundaries:', error);
        showAlert('Error getting highlight boundaries: ' + error.message, 'danger');
    });
}

function drawHighlightBoundariesWithVideo(highlightSegments, videoDuration, videoFps, strokeDensity) {
    if (!highlightSegments || highlightSegments.length === 0) {
        showAlert('No highlight boundaries to draw', 'warning');
        return;
    }
    
    // Show animation section
    showAnimationSection();
    
    // Show drawing progress and manage buttons
    const progressDiv = document.getElementById('drawingProgress');
    const drawAllBtn = document.getElementById('drawAllBtn');
    const drawLargeBtn = document.getElementById('drawLargeBtn');
    const drawMediumBtn = document.getElementById('drawMediumBtn');
    const drawSmallBtn = document.getElementById('drawSmallBtn');
    const highlightBtn = document.getElementById('highlightBoundariesBtn');
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
    if (highlightBtn) {
        highlightBtn.disabled = true;
        highlightBtn.style.display = 'none';
    }
    if (stopBtn) {
        stopBtn.style.display = 'inline-block';
    }
    
    // Update brush type display
    updateBrushTypeDisplay();
    
    // Reset and initialize progress bars
    resetProgressBars();
    updateDrawingProgress(0, highlightSegments.length);
    
    startDrawingTimer();
    
    showAlert(`Drawing ${highlightSegments.length} highlight boundaries`, 'info');
    
    // Initialize video recorder
    const videoRecorder = new VideoRecorder({
        videoDuration: videoDuration,
        segmentCount: highlightSegments.length,
        canvas: document.getElementById('drawingCanvas'),
        frameRate: videoFps,
        timeCompression: true,
        isCumulative: true
    });
    
    // Start video recording
    videoRecorder.start();
    
    // Initialize drawing state
    let currentIndex = 0;
    const currentSegmentSpan = document.getElementById('currentDrawingSegment');
    const progressTextSpan = document.getElementById('drawingProgressInfo');
    
    function drawNextHighlightBoundary() {
        // Check if drawing was interrupted
        if (canvasData.drawingInterrupted || !canvasData.isDrawingAll) {
            stopDrawingTimer();
            if (videoRecorder) {
                videoRecorder.stop();
            }
            return;
        }
        
        // Check if we've drawn all boundaries
        if (currentIndex >= highlightSegments.length) {
            // Finished drawing all boundaries - restore UI
            const progressDiv = document.getElementById('drawingProgress');
            const drawAllBtn = document.getElementById('drawAllBtn');
            const drawLargeBtn = document.getElementById('drawLargeBtn');
            const drawMediumBtn = document.getElementById('drawMediumBtn');
            const drawSmallBtn = document.getElementById('drawSmallBtn');
            const highlightBtn = document.getElementById('highlightBoundariesBtn');
            const stopBtn = document.getElementById('stopDrawingBtn');
            
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
            if (highlightBtn) {
                highlightBtn.disabled = false;
                highlightBtn.innerHTML = '<i class="fas fa-highlighter"></i> Highlight Boundaries';
                highlightBtn.style.display = 'inline-block';
            }
            if (stopBtn) stopBtn.style.display = 'none';
            stopDrawingTimer();
            
            // Finalize video
            if (videoRecorder) {
                videoRecorder.captureFrame(); // Final frame
                const stopResult = videoRecorder.stop();
                
                // Check if stop() returns a Promise
                if (stopResult && typeof stopResult.then === 'function') {
                    stopResult.then(videoBlob => {
                        if (videoBlob) {
                            showVideoResults(videoBlob, 'highlight_boundaries');
                        }
                    }).catch(error => {
                        console.error('Error stopping video recorder:', error);
                    });
                } else {
                    // If stop() doesn't return a Promise, handle synchronously
                    console.log('Video recorder stopped synchronously');
                    if (stopResult) {
                        showVideoResults(stopResult, 'highlight_boundaries');
                    }
                }
            }
            
            showAlert(`Successfully drew ${highlightSegments.length} highlight boundaries`, 'success');
            return;
        }
        
        const segment = highlightSegments[currentIndex];
        
        // Update progress
        if (currentSegmentSpan) currentSegmentSpan.textContent = segment.id;
        if (progressTextSpan) progressTextSpan.textContent = `${currentIndex + 1} / ${highlightSegments.length}`;
        
        // Update drawing progress bar
        updateDrawingProgress(currentIndex + 1, highlightSegments.length);
        
        // Capture frame before drawing this boundary
        if (videoRecorder) {
            videoRecorder.captureFrame();
        }
        
        // Draw this highlight boundary using the stored stroke data
        const stroke = segment.highlight_stroke;
        if (stroke && stroke.points) {
            // Draw the tapered stroke directly
            drawTaperedStrokes([stroke]);
            
            // Capture frame after drawing
            if (videoRecorder) {
                videoRecorder.captureFrame();
            }
            
            // Calculate dynamic delay based on boundary complexity
            let delay = Math.max(100, Math.min(1000, stroke.points.length * 10)); // 10ms per point
            
            // Move to next boundary after calculated delay
            currentIndex++;
            setTimeout(drawNextHighlightBoundary, delay);
        } else {
            console.warn(`Highlight boundary ${segment.id} has no stroke data`);
            currentIndex++;
            setTimeout(drawNextHighlightBoundary, 50); // Short delay on error
        }
    }
    
    // Start drawing the first boundary
    drawNextHighlightBoundary();
}

function drawSmallFragments() {
    if (!canvasData.segmentInfo || !canvasData.segmentInfo.segments || !currentFileId) {
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
    if (!canvasData.segmentInfo || !canvasData.segmentInfo.segments || !currentFileId) {
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
    if (!canvasData.segmentInfo || !canvasData.segmentInfo.segments || !currentFileId) {
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
        console.log(`🎨 Drawing segment ${segment.id} (${currentIndex + 1}/${sortedSegments.length})`);
        console.log(`📊 Segment data:`, {
            id: segment.id,
            pixel_count: segment.pixel_count,
            average_color: segment.average_color
        });
        
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
        .then(response => {
            console.log(`📡 Backend response for segment ${segment.id}:`, response.status);
            return response.json();
        })
        .then(data => {
            console.log(`Backend data for segment ${segment.id}:`, data);
            
            // Check again if drawing was interrupted
            if (canvasData.drawingInterrupted || !canvasData.isDrawingAll) {
                stopDrawingTimer();
                if (videoRecorder) {
                    videoRecorder.stop();
                }
                return;
            }
            
            if (data.success) {
                console.log(`Successfully got brush strokes for segment ${segment.id}:`, data.brush_strokes?.length || 0, 'strokes');

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

function hideBoundaryResults() {
    document.getElementById('boundaryResults').style.display = 'none';
}

function resetSegmentButton() {
    const segmentBtn = document.getElementById('segmentBtn');
    segmentBtn.disabled = false;
    segmentBtn.innerHTML = '<i class="fas fa-eye"></i> Segmentation';
    document.getElementById('processingStatus').style.display = 'none';
}

function resetApp() {
    currentFileId = null;
    clearImage();
    hideResults();
    hideSegmentationResults();
    hideBoundaryResults();
    
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
    
    // Show animation section if not visible
    const animationSection = document.getElementById('animationSection');
    if (animationSection) {
        animationSection.style.display = 'block';
    }
    
    const videoDuration = parseInt(document.getElementById('videoDuration').value);
    const videoFps = parseInt(document.getElementById('videoFps').value);
    
    // Calculate frame distribution based on applied effects
    const frameDistribution = calculateEffectBasedFrameDistribution(videoDuration, videoFps);
    
    showAlert(`Generating video from ${canvasData.cumulativeFrames.length} captured frames with effect-based timing...`, 'info');
    console.log('Frame distribution:', frameDistribution);
    
    // Disable the generate video button during processing
    const generateBtn = document.getElementById('generateVideoBtn');
    if (generateBtn) {
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    }
    
    // Show progress with detailed tracking
    showVideoGenerationProgressWithPercent(0);
    
    // Create a video recorder for manual generation with effect-based distribution
    const videoRecorder = new VideoRecorder({
        videoDuration: videoDuration,
        segmentCount: canvasData.cumulativeFrames.length,
        canvas: document.getElementById('drawingCanvas'),
        frameRate: videoFps,
        timeCompression: true,
        isCumulative: true,
        frameDistribution: frameDistribution // Pass the calculated distribution
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
        canvasData.appliedEffects = []; // Clear applied effects tracking
        
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
    const styleSelect = document.getElementById('styleSelect');
    const description = document.getElementById('styleDescription');
    const stylePreview = document.querySelector('.style-preview');

    if (!styleSelect) return;
    
    const style = styleSelect.value;

    if (description && styleDescriptions[style]) {
        description.textContent = styleDescriptions[style];
    }
    
    if (stylePreview) {
        // Update style-specific styling
        stylePreview.className = `style-preview p-3 rounded bg-light style-${style}`;
    }
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
        this.frameDistribution = options.frameDistribution || null; // Effect-based frame distribution
        
        this.capturedFrames = [];
        this.isRecording = false;
        this.totalFramesNeeded = this.videoDuration * this.frameRate;
        
        if (this.frameDistribution) {
            // Use effect-based distribution
            console.log('Using effect-based frame distribution');
            console.log('Distribution:', this.frameDistribution);
        } else {
            // Use traditional uniform distribution
            this.framesPerSegment = Math.max(1, Math.floor(this.totalFramesNeeded / this.segmentCount));
            console.log(`Video setup: ${this.videoDuration}s @ ${this.frameRate}fps = ${this.totalFramesNeeded} frames`);
            console.log(`Frames per segment: ${this.framesPerSegment}`);
        }
        
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
        
        if (this.frameDistribution) {
            // Use effect-based frame distribution
            console.log('Using effect-based frame distribution for video generation');
            await this.playbackFramesWithEffectDistribution(videoCanvas, ctx, progressCallback, frameInterval);
        } else {
            // Use traditional uniform distribution
            await this.playbackFramesWithUniformDistribution(videoCanvas, ctx, progressCallback, frameInterval);
        }
    }
    
    async playbackFramesWithUniformDistribution(videoCanvas, ctx, progressCallback, frameInterval) {
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
    
    async playbackFramesWithEffectDistribution(videoCanvas, ctx, progressCallback, frameInterval) {
        console.log('Starting effect-based frame distribution playback');
        
        // Calculate total video frames from distribution (not from this.totalFramesNeeded)
        const totalVideoFramesNeeded = this.frameDistribution.reduce((sum, range) => sum + range.videoFrameCount, 0);
        let totalVideoFramesGenerated = 0;
        
        console.log(`Total video frames to generate: ${totalVideoFramesNeeded} (duration: ${totalVideoFramesNeeded / this.frameRate}s)`);
        
        for (const range of this.frameDistribution) {
            console.log(`Processing ${range.type} range: source frames ${range.sourceStartFrame}-${range.sourceEndFrame}, video frames: ${range.videoFrameCount}, duration: ${range.duration}s`);
            
            const sourceFrames = [];
            for (let i = range.sourceStartFrame; i <= range.sourceEndFrame; i++) {
                if (i < this.capturedFrames.length) {
                    sourceFrames.push(this.capturedFrames[i]);
                }
            }
            
            if (sourceFrames.length === 0) {
                console.warn(`No source frames found for range ${range.sourceStartFrame}-${range.sourceEndFrame}`);
                continue;
            }
            
            // Generate EXACTLY the specified number of video frames for this range
            const videoFramesForRange = range.videoFrameCount;
            const sourceFrameStep = sourceFrames.length > 1 ? (sourceFrames.length - 1) / (videoFramesForRange - 1) : 0;
            
            console.log(`Generating exactly ${videoFramesForRange} video frames from ${sourceFrames.length} source frames (step: ${sourceFrameStep})`);
            
            for (let i = 0; i < videoFramesForRange; i++) {
                // Calculate which source frame to use with better interpolation
                let sourceFrameIndex;
                if (sourceFrames.length === 1) {
                    sourceFrameIndex = 0;
                } else if (i === videoFramesForRange - 1) {
                    sourceFrameIndex = sourceFrames.length - 1; // Always use last frame for final video frame
                } else {
                    sourceFrameIndex = Math.round(i * sourceFrameStep);
                }
                
                const frameIndex = Math.min(sourceFrameIndex, sourceFrames.length - 1);
                
                // Draw the frame
                await this.drawFrameToCanvas(ctx, sourceFrames[frameIndex]);
                
                totalVideoFramesGenerated++;
                
                // Update progress based on actual total frames needed
                const progress = (totalVideoFramesGenerated / totalVideoFramesNeeded) * 100;
                if (progressCallback) {
                    progressCallback(progress);
                }
                
                // Wait for next frame timing (except for last frame)
                if (totalVideoFramesGenerated < totalVideoFramesNeeded) {
                    await new Promise(resolve => setTimeout(resolve, frameInterval));
                }
            }
        }
        
        console.log(`Effect-based frame playback completed. Generated ${totalVideoFramesGenerated} video frames (expected: ${totalVideoFramesNeeded})`);
        
        // Verify we generated the correct number of frames
        if (totalVideoFramesGenerated !== totalVideoFramesNeeded) {
            console.warn(`Frame count mismatch! Generated: ${totalVideoFramesGenerated}, Expected: ${totalVideoFramesNeeded}`);
        }
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
    
    // Show video in Animation section first, then fallback to old container
    const animationVideoContainer = document.getElementById('videoResultContainer');
    const animationVideo = document.getElementById('embeddedResultVideo');
    const downloadVideoBtn = document.getElementById('embeddedDownloadBtn');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    
    if (animationVideoContainer && animationVideo) {
        // Hide placeholder and show video container
        if (videoPlaceholder) {
            videoPlaceholder.style.display = 'none';
        }
        
        // Show video in Animation section
        animationVideoContainer.style.display = 'block';
        animationVideo.src = videoUrl;
        animationVideo.load();
        
        // Show and setup download button
        if (downloadVideoBtn) {
            downloadVideoBtn.style.display = 'inline-block';
            downloadVideoBtn.onclick = () => {
                const a = document.createElement('a');
                a.href = videoUrl;
                a.download = `drawing_animation_${Date.now()}.mp4`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
        }
        
        // Scroll to video container
        animationVideoContainer.scrollIntoView({ behavior: 'smooth' });
        
        console.log('Video displayed in Animation section');
    } else {
        // Fallback to old video container system
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
    const resultVideo = resultsSection ? resultsSection.querySelector('#resultVideo') : null;
    const downloadBtn = resultsSection ? resultsSection.querySelector('#downloadBtn') : null;
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
        
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
}

// Boundary Detection Functions
function detectBoundaries() {
    if (!currentFileId) {
        showAlert('Please upload an image first', 'warning');
        return;
    }

    const sensitivity = document.getElementById('boundarySensitivity').value;
    const fragmentation = document.getElementById('boundaryFragmentation').value;
    const boundariesBtn = document.getElementById('boundariesBtn');
    
    // Disable button and show loading state
    boundariesBtn.disabled = true;
    boundariesBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting Boundaries...';
    
    showAlert('Starting boundary detection...', 'info');

    fetch('/detect_boundaries', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            sensitivity: parseInt(sensitivity),
            fragmentation: parseInt(fragmentation)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Boundary detection started. Please wait...', 'info');
            // Start polling for boundary detection status
            pollBoundaryStatus(currentFileId);
        } else {
            showAlert(data.error || 'Failed to start boundary detection', 'danger');
            resetBoundaryButton();
        }
    })
    .catch(error => {
        console.error('Boundary detection error:', error);
        showAlert('Failed to start boundary detection. Please try again.', 'danger');
        resetBoundaryButton();
    });
}

function pollBoundaryStatus(fileId) {
    const statusInterval = setInterval(() => {
        fetch(`/boundary_status/${fileId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                clearInterval(statusInterval);
                showAlert('Boundary detection completed!', 'success');
                loadBoundaryResults(fileId);
                resetBoundaryButton();
            } else if (data.status === 'error') {
                clearInterval(statusInterval);
                showAlert(data.error || 'Boundary detection failed', 'danger');
                resetBoundaryButton();
            } else if (data.status === 'processing') {
                // Continue polling
                showAlert('Detecting boundaries... Please wait.', 'info');
            }
        })
        .catch(error => {
            console.error('Status polling error:', error);
            clearInterval(statusInterval);
            showAlert('Failed to check boundary detection status', 'danger');
            resetBoundaryButton();
        });
    }, 2000); // Poll every 2 seconds
}

function loadBoundaryResults(fileId) {
    const sensitivity = document.getElementById('boundarySensitivity').value;
    const fragmentation = document.getElementById('boundaryFragmentation').value;
    
//    console.log(`DEBUG: loadBoundaryResults called for fileId=${fileId}, sensitivity=${sensitivity}, fragmentation=${fragmentation}`);
    
    fetch(`/boundaries/${fileId}?sensitivity=${sensitivity}&fragmentation=${fragmentation}`)
    .then(response => response.json())
    .then(data => {
//        console.log(`DEBUG: Frontend API response:`, data);
//        console.log(`DEBUG: Frontend received ${data.boundary_data ? data.boundary_data.length : 0} boundaries`);
        if (data.boundary_data && data.boundary_data.length >= 0) {
            canvasData.boundaries = data.boundary_data;
            displayBoundaryResults(data.boundary_data);
        } else {
            showAlert(data.error || 'Failed to load boundary results', 'danger');
        }
    })
    .catch(error => {
        console.error('Load boundary results error:', error);
        showAlert('Failed to load boundary results', 'danger');
    });
}

function displayBoundaryResults(boundaries) {
//    console.log(`DEBUG: displayBoundaryResults called with ${boundaries.length} boundaries`);
    const boundaryResults = document.getElementById('boundaryResults');
    const boundaryList = document.getElementById('boundaryList');
    
    // Clear existing results
    boundaryList.innerHTML = '';
    
    if (boundaries.length === 0) {
        boundaryList.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No boundaries detected</td></tr>';
    } else {
        boundaries.forEach((boundary, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${boundary.id}</td>
                <td>${boundary.length}</td>
                <td>${(boundary.contrast_ratio * 100).toFixed(1)}%</td>
                <td>
                    <div class="d-flex gap-2">
                        <button class="btn btn-primary btn-sm" onclick="drawBoundary('${boundary.id}', 'brightest')">
                            <i class="fas fa-brush"></i> Brightest
                        </button>
                        <button class="btn btn-secondary btn-sm" onclick="drawBoundary('${boundary.id}', 'darkest')">
                            <i class="fas fa-brush"></i> Darkest
                        </button>
                    </div>
                </td>
                <td>
                    <small class="text-muted">
                        Bright: rgb(${boundary.brightest_color.r}, ${boundary.brightest_color.g}, ${boundary.brightest_color.b})<br>
                        Dark: rgb(${boundary.darkest_color.r}, ${boundary.darkest_color.g}, ${boundary.darkest_color.b})
                    </small>
                </td>
            `;
            boundaryList.appendChild(row);
        });
    }
    
    // Show/hide Draw All Boundaries button
    const drawAllBtn = document.getElementById('drawAllBoundariesBtn');
    if (drawAllBtn) {
        drawAllBtn.style.display = boundaries.length > 0 ? 'inline-block' : 'none';
    }
    
    // Show results section
    boundaryResults.style.display = 'block';
    showAlert(`Found ${boundaries.length} contrast boundaries`, 'success');
}

function drawBoundary(boundaryId, colorType) {
    if (!currentFileId) {
        showAlert('No image loaded', 'warning');
        return;
    }

    const sensitivity = document.getElementById('boundarySensitivity').value;
    
    // Initialize canvas if needed
    if (!canvasData.canvasInitialized) {
        initializeInteractiveCanvas(currentFileId);
    }
    
    showAlert(`Drawing boundary ${boundaryId} with ${colorType} color...`, 'info');

    fetch('/draw_boundary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            boundary_id: parseInt(boundaryId),
            color_type: colorType,
            sensitivity: parseInt(sensitivity)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.boundary_strokes) {
            // Draw the boundary strokes on the canvas
            applyBrushStrokesFast(data.boundary_strokes, `boundary_${boundaryId}`);
            showAlert(`Boundary ${boundaryId} drawn successfully!`, 'success');
        } else {
            showAlert(data.error || 'Failed to draw boundary', 'danger');
        }
    })
    .catch(error => {
        console.error('Draw boundary error:', error);
        showAlert('Failed to draw boundary. Please try again.', 'danger');
    });
}

function resetBoundaryButton() {
    const boundariesBtn = document.getElementById('boundariesBtn');
    boundariesBtn.disabled = false;
    boundariesBtn.innerHTML = '<i class="fas fa-search"></i> Detect Boundaries';
}

function drawAllBoundaries() {
    if (!currentFileId || !canvasData.boundaries || canvasData.boundaries.length === 0) {
        showAlert('No boundaries available to draw', 'warning');
        return;
    }

    const drawAllBtn = document.getElementById('drawAllBoundariesBtn');
    if (drawAllBtn) {
        drawAllBtn.disabled = true;
        drawAllBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Drawing All Boundaries...';
    }

    // Initialize canvas if needed
    if (!canvasData.canvasInitialized) {
        initializeInteractiveCanvas(currentFileId);
    }

    let currentIndex = 0;
    const boundaries = canvasData.boundaries;
    let successCount = 0;
    let errorCount = 0;

    function drawNextBoundary() {
        if (currentIndex >= boundaries.length) {
            // All boundaries processed
            if (drawAllBtn) {
                drawAllBtn.disabled = false;
                drawAllBtn.innerHTML = '<i class="fas fa-paint-brush"></i> Draw All Boundaries';
            }
            
            const message = `Completed drawing all boundaries! Success: ${successCount}, Errors: ${errorCount}`;
            showAlert(message, errorCount === 0 ? 'success' : 'warning');
            return;
        }

        const boundary = boundaries[currentIndex];
        const boundaryId = boundary.id;
        const colorType = 'brightest'; // Default to brightest color

        showAlert(`Drawing boundary ${currentIndex + 1} of ${boundaries.length}...`, 'info');

        // Draw current boundary
        fetch('/draw_boundary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: currentFileId,
                boundary_id: parseInt(boundaryId),
                color_type: colorType,
                sensitivity: parseInt(document.getElementById('boundarySensitivity').value)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.boundary_strokes) {
                // Apply strokes to canvas
                applyBrushStrokesFast(data.boundary_strokes);
                successCount++;
            } else {
                console.error('Failed to draw boundary:', data.error);
                errorCount++;
            }
            
            currentIndex++;
            // Small delay between boundaries for visual effect
            setTimeout(drawNextBoundary, 200);
        })
        .catch(error => {
            console.error('Error drawing boundary:', error);
            errorCount++;
            currentIndex++;
            setTimeout(drawNextBoundary, 200);
        });
    }

    // Start drawing boundaries
    drawNextBoundary();
}

// Initialize collapsible tables functionality
function initializeCollapsibleTables() {
    // Handle boundary table collapse
    const boundaryCollapse = document.getElementById('boundaryDetailsCollapse');
    const boundaryChevron = document.getElementById('boundaryChevron');
    
    if (boundaryCollapse && boundaryChevron) {
        boundaryCollapse.addEventListener('show.bs.collapse', function() {
            boundaryChevron.classList.remove('fa-chevron-down');
            boundaryChevron.classList.add('fa-chevron-up');
        });
        
        boundaryCollapse.addEventListener('hide.bs.collapse', function() {
            boundaryChevron.classList.remove('fa-chevron-up');
            boundaryChevron.classList.add('fa-chevron-down');
        });
    }
    
    // Handle segmentation table collapse
    const segmentationCollapse = document.getElementById('segmentationDetailsCollapse');
    const segmentationChevron = document.getElementById('segmentationChevron');
    
    if (segmentationCollapse && segmentationChevron) {
        segmentationCollapse.addEventListener('show.bs.collapse', function() {
            segmentationChevron.classList.remove('fa-chevron-down');
            segmentationChevron.classList.add('fa-chevron-up');
        });
        
        segmentationCollapse.addEventListener('hide.bs.collapse', function() {
            segmentationChevron.classList.remove('fa-chevron-up');
            segmentationChevron.classList.add('fa-chevron-down');
        });
    }
}

// Initialize animation controls functionality
function initializeAnimationControls() {
    // Light Effect button (only remaining button in Effects section)
    const lightEffectBtn = document.getElementById('lightEffectBtn');
    if (lightEffectBtn) {
        lightEffectBtn.addEventListener('click', function() {
            startLightEffectAnimation();
        });
    }
    
    // Generate Video button
    const generateVideoBtn = document.getElementById('generateVideoBtn');
    if (generateVideoBtn) {
        generateVideoBtn.addEventListener('click', function() {
            generateVideoFromFrames();
        });
    }
    
    // Download Video button
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    if (downloadVideoBtn) {
        downloadVideoBtn.addEventListener('click', function() {
            downloadGeneratedVideo();
        });
    }
}

// Show animation section when needed
function showAnimationSection() {
    const animationSection = document.getElementById('animationSection');
    if (animationSection) {
        animationSection.style.display = 'block';
        // Scroll to animation section
        animationSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Start drawing animation
function startDrawingAnimation() {
    const startBtn = document.getElementById('startDrawingBtn');
    const pauseBtn = document.getElementById('pauseDrawingBtn');
    
    if (startBtn) {
        startBtn.style.display = 'none';
    }
    if (pauseBtn) {
        pauseBtn.style.display = 'inline-block';
    }
    
    // Connect to existing automatic drawing functionality
    const drawAllBtn = document.getElementById('drawAllBtn');
    if (drawAllBtn) {
        drawAllBtn.click(); // Trigger existing draw all functionality
    } else {
        // Fallback: start automatic drawing if available
        if (typeof startAutomaticDrawing === 'function') {
            startAutomaticDrawing();
        } else {
            showAlert('Drawing animation started! Use boundary detection to begin drawing.', 'info');
        }
    }
}

// Pause drawing animation
function pauseDrawingAnimation() {
    const startBtn = document.getElementById('startDrawingBtn');
    const pauseBtn = document.getElementById('pauseDrawingBtn');
    
    if (startBtn) {
        startBtn.style.display = 'inline-block';
        startBtn.innerHTML = '<i class="fas fa-play"></i> Resume Drawing';
    }
    if (pauseBtn) {
        pauseBtn.style.display = 'none';
    }
    
    // Connect to existing stop drawing functionality
    const stopDrawingBtn = document.getElementById('stopDrawingBtn');
    if (stopDrawingBtn) {
        stopDrawingBtn.click(); // Trigger existing stop functionality
    }
    
    showAlert('Drawing animation paused', 'info');
}

// Reset drawing canvas
function resetDrawingCanvas() {
    const startBtn = document.getElementById('startDrawingBtn');
    const pauseBtn = document.getElementById('pauseDrawingBtn');
    
    if (startBtn) {
        startBtn.style.display = 'inline-block';
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start Drawing Animation';
    }
    if (pauseBtn) {
        pauseBtn.style.display = 'none';
    }
    
    // Connect to existing clear canvas functionality
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');
    if (clearCanvasBtn) {
        clearCanvasBtn.click(); // Trigger existing clear functionality
    }
    
    // Clear canvas data
    if (canvasData) {
        canvasData.cumulativeFrames = [];
        canvasData.videoResults = [];
        canvasData.appliedEffects = []; // Clear applied effects tracking
    }
    
    // Hide video preview area
    const videoPreviewArea = document.getElementById('videoPreviewArea');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    
    if (videoPreviewArea) {
        videoPreviewArea.style.display = 'none';
    }
    if (downloadVideoBtn) {
        downloadVideoBtn.style.display = 'none';
    }
    
    showAlert('Canvas reset successfully', 'success');
}

// Download generated video
function downloadGeneratedVideo() {
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.click(); // Use existing download functionality
    }
}

// Create canvas container directly in Animation section (no moving required)
function createCanvasInAnimationSection(canvasContainer) {
    const animationCanvasArea = document.getElementById('interactiveCanvasArea');
    
    if (animationCanvasArea && canvasContainer) {
        // Clear existing content in animation canvas area
        animationCanvasArea.innerHTML = '';
        
        // Set proper styling for animation section
        canvasContainer.className = 'canvas-container';
        
        // Add the canvas container directly to animation section
        animationCanvasArea.appendChild(canvasContainer);
        
        // Show animation section
        showAnimationSection();
        
        console.log('Canvas created directly in Animation section');
    } else {
        console.warn('Could not create canvas in Animation section - elements not found');
        
        // Fallback: try to find segmentation results and append there
        const segmentationResults = document.getElementById('segmentationResults');
        if (segmentationResults && canvasContainer) {
            const infoBody = segmentationResults.querySelector('.card-body .col-12.mt-3');
            if (infoBody) {
                infoBody.appendChild(canvasContainer);
                console.log('Canvas added to segmentation results as fallback');
            }
        }
    }
}

// Light Effect Animation Integration
function startLightEffectAnimation() {
    console.log('Starting Light Effect Animation...');
    
    if (!canvasData.canvasInitialized) {
        showAlert('Канвас не инициализирован. Сначала загрузите изображение и создайте сегментацию.', 'warning');
        return;
    }
    
    const canvas = document.getElementById('drawingCanvas');
    if (!canvas) {
        showAlert('Канвас не найден', 'error');
        return;
    }
    
    // Show progress
    showAlert('Генерируем эффект освещения...', 'info');
    
    // Get canvas data as base64
    const canvasImageData = canvas.toDataURL('image/jpeg', 0.95);
    
    // Call our own API endpoint instead of external flash service
    fetch('/light_effect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            canvas_data: canvasImageData,
            relief_strength: 0.05
        })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            showAlert('Ошибка генерации эффекта: ' + data.error, 'error');
            return;
        }
        
        console.log('Light effect generated:', data.total_frames, 'frames');
        showAlert('Эффект освещения сгенерирован. Воспроизводим анимацию...', 'success');
        
        // Play animation and add frames to video
        playLightEffectAnimation(data.frames);
    })
    .catch(error => {
        console.error('Error generating light effect:', error);
        showAlert('Ошибка генерации эффекта освещения', 'error');
    });
}

function playLightEffectAnimation(frames) {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    
    let currentFrame = 0;
    const totalFrames = frames.length;
    
    // Initialize cumulative frames if not exists
    if (!canvasData.cumulativeFrames) {
        canvasData.cumulativeFrames = [];
    }
    
    // Record the starting frame index for this effect
    const effectStartFrame = canvasData.cumulativeFrames.length;
    
    function displayNextFrame() {
        if (currentFrame >= totalFrames) {
            // Animation complete - record the effect application
            const effectEndFrame = canvasData.cumulativeFrames.length - 1;
            
            // Track this effect application
            canvasData.appliedEffects.push({
                type: 'light_effect',
                startFrame: effectStartFrame,
                endFrame: effectEndFrame,
                frameCount: totalFrames,
                timestamp: new Date().toISOString(),
                duration: 3 // Each effect gets 3 seconds in video
            });
            
            console.log('Light effect animation completed');
            console.log(`Effect tracked: frames ${effectStartFrame}-${effectEndFrame} (${totalFrames} frames)`);
            showAlert('Анимация эффекта освещения завершена! Кадры добавлены для видео.', 'success');
            
            // Display effect tracking info
            displayEffectTrackingInfo();
            return;
        }
        
        // Display frame on canvas
        const img = new Image();
        img.onload = function() {
            // Clear canvas and draw new frame
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Add frame to cumulative frames for video generation
            canvasData.cumulativeFrames.push(canvas.toDataURL('image/jpeg', 0.8));
            
            // Update progress
            const progress = Math.round((currentFrame / totalFrames) * 100);
            showAlert(`Эффект освещения: кадр ${currentFrame + 1}/${totalFrames} (${progress}%)`, 'info');
            
            currentFrame++;
            
            // Schedule next frame
            setTimeout(displayNextFrame, 100); // 100ms delay between frames
        };
        
        img.src = 'data:image/jpeg;base64,' + frames[currentFrame];
    }
    
    // Start animation
    displayNextFrame();
}

// Function to display effect tracking information
function displayEffectTrackingInfo() {
    if (canvasData.appliedEffects.length === 0) {
        return;
    }
    
    console.log('Applied Effects Summary:');
    canvasData.appliedEffects.forEach((effect, index) => {
        console.log(`${index + 1}. ${effect.type}: frames ${effect.startFrame}-${effect.endFrame} (${effect.frameCount} frames, ${effect.duration}s allocated)`);
    });
    
    const totalEffectTime = canvasData.appliedEffects.reduce((sum, effect) => sum + effect.duration, 0);
    console.log(`Total effect time allocated: ${totalEffectTime} seconds`);
}

// Function to calculate frame distribution based on applied effects
function calculateEffectBasedFrameDistribution(videoDuration, videoFps) {
    const totalFrames = canvasData.cumulativeFrames.length;
    const appliedEffects = canvasData.appliedEffects;
    
    // Calculate total time allocated to effects (3 seconds each)
    const totalEffectTime = appliedEffects.length * 3;
    
    // Calculate remaining time for regular drawing frames
    const remainingTime = Math.max(0, videoDuration - totalEffectTime);
    
    console.log(`Video duration: ${videoDuration}s, Effects: ${appliedEffects.length} (${totalEffectTime}s), Remaining: ${remainingTime}s`);
    
    // IMPORTANT: Calculate total video frames based on REQUESTED duration, not source frames
    const totalVideoFrames = videoDuration * videoFps;
    const effectVideoFrames = 3 * videoFps; // Each effect gets exactly 3 seconds worth of video frames
    
    console.log(`Total video frames needed: ${totalVideoFrames}, Effect frames each: ${effectVideoFrames}`);
    
    // Create frame distribution array
    const distribution = [];
    
    // Sort effects by their start frame to process in order
    const sortedEffects = [...appliedEffects].sort((a, b) => a.startFrame - b.startFrame);
    
    let currentSourceFrame = 0;
    
    for (const effect of sortedEffects) {
        // Add regular frames before this effect (if any)
        if (currentSourceFrame < effect.startFrame) {
            const regularFrameCount = effect.startFrame - currentSourceFrame;
            distribution.push({
                type: 'regular',
                sourceStartFrame: currentSourceFrame,
                sourceEndFrame: effect.startFrame - 1,
                sourceFrameCount: regularFrameCount,
                duration: 0, // Will be calculated later
                videoFrameCount: 0 // Will be calculated later
            });
        }
        
        // Add the effect with exactly 3 seconds allocation
        distribution.push({
            type: 'effect',
            sourceStartFrame: effect.startFrame,
            sourceEndFrame: effect.endFrame,
            sourceFrameCount: effect.frameCount,
            videoFrameCount: effectVideoFrames, // Exactly 3 seconds worth
            duration: 3,
            effectType: effect.type
        });
        
        currentSourceFrame = effect.endFrame + 1;
    }
    
    // Add remaining regular frames after all effects
    if (currentSourceFrame < totalFrames) {
        const regularFrameCount = totalFrames - currentSourceFrame;
        distribution.push({
            type: 'regular',
            sourceStartFrame: currentSourceFrame,
            sourceEndFrame: totalFrames - 1,
            sourceFrameCount: regularFrameCount,
            duration: 0, // Will be calculated later
            videoFrameCount: 0 // Will be calculated later
        });
    }
    
    // Calculate video frames for regular ranges
    const regularRanges = distribution.filter(range => range.type === 'regular');
    const totalEffectVideoFrames = appliedEffects.length * effectVideoFrames;
    const remainingVideoFrames = Math.max(0, totalVideoFrames - totalEffectVideoFrames);
    const totalRegularFrames = regularRanges.reduce((sum, range) => sum + range.sourceFrameCount, 0);
    
    console.log(`Remaining video frames for regular content: ${remainingVideoFrames}`);
    
    if (totalRegularFrames > 0 && remainingVideoFrames > 0) {
        regularRanges.forEach(range => {
            const proportion = range.sourceFrameCount / totalRegularFrames;
            range.videoFrameCount = Math.max(1, Math.round(remainingVideoFrames * proportion));
            range.duration = range.videoFrameCount / videoFps;
        });
    } else {
        // If no remaining frames, give minimal allocation to regular ranges
        regularRanges.forEach(range => {
            range.videoFrameCount = Math.max(1, Math.round(0.1 * videoFps)); // 100ms minimum
            range.duration = range.videoFrameCount / videoFps;
        });
    }
    
    // Verify total duration matches requested duration
    const calculatedTotalFrames = distribution.reduce((sum, range) => sum + range.videoFrameCount, 0);
    const calculatedDuration = calculatedTotalFrames / videoFps;
    
    console.log(`Calculated total frames: ${calculatedTotalFrames}, duration: ${calculatedDuration}s (requested: ${videoDuration}s)`);
    
    return distribution;
}

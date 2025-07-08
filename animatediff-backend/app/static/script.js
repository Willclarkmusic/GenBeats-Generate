// Global state
let currentJobs = new Map();
let pollingInterval;

// Authentication
function getAuthHeader() {
    const token = localStorage.getItem('token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/';
        return false;
    }
    return true;
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    if (!checkAuth()) return;
    
    initializeApp();
    setupEventListeners();
    loadVideos();
    checkSystemStatus();
    
    // Start polling for job updates
    startPolling();
});

function initializeApp() {
    updateSliderValues();
    setupPresets();
}

function setupEventListeners() {
    // Form submission
    document.getElementById('videoForm').addEventListener('submit', handleVideoGeneration);
    
    // Logout
    document.getElementById('logout-btn').addEventListener('click', logout);
    
    // Slider updates
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', updateSliderValues);
    });
    
    // Prompt character counter
    document.getElementById('prompt').addEventListener('input', updateCharCounter);
    
    // Modal close
    document.getElementById('modal-close').addEventListener('click', closeModal);
    
    // Close modal on outside click
    document.getElementById('video-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeModal();
        }
    });
}

function updateSliderValues() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const valueSpan = document.getElementById(slider.id + '-value');
        if (valueSpan) {
            valueSpan.textContent = slider.value;
        }
    });
}

function updateCharCounter() {
    const prompt = document.getElementById('prompt');
    const counter = document.getElementById('prompt-counter');
    counter.textContent = prompt.value.length;
}

function setupPresets() {
    const presets = {
        cinematic: {
            prompt: "cinematic wide shot of a majestic mountain landscape at golden hour, dramatic clouds, professional film quality",
            negative_prompt: "blurry, low quality, amateur, shaky",
            steps: 35,
            guidance_scale: 8.0,
            motion_scale: 0.8
        },
        nature: {
            prompt: "peaceful forest stream with flowing water, gentle breeze through trees, natural lighting",
            negative_prompt: "artificial, fake, processed, over-saturated",
            steps: 28,
            guidance_scale: 7.5,
            motion_scale: 1.2
        },
        portrait: {
            prompt: "close-up portrait of a person with natural expressions, soft lighting, professional photography",
            negative_prompt: "distorted face, blurry, low quality, artificial",
            steps: 32,
            guidance_scale: 9.0,
            motion_scale: 0.6
        },
        abstract: {
            prompt: "abstract flowing colors and shapes, mesmerizing patterns, artistic visualization",
            negative_prompt: "realistic, photographic, mundane",
            steps: 25,
            guidance_scale: 6.0,
            motion_scale: 1.5
        }
    };
    
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const presetName = this.dataset.preset;
            const preset = presets[presetName];
            if (preset) {
                applyPreset(preset);
            }
        });
    });
}

function applyPreset(preset) {
    document.getElementById('prompt').value = preset.prompt;
    document.getElementById('negative_prompt').value = preset.negative_prompt;
    document.getElementById('steps').value = preset.steps;
    document.getElementById('guidance_scale').value = preset.guidance_scale;
    document.getElementById('motion_scale').value = preset.motion_scale;
    updateSliderValues();
    updateCharCounter();
}

async function handleVideoGeneration(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const requestData = {
        prompt: formData.get('prompt'),
        negative_prompt: formData.get('negative_prompt') || '',
        steps: parseInt(formData.get('steps')),
        guidance_scale: parseFloat(formData.get('guidance_scale')),
        width: parseInt(formData.get('width')),
        height: parseInt(formData.get('height')),
        duration: parseInt(formData.get('duration')),
        motion_scale: parseFloat(formData.get('motion_scale')),
        seed: parseInt(formData.get('seed')) || -1
    };
    
    try {
        const response = await fetch('/generate-video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeader()
            },
            body: JSON.stringify(requestData)
        });
        
        if (response.status === 401) {
            handleAuthError();
            return;
        }
        
        const result = await response.json();
        
        if (response.ok) {
            // Add job to tracking
            currentJobs.set(result.job_id, {
                ...requestData,
                status: result.status,
                progress: 0,
                message: result.message
            });
            
            // Show progress
            showProgress();
            
            // Disable form
            toggleForm(false);
            
        } else {
            showError(result.detail || 'Failed to start video generation');
        }
        
    } catch (error) {
        showError('Network error. Please try again.');
    }
}

function showProgress() {
    const container = document.getElementById('progress-container');
    container.style.display = 'block';
    
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    const percent = document.getElementById('progress-percent');
    
    fill.style.width = '0%';
    text.textContent = 'Initializing...';
    percent.textContent = '0%';
}

function updateProgress(progress, message) {
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    const percent = document.getElementById('progress-percent');
    
    const progressPercent = Math.round(progress * 100);
    
    fill.style.width = `${progressPercent}%`;
    text.textContent = message || 'Processing...';
    percent.textContent = `${progressPercent}%`;
}

function hideProgress() {
    const container = document.getElementById('progress-container');
    container.style.display = 'none';
}

function toggleForm(enabled) {
    const form = document.getElementById('videoForm');
    const inputs = form.querySelectorAll('input, textarea, select, button');
    
    inputs.forEach(input => {
        input.disabled = !enabled;
    });
}

function startPolling() {
    pollingInterval = setInterval(async () => {
        for (const [jobId, job] of currentJobs) {
            try {
                const response = await fetch(`/status/${jobId}`, {
                    headers: getAuthHeader()
                });
                
                if (response.ok) {
                    const status = await response.json();
                    
                    // Update job status
                    currentJobs.set(jobId, {
                        ...job,
                        status: status.status,
                        progress: status.progress,
                        message: status.message
                    });
                    
                    // Update UI
                    updateProgress(status.progress, status.message);
                    
                    // Check if completed
                    if (status.status === 'completed') {
                        currentJobs.delete(jobId);
                        hideProgress();
                        toggleForm(true);
                        loadVideos(); // Refresh video list
                    } else if (status.status === 'failed') {
                        currentJobs.delete(jobId);
                        hideProgress();
                        toggleForm(true);
                        showError(status.message);
                    }
                }
            } catch (error) {
                console.error('Error polling job status:', error);
            }
        }
        
        // Check system status periodically
        if (Math.random() < 0.1) { // 10% chance each poll
            checkSystemStatus();
        }
    }, 2000); // Poll every 2 seconds
}

async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        
        if (health.status === 'healthy') {
            statusText.textContent = health.model_loaded ? 'Ready' : 'Loading model...';
            statusDot.className = 'status-dot online';
        } else {
            statusText.textContent = 'System offline';
            statusDot.className = 'status-dot offline';
        }
    } catch (error) {
        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        
        statusText.textContent = 'Connection error';
        statusDot.className = 'status-dot offline';
    }
}

async function loadVideos() {
    try {
        const response = await fetch('/videos', {
            headers: getAuthHeader()
        });
        
        if (response.status === 401) {
            handleAuthError();
            return;
        }
        
        const data = await response.json();
        const container = document.getElementById('videos-container');
        
        if (data.videos.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #888;">No videos generated yet.</p>';
            return;
        }
        
        container.innerHTML = data.videos.map(video => `
            <div class="video-card">
                <img src="${video.thumbnail_url}" alt="Video thumbnail" class="video-thumbnail" 
                     onclick="openVideo('${video.job_id}')">
                <div class="video-info">
                    <div class="video-prompt">${video.metadata.prompt}</div>
                    <div class="video-meta">
                        <span>${formatDate(video.created_at)}</span>
                        <span>${video.metadata.width}x${video.metadata.height}</span>
                    </div>
                    <div class="video-actions">
                        <button class="btn btn-outline" onclick="downloadVideo('${video.job_id}')">Download</button>
                        <button class="btn btn-outline" onclick="deleteVideo('${video.job_id}')">Delete</button>
                    </div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error loading videos:', error);
    }
}

function openVideo(jobId) {
    const modal = document.getElementById('video-modal');
    const modalBody = document.getElementById('modal-body');
    
    modalBody.innerHTML = `
        <video class="modal-video" controls autoplay>
            <source src="/result/${jobId}/video" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    `;
    
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('video-modal');
    modal.style.display = 'none';
    
    // Stop video playback
    const video = modal.querySelector('video');
    if (video) {
        video.pause();
    }
}

function downloadVideo(jobId) {
    const link = document.createElement('a');
    link.href = `/result/${jobId}/video`;
    link.download = `video_${jobId}.mp4`;
    link.click();
}

async function deleteVideo(jobId) {
    if (!confirm('Are you sure you want to delete this video?')) {
        return;
    }
    
    try {
        const response = await fetch(`/videos/${jobId}`, {
            method: 'DELETE',
            headers: getAuthHeader()
        });
        
        if (response.ok) {
            loadVideos(); // Refresh video list
        } else {
            showError('Failed to delete video');
        }
    } catch (error) {
        showError('Network error while deleting video');
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function showError(message) {
    // Create or update error message
    let errorDiv = document.getElementById('error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'error-message';
        errorDiv.className = 'error-message';
        document.querySelector('.sidebar').appendChild(errorDiv);
    }
    
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Hide after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function handleAuthError() {
    localStorage.removeItem('token');
    window.location.href = '/';
}

function logout() {
    localStorage.removeItem('token');
    window.location.href = '/';
}

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
});
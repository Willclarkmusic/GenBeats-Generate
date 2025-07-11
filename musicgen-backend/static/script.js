/**
 * MusicGen AI Frontend JavaScript with CUDA Support
 * Handles UI interactions, API calls, audio playback, and performance monitoring
 */

class MusicGenApp {
    constructor() {
        this.currentJob = null;
        this.pollInterval = null;
        this.performanceInterval = null;
        this.systemInfo = null;
        this.availableDevices = [];
        this.performanceVisible = false;
        
        this.initializeElements();
        this.bindEvents();
        this.loadSystemInfo();
        this.loadMusicLibrary();
        this.startPerformanceMonitoring();
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        // Form elements
        this.form = document.getElementById('generationForm');
        this.promptInput = document.getElementById('prompt');
        this.durationSlider = document.getElementById('duration');
        this.durationValue = document.getElementById('durationValue');
        this.modelSelect = document.getElementById('model');
        this.seedInput = document.getElementById('seed');
        this.generateBtn = document.getElementById('generateBtn');
        this.btnText = this.generateBtn.querySelector('.btn-text');
        this.btnLoading = this.generateBtn.querySelector('.btn-loading');

        // Advanced settings
        this.advancedToggle = document.getElementById('advancedToggle');
        this.advancedSettings = document.getElementById('advancedSettings');
        this.deviceSelect = document.getElementById('deviceSelect');
        this.deviceHelp = document.getElementById('deviceHelp');
        this.gpuMonitorGroup = document.getElementById('gpuMonitorGroup');
        this.memoryUsed = document.getElementById('memoryUsed');
        this.memoryText = document.getElementById('memoryText');
        this.clearCacheBtn = document.getElementById('clearCacheBtn');

        // Status elements
        this.statusSection = document.getElementById('statusSection');
        this.statusCard = document.getElementById('statusCard');
        this.statusLabel = document.getElementById('statusLabel');
        this.statusMessage = document.getElementById('statusMessage');
        this.jobId = document.getElementById('jobId');
        this.progressFill = document.getElementById('progressFill');

        // Player elements
        this.playerSection = document.getElementById('playerSection');
        this.audioPlayer = document.getElementById('audioPlayer');
        this.playerInfo = document.getElementById('playerInfo');
        this.trackTitle = document.getElementById('trackTitle');
        this.trackDetails = document.getElementById('trackDetails');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.shareBtn = document.getElementById('shareBtn');

        // Performance monitoring
        this.performanceSection = document.getElementById('performanceSection');
        this.performanceInfo = document.getElementById('performanceInfo');
        this.deviceInfo = document.getElementById('deviceInfo');
        this.statsGrid = document.getElementById('statsGrid');
        this.insightsList = document.getElementById('insightsList');
        this.showPerformanceBtn = document.getElementById('showPerformanceBtn');
        this.refreshPerformance = document.getElementById('refreshPerformance');
        this.clearGpuCache = document.getElementById('clearGpuCache');

        // Performance metrics in header
        this.generationTime = document.getElementById('generationTime');
        this.gpuMemory = document.getElementById('gpuMemory');
        this.deviceTemp = document.getElementById('deviceTemp');

        // Library elements
        this.musicGrid = document.getElementById('musicGrid');
        this.refreshLibrary = document.getElementById('refreshLibrary');
        this.libraryCount = document.getElementById('libraryCount');

        // System info
        this.systemInfoEl = document.getElementById('systemInfo');

        // Toast container
        this.toastContainer = document.getElementById('toastContainer');
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Form submission
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));

        // Duration slider
        this.durationSlider.addEventListener('input', (e) => {
            this.durationValue.textContent = e.target.value;
        });

        // Advanced settings toggle
        if (this.advancedToggle) {
            this.advancedToggle.addEventListener('click', () => this.toggleAdvancedSettings());
        }

        // Device selection
        if (this.deviceSelect) {
            this.deviceSelect.addEventListener('change', (e) => this.handleDeviceChange(e));
        }

        // Clear cache button
        if (this.clearCacheBtn) {
            this.clearCacheBtn.addEventListener('click', () => this.clearGpuCache());
        }

        // Performance monitoring buttons
        if (this.showPerformanceBtn) {
            this.showPerformanceBtn.addEventListener('click', () => this.togglePerformanceSection());
        }
        if (this.refreshPerformance) {
            this.refreshPerformance.addEventListener('click', () => this.refreshPerformanceData());
        }
        if (this.clearGpuCache) {
            this.clearGpuCache.addEventListener('click', () => this.clearGpuCache());
        }

        // Example prompt tags
        document.querySelectorAll('.prompt-tag').forEach(tag => {
            tag.addEventListener('click', (e) => {
                const prompt = e.target.dataset.prompt;
                this.promptInput.value = prompt;
                this.promptInput.focus();
                this.showToast('Prompt loaded!', 'info');
            });
        });

        // Library refresh
        this.refreshLibrary.addEventListener('click', () => this.loadMusicLibrary());

        // Player download button
        this.downloadBtn.addEventListener('click', () => this.downloadCurrentTrack());

        // Player share button
        this.shareBtn.addEventListener('click', () => this.shareCurrentTrack());
    }

    /**
     * Load system information
     */
    async loadSystemInfo() {
        try {
            const response = await fetch('/system-info');
            const data = await response.json();
            this.systemInfo = data;

            // Update main system info display
            const currentDevice = data.current_device?.device || 'Unknown';
            const deviceName = data.current_device?.device_info?.device_name || 'Unknown Device';
            const isGpu = currentDevice.startsWith('cuda');
            
            const deviceInfo = isGpu ? 
                `🚀 GPU: ${deviceName}` : 
                `⚡ CPU: ${deviceName}`;
            
            const modelInfo = data.model_info?.current_model ? 
                ` | Model: ${data.model_info.current_model}` : 
                '';

            this.systemInfoEl.innerHTML = `${deviceInfo}${modelInfo}`;
            this.systemInfoEl.className = `system-info ${isGpu ? 'gpu' : 'cpu'}`;

            // Update device selection dropdown
            this.updateDeviceSelection(data);

            // Update performance info in header
            this.updateHeaderPerformanceInfo(data);

            // Show GPU monitoring if GPU is available
            if (isGpu && this.gpuMonitorGroup) {
                this.gpuMonitorGroup.style.display = 'block';
                this.updateGpuMemoryMonitor(data.current_device?.memory);
            }

        } catch (error) {
            console.error('Failed to load system info:', error);
            this.systemInfoEl.innerHTML = '❌ System info unavailable';
            this.systemInfoEl.className = 'system-info';
        }
    }

    /**
     * Update device selection dropdown
     */
    updateDeviceSelection(systemInfo) {
        if (!this.deviceSelect) return;

        // Clear existing options except auto and force_cpu
        const staticOptions = ['auto', 'force_cpu'];
        Array.from(this.deviceSelect.options).forEach(option => {
            if (!staticOptions.includes(option.value)) {
                option.remove();
            }
        });

        // Add GPU devices if available
        if (systemInfo.cuda?.cuda_available && systemInfo.cuda.devices) {
            systemInfo.cuda.devices.forEach(device => {
                const option = document.createElement('option');
                option.value = `cuda:${device.id}`;
                option.textContent = `GPU ${device.id}: ${device.name} (${device.total_memory_gb}GB)`;
                this.deviceSelect.appendChild(option);
            });
        }

        // Update help text
        const currentDevice = systemInfo.current_device?.device;
        if (currentDevice && this.deviceHelp) {
            const deviceInfo = systemInfo.current_device.device_info;
            this.deviceHelp.textContent = `Currently using: ${deviceInfo.device_name} (${deviceInfo.reason})`;
        }
    }

    /**
     * Update header performance info
     */
    updateHeaderPerformanceInfo(systemInfo) {
        if (!this.performanceInfo) return;

        const stats = systemInfo.performance_stats;
        const isVisible = stats && stats.total_generations > 0;

        this.performanceInfo.style.display = isVisible ? 'flex' : 'none';

        if (isVisible) {
            // Update generation time
            const lastTime = stats.last_generation_time;
            const timeText = lastTime > 0 ? `${lastTime.toFixed(1)}s` : '-';
            this.generationTime.querySelector('.metric-value').textContent = timeText;

            // Update GPU memory
            const currentMemory = systemInfo.current_device?.memory;
            if (currentMemory && currentMemory.total > 0) {
                const usedPercent = ((currentMemory.allocated / currentMemory.total) * 100).toFixed(0);
                this.gpuMemory.querySelector('.metric-value').textContent = `${usedPercent}%`;
            } else {
                this.gpuMemory.querySelector('.metric-value').textContent = '-';
            }

            // Update temperature (if available via GPU monitoring)
            this.updateTemperatureInfo();
        }
    }

    /**
     * Update GPU memory monitor
     */
    updateGpuMemoryMonitor(memoryInfo) {
        if (!memoryInfo || !this.memoryUsed || !this.memoryText) return;

        const { total, allocated, free } = memoryInfo;
        const usedPercent = total > 0 ? (allocated / total) * 100 : 0;

        this.memoryUsed.style.width = `${usedPercent}%`;
        this.memoryText.textContent = `${allocated.toFixed(1)}GB / ${total.toFixed(1)}GB (${usedPercent.toFixed(0)}%)`;

        // Update color based on usage
        if (usedPercent > 90) {
            this.memoryUsed.style.background = 'var(--error-color)';
        } else if (usedPercent > 70) {
            this.memoryUsed.style.background = 'var(--warning-color)';
        } else {
            this.memoryUsed.style.background = 'var(--success-color)';
        }
    }

    /**
     * Toggle advanced settings visibility
     */
    toggleAdvancedSettings() {
        if (!this.advancedSettings) return;

        const isVisible = this.advancedSettings.style.display !== 'none';
        this.advancedSettings.style.display = isVisible ? 'none' : 'block';
        
        if (this.advancedToggle) {
            this.advancedToggle.textContent = isVisible ? '⚙️ Advanced Settings' : '⚙️ Hide Advanced';
        }
    }

    /**
     * Handle device selection change
     */
    handleDeviceChange(event) {
        const selectedValue = event.target.value;
        
        if (this.deviceHelp) {
            switch (selectedValue) {
                case 'auto':
                    this.deviceHelp.textContent = 'Automatically selects the best available device.';
                    break;
                case 'force_cpu':
                    this.deviceHelp.textContent = 'Forces CPU processing. Slower but works on any system.';
                    break;
                default:
                    if (selectedValue.startsWith('cuda:')) {
                        this.deviceHelp.textContent = `Using specific GPU device: ${selectedValue}`;
                    }
                    break;
            }
        }
    }

    /**
     * Clear GPU cache
     */
    async clearGpuCache() {
        try {
            const response = await fetch('/clear-gpu-cache', { method: 'POST' });
            const result = await response.json();

            if (result.success) {
                this.showToast(`GPU cache cleared! Freed ${result.memory_freed_gb}GB`, 'success');
                this.loadSystemInfo(); // Refresh memory info
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Failed to clear GPU cache:', error);
            this.showToast('Failed to clear GPU cache', 'error');
        }
    }

    /**
     * Start performance monitoring
     */
    startPerformanceMonitoring() {
        // Update every 5 seconds when performance section is visible
        this.performanceInterval = setInterval(() => {
            if (this.performanceVisible) {
                this.refreshPerformanceData();
            }
            this.updateTemperatureInfo();
            
            // Refresh system info every 30 seconds to catch device changes
            if (Date.now() % 30000 < 5000) {
                this.loadSystemInfo();
            }
        }, 5000);
    }

    /**
     * Update temperature info from GPU status
     */
    async updateTemperatureInfo() {
        try {
            const response = await fetch('/gpu-status');
            const data = await response.json();

            if (data.cuda_available && data.devices && data.devices.length > 0) {
                const device = data.devices[0]; // Use first device
                if (device.temperature && this.deviceTemp) {
                    this.deviceTemp.querySelector('.metric-value').textContent = `${device.temperature}°C`;
                }
            }
        } catch (error) {
            // Silent fail for temperature monitoring
        }
    }

    /**
     * Toggle performance section visibility
     */
    togglePerformanceSection() {
        if (!this.performanceSection) return;

        this.performanceVisible = !this.performanceVisible;
        this.performanceSection.style.display = this.performanceVisible ? 'block' : 'none';

        if (this.showPerformanceBtn) {
            this.showPerformanceBtn.textContent = this.performanceVisible ? '📊 Hide Performance' : '📈 Show Performance';
        }

        if (this.performanceVisible) {
            this.refreshPerformanceData();
        }
    }

    /**
     * Refresh performance data
     */
    async refreshPerformanceData() {
        try {
            // Load system info, GPU status, and performance metrics in parallel
            const [systemResponse, gpuResponse, metricsResponse] = await Promise.all([
                fetch('/system-info'),
                fetch('/gpu-status'),
                fetch('/performance-metrics')
            ]);

            const [systemData, gpuData, metricsData] = await Promise.all([
                systemResponse.json(),
                gpuResponse.json(),
                metricsResponse.json()
            ]);

            this.updateDeviceInfoDisplay(systemData, gpuData);
            this.updatePerformanceStats(metricsData);
            this.updateGpuMemoryChart(gpuData);
            this.updateInsights(metricsData);

        } catch (error) {
            console.error('Failed to refresh performance data:', error);
        }
    }

    /**
     * Update device info display
     */
    updateDeviceInfoDisplay(systemData, gpuData) {
        if (!this.deviceInfo) return;

        const currentDevice = systemData.current_device;
        const platform = systemData.platform;

        let deviceInfoHTML = `
            <div><strong>Platform:</strong> ${platform.system} ${platform.release}</div>
            <div><strong>Python:</strong> ${platform.python_version}</div>
            <div><strong>PyTorch:</strong> ${systemData.pytorch.version}</div>
            <div><strong>Current Device:</strong> ${currentDevice.device}</div>
        `;

        if (gpuData.cuda_available) {
            deviceInfoHTML += `<div><strong>CUDA:</strong> ${systemData.cuda.cuda_version}</div>`;
            deviceInfoHTML += `<div><strong>GPU Driver:</strong> ${systemData.cuda.driver_version || 'Unknown'}</div>`;
            
            if (gpuData.devices && gpuData.devices.length > 0) {
                const device = gpuData.devices[0];
                deviceInfoHTML += `<div><strong>GPU Memory:</strong> ${device.memory.allocated.toFixed(1)}GB / ${device.memory.total.toFixed(1)}GB</div>`;
                if (device.temperature) {
                    deviceInfoHTML += `<div><strong>Temperature:</strong> ${device.temperature}°C</div>`;
                }
                if (device.utilization_percent !== undefined) {
                    deviceInfoHTML += `<div><strong>GPU Usage:</strong> ${device.utilization_percent.toFixed(0)}%</div>`;
                }
            }
        }

        this.deviceInfo.innerHTML = deviceInfoHTML;
    }

    /**
     * Update performance statistics
     */
    updatePerformanceStats(metricsData) {
        const stats = metricsData.statistics;

        const elements = {
            totalGenerated: document.getElementById('totalGenerated'),
            gpuGenerated: document.getElementById('gpuGenerated'),
            avgGpuTime: document.getElementById('avgGpuTime'),
            avgCpuTime: document.getElementById('avgCpuTime')
        };

        if (elements.totalGenerated) elements.totalGenerated.textContent = stats.total_generations;
        if (elements.gpuGenerated) elements.gpuGenerated.textContent = stats.gpu_generations;
        if (elements.avgGpuTime) elements.avgGpuTime.textContent = stats.average_gpu_time > 0 ? `${stats.average_gpu_time.toFixed(1)}s` : '-';
        if (elements.avgCpuTime) elements.avgCpuTime.textContent = stats.average_cpu_time > 0 ? `${stats.average_cpu_time.toFixed(1)}s` : '-';
    }

    /**
     * Update GPU memory chart
     */
    updateGpuMemoryChart(gpuData) {
        const elements = {
            allocated: document.getElementById('memoryAllocated'),
            cached: document.getElementById('memoryCached'),
            free: document.getElementById('memoryFree')
        };

        if (!gpuData.cuda_available || !gpuData.devices || !gpuData.devices.length) {
            // Hide GPU memory card if no GPU
            const gpuMemoryCard = document.getElementById('gpuMemoryCard');
            if (gpuMemoryCard) gpuMemoryCard.style.display = 'none';
            return;
        }

        const memory = gpuData.devices[0].memory;
        const total = memory.total;

        if (total > 0) {
            const allocatedPercent = (memory.allocated / total) * 100;
            const cachedPercent = (memory.cached / total) * 100;
            const freePercent = (memory.free / total) * 100;

            if (elements.allocated) elements.allocated.style.width = `${allocatedPercent}%`;
            if (elements.cached) elements.cached.style.width = `${cachedPercent}%`;
            if (elements.free) elements.free.style.width = `${freePercent}%`;
        }
    }

    /**
     * Update insights and recommendations
     */
    updateInsights(metricsData) {
        if (!this.insightsList) return;

        const insights = metricsData.insights || [];
        const recommendations = metricsData.recommendations || [];
        const allInsights = [...insights, ...recommendations];

        if (allInsights.length === 0) {
            this.insightsList.innerHTML = '<div class="loading-placeholder">No insights available yet. Generate some music to see performance data!</div>';
            return;
        }

        this.insightsList.innerHTML = allInsights.map(insight => 
            `<div class="insight-item ${insight.type}">${insight.message}</div>`
        ).join('');
    }

    /**
     * Handle form submission
     */
    async handleFormSubmit(e) {
        e.preventDefault();

        const formData = new FormData(this.form);
        const deviceSelection = this.deviceSelect ? this.deviceSelect.value : 'auto';
        
        const request = {
            prompt: formData.get('prompt').trim(),
            duration: parseFloat(formData.get('duration')),
            model: formData.get('model'),
            seed: formData.get('seed') ? parseInt(formData.get('seed')) : null,
            force_cpu: deviceSelection === 'force_cpu',
            preferred_device: deviceSelection.startsWith('cuda:') ? deviceSelection : null
        };

        // Validation
        if (!request.prompt) {
            this.showToast('Please enter a music description', 'error');
            return;
        }

        if (request.duration < 2 || request.duration > 300) {
            this.showToast('Duration must be between 2 and 300 seconds', 'error');
            return;
        }

        try {
            this.setGenerationState(true);
            this.showStatusSection();

            const response = await fetch('/generate-music', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Generation failed');
            }

            const job = await response.json();
            this.currentJob = job;
            this.updateStatusDisplay(job);
            this.startPolling();

            this.showToast('Generation started!', 'success');

        } catch (error) {
            console.error('Generation error:', error);
            this.showToast(error.message, 'error');
            this.setGenerationState(false);
            this.hideStatusSection();
        }
    }

    /**
     * Set generation button state
     */
    setGenerationState(isGenerating) {
        this.generateBtn.disabled = isGenerating;
        
        if (isGenerating) {
            this.btnText.style.display = 'none';
            this.btnLoading.style.display = 'flex';
        } else {
            this.btnText.style.display = 'block';
            this.btnLoading.style.display = 'none';
        }
    }

    /**
     * Show status section
     */
    showStatusSection() {
        this.statusSection.style.display = 'block';
        this.statusSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    /**
     * Hide status section
     */
    hideStatusSection() {
        this.statusSection.style.display = 'none';
    }

    /**
     * Update status display
     */
    updateStatusDisplay(job) {
        this.statusLabel.textContent = job.status;
        this.statusLabel.className = `status-label ${job.status}`;
        this.statusMessage.textContent = job.message;
        this.jobId.textContent = `Job: ${job.job_id.substring(0, 8)}...`;

        // Update progress bar
        let progress = 0;
        switch (job.status) {
            case 'queued':
                progress = 10;
                break;
            case 'processing':
                progress = 50;
                break;
            case 'completed':
                progress = 100;
                break;
            case 'failed':
                progress = 0;
                break;
        }
        this.progressFill.style.width = `${progress}%`;
    }

    /**
     * Start polling for job status
     */
    startPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(async () => {
            try {
                await this.checkJobStatus();
            } catch (error) {
                console.error('Polling error:', error);
                this.stopPolling();
                this.showToast('Lost connection to server', 'error');
            }
        }, 2000);
    }

    /**
     * Stop polling for job status
     */
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    /**
     * Check job status
     */
    async checkJobStatus() {
        if (!this.currentJob) return;

        const response = await fetch(`/job/${this.currentJob.job_id}`);
        if (!response.ok) {
            throw new Error('Failed to get job status');
        }

        const job = await response.json();
        this.updateStatusDisplay(job);

        if (job.status === 'completed') {
            this.stopPolling();
            this.setGenerationState(false);
            this.showToast('Music generation completed!', 'success');
            this.loadAudio(job);
            this.loadMusicLibrary(); // Refresh library
        } else if (job.status === 'failed') {
            this.stopPolling();
            this.setGenerationState(false);
            this.showToast(`Generation failed: ${job.message}`, 'error');
        }
    }

    /**
     * Load generated audio
     */
    async loadAudio(job) {
        try {
            const audioUrl = `/download/${job.job_id}`;
            this.audioPlayer.src = audioUrl;
            
            // Update player info
            this.trackTitle.textContent = 'Generated Music';
            
            let detailsHTML = `<div>Duration: ${job.duration}s | Model: ${job.model_used}`;
            if (job.device_used) {
                detailsHTML += ` | Device: ${job.device_used}`;
            }
            detailsHTML += `</div>`;
            
            detailsHTML += `<div>Created: ${new Date(job.created_at).toLocaleString()}`;
            if (job.generation_time) {
                detailsHTML += ` | Generation: ${job.generation_time.toFixed(1)}s`;
            }
            if (job.gpu_memory_used) {
                detailsHTML += ` | GPU Memory: ${job.gpu_memory_used.toFixed(1)}GB`;
            }
            detailsHTML += `</div>`;
            
            this.trackDetails.innerHTML = detailsHTML;

            // Store download URL for later use
            this.audioPlayer.dataset.downloadUrl = audioUrl;
            this.audioPlayer.dataset.jobId = job.job_id;

            this.showPlayerSection();

        } catch (error) {
            console.error('Failed to load audio:', error);
            this.showToast('Failed to load generated audio', 'error');
        }
    }

    /**
     * Show player section
     */
    showPlayerSection() {
        this.playerSection.style.display = 'block';
        this.playerSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    /**
     * Download current track
     */
    downloadCurrentTrack() {
        const downloadUrl = this.audioPlayer.dataset.downloadUrl;
        if (downloadUrl) {
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = `musicgen_${Date.now()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            this.showToast('Download started!', 'success');
        }
    }

    /**
     * Share current track
     */
    async shareCurrentTrack() {
        const downloadUrl = this.audioPlayer.dataset.downloadUrl;
        if (downloadUrl) {
            const fullUrl = window.location.origin + downloadUrl;
            
            try {
                await navigator.clipboard.writeText(fullUrl);
                this.showToast('Link copied to clipboard!', 'success');
            } catch (error) {
                console.error('Failed to copy to clipboard:', error);
                this.showToast('Failed to copy link', 'error');
            }
        }
    }

    /**
     * Load music library
     */
    async loadMusicLibrary() {
        try {
            this.musicGrid.innerHTML = '<div class="loading-placeholder">Loading your music library...</div>';

            const response = await fetch('/list-generated');
            if (!response.ok) {
                throw new Error('Failed to load library');
            }

            const data = await response.json();
            this.renderMusicLibrary(data.files);
            this.libraryCount.textContent = `${data.files.length} tracks`;

        } catch (error) {
            console.error('Failed to load library:', error);
            this.musicGrid.innerHTML = '<div class="loading-placeholder">Failed to load library</div>';
            this.libraryCount.textContent = 'Error';
            this.showToast('Failed to load music library', 'error');
        }
    }

    /**
     * Render music library
     */
    renderMusicLibrary(files) {
        if (files.length === 0) {
            this.musicGrid.innerHTML = '<div class="loading-placeholder">No music generated yet. Create your first track!</div>';
            return;
        }

        this.musicGrid.innerHTML = files.map(file => this.createMusicItem(file)).join('');

        // Bind events for music items
        this.musicGrid.querySelectorAll('.play-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const path = e.target.dataset.path;
                this.playTrack(path, e.target.dataset.filename);
            });
        });

        this.musicGrid.querySelectorAll('.download-btn-small').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const path = e.target.dataset.path;
                const filename = e.target.dataset.filename;
                this.downloadTrack(path, filename);
            });
        });
    }

    /**
     * Create music item HTML
     */
    createMusicItem(file) {
        const sizeFormatted = this.formatFileSize(file.size);
        const dateFormatted = new Date(file.created).toLocaleDateString();
        const timeFormatted = new Date(file.created).toLocaleTimeString();

        return `
            <div class="music-item">
                <div class="music-item-header">
                    <div class="music-item-title">${file.filename}</div>
                    <div class="music-item-size">${sizeFormatted}</div>
                </div>
                <div class="music-item-date">${dateFormatted} ${timeFormatted}</div>
                <div class="music-item-controls">
                    <button class="play-btn" data-path="${file.path}" data-filename="${file.filename}">
                        ▶️ Play
                    </button>
                    <button class="download-btn-small" data-path="${file.path}" data-filename="${file.filename}">
                        📥 Download
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Play track from library
     */
    playTrack(path, filename) {
        this.audioPlayer.src = path;
        this.trackTitle.textContent = filename;
        this.trackDetails.innerHTML = `<div>Playing from library</div>`;
        this.audioPlayer.dataset.downloadUrl = path;
        this.showPlayerSection();
        this.audioPlayer.play();
        this.showToast(`Playing: ${filename}`, 'info');
    }

    /**
     * Download track from library
     */
    downloadTrack(path, filename) {
        const link = document.createElement('a');
        link.href = path;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        this.showToast('Download started!', 'success');
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        this.toastContainer.appendChild(toast);

        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);

        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MusicGenApp();
});

// Handle page visibility changes to pause polling when tab is hidden
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, could pause polling to save resources
        console.log('Page hidden');
    } else {
        // Page is visible again
        console.log('Page visible');
    }
});

// Handle beforeunload to clean up
window.addEventListener('beforeunload', () => {
    // Clean up intervals and connections
    console.log('Page unloading');
});
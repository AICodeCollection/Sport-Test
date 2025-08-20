class HeartRateDetector {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isDetecting = false;
        this.faceDetected = false;
        this.signalProcessor = new SignalProcessor(30);
        this.faceApi = null;
        this.detectionInterval = null;
        this.chartData = [];
        this.chartCanvas = null;
        this.chartCtx = null;
        
        this.roiRegion = null; // 感兴趣区域
        this.frameCount = 0;
        
        // 校准机制相关
        this.calibrationPeriod = 15000; // 15秒校准期
        this.displayDelay = 5000; // 5秒显示延迟
        this.heartRateHistory = []; // 心率历史记录
        this.calibrationStartTime = null;
        this.isCalibrating = false;
        
        this.initChart();
    }
    
    async init() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('canvasElement');
        this.ctx = this.canvas.getContext('2d');
        this.chartCanvas = document.getElementById('chartCanvas');
        this.chartCtx = this.chartCanvas.getContext('2d');
        
        await this.loadFaceApi();
        await this.setupCamera();
    }
    
    async loadFaceApi() {
        try {
            await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
            await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
            console.log('Face API loaded successfully');
        } catch (error) {
            console.warn('Face API not available, using fallback detection');
            this.faceApi = null;
        }
    }
    
    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                }
            });
            
            this.video.srcObject = stream;
            this.video.onloadedmetadata = () => {
                this.video.play();
                this.updateStatus('摄像头已准备就绪', 'success');
            };
        } catch (error) {
            this.updateStatus('无法访问摄像头: ' + error.message, 'error');
            throw error;
        }
    }
    
    startDetection() {
        if (this.isDetecting) return;
        
        this.isDetecting = true;
        this.signalProcessor.reset();
        this.chartData = [];
        this.frameCount = 0;
        
        // 重置校准机制
        this.heartRateHistory = [];
        this.calibrationStartTime = Date.now();
        this.isCalibrating = true;
        
        this.updateStatus('正在校准和检测心率...', 'success');
        
        this.detectionInterval = setInterval(() => {
            this.processFrame();
        }, 1000 / 30); // 30 FPS
    }
    
    stopDetection() {
        if (!this.isDetecting) return;
        
        this.isDetecting = false;
        clearInterval(this.detectionInterval);
        this.updateStatus('检测已停止', '');
        
        // 清空canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    async processFrame() {
        if (!this.video || this.video.readyState !== 4) return;
        
        this.frameCount++;
        
        // 绘制视频帧到canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // 检测人脸
        await this.detectFace();
        
        // 如果检测到人脸，提取ROI并分析
        if (this.faceDetected && this.roiRegion) {
            const rgbValue = this.extractRGBFromROI();
            if (rgbValue !== null) {
                this.signalProcessor.addDataPoint(rgbValue);
                this.updateHeartRate();
                this.updateChart(rgbValue);
                
                // 调试信息
                if (this.frameCount % 30 === 0) { // 每秒输出一次
                    console.log(`Frame ${this.frameCount}: RGB=${rgbValue.toFixed(2)}, Buffer size=${this.signalProcessor.signalBuffer.length}`);
                }
            } else {
                console.warn('RGB value is null');
            }
        } else {
            if (!this.faceDetected) {
                console.warn('No face detected');
            }
            if (!this.roiRegion) {
                console.warn('No ROI region');
            }
        }
        
        // 更新信号质量显示
        this.updateSignalQuality();
    }
    
    async detectFace() {
        try {
            if (this.faceApi && faceapi.nets.tinyFaceDetector.isLoaded) {
                const detections = await faceapi.detectAllFaces(this.video, 
                    new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
                
                if (detections.length > 0) {
                    this.faceDetected = true;
                    const detection = detections[0];
                    
                    // 绘制人脸边界框
                    this.drawFaceBox(detection.detection.box);
                    
                    // 设置ROI区域（前额区域）
                    this.setROIFromLandmarks(detection.landmarks);
                } else {
                    this.faceDetected = false;
                    this.roiRegion = null;
                }
            } else {
                // 使用简单的中心区域作为ROI
                this.setDefaultROI();
            }
        } catch (error) {
            console.warn('Face detection error:', error);
            this.setDefaultROI();
        }
    }
    
    drawFaceBox(box) {
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
    
    setROIFromLandmarks(landmarks) {
        // 使用脸颊和额头区域作为ROI，这些区域血流信号更强
        const points = landmarks.positions;
        
        // 定义关键点索引 (基于68点人脸特征)
        // 额头区域：眉毛上方
        // 脸颊区域：从鼻翼到耳朵的区域
        const foreheadPoints = [
            // 左眉毛上方区域
            points[19], points[20], points[21], points[22], points[23], points[24]
        ];
        
        const leftCheekPoints = [
            // 左脸颊区域
            points[1], points[2], points[3], points[4], points[5], points[6],
            points[31], points[32], points[33], points[34], points[35]
        ];
        
        const rightCheekPoints = [
            // 右脸颊区域
            points[10], points[11], points[12], points[13], points[14], points[15],
            points[31], points[32], points[33], points[34], points[35]
        ];
        
        // 计算多个ROI区域
        this.roiRegions = [];
        
        // 额头区域
        if (foreheadPoints.length > 0) {
            const foreheadROI = this.calculateROIFromPoints(foreheadPoints, 'forehead');
            if (foreheadROI) this.roiRegions.push(foreheadROI);
        }
        
        // 左脸颊区域
        if (leftCheekPoints.length > 0) {
            const leftCheekROI = this.calculateROIFromPoints(leftCheekPoints, 'left_cheek');
            if (leftCheekROI) this.roiRegions.push(leftCheekROI);
        }
        
        // 右脸颊区域
        if (rightCheekPoints.length > 0) {
            const rightCheekROI = this.calculateROIFromPoints(rightCheekPoints, 'right_cheek');
            if (rightCheekROI) this.roiRegions.push(rightCheekROI);
        }
        
        // 为了兼容现有代码，设置主要ROI为额头区域
        this.roiRegion = this.roiRegions.find(roi => roi.type === 'forehead') || this.roiRegions[0];
        
        this.drawROI();
    }
    
    calculateROIFromPoints(points, type) {
        if (points.length === 0) return null;
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        points.forEach(point => {
            if (point) {
                minX = Math.min(minX, point.x);
                maxX = Math.max(maxX, point.x);
                minY = Math.min(minY, point.y);
                maxY = Math.max(maxY, point.y);
            }
        });
        
        if (minX === Infinity) return null;
        
        // 根据区域类型调整边界
        let padding = 15;
        let heightAdjustment = 0;
        
        if (type === 'forehead') {
            // 额头区域：向上扩展更多
            heightAdjustment = -20;
            padding = 20;
        } else if (type === 'left_cheek' || type === 'right_cheek') {
            // 脸颊区域：更集中在中心
            padding = 10;
        }
        
        const width = maxX - minX + 2 * padding;
        const height = maxY - minY + 2 * padding;
        
        return {
            x: minX - padding,
            y: minY - padding + heightAdjustment,
            width: width,
            height: height,
            type: type
        };
    }
    
    setDefaultROI() {
        // 使用更大的屏幕中心区域作为默认ROI
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const width = 300; // 进一步增加宽度
        const height = 300; // 进一步增加高度
        
        // 创建多个默认ROI区域
        this.roiRegions = [
            {
                // 额头区域
                x: centerX - width / 3,
                y: centerY - height / 2,
                width: width / 1.5,
                height: height / 3,
                type: 'forehead'
            },
            {
                // 左脸颊区域
                x: centerX - width / 2,
                y: centerY - height / 6,
                width: width / 3,
                height: height / 3,
                type: 'left_cheek'
            },
            {
                // 右脸颊区域
                x: centerX + width / 6,
                y: centerY - height / 6,
                width: width / 3,
                height: height / 3,
                type: 'right_cheek'
            }
        ];
        
        // 设置主要ROI为额头区域
        this.roiRegion = this.roiRegions[0];
        
        this.faceDetected = true; // 假设有人脸
        this.drawROI();
        
        console.log('Default ROI regions created:', this.roiRegions);
    }
    
    drawROI() {
        if (!this.roiRegions || this.roiRegions.length === 0) return;
        
        const colors = {
            'forehead': '#ff0000',
            'left_cheek': '#00ff00',
            'right_cheek': '#0000ff'
        };
        
        // 绘制所有ROI区域
        this.roiRegions.forEach(roi => {
            const color = colors[roi.type] || '#ffff00';
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
            
            // 添加标签
            this.ctx.fillStyle = color;
            this.ctx.font = '12px Arial';
            this.ctx.fillText(roi.type, roi.x, roi.y - 5);
        });
    }
    
    extractRGBFromROI() {
        if (!this.roiRegions || this.roiRegions.length === 0) {
            if (!this.roiRegion) return null;
            // 兼容性处理：如果没有多个ROI，使用单个ROI
            return this.extractRGBFromSingleROI(this.roiRegion);
        }
        
        // 从多个ROI区域提取RGB信号
        const roiSignals = [];
        
        this.roiRegions.forEach(roi => {
            const signal = this.extractRGBFromSingleROI(roi);
            if (signal) {
                roiSignals.push({
                    type: roi.type,
                    rgb: signal
                });
            }
        });
        
        if (roiSignals.length === 0) return null;
        
        // 使用改进的PPG算法处理多个ROI信号
        return this.processMultiROISignals(roiSignals);
    }
    
    extractRGBFromSingleROI(roi) {
        try {
            const imageData = this.ctx.getImageData(roi.x, roi.y, roi.width, roi.height);
            const data = imageData.data;
            let r = 0, g = 0, b = 0;
            let pixelCount = 0;
            
            // 计算ROI区域的平均RGB值，增加有效性检查
            for (let i = 0; i < data.length; i += 4) {
                const red = data[i];
                const green = data[i + 1];
                const blue = data[i + 2];
                const alpha = data[i + 3];
                
                // 只计算有效像素（不透明且有颜色信息）
                if (alpha > 0 && (red + green + blue) > 30) {
                    r += red;
                    g += green;
                    b += blue;
                    pixelCount++;
                }
            }
            
            if (pixelCount === 0) {
                console.warn('No valid pixels in ROI');
                return null;
            }
            
            const result = {
                r: r / pixelCount,
                g: g / pixelCount,
                b: b / pixelCount
            };
            
            // 调试信息
            if (this.frameCount % 60 === 0) { // 每2秒输出一次
                console.log(`ROI RGB: R=${result.r.toFixed(1)}, G=${result.g.toFixed(1)}, B=${result.b.toFixed(1)}, Pixels=${pixelCount}`);
            }
            
            return result;
        } catch (error) {
            console.error('Error extracting RGB from ROI:', error);
            return null;
        }
    }
    
    processMultiROISignals(roiSignals) {
        if (roiSignals.length === 0) return null;
        
        // 权重：额头区域权重最高，脸颊区域次之
        const weights = {
            'forehead': 0.6,
            'left_cheek': 0.2,
            'right_cheek': 0.2
        };
        
        let totalWeight = 0;
        let weightedSignal = 0;
        
        // 简化处理：直接使用绿色通道的加权平均
        roiSignals.forEach(signal => {
            const weight = weights[signal.type] || 0.1;
            totalWeight += weight;
            // 绿色通道对血流最敏感
            weightedSignal += signal.rgb.g * weight;
        });
        
        if (totalWeight === 0) return null;
        
        const avgSignal = weightedSignal / totalWeight;
        
        // 调试信息
        if (this.frameCount % 90 === 0) { // 每3秒输出一次
            console.log(`Processed signal: ${avgSignal.toFixed(2)}, ROI count: ${roiSignals.length}`);
        }
        
        return avgSignal;
    }
    
    updateHeartRate() {
        const currentTime = Date.now();
        const heartRate = this.signalProcessor.calculateHeartRate();
        const heartRateElement = document.getElementById('heartRate');
        
        if (heartRate) {
            // 记录心率历史，包含时间戳
            this.heartRateHistory.push({
                value: heartRate,
                timestamp: currentTime
            });
            
            // 清理过期数据（超过校准期的数据）
            this.heartRateHistory = this.heartRateHistory.filter(
                record => currentTime - record.timestamp < this.calibrationPeriod + this.displayDelay
            );
        }
        
        // 检查是否还在校准期
        if (this.isCalibrating && currentTime - this.calibrationStartTime >= this.calibrationPeriod) {
            this.isCalibrating = false;
            this.updateStatus('校准完成，显示实时心率', 'success');
        }
        
        // 显示心率逻辑
        if (this.isCalibrating) {
            // 校准期间显示进度
            const progress = Math.floor((currentTime - this.calibrationStartTime) / this.calibrationPeriod * 100);
            heartRateElement.textContent = `校准中... ${progress}%`;
            heartRateElement.style.color = '#ffa500';
        } else {
            // 使用延迟显示的心率
            const delayedHeartRate = this.getDelayedHeartRate(currentTime);
            if (delayedHeartRate) {
                heartRateElement.textContent = `${delayedHeartRate} BPM`;
                heartRateElement.style.color = '#ff6b6b';
            } else {
                heartRateElement.textContent = '-- BPM';
                heartRateElement.style.color = '#ccc';
            }
        }
    }
    
    getDelayedHeartRate(currentTime) {
        // 获取延迟显示的心率（使用5秒前的数据）
        const targetTime = currentTime - this.displayDelay;
        
        // 找到最接近目标时间的心率记录
        let bestRecord = null;
        let minTimeDiff = Infinity;
        
        for (const record of this.heartRateHistory) {
            const timeDiff = Math.abs(record.timestamp - targetTime);
            if (timeDiff < minTimeDiff) {
                minTimeDiff = timeDiff;
                bestRecord = record;
            }
        }
        
        // 如果找到合适的记录且时间差不超过2秒，返回该心率
        if (bestRecord && minTimeDiff < 2000) {
            return this.getStableHeartRate(targetTime);
        }
        
        return null;
    }
    
    getStableHeartRate(targetTime) {
        // 获取目标时间前后2秒内的心率记录
        const timeWindow = 2000; // 2秒窗口
        const relevantRecords = this.heartRateHistory.filter(
            record => Math.abs(record.timestamp - targetTime) < timeWindow
        );
        
        if (relevantRecords.length === 0) return null;
        
        // 计算平均心率并进行稳定性检查
        const values = relevantRecords.map(record => record.value);
        const avgHeartRate = values.reduce((sum, val) => sum + val, 0) / values.length;
        
        // 检查数据稳定性（标准差不应过大）
        const variance = values.reduce((sum, val) => sum + Math.pow(val - avgHeartRate, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        // 如果标准差过大，说明数据不稳定
        if (stdDev > 15) {
            return null;
        }
        
        return Math.round(avgHeartRate);
    }
    
    updateSignalQuality() {
        const quality = this.signalProcessor.getSignalQuality();
        const qualityElement = document.getElementById('signalQuality');
        
        if (!qualityElement) {
            console.warn('signalQuality element not found');
            return;
        }
        
        const qualityMap = {
            'insufficient': { text: '数据不足', color: '#ccc' },
            'poor': { text: '信号较差', color: '#ff6b6b' },
            'fair': { text: '信号一般', color: '#ffa500' },
            'good': { text: '信号良好', color: '#90EE90' },
            'excellent': { text: '信号优秀', color: '#00ff00' }
        };
        
        const qualityInfo = qualityMap[quality] || qualityMap['poor'];
        
        // 清除现有内容，防止重复
        qualityElement.innerHTML = '';
        qualityElement.textContent = qualityInfo.text;
        qualityElement.style.color = qualityInfo.color;
    }
    
    initChart() {
        this.chartData = [];
        this.maxChartPoints = 150; // 5秒数据 (30fps)
    }
    
    updateChart(value) {
        this.chartData.push(value);
        
        if (this.chartData.length > this.maxChartPoints) {
            this.chartData.shift();
        }
        
        this.drawChart();
    }
    
    drawChart() {
        if (!this.chartCtx || this.chartData.length === 0) return;
        
        const width = this.chartCanvas.width;
        const height = this.chartCanvas.height;
        
        // 清空canvas
        this.chartCtx.clearRect(0, 0, width, height);
        
        // 找到数据的最大值和最小值
        const min = Math.min(...this.chartData);
        const max = Math.max(...this.chartData);
        const range = max - min;
        
        if (range === 0) return;
        
        // 绘制波形
        this.chartCtx.strokeStyle = '#00ff00';
        this.chartCtx.lineWidth = 2;
        this.chartCtx.beginPath();
        
        for (let i = 0; i < this.chartData.length; i++) {
            const x = (i / (this.chartData.length - 1)) * width;
            const y = height - ((this.chartData[i] - min) / range) * height;
            
            if (i === 0) {
                this.chartCtx.moveTo(x, y);
            } else {
                this.chartCtx.lineTo(x, y);
            }
        }
        
        this.chartCtx.stroke();
        
        // 绘制网格
        this.drawGrid();
    }
    
    drawGrid() {
        this.chartCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        this.chartCtx.lineWidth = 1;
        
        const width = this.chartCanvas.width;
        const height = this.chartCanvas.height;
        
        // 垂直网格线
        for (let i = 0; i <= 10; i++) {
            const x = (i / 10) * width;
            this.chartCtx.beginPath();
            this.chartCtx.moveTo(x, 0);
            this.chartCtx.lineTo(x, height);
            this.chartCtx.stroke();
        }
        
        // 水平网格线
        for (let i = 0; i <= 5; i++) {
            const y = (i / 5) * height;
            this.chartCtx.beginPath();
            this.chartCtx.moveTo(0, y);
            this.chartCtx.lineTo(width, y);
            this.chartCtx.stroke();
        }
    }
    
    updateStatus(message, type = '') {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = 'status ' + type;
    }
}
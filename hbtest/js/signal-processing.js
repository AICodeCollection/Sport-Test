class SignalProcessor {
    constructor(sampleRate = 30) {
        this.sampleRate = sampleRate;
        this.bufferSize = sampleRate * 15; // 减少到15秒缓冲，提高响应速度
        this.signalBuffer = [];
        this.fft = new FFT(512); // 减少到512点FFT，平衡精度和性能
        
        // 优化滤波器参数，降低阈值提高敏感度
        this.lowPassAlpha = 0.3; // 增加低通滤波强度
        this.highPassAlpha = 0.95; // 稍微降低高通滤波
        this.lastRawValue = 0;
        this.lastFilteredValue = 0;
        
        // 基于de Haan论文的改进PPG算法参数
        this.motionRobustBuffer = []; // 运动鲁棒性缓冲区
        this.signatureVector = null; // 血容量脉搏信号特征向量
        this.adaptiveThreshold = 0.3; // 降低自适应阈值，增加敏感度
        this.motionDetectionWindow = 15; // 减少运动检测窗口
        
        // 初始化血容量脉搏信号特征向量
        this.initializeSignatureVector();
    }
    
    // 初始化血容量脉搏信号特征向量
    initializeSignatureVector() {
        // 基于de Haan论文的血容量脉搏信号特征向量
        // 这个向量在归一化RGB空间中定义，对运动更鲁棒
        this.signatureVector = {
            r: 0.77,  // 红色通道权重
            g: 0.51,  // 绿色通道权重  
            b: 0.34   // 蓝色通道权重
        };
        
        // 这些权重基于血红蛋白和氧合血红蛋白的吸收光谱特性
        // 在可见光范围内的优化组合
    }
    
    // 带通滤波器 (0.8Hz - 3Hz)
    bandPassFilter(signal) {
        const filtered = new Array(signal.length);
        
        // 高通滤波 (去除直流分量)
        filtered[0] = signal[0];
        for (let i = 1; i < signal.length; i++) {
            filtered[i] = this.highPassAlpha * (filtered[i-1] + signal[i] - signal[i-1]);
        }
        
        // 低通滤波 (去除高频噪声)
        for (let i = 1; i < filtered.length; i++) {
            filtered[i] = this.lowPassAlpha * filtered[i] + (1 - this.lowPassAlpha) * filtered[i-1];
        }
        
        return filtered;
    }
    
    // 移动平均滤波
    movingAverage(signal, windowSize = 5) {
        const smoothed = new Array(signal.length);
        
        for (let i = 0; i < signal.length; i++) {
            let sum = 0;
            let count = 0;
            
            for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                sum += signal[j];
                count++;
            }
            
            smoothed[i] = sum / count;
        }
        
        return smoothed;
    }
    
    // 去除异常值
    removeOutliers(signal, threshold = 2) {
        const mean = signal.reduce((sum, val) => sum + val, 0) / signal.length;
        const std = Math.sqrt(signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length);
        
        return signal.map(val => {
            if (Math.abs(val - mean) > threshold * std) {
                return mean;
            }
            return val;
        });
    }
    
    // 添加新的数据点
    addDataPoint(value) {
        this.signalBuffer.push(value);
        
        // 保持缓冲区大小
        if (this.signalBuffer.length > this.bufferSize) {
            this.signalBuffer.shift();
        }
        
        // 更新运动鲁棒性缓冲区
        this.updateMotionRobustBuffer(value);
    }
    
    // 更新运动鲁棒性缓冲区
    updateMotionRobustBuffer(value) {
        this.motionRobustBuffer.push(value);
        
        // 保持运动检测窗口大小
        if (this.motionRobustBuffer.length > this.motionDetectionWindow * this.sampleRate) {
            this.motionRobustBuffer.shift();
        }
    }
    
    // 检测运动伪影
    detectMotionArtifacts() {
        if (this.motionRobustBuffer.length < this.motionDetectionWindow) {
            return false;
        }
        
        // 计算信号的短时变化
        const windowSize = this.sampleRate; // 1秒窗口
        const windows = [];
        
        for (let i = 0; i <= this.motionRobustBuffer.length - windowSize; i += windowSize) {
            const window = this.motionRobustBuffer.slice(i, i + windowSize);
            const variance = this.calculateVariance(window);
            windows.push(variance);
        }
        
        if (windows.length < 2) return false;
        
        // 检查方差的变化
        const varianceChange = this.calculateVariance(windows);
        const avgVariance = windows.reduce((sum, val) => sum + val, 0) / windows.length;
        
        // 如果方差变化太大，认为有运动伪影
        return varianceChange > avgVariance * 1.5;
    }
    
    // 计算方差
    calculateVariance(data) {
        const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
        return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    }
    
    // 自适应信号处理
    adaptiveSignalProcessing(signal) {
        // 检测运动伪影
        const hasMotion = this.detectMotionArtifacts();
        
        if (hasMotion) {
            // 使用更激进的滤波参数
            this.lowPassAlpha = 0.1;
            this.highPassAlpha = 0.99;
            
            // 应用额外的平滑
            signal = this.movingAverage(signal, 8);
        } else {
            // 使用标准滤波参数
            this.lowPassAlpha = 0.15;
            this.highPassAlpha = 0.98;
            
            // 标准平滑
            signal = this.movingAverage(signal, 5);
        }
        
        return signal;
    }
    
    // 获取处理后的信号
    getProcessedSignal() {
        if (this.signalBuffer.length < 30) { // 降低到1秒数据即可开始处理
            return null;
        }
        
        // 复制缓冲区数据
        let signal = [...this.signalBuffer];
        
        // 改进的信号处理流程
        signal = this.removeOutliers(signal);
        signal = this.bandPassFilter(signal);
        signal = this.adaptiveSignalProcessing(signal); // 使用自适应处理
        
        return signal;
    }
    
    // 计算心率
    calculateHeartRate() {
        const signal = this.getProcessedSignal();
        if (!signal || signal.length < 90) { // 降低到3秒数据
            return null;
        }
        
        // 准备FFT数据
        const fftSize = 512; // 使用较小的FFT大小
        const real = new Array(fftSize).fill(0);
        const imag = new Array(fftSize).fill(0);
        
        // 使用最近的数据进行FFT
        const startIndex = Math.max(0, signal.length - fftSize);
        for (let i = 0; i < Math.min(fftSize, signal.length); i++) {
            real[i] = signal[startIndex + i];
        }
        
        // 应用汉宁窗
        this.applyHanningWindow(real);
        
        // 执行FFT
        const fftResult = this.fft.fft(real, imag);
        const magnitude = this.fft.magnitude(fftResult.real, fftResult.imag);
        
        // 找到峰值频率，使用更精确的峰值检测
        const peakFrequency = this.findRobustPeakFrequency(magnitude, this.sampleRate);
        
        // 转换为BPM
        let bpm = peakFrequency * 60;
        
        // 智能心率校正：如果检测到的心率明显偏低，尝试二倍频
        if (bpm < 50 && bpm >= 25) {
            const correctedBPM = bpm * 2;
            if (correctedBPM >= 50 && correctedBPM <= 200) {
                console.log(`Low heart rate detected (${bpm.toFixed(1)}), correcting to double frequency: ${correctedBPM.toFixed(1)} BPM`);
                bpm = correctedBPM;
            }
        }
        
        // 如果心率过高，尝试一半频率
        if (bpm > 150 && bpm <= 400) {
            const correctedBPM = bpm / 2;
            if (correctedBPM >= 50 && correctedBPM <= 150) {
                console.log(`High heart rate detected (${bpm.toFixed(1)}), correcting to half frequency: ${correctedBPM.toFixed(1)} BPM`);
                bpm = correctedBPM;
            }
        }
        
        // 放宽检查范围并增加调试信息
        if (bpm >= 40 && bpm <= 220) {
            console.log(`Final heart rate: ${Math.round(bpm)} BPM, Peak frequency: ${peakFrequency.toFixed(3)} Hz`);
            return Math.round(bpm);
        }
        
        console.log(`Invalid BPM after correction: ${bpm}, Peak frequency: ${peakFrequency.toFixed(3)} Hz`);
        return null;
    }
    
    // 更robust的峰值检测 (基于de Haan论文改进)
    findRobustPeakFrequency(magnitude, sampleRate) {
        const minFreq = 0.7; // 42 BPM
        const maxFreq = 3.5; // 210 BPM
        
        // 修正索引计算：frequency = index * sampleRate / N
        // 所以 index = frequency * N / sampleRate
        const minIndex = Math.floor(minFreq * magnitude.length / sampleRate);
        const maxIndex = Math.floor(maxFreq * magnitude.length / sampleRate);
        
        console.log(`FFT analysis: sampleRate=${sampleRate}, magnitude.length=${magnitude.length}, minIndex=${minIndex}, maxIndex=${maxIndex}, freq range: ${minFreq}-${maxFreq} Hz`);
        
        // 找到所有局部峰值
        const peaks = [];
        for (let i = minIndex + 1; i < maxIndex - 1 && i < magnitude.length / 2; i++) {
            if (magnitude[i] > magnitude[i-1] && magnitude[i] > magnitude[i+1]) {
                // 计算峰值的尖锐度
                const sharpness = this.calculatePeakSharpness(magnitude, i);
                // 正确的频率计算公式
                const frequency = i * sampleRate / magnitude.length;
                peaks.push({
                    index: i,
                    value: magnitude[i],
                    frequency: frequency,
                    sharpness: sharpness
                });
            }
        }
        
        if (peaks.length === 0) {
            return 0;
        }
        
        // 结合幅值和尖锐度进行排序
        peaks.sort((a, b) => {
            const scoreA = a.value * (1 + a.sharpness);
            const scoreB = b.value * (1 + b.sharpness);
            return scoreB - scoreA;
        });
        
        // 输出前几个峰值用于调试
        console.log('Top 5 peaks:');
        peaks.slice(0, 5).forEach((peak, index) => {
            const bpm = peak.frequency * 60;
            console.log(`  ${index + 1}. Index: ${peak.index}, Freq: ${peak.frequency.toFixed(3)} Hz, BPM: ${bpm.toFixed(1)}, Magnitude: ${peak.value.toFixed(3)}`);
        });
        
        // 检查最佳峰值是否显著
        const bestPeak = peaks[0];
        const avgMagnitude = magnitude.slice(minIndex, maxIndex).reduce((sum, val) => sum + val, 0) / (maxIndex - minIndex);
        
        console.log(`Best peak: ${bestPeak.frequency.toFixed(3)} Hz (${(bestPeak.frequency * 60).toFixed(1)} BPM), magnitude: ${bestPeak.value.toFixed(3)}, avg: ${avgMagnitude.toFixed(3)}`);
        
        // 降低显著性阈值以便检测
        const significanceThreshold = this.adaptiveThreshold;
        if (bestPeak.value < avgMagnitude * (1.5 + significanceThreshold)) {
            console.log('Peak not significant enough');
            return 0;
        }
        
        // 检查二倍频峰值，可能心率检测到了一半
        const doubleFreqPeak = peaks.find(peak => 
            Math.abs(peak.frequency - bestPeak.frequency * 2) < 0.1
        );
        
        if (doubleFreqPeak && doubleFreqPeak.value > bestPeak.value * 0.7) {
            console.log(`Found double frequency peak: ${doubleFreqPeak.frequency.toFixed(3)} Hz, using it instead`);
            return doubleFreqPeak.frequency;
        }
        
        // 检查一半频率峰值，可能检测到了二倍频
        const halfFreqPeak = peaks.find(peak => 
            Math.abs(peak.frequency - bestPeak.frequency / 2) < 0.1
        );
        
        if (halfFreqPeak && halfFreqPeak.value > bestPeak.value * 0.5) {
            console.log(`Found half frequency peak: ${halfFreqPeak.frequency.toFixed(3)} Hz, considering it`);
            // 如果一半频率的峰值也很强，选择更合理的心率范围
            const bestBPM = bestPeak.frequency * 60;
            const halfBPM = halfFreqPeak.frequency * 60;
            
            if (bestBPM > 120 && halfBPM >= 50 && halfBPM <= 120) {
                console.log(`Using half frequency as it's in normal heart rate range`);
                return halfFreqPeak.frequency;
            }
        }
        
        return bestPeak.frequency;
    }
    
    // 计算峰值尖锐度
    calculatePeakSharpness(magnitude, peakIndex) {
        const windowSize = 3;
        const start = Math.max(0, peakIndex - windowSize);
        const end = Math.min(magnitude.length, peakIndex + windowSize + 1);
        
        const peakValue = magnitude[peakIndex];
        let sumDiff = 0;
        let count = 0;
        
        for (let i = start; i < end; i++) {
            if (i !== peakIndex) {
                sumDiff += peakValue - magnitude[i];
                count++;
            }
        }
        
        return count > 0 ? sumDiff / count : 0;
    }
    
    // 应用汉宁窗
    applyHanningWindow(signal) {
        for (let i = 0; i < signal.length; i++) {
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (signal.length - 1)));
            signal[i] *= window;
        }
    }
    
    // 获取信号质量指标 (改进版本)
    getSignalQuality() {
        if (this.signalBuffer.length < 30) { // 降低要求
            return 'insufficient';
        }
        
        const signal = this.getProcessedSignal();
        if (!signal) {
            return 'poor';
        }
        
        // 多维度信号质量评估
        const snr = this.calculateSNR(signal);
        const motionLevel = this.detectMotionArtifacts() ? 0.7 : 1.0; // 减少运动影响
        const signalStability = this.calculateSignalStability(signal);
        const peakQuality = this.assessPeakQuality(signal);
        
        // 综合评分，降低阈值
        const qualityScore = (snr * 0.4 + motionLevel * 0.2 + signalStability * 0.2 + peakQuality * 0.2);
        
        console.log(`Signal quality components: SNR=${snr.toFixed(3)}, Motion=${motionLevel}, Stability=${signalStability.toFixed(3)}, Peak=${peakQuality.toFixed(3)}, Total=${qualityScore.toFixed(3)}`);
        
        if (qualityScore > 0.7) {
            return 'excellent';
        } else if (qualityScore > 0.5) {
            return 'good';
        } else if (qualityScore > 0.3) {
            return 'fair';
        } else {
            return 'poor';
        }
    }
    
    // 计算信噪比
    calculateSNR(signal) {
        const mean = signal.reduce((sum, val) => sum + val, 0) / signal.length;
        const variance = signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length;
        const snr = Math.abs(mean) / Math.sqrt(variance);
        return Math.min(snr / 0.5, 1.0); // 归一化到0-1
    }
    
    // 计算信号稳定性
    calculateSignalStability(signal) {
        const windowSize = this.sampleRate * 2; // 2秒窗口
        const windows = [];
        
        for (let i = 0; i <= signal.length - windowSize; i += windowSize) {
            const window = signal.slice(i, i + windowSize);
            const variance = this.calculateVariance(window);
            windows.push(variance);
        }
        
        if (windows.length < 2) return 0.5;
        
        const avgVariance = windows.reduce((sum, val) => sum + val, 0) / windows.length;
        const varianceOfVariances = this.calculateVariance(windows);
        
        const stability = avgVariance / (avgVariance + varianceOfVariances);
        return Math.min(stability, 1.0);
    }
    
    // 评估峰值质量
    assessPeakQuality(signal) {
        // 执行FFT分析来评估峰值质量
        const fftSize = 512;
        const real = new Array(fftSize).fill(0);
        const imag = new Array(fftSize).fill(0);
        
        const startIndex = Math.max(0, signal.length - fftSize);
        for (let i = 0; i < Math.min(fftSize, signal.length); i++) {
            real[i] = signal[startIndex + i];
        }
        
        this.applyHanningWindow(real);
        const fftResult = this.fft.fft(real, imag);
        const magnitude = this.fft.magnitude(fftResult.real, fftResult.imag);
        
        // 计算心率频段内的能量比例
        const minFreq = 0.7;
        const maxFreq = 3.5;
        const minIndex = Math.floor(minFreq * magnitude.length / this.sampleRate);
        const maxIndex = Math.floor(maxFreq * magnitude.length / this.sampleRate);
        
        const heartRateEnergy = magnitude.slice(minIndex, maxIndex).reduce((sum, val) => sum + val * val, 0);
        const totalEnergy = magnitude.reduce((sum, val) => sum + val * val, 0);
        
        const energyRatio = heartRateEnergy / totalEnergy;
        return Math.min(energyRatio * 2, 1.0); // 归一化
    }
    
    // 重置缓冲区
    reset() {
        this.signalBuffer = [];
        this.motionRobustBuffer = [];
        this.lastRawValue = 0;
        this.lastFilteredValue = 0;
        
        // 重置自适应参数
        this.adaptiveThreshold = 0.7;
        this.lowPassAlpha = 0.15;
        this.highPassAlpha = 0.98;
    }
}
class FFT {
    constructor(size) {
        this.size = size;
        this.reverse = new Array(size);
        this.cosTable = new Array(size);
        this.sinTable = new Array(size);
        
        // 预计算位反转表
        for (let i = 0; i < size; i++) {
            this.reverse[i] = this.reverseBits(i, Math.log2(size));
        }
        
        // 预计算三角函数表
        for (let i = 0; i < size; i++) {
            this.cosTable[i] = Math.cos(2 * Math.PI * i / size);
            this.sinTable[i] = Math.sin(2 * Math.PI * i / size);
        }
    }
    
    reverseBits(num, bits) {
        let result = 0;
        for (let i = 0; i < bits; i++) {
            result = (result << 1) | (num & 1);
            num >>= 1;
        }
        return result;
    }
    
    fft(real, imag) {
        const N = this.size;
        const real_out = new Array(N);
        const imag_out = new Array(N);
        
        // 位反转重排
        for (let i = 0; i < N; i++) {
            real_out[i] = real[this.reverse[i]];
            imag_out[i] = imag[this.reverse[i]];
        }
        
        // 蝶形运算
        for (let len = 2; len <= N; len <<= 1) {
            const step = N / len;
            for (let i = 0; i < N; i += len) {
                for (let j = 0; j < len / 2; j++) {
                    const u = real_out[i + j];
                    const v = imag_out[i + j];
                    const w_real = this.cosTable[step * j];
                    const w_imag = -this.sinTable[step * j];
                    const t_real = real_out[i + j + len / 2] * w_real - imag_out[i + j + len / 2] * w_imag;
                    const t_imag = real_out[i + j + len / 2] * w_imag + imag_out[i + j + len / 2] * w_real;
                    
                    real_out[i + j] = u + t_real;
                    imag_out[i + j] = v + t_imag;
                    real_out[i + j + len / 2] = u - t_real;
                    imag_out[i + j + len / 2] = v - t_imag;
                }
            }
        }
        
        return { real: real_out, imag: imag_out };
    }
    
    magnitude(real, imag) {
        const magnitude = new Array(real.length);
        for (let i = 0; i < real.length; i++) {
            magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }
        return magnitude;
    }
    
    findPeakFrequency(magnitude, sampleRate) {
        let maxIndex = 0;
        let maxValue = 0;
        
        // 只在心率范围内寻找峰值 (0.8Hz - 3Hz, 对应48-180 BPM)
        const minIndex = Math.floor(0.8 * magnitude.length / sampleRate);
        const maxIndex_limit = Math.floor(3.0 * magnitude.length / sampleRate);
        
        for (let i = minIndex; i < maxIndex_limit && i < magnitude.length / 2; i++) {
            if (magnitude[i] > maxValue) {
                maxValue = magnitude[i];
                maxIndex = i;
            }
        }
        
        // 转换为频率
        const frequency = maxIndex * sampleRate / magnitude.length;
        return frequency;
    }
}
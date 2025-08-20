class HeartRateApp {
    constructor() {
        this.detector = new HeartRateDetector();
        this.isInitialized = false;
        
        this.bindEvents();
    }
    
    bindEvents() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        startBtn.addEventListener('click', () => this.startDetection());
        stopBtn.addEventListener('click', () => this.stopDetection());
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', () => {
            this.init();
        });
    }
    
    async init() {
        try {
            await this.detector.init();
            this.isInitialized = true;
            this.updateStatus('系统已准备就绪，点击开始检测', 'success');
        } catch (error) {
            this.updateStatus('初始化失败: ' + error.message, 'error');
            console.error('Initialization error:', error);
        }
    }
    
    async startDetection() {
        if (!this.isInitialized) {
            this.updateStatus('系统未初始化，正在重试...', 'error');
            await this.init();
            return;
        }
        
        try {
            this.detector.startDetection();
            this.updateButtonStates(true);
        } catch (error) {
            this.updateStatus('启动检测失败: ' + error.message, 'error');
            console.error('Detection start error:', error);
        }
    }
    
    stopDetection() {
        this.detector.stopDetection();
        this.updateButtonStates(false);
    }
    
    updateButtonStates(isDetecting) {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        startBtn.disabled = isDetecting;
        stopBtn.disabled = !isDetecting;
    }
    
    updateStatus(message, type = '') {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = 'status ' + type;
    }
}

// 创建应用实例
const app = new HeartRateApp();

// 检查浏览器兼容性
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('您的浏览器不支持摄像头访问，请使用现代浏览器（Chrome、Firefox、Safari等）');
}

// 错误处理
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    app.updateStatus('发生错误: ' + event.error.message, 'error');
});

// 未捕获的Promise错误
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    app.updateStatus('发生错误: ' + event.reason, 'error');
});
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import cv2
import json
from pose_detector import RunningPoseDetector
from gait_analyzer import GaitAnalyzer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# 创建上传文件夹
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# 初始化分析器
pose_detector = RunningPoseDetector()
gait_analyzer = GaitAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传视频文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 分析视频
        try:
            analysis_result = analyze_video(filepath, filename)
            return jsonify(analysis_result)
        except Exception as e:
            return jsonify({'error': f'分析视频时出错: {str(e)}'}), 500

def analyze_video(video_path, filename):
    """分析视频文件"""
    # 检测姿态
    frame_data = pose_detector.process_video(video_path)
    
    if not frame_data:
        return {'error': '无法检测到人体姿态'}
    
    # 分析步态
    gait_data = gait_analyzer.analyze_gait_cycle(frame_data)
    form_data = gait_analyzer.analyze_running_form(frame_data)
    
    # 生成建议
    recommendations = gait_analyzer.generate_recommendations(gait_data, form_data)
    
    # 生成可视化图表
    charts = generate_charts(gait_data, form_data)
    
    return {
        'video_url': f'/uploads/{filename}',
        'gait_metrics': {
            'cadence': round(gait_data['cadence'], 1),
            'stride_length': round(gait_data['stride_length'], 3),
            'total_steps': gait_data['total_steps']
        },
        'form_analysis': {
            'avg_knee_angle': round(np.mean(form_data['knee_angles']), 1) if form_data['knee_angles'] else 0,
            'avg_forward_lean': round(np.mean(form_data['forward_lean']), 1) if form_data['forward_lean'] else 0
        },
        'recommendations': recommendations,
        'charts': charts
    }

def generate_charts(gait_data, form_data):
    """生成分析图表"""
    charts = {}
    
    # 步频分析图
    if gait_data['left_foot_strikes'] and gait_data['right_foot_strikes']:
        plt.figure(figsize=(10, 6))
        plt.plot(gait_data['left_foot_strikes'], [1] * len(gait_data['left_foot_strikes']), 'bo', label='Left Foot Strike')
        plt.plot(gait_data['right_foot_strikes'], [2] * len(gait_data['right_foot_strikes']), 'ro', label='Right Foot Strike')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Foot')
        plt.title('Gait Cycle Analysis')
        plt.legend()
        plt.grid(True)
        plt.yticks([1, 2], ['Left', 'Right'])
        
        # 保存图表为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        charts['gait_cycle'] = base64.b64encode(chart_data).decode()
    
    # 膝关节角度变化图
    if form_data['knee_angles']:
        plt.figure(figsize=(10, 6))
        plt.plot(form_data['knee_angles'], 'b-', linewidth=2)
        plt.xlabel('Frame Number')
        plt.ylabel('Knee Angle (degrees)')
        plt.title('Knee Angle Changes')
        plt.grid(True, alpha=0.3)
        
        # 添加理想范围
        plt.axhline(y=140, color='r', linestyle='--', alpha=0.7, label='Min Recommended Angle')
        plt.axhline(y=170, color='r', linestyle='--', alpha=0.7, label='Max Recommended Angle')
        plt.legend()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        charts['knee_angles'] = base64.b64encode(chart_data).decode()
    
    return charts

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8888)
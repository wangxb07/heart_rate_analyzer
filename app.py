from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename
from analyzer import HeartRateAnalyzer
import plotly.subplots

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'csv'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'heart_rate_file' not in request.files or 'breath_rate_file' not in request.files:
            return jsonify({'error': '请上传两个文件'}), 400
        
        heart_rate_file = request.files['heart_rate_file']
        breath_rate_file = request.files['breath_rate_file']
        
        if heart_rate_file.filename == '' or breath_rate_file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        if not (allowed_file(heart_rate_file.filename) and allowed_file(breath_rate_file.filename)):
            return jsonify({'error': '文件类型不支持'}), 400
        
        try:
            # 读取数据
            heart_rate_data = [line.decode('utf-8').strip() for line in heart_rate_file.readlines() if line.strip()]
            breath_rate_data = [line.decode('utf-8').strip() for line in breath_rate_file.readlines() if line.strip()]
            
            if not heart_rate_data or not breath_rate_data:
                return jsonify({'error': '文件内容为空'}), 400
                
            # 打印调试信息
            # print("心率数据前几行:")
            # for line in heart_rate_data[:3]:
            #     print(f"原始行: '{line}'")
            #     print(f"分割后: {line.split('\t')}")
                
            # print("\n呼吸率数据前几行:")
            # for line in breath_rate_data[:3]:
            #     print(f"原始行: '{line}'")
            #     print(f"分割后: {line.split('\t')}")
            
            # 合并数据
            combined_data = heart_rate_data + ['这是一段 呼吸率的记录'] + breath_rate_data
            
            # 分析数据
            analyzer = HeartRateAnalyzer()
            results = analyzer.analyze(combined_data)
            
            # 创建绘图
            fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
            
            # 添加原始心率数据
            fig.add_trace(
                go.Scatter(
                    x=results.index,
                    y=results['heart_rate'],
                    name="原始心率",
                    line=dict(color='blue', width=1),
                    opacity=0.7
                ),
                secondary_y=False
            )
            
            # 添加修正后的心率数据
            fig.add_trace(
                go.Scatter(
                    x=results.index,
                    y=results['corrected_heart_rate'],
                    name="修正后心率",
                    line=dict(color='red', width=1)
                ),
                secondary_y=False
            )
            
            # 添加呼吸率数据（使用次坐标轴）
            fig.add_trace(
                go.Scatter(
                    x=results.index,
                    y=results['breath_rate'],
                    name="呼吸率",
                    line=dict(color='green', width=1, dash='dash'),
                    opacity=0.7
                ),
                secondary_y=True
            )
            
            # 添加可靠性分数低的点
            low_reliability = results[results['reliability_score'] < analyzer.reliability_threshold]
            fig.add_trace(
                go.Scatter(
                    x=low_reliability.index,
                    y=low_reliability['heart_rate'],
                    mode='markers',
                    name='低可靠性点',
                    marker=dict(color='orange', size=8, symbol='x'),
                    opacity=0.5
                ),
                secondary_y=False
            )
            
            # 更新布局
            fig.update_layout(
                title='心率和呼吸率分析',
                xaxis_title='时间',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # 更新Y轴标题
            fig.update_yaxes(title_text="心率 (次/分钟)", secondary_y=False)
            fig.update_yaxes(title_text="呼吸率 (次/分钟)", secondary_y=True)
            
            # 生成统计报告
            stats = {
                'original_stats': {
                    'mean': float(round(results['heart_rate'].mean(), 2)),
                    'std': float(round(results['heart_rate'].std(), 2)),
                    'min': int(results['heart_rate'].min()),
                    'max': int(results['heart_rate'].max())
                },
                'corrected_stats': {
                    'mean': float(round(results['corrected_heart_rate'].mean(), 2)),
                    'std': float(round(results['corrected_heart_rate'].std(), 2)),
                    'min': int(results['corrected_heart_rate'].min()),
                    'max': int(results['corrected_heart_rate'].max())
                },
                'low_reliability_points': int(len(results[results['reliability_score'] < 70]))
            }
            
            # 确保时间戳是字符串格式
            results.index = results.index.astype(str)
            
            return jsonify({
                'plot': json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)),
                'stats': stats
            })
            
        except Exception as e:
            print("处理文件时出错:", str(e))
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500
            
    except Exception as e:
        print("上传文件时出错:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'上传文件时出错: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)

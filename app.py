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

            # 合并数据
            combined_data = heart_rate_data + ['这是一段 呼吸率的记录'] + breath_rate_data
            
            print(f"开始处理数据，共 {len(combined_data)} 条记录")
            
            def parse_data_point(data_point):
                """解析数据点，处理字符串格式的数据"""
                if isinstance(data_point, str) and ',' not in data_point:
                    return data_point
                try:
                    if isinstance(data_point, (list, tuple)):
                        return data_point
                    # 处理字符串形式的数据点
                    parts = str(data_point).split(',')
                    return [
                        parts[0].strip(),  # id
                        parts[1].strip().strip("'"),  # timestamp
                        parts[2].strip().strip("'"),  # device_time
                        int(parts[3].strip())  # heart_rate/breath_rate
                    ]
                except Exception as e:
                    print(f"解析数据点失败: {str(e)}, 原始数据: {data_point}")
                    return None

            # 分离心率和呼吸率数据
            split_index = combined_data.index('这是一段 呼吸率的记录')
            heart_data = [parse_data_point(d) for d in combined_data[:split_index]]
            breath_data = [parse_data_point(d) for d in combined_data[split_index + 1:]]
            
            # 过滤掉解析失败的数据点
            heart_data = [d for d in heart_data if d is not None]
            breath_data = [d for d in breath_data if d is not None]
            
            print(f"处理后的心率数据: {len(heart_data)} 条")
            print(f"处理后的呼吸数据: {len(breath_data)} 条")
            
            def parse_timestamp(ts_str):
                """解析ISO 8601格式的时间戳"""
                try:
                    return pd.to_datetime(ts_str, format='%Y-%m-%dT%H:%M:%S%z')
                except ValueError:
                    # 处理带有小数秒的情况
                    return pd.to_datetime(ts_str)
            
            # 首先对数据按时间排序（倒序）
            sorted_heart_data = sorted(heart_data, key=lambda x: parse_timestamp(x[2]), reverse=True)
            sorted_breath_data = sorted(breath_data, key=lambda x: parse_timestamp(x[2]), reverse=True)
            
            # 处理心率数据分段
            heart_segments = []
            current_segment = []
            last_timestamp = None
            
            for data_point in sorted_heart_data:
                try:
                    current_timestamp = parse_timestamp(data_point[2])
                    
                    if last_timestamp is not None:
                        time_diff = abs((current_timestamp - last_timestamp).total_seconds() / 60)
                        if time_diff > 10:
                            if current_segment:
                                print(f"发现时间间隔 {time_diff:.1f} 分钟，创建新的数据段")
                                heart_segments.append(current_segment)
                            current_segment = []
                    
                    current_segment.append(data_point)
                    last_timestamp = current_timestamp
                except Exception as e:
                    print(f"处理数据点时出错: {str(e)}, 数据点: {data_point}")
                    continue
            
            if current_segment:
                heart_segments.append(current_segment)
            
            # 处理呼吸率数据分段（使用相同的分段点）
            breath_segments = []
            for heart_segment in heart_segments:
                start_time = parse_timestamp(heart_segment[-1][2])  # 使用心率段的开始时间
                end_time = parse_timestamp(heart_segment[0][2])    # 使用心率段的结束时间
                
                # 找到对应时间范围内的呼吸率数据
                breath_segment = [
                    point for point in sorted_breath_data
                    if start_time <= parse_timestamp(point[2]) <= end_time
                ]
                breath_segments.append(breath_segment)
            
            # 组合每个段的数据
            all_segments = []
            for heart_segment, breath_segment in zip(heart_segments, breath_segments):
                # 将数据转换为字符串格式，并添加标题行
                header = "id,timestamp,device_time,value"
                heart_str_data = [header] + [f"{d[0]},{d[1]},{d[2]},{d[3]}" for d in heart_segment]
                breath_str_data = [header] + [f"{d[0]},{d[1]},{d[2]},{d[3]}" for d in breath_segment]
                combined_segment = heart_str_data + ['这是一段 呼吸率的记录'] + breath_str_data
                all_segments.append(combined_segment)
            
            # 分析所有段，筛选出有效的段
            valid_segments = []
            analyzer = HeartRateAnalyzer()
            
            print("\n开始分析所有数据段:")
            for i, segment in enumerate(all_segments):
                try:
                    results = analyzer.analyze(segment)
                    if results.empty:
                        print(f"段 {i + 1}: 无有效数据，跳过")
                        continue
                    
                    # 计算有效数据点数（同时有心率和呼吸率的点）
                    valid_points = results[results['heart_rate'].notna() & results['breath_rate'].notna()]
                    valid_count = len(valid_points)
                    
                    # 计算时间跨度
                    time_span = (valid_points.index.max() - valid_points.index.min()).total_seconds() / 60
                    
                    # 计算平均心率
                    avg_heart_rate = valid_points['heart_rate'].mean()
                    avg_breath_rate = valid_points['breath_rate'].mean()
                    
                    print(f"段 {i + 1}:")
                    print(f"  - 总数据点: {len(results)}")
                    print(f"  - 有效数据点: {valid_count}")
                    print(f"  - 时间跨度: {time_span:.1f}分钟")
                    print(f"  - 平均心率: {avg_heart_rate:.1f}")
                    print(f"  - 平均呼吸率: {avg_breath_rate:.1f}")
                    
                    # 检查数据质量
                    if valid_count >= 10 and time_span >= 1:  # 至少100个点且跨度至少5分钟
                        valid_segments.append((segment, valid_count, time_span, avg_heart_rate, avg_breath_rate))
                        print(f"  - 状态: 有效")
                    else:
                        print(f"  - 状态: 无效 (需要>=100个点且>=5分钟)")
                except Exception as e:
                    print(f"段 {i + 1} 处理出错: {str(e)}")
                    continue
            
            if not valid_segments:
                return jsonify({'error': '没有找到足够的有效数据段（每段至少需要100个有效数据点且时间跨度>=5分钟）'}), 400
            
            # 按有效点数排序并限制最多5段
            valid_segments.sort(key=lambda x: x[1], reverse=True)
            segments_to_plot = [segment for segment, _, _, _, _ in valid_segments[:5]]
            
            print(f"\n将显示 {len(segments_to_plot)} 个有效数据段")
            
            # 创建子图，每个有效段一个图表
            fig = plotly.subplots.make_subplots(
                rows=len(segments_to_plot), 
                cols=1,
                subplot_titles=[
                    f"数据段 {i+1} (有效点数: {count}, 时长: {span:.1f}分钟, 平均心率: {hr:.1f}, 平均呼吸率: {br:.1f})" 
                    for i, (_, count, span, hr, br) in enumerate(valid_segments[:5])
                ],
                specs=[[{"secondary_y": True}] for _ in range(len(segments_to_plot))],
                vertical_spacing=0.1
            )
            
            # 分析每个段并绘图
            for segment_idx, segment_data in enumerate(segments_to_plot):
                print(f"\n正在处理第 {segment_idx + 1}/{len(segments_to_plot)} 段数据")
                row = segment_idx + 1
                
                # 分析当前段的数据
                analyzer = HeartRateAnalyzer()
                results = analyzer.analyze(segment_data)
                if results.empty:
                    print(f"段 {segment_idx + 1} 没有有效数据，跳过")
                    continue

                # 打印数据统计
                print(f"\n=== 段 {segment_idx + 1} 数据统计 ===")
                print(f"数据点总数: {len(results)}")
                print(f"有效心率数据点数: {results['heart_rate'].notna().sum()}")
                print(f"有效呼吸率数据点数: {results['breath_rate'].notna().sum()}")
                
                # 计算并打印一些基本统计信息
                if not results.empty:
                    hr_stats = results['heart_rate'].describe()
                    br_stats = results['breath_rate'].describe()
                    print("\n心率统计:")
                    print(f"平均值: {hr_stats['mean']:.2f}")
                    print(f"标准差: {hr_stats['std']:.2f}")
                    print(f"最小值: {hr_stats['min']:.2f}")
                    print(f"最大值: {hr_stats['max']:.2f}")
                    
                    print("\n呼吸率统计:")
                    print(f"平均值: {br_stats['mean']:.2f}")
                    print(f"标准差: {br_stats['std']:.2f}")
                    print(f"最小值: {br_stats['min']:.2f}")
                    print(f"最大值: {br_stats['max']:.2f}")
                
                # 绘制心率数据
                fig.add_trace(
                    go.Scatter(
                        x=results.index,
                        y=results['heart_rate'],
                        name=f'心率 (段 {segment_idx + 1})',
                        mode='lines',
                        line=dict(color='blue')
                    ),
                    row=row, col=1
                )
                
                # 绘制修正后的心率数据
                fig.add_trace(
                    go.Scatter(
                        x=results.index,
                        y=results['corrected_heart_rate'],
                        name=f'修正后心率 (段 {segment_idx + 1})',
                        mode='lines',
                        line=dict(color='green', dash='dash')
                    ),
                    row=row, col=1
                )
                
                # 在同一图上绘制呼吸率数据（使用次坐标轴）
                fig.add_trace(
                    go.Scatter(
                        x=results.index,
                        y=results['breath_rate'],
                        name=f'呼吸率 (段 {segment_idx + 1})',
                        mode='lines',
                        line=dict(color='red'),
                        yaxis='y2'
                    ),
                    row=row, col=1, secondary_y=True
                )
            
            # 更新布局
            fig.update_layout(
                title=f'心率和呼吸率分析 (显示{len(segments_to_plot)}个有效数据段)',
                showlegend=True,
                height=300 * len(segments_to_plot),  # 根据段数调整图表高度
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                hovermode='x unified'
            )
            
            # 为每个子图设置y轴标题
            for i in range(len(segments_to_plot)):
                fig.update_yaxes(title_text="心率 (次/分钟)", secondary_y=False, row=i+1, col=1)
                fig.update_yaxes(title_text="呼吸率 (次/分钟)", secondary_y=True, row=i+1, col=1)
            
            # 生成统计报告（合并所有段的数据）
            all_results = pd.concat([analyzer.analyze(segment) for segment, _, _, _, _ in valid_segments])
            stats = {
                'original_stats': {
                    'mean': float(round(all_results['heart_rate'].mean(), 2)),
                    'std': float(round(all_results['heart_rate'].std(), 2)),
                    'min': int(all_results['heart_rate'].min()),
                    'max': int(all_results['heart_rate'].max())
                },
                'corrected_stats': {
                    'mean': float(round(all_results['corrected_heart_rate'].mean(), 2)),
                    'std': float(round(all_results['corrected_heart_rate'].std(), 2)),
                    'min': int(all_results['corrected_heart_rate'].min()),
                    'max': int(all_results['corrected_heart_rate'].max())
                },
                'low_reliability_points': int(len(all_results[all_results['reliability_score'] < 70])),
                'segment_count': len(valid_segments),
                'segments_stats': [
                    {
                        'segment_number': i + 1,
                        'valid_points': count,
                        'duration_minutes': span,
                        'avg_heart_rate': round(hr, 1),
                        'avg_breath_rate': round(br, 1)
                    }
                    for i, (_, count, span, hr, br) in enumerate(valid_segments)
                ]
            }
            
            # 确保时间戳是字符串格式
            all_results.index = all_results.index.astype(str)
            
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
    app.run(host='0.0.0.0', port=3000)

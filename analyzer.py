import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore, pearsonr
from scipy.signal import correlate
from io import StringIO

class HeartRateAnalyzer:
    def __init__(self, window_size=5, reliability_threshold=70, correlation_half_window=10, correlation_threshold=0.8):
        self.window_size = window_size
        self.reliability_threshold = reliability_threshold
        self.correlation_half_window = correlation_half_window  # 相关性分析窗口的半径，实际窗口大小为 2*half_window+1
        self.correlation_threshold = correlation_threshold  # 相关性阈值
    
    def parse_datetime(self, dt_str):
        """处理时间字符串，移除多余的引号"""
        dt_str = dt_str.strip("'\"")  # 移除可能存在的引号
        return datetime.fromisoformat(dt_str)
        
    def preprocess_data(self, data):
        # 将数据解析为DataFrame
        split_index = data.index('这是一段 呼吸率的记录')
        heart_rate_data = data[:split_index]
        breath_rate_data = data[split_index+1:]
        
        # 跳过标题行
        heart_rate_data = heart_rate_data[1:]  # 跳过第一行（标题行）
        breath_rate_data = breath_rate_data[1:]  # 跳过第一行（标题行）
        
        # 创建心率DataFrame
        heart_rate_df = pd.DataFrame([
            {
                'id': int(row[0]),
                'timestamp': self.parse_datetime(row[1]),
                'device_time': self.parse_datetime(row[2]),
                'heart_rate': int(row[3])  # 保持列名为heart_rate
            } for row in [line.strip().split(',') for line in heart_rate_data if line.strip() and not line.startswith('id')]
        ])
        
        # 创建呼吸率DataFrame
        breath_rate_df = pd.DataFrame([
            {
                'id': int(row[0]),
                'timestamp': self.parse_datetime(row[1]),
                'device_time': self.parse_datetime(row[2]),
                'breath_rate': int(row[3])  # 保持列名为breath_rate
            } for row in [line.strip().split(',') for line in breath_rate_data if line.strip() and not line.startswith('id')]
        ])
        
        return heart_rate_df, breath_rate_df

    def calculate_hr_br_correlation(self, heart_rate_df, breath_rate_df):
        """计算心率和呼吸率的相关性分数
        
        使用滑动窗口计算局部相关性。窗口大小为 2*correlation_half_window+1，
        即在当前点两侧各取 correlation_half_window 个点。
        """
        # 确保两个DataFrame都使用device_time作为索引
        if 'device_time' not in heart_rate_df.columns:
            heart_rate_df = heart_rate_df.reset_index().rename(columns={'index': 'device_time'})
        if 'device_time' not in breath_rate_df.columns:
            breath_rate_df = breath_rate_df.reset_index().rename(columns={'index': 'device_time'})
            
        # 设置device_time为索引
        heart_rate_df = heart_rate_df.set_index('device_time')
        breath_rate_df = breath_rate_df.set_index('device_time')
        
        # 使用时间索引合并数据
        merged_df = pd.merge(
            heart_rate_df[['heart_rate']], 
            breath_rate_df[['breath_rate']], 
            left_index=True, 
            right_index=True,
            how='inner'
        )
        
        # 创建一个Series来存储相关性分数，使用与heart_rate_df相同的索引
        correlations = pd.Series(0.0, index=heart_rate_df.index)
        
        # 使用滑动窗口计算局部相关性
        for idx in heart_rate_df.index:
            # 在merged_df中找到对应时间点的位置
            if idx in merged_df.index:
                pos = merged_df.index.get_loc(idx)
                # 使用半窗口大小计算实际的窗口范围
                start_idx = max(0, pos - self.correlation_half_window)
                end_idx = min(len(merged_df), pos + self.correlation_half_window + 1)
                window_data = merged_df.iloc[start_idx:end_idx]
                
                if len(window_data) >= 3:  # 需要至少3个点才能计算相关性
                    try:
                        # 检查数据是否为常量
                        if (window_data['heart_rate'].nunique() == 1 or 
                            window_data['breath_rate'].nunique() == 1):
                            correlations[idx] = 0.0  # 如果任一数组是常量，相关系数设为0
                        else:
                            corr, _ = pearsonr(window_data['heart_rate'], window_data['breath_rate'])
                            correlations[idx] = abs(corr)  # 使用相关系数的绝对值
                    except:
                        correlations[idx] = 0.0
        
        return correlations

    def filter_invalid_data(self, heart_rate_df, breath_rate_df):
        """过滤无效数据"""
        # 处理心率数据
        if not heart_rate_df.empty:
            valid_heart = heart_rate_df['heart_rate'] > 0
            heart_rate_df = heart_rate_df[valid_heart].copy()
        
        # 处理呼吸率数据（如果存在）
        if not breath_rate_df.empty and 'breath_rate' in breath_rate_df.columns:
            valid_breath = breath_rate_df['breath_rate'] > 0
            breath_rate_df = breath_rate_df[valid_breath].copy()
        else:
            # 如果没有呼吸率数据，创建一个空的DataFrame保持一致的结构
            breath_rate_df = pd.DataFrame(columns=['id', 'timestamp', 'device_time', 'breath_rate'])
        
        return heart_rate_df, breath_rate_df

    def analyze(self, data):
        """分析心率和呼吸率数据"""
        try:
            # 分离心率和呼吸率数据
            split_index = data.index('这是一段 呼吸率的记录')
            heart_rate_data = pd.read_csv(StringIO('\n'.join(data[:split_index])))
            breath_rate_data = pd.read_csv(StringIO('\n'.join(data[split_index + 1:])))
            
            # 确保数据帧有必要的列
            required_columns = {
                'heart_rate_data': ['device_time', 'value'],
                'breath_rate_data': ['device_time', 'value']
            }
            
            for df_name, columns in required_columns.items():
                df = locals()[df_name]
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    print(f"{df_name} 缺少必要的列: {missing_cols}")
                    return pd.DataFrame()
            
            # 过滤掉值为0的数据点
            heart_rate_data = heart_rate_data[heart_rate_data['value'] > 0].copy()  # 创建副本避免警告
            breath_rate_data = breath_rate_data[breath_rate_data['value'] > 0].copy()
            
            if heart_rate_data.empty or breath_rate_data.empty:
                print("过滤0值后数据为空")
                return pd.DataFrame()
            
            # 设置时间索引 (链式操作避免警告)
            heart_rate_data = (heart_rate_data
                             .assign(device_time=lambda x: pd.to_datetime(x['device_time']))
                             .set_index('device_time')
                             .sort_index())
            
            # 对齐呼吸率数据
            aligned_breath_rate = self.align_breath_rate(heart_rate_data, breath_rate_data)
            
            # 创建结果DataFrame
            results = pd.DataFrame(index=heart_rate_data.index)
            results['heart_rate'] = heart_rate_data['value']
            results['breath_rate'] = aligned_breath_rate
            
            # 计算可靠性分数
            results['reliability_score'] = self.calculate_reliability_scores(results[['heart_rate']])
            
            # 修正心率数据
            results['corrected_heart_rate'] = self.correct_heart_rate(results['heart_rate'])
            
            # 计算相关性分数
            correlations = self.calculate_hr_br_correlation(
                results[['heart_rate']],
                results[['breath_rate']]
            )
            results['correlation'] = correlations
            
            # 根据相关性阈值、可靠性阈值和其他条件过滤数据
            valid = (
                (results['heart_rate'] > 0) & 
                (results['breath_rate'] > 0) & 
                (results['corrected_heart_rate'] > 0) &
                (results['correlation'] >= self.correlation_threshold) &
                (results['reliability_score'] >= self.reliability_threshold)
            )
            results = results[valid]
            
            return results
            
        except Exception as e:
            print(f"分析数据时出错: {str(e)}")
            return pd.DataFrame()

    def align_breath_rate(self, heart_rate_df, breath_rate_df):
        """将呼吸率数据对齐到心率数据的时间点"""
        try:
            # 检查输入数据是否为空
            if heart_rate_df.empty or breath_rate_df.empty:
                print("心率或呼吸率数据为空，返回空Series")
                return pd.Series(index=heart_rate_df.index, dtype=float)
            
            # 打印原始数据的样本
            print("\n=== 对齐前的数据样本 ===")
            print("\n心率数据样本（前3条）:")
            print(heart_rate_df.head(3))
            print("\n呼吸率数据样本（前3条）:")
            print(breath_rate_df[['device_time', 'value']].head(3))
            
            # 准备呼吸率数据
            breath_rate_df['device_time'] = pd.to_datetime(breath_rate_df['device_time'])
            breath_rate_df.sort_values('device_time', inplace=True)
            
            # 使用pandas的merge_asof进行最近邻匹配
            merged_df = pd.merge_asof(
                pd.DataFrame({'time': heart_rate_df.index}),
                pd.DataFrame({
                    'time': breath_rate_df['device_time'],
                    'breath_rate': breath_rate_df['value']
                }),
                on='time',
                tolerance=pd.Timedelta('30s'),
                direction='nearest'
            )
            
            # 创建结果Series，过滤掉0值
            aligned_breath = pd.Series(
                merged_df['breath_rate'].values,
                index=heart_rate_df.index
            )
            aligned_breath = aligned_breath[aligned_breath > 0]
            
            # 打印对齐后的结果样本
            print("\n=== 对齐后的数据样本 ===")
            result_df = pd.DataFrame({
                'device_time': heart_rate_df.index,
                'heart_rate': heart_rate_df['value'],
                'breath_rate': aligned_breath
            })
            print("\n对齐后的数据（前3条）:")
            print(result_df.head(3))
            
            # 打印一些统计信息
            print("\n=== 数据统计 ===")
            print(f"心率数据点数: {len(heart_rate_df)}")
            print(f"呼吸率原始数据点数: {len(breath_rate_df)}")
            print(f"对齐后的数据点数: {len(aligned_breath)}")
            print(f"对齐后的非空数据点数: {aligned_breath.notna().sum()}")
            print(f"对齐率: {(aligned_breath.notna().sum() / len(aligned_breath)) * 100:.2f}%")
            
            return aligned_breath
            
        except Exception as e:
            print(f"对齐呼吸率数据时出错: {str(e)}")
            return pd.Series(index=heart_rate_df.index, dtype=float)

    def calculate_reliability_scores(self, heart_rate_df):
        """计算心率数据的可靠性分数"""
        # 使用移动窗口计算统计数据
        window_mean = heart_rate_df['heart_rate'].rolling(
            window=self.window_size, center=True, min_periods=1
        ).mean()
        window_std = heart_rate_df['heart_rate'].rolling(
            window=self.window_size, center=True, min_periods=1
        ).std()
        
        # 计算差分
        diffs = heart_rate_df['heart_rate'].diff()
        
        # 计算每个点的可靠性分数
        scores = []
        for i in range(len(heart_rate_df)):
            # 获取当前点的统计数据
            mean = window_mean.iloc[i]
            std = window_std.iloc[i] if not pd.isna(window_std.iloc[i]) else 0
            diff = abs(diffs.iloc[i]) if i > 0 else 0
            
            # 计算分数
            score = 100
            
            # 基于与窗口均值的偏差
            if mean > 0:  # 避免除以零
                deviation = abs(heart_rate_df['heart_rate'].iloc[i] - mean) / mean
                score -= min(30, deviation * 100)
            
            # 基于窗口标准差
            if mean > 0:  # 避免除以零
                score -= min(30, (std / mean) * 100)
            
            # 基于与前一个点的差异
            if mean > 0:  # 避免除以零
                score -= min(40, (diff / mean) * 100)
            
            scores.append(max(0, min(100, score)))
        
        return scores

    def apply_moving_average(self, series, window_size=5):
        """应用移动平均来平滑数据"""
        return series.rolling(window=window_size, center=True, min_periods=1).mean()

    def correct_heart_rate(self, heart_rate):
        """修正心率数据"""
        # 使用移动平均来平滑心率数据
        smoothed_heart_rate = self.apply_moving_average(heart_rate)
        
        return smoothed_heart_rate

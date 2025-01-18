import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore, pearsonr
from scipy.signal import correlate

class HeartRateAnalyzer:
    def __init__(self, window_size=5, reliability_threshold=70, correlation_window=10, correlation_threshold=0.6):
        self.window_size = window_size
        self.reliability_threshold = reliability_threshold
        self.correlation_window = correlation_window  # 相关性分析的窗口大小
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
                'heart_rate': int(row[3])
            } for row in [line.strip().split(',') for line in heart_rate_data if line.strip()]
        ])
        
        # 创建呼吸率DataFrame
        breath_rate_df = pd.DataFrame([
            {
                'id': int(row[0]),
                'timestamp': self.parse_datetime(row[1]),
                'device_time': self.parse_datetime(row[2]),
                'breath_rate': int(row[3])
            } for row in [line.strip().split(',') for line in breath_rate_data if line.strip()]
        ])
        
        return heart_rate_df, breath_rate_df

    def calculate_hr_br_correlation(self, heart_rate_df, breath_rate_df):
        """计算心率和呼吸率的相关性分数"""
        # 确保两个DataFrame使用相同的时间索引
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
                start_idx = max(0, pos - self.correlation_window)
                end_idx = min(len(merged_df), pos + self.correlation_window + 1)
                window_data = merged_df.iloc[start_idx:end_idx]
                
                if len(window_data) >= 3:  # 需要至少3个点才能计算相关性
                    try:
                        corr, _ = pearsonr(window_data['heart_rate'], window_data['breath_rate'])
                        correlations[idx] = abs(corr)  # 使用相关系数的绝对值
                    except:
                        correlations[idx] = 0.0
        
        return correlations

    def calculate_reliability_score(self, row, window_stats, diffs, hr_br_correlation):
        """计算单个数据点的可靠性分数"""
        # 1. 基于窗口统计的分数 (0-40分)
        window_mean = window_stats.loc[row.name, 'window_mean']
        window_std = window_stats.loc[row.name, 'window_std']
        z_score = abs((row['heart_rate'] - window_mean) / window_std) if window_std != 0 else 0
        stats_score = max(0, 40 - z_score * 10)
        
        # 2. 基于差分的分数 (0-30分)
        diff_score = 30
        if row.name in diffs.index:
            diff = abs(diffs.loc[row.name])
            if diff > 20:  # 如果心率变化超过20，开始扣分
                diff_score = max(0, 30 - (diff - 20))
        
        # 3. 基于心率-呼吸率相关性的分数 (0-30分)
        correlation_score = hr_br_correlation * 30
        
        # 合并所有分数
        total_score = stats_score + diff_score + correlation_score
        return min(100, total_score)  # 确保分数不超过100
    
    def filter_invalid_data(self, heart_rate_df, breath_rate_df):
        """过滤掉无效的数据点"""
        # 1. 过滤掉心率或呼吸率为0的数据
        valid_heart = heart_rate_df['heart_rate'] > 0
        valid_breath = breath_rate_df['breath_rate'] > 0
        
        heart_rate_df = heart_rate_df[valid_heart].copy()
        breath_rate_df = breath_rate_df[valid_breath].copy()
        
        return heart_rate_df, breath_rate_df
    
    def analyze(self, data):
        """分析心率数据
        处理顺序：
        1. 预处理数据
        2. 过滤无效数据（零值）
        3. 计算并过滤呼吸相关性
        4. 对剩余数据进行可靠性分析和修正
        """
        # 1. 预处理数据
        heart_rate_df, breath_rate_df = self.preprocess_data(data)
        
        # 2. 过滤无效数据
        heart_rate_df, breath_rate_df = self.filter_invalid_data(heart_rate_df, breath_rate_df)
        
        # 3. 计算心率和呼吸率的相关性并过滤
        hr_br_correlations = self.calculate_hr_br_correlation(heart_rate_df, breath_rate_df)
        heart_rate_df['correlation'] = hr_br_correlations
        
        # 只保留相关性强的数据
        valid_correlation = heart_rate_df['correlation'] >= self.correlation_threshold
        heart_rate_df = heart_rate_df[valid_correlation].copy()
        
        # 如果过滤后没有数据，直接返回
        if len(heart_rate_df) == 0:
            return heart_rate_df
        
        # 4. 对相关性强的数据进行进一步分析
        # 4.1 计算滑动窗口统计
        heart_rate_df['window_mean'] = heart_rate_df['heart_rate'].rolling(
            window=self.window_size, center=True, min_periods=1
        ).mean()
        heart_rate_df['window_std'] = heart_rate_df['heart_rate'].rolling(
            window=self.window_size, center=True, min_periods=1
        ).std()
        window_stats = heart_rate_df[['window_mean', 'window_std']]
        
        # 4.2 计算差分
        diffs = heart_rate_df['heart_rate'].diff()
        
        # 4.3 计算可靠性分数
        heart_rate_df['reliability_score'] = heart_rate_df.apply(
            lambda row: self.calculate_reliability_score(
                row, 
                window_stats, 
                diffs,
                hr_br_correlations[row.name]
            ), 
            axis=1
        )
        
        # 4.4 修正不可靠的数据点
        heart_rate_df['corrected_heart_rate'] = heart_rate_df.apply(
            lambda row: row['window_mean'] if row['reliability_score'] < self.reliability_threshold 
            else row['heart_rate'],
            axis=1
        )
        
        # 5. 添加呼吸率数据到结果中
        heart_rate_df['breath_rate'] = pd.Series(dtype=float)
        for idx in heart_rate_df.index:
            if idx in breath_rate_df.index:
                heart_rate_df.at[idx, 'breath_rate'] = breath_rate_df.at[idx, 'breath_rate']
        
        return heart_rate_df

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore, pearsonr
from scipy.signal import correlate
from io import StringIO
import signal

class HeartRateAnalyzer:
    def __init__(self, window_size=15, reliability_threshold=70, correlation_half_window=30, correlation_threshold=0.3):
        self.window_size = window_size
        self.reliability_threshold = reliability_threshold
        self.correlation_half_window = correlation_half_window  # 相关性分析窗口的半径，实际窗口大小为 2*half_window+1
        self.normal_heart_rate = 68  # 正常心率值
        self.min_correlation_threshold = 0.2  # 最低相关性阈值
        self.max_time_gap = 60  # 最大允许的时间间隔（秒）

    def parse_datetime(self, dt_str):
        """处理时间字符串，移除多余的引号"""
        dt_str = dt_str.strip("'\"")  # 移除可能存在的引号
        return datetime.fromisoformat(dt_str)
        
    def preprocess_data(self, data):
        """将原始数据解析为心率和呼吸率的DataFrame"""
        # 将数据解析为DataFrame
        split_index = data.index('这是一段 呼吸率的记录')
        heart_rate_data = data[:split_index]
        breath_rate_data = data[split_index+1:]
        
        # 跳过标题行
        heart_rate_data = heart_rate_data[1:]
        breath_rate_data = breath_rate_data[1:]
        
        # 创建心率DataFrame
        heart_rate_df = pd.DataFrame([
            {
                'id': int(row[0]),
                'device_time': self.parse_datetime(row[2]),
                'heart_rate': int(row[3])
            } for row in [line.strip().split(',') for line in heart_rate_data if line.strip() and not line.startswith('id')]
        ])
        
        # 创建呼吸率DataFrame
        breath_rate_df = pd.DataFrame([
            {
                'id': int(row[0]),
                'device_time': self.parse_datetime(row[2]),
                'breath_rate': int(row[3])
            } for row in [line.strip().split(',') for line in breath_rate_data if line.strip() and not line.startswith('id')]
        ])
        
        # 确保device_time列是datetime类型并处理重复值
        heart_rate_df = self._process_time_column(heart_rate_df)
        breath_rate_df = self._process_time_column(breath_rate_df)
        
        return heart_rate_df, breath_rate_df
    
    def _process_time_column(self, df):
        """处理DataFrame的时间列，包括类型转换和重复值处理"""
        if not df.empty and 'device_time' in df.columns:
            # 转换为datetime类型
            df['device_time'] = pd.to_datetime(df['device_time'])
            
            # 处理None值
            none_mask = df['device_time'].isna()
            if none_mask.any() and 'timestamp' in df.columns:
                df.loc[none_mask, 'device_time'] = df.loc[none_mask, 'timestamp']
            
            # 处理重复值
            if df['device_time'].duplicated().any():
                df = df.sort_values('device_time').drop_duplicates('device_time', keep='last')
        
        return df

    def filter_invalid_data(self, heart_rate_df, breath_rate_df):
        """过滤无效数据
        
        过滤规则：
        1. 心率必须大于0
        2. 呼吸率必须大于0
        3. 呼吸率必须在12-24之间（不包含12和24）
        4. 如果某个时间点的呼吸率无效，对应的心率数据也会被过滤
        5. 所有列都不能包含NaN值
        """
        if heart_rate_df.empty or breath_rate_df.empty:
            return heart_rate_df, breath_rate_df
            
        # 处理心率数据
        valid_heart = (heart_rate_df['heart_rate'] > 0) & (~heart_rate_df.isna().any(axis=1))
        heart_rate_df = heart_rate_df[valid_heart].copy()
        
        # 处理呼吸率数据
        if 'breath_rate' in breath_rate_df.columns:
            valid_breath = (
                (breath_rate_df['breath_rate'] > 0) & 
                (breath_rate_df['breath_rate'] > 12) & 
                (breath_rate_df['breath_rate'] < 24) & 
                (~breath_rate_df.isna().any(axis=1))
            )
            breath_rate_df = breath_rate_df[valid_breath].copy()
            
            if not heart_rate_df.empty:
                # 对每个心率数据点，找到最近的呼吸率数据点
                matched_indices = self._find_matching_timestamps(
                    heart_rate_df['device_time'],
                    breath_rate_df['device_time'],
                    pd.Timedelta(seconds=30)
                )
                
                # 过滤掉未匹配的心率数据
                heart_rate_df = heart_rate_df[matched_indices].copy()
        else:
            breath_rate_df = pd.DataFrame(columns=['id', 'device_time', 'breath_rate'])
        
        return heart_rate_df, breath_rate_df
    
    def _find_matching_timestamps(self, source_times, target_times, threshold):
        """查找匹配的时间戳
        
        Args:
            source_times: 源时间序列
            target_times: 目标时间序列
            threshold: 时间阈值
            
        Returns:
            匹配结果的布尔数组
        """
        matched_indices = []
        for time in source_times:
            time_diff = abs(target_times - time)
            min_diff = time_diff.min()
            matched_indices.append(min_diff <= threshold)
        return matched_indices

    def calculate_hr_br_correlation(self, heart_rate_df, breath_rate_df):
        """计算心率和呼吸率的相关性分数"""
        # 准备数据
        heart_rate_df, breath_rate_df = self._prepare_dataframes_for_correlation(heart_rate_df, breath_rate_df)
        
        # 使用时间索引合并数据
        merged_df = pd.merge(
            heart_rate_df[['corrected_heart_rate']], 
            breath_rate_df[['corrected_breath_rate']], 
            left_index=True, 
            right_index=True,
            how='inner'
        )
        
        # 计算相关性
        window_size = 2 * self.correlation_half_window + 1
        correlations = self._calculate_rolling_correlation(
            merged_df['corrected_heart_rate'],
            merged_df['corrected_breath_rate'],
            window_size
        )
        
        # 创建结果Series
        full_correlations = pd.Series(0.0, index=heart_rate_df.index)
        full_correlations.loc[correlations.index] = correlations
        
        return full_correlations
    
    def _prepare_dataframes_for_correlation(self, heart_rate_df, breath_rate_df):
        """准备用于相关性计算的DataFrame"""
        # 确保两个DataFrame都使用device_time作为索引
        for df in [heart_rate_df, breath_rate_df]:
            if 'device_time' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'device_time'})
            df.set_index('device_time', inplace=True)
        return heart_rate_df, breath_rate_df
    
    def _calculate_rolling_correlation(self, series1, series2, window_size):
        """计算滚动相关性
        
        只保留正相关，负相关被视为无效（设为0）
        """
        correlations = series1.rolling(
            window=window_size,
            min_periods=3,
            center=True
        ).corr(series2)
        
        # 处理常量数据
        std1 = series1.rolling(window=window_size, min_periods=3, center=True).std()
        std2 = series2.rolling(window=window_size, min_periods=3, center=True).std()
        correlations = correlations.where((std1 > 0) & (std2 > 0), 0.0)
        
        # 将负相关设为0，只保留正相关
        correlations = correlations.where(correlations > 0, 0.0)
        
        return correlations.fillna(0.0)

    def analyze(self, data):
        """分析心率和呼吸率数据"""
        try:
            # 使用 preprocess_data 处理原始数据
            heart_rate_df, breath_rate_df = self.preprocess_data(data)
            
            if heart_rate_df.empty or breath_rate_df.empty:
                return pd.DataFrame()
            
            # 过滤无效数据
            heart_rate_df, breath_rate_df = self.filter_invalid_data(heart_rate_df, breath_rate_df)
            
            if heart_rate_df.empty or breath_rate_df.empty:
                return pd.DataFrame()
            
            try:
                # 创建结果DataFrame
                results = pd.DataFrame()
                results['device_time'] = heart_rate_df['device_time']
                results['heart_rate'] = heart_rate_df['heart_rate']
                results['breath_rate'] = breath_rate_df['breath_rate']
                print("\n1. 初始DataFrame:")
                print(results.head())
                
                # 设置时间戳为索引
                results = results.set_index('device_time')
                
                # 确保没有NaN值
                results = results.ffill().bfill()
                
                # 确保结果按时间排序
                results = results.sort_index()
                print("\n2. 设置索引后的DataFrame:")
                print(results.head())
                
                # 修正呼吸率数据
                results['corrected_breath_rate'] = self.correct_breath_rate(results['breath_rate'])
                
                # 计算可靠性分数
                results['reliability_score'] = self.calculate_reliability_scores(results[['heart_rate']])
                print("\n3. 添加可靠性分数后的DataFrame:")
                print(results.head())
                
                # 修正心率数据
                results['corrected_heart_rate'] = self.correct_heart_rate(results['heart_rate'])
                print("\n4. 添加修正后心率的DataFrame:")
                print(results.head())
                
                # 临时重置索引以访问device_time列
                results_with_time = results.reset_index()
                
                # 计算相关性分数
                correlations = self.calculate_hr_br_correlation(
                    results_with_time[['device_time', 'corrected_heart_rate']],
                    results_with_time[['device_time', 'corrected_breath_rate']]
                )
                
                # 将相关性结果添加到DataFrame中
                results['correlation'] = correlations
                results['correlation'] = results['correlation'].fillna(0)  # 填充NaN值
                print("\n5. 添加相关性分数后的DataFrame:")
                print(results.head())
                
                # 计算每个点的动态相关性阈值
                results['dynamic_correlation_threshold'] = results['heart_rate'].apply(self._calculate_dynamic_correlation_threshold)
                
                # 根据可靠性和动态相关性阈值一起过滤数据
                valid_data = (
                    (results['reliability_score'] >= self.reliability_threshold) &
                    (results['correlation'] >= results['dynamic_correlation_threshold'])
                )
                results = results[valid_data].copy()
                
                # 标记数据质量
                results['data_quality'] = 'high'  # 因为我们已经过滤掉了所有低质量数据
                print("\n6. 最终过滤后的DataFrame:")
                print(results.head())
                
                return results
                
            except Exception as e:
                import sys
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"创建结果DataFrame时出错: {str(e)}")
                print(f"错误位置: 第 {exc_traceback.tb_lineno} 行")
                return pd.DataFrame()
            
        except Exception as e:
            import traceback
            print(f"分析数据时出错: {str(e)}")
            print("错误详情:")
            tb = traceback.extract_tb(e.__traceback__)
            print(f"文件 {tb[-1].filename}, 第 {tb[-1].lineno} 行")
            print(f"出错代码: {tb[-1].line}")
            return pd.DataFrame()

    def correct_heart_rate(self, heart_rate):
        """修正心率数据，使用指数加权移动平均"""
        # 先使用简单移动平均去除极端值
        smoothed_heart_rate = self.apply_moving_average(heart_rate)
        
        # 再使用指数加权移动平均获得更平滑的结果
        final_heart_rate = smoothed_heart_rate.ewm(span=self.window_size).mean()
        
        return final_heart_rate

    def correct_breath_rate(self, breath_rate):
        """修正呼吸率数据，使用指数加权移动平均
        
        Args:
            breath_rate: 原始呼吸率数据Series
            
        Returns:
            平滑后的呼吸率数据Series
        """
        # 先使用简单移动平均去除极端值
        smoothed_breath_rate = self.apply_moving_average(breath_rate)
        
        # 再使用指数加权移动平均获得更平滑的结果
        final_breath_rate = smoothed_breath_rate.ewm(span=self.window_size).mean()
        
        return final_breath_rate

    def apply_moving_average(self, series, window_size=5):
        """应用移动平均来平滑数据"""
        return series.rolling(window=window_size, center=True, min_periods=1).mean()

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
                score -= min(30, deviation * 100 * 0.7)
            
            # 基于窗口标准差
            if mean > 0:  # 避免除以零
                score -= min(30, (std / mean) * 100)
            
            # 基于与前一个点的差异
            if mean > 0:  # 避免除以零
                score -= min(40, (diff / mean) * 100)
            
            scores.append(max(0, min(100, score)))
        
        return scores

    def evaluate_segments_quality(self, segments):
        """评估已分段数据的质量
        
        Args:
            segments: 已经分好的数据段列表，每个段是一个DataFrame
            
        Returns:
            dict: 包含所有段落质量评估结果的字典
        """
        if not segments:
            return {
                '总段数': 0,
                '段落详情': []
            }
        
        all_metrics = []
        for i, segment_data in enumerate(segments, 1):
            # 从segment_data中提取心率和呼吸率数据
            heart_rate_df = pd.DataFrame(segment_data, columns=['id', 'heart_rate', 'device_time'])
            heart_rate_df['heart_rate'] = pd.to_numeric(heart_rate_df['heart_rate'], errors='coerce')
            heart_rate_df['device_time'] = pd.to_datetime(heart_rate_df['device_time'])
            heart_rate_df = heart_rate_df.set_index('device_time')
            
            # 创建对应的呼吸率数据
            breath_rate_df = pd.DataFrame(segment_data, columns=['id', 'breath_rate', 'device_time'])
            breath_rate_df['breath_rate'] = pd.to_numeric(breath_rate_df['breath_rate'], errors='coerce')
            breath_rate_df['device_time'] = pd.to_datetime(breath_rate_df['device_time'])
            breath_rate_df = breath_rate_df.set_index('device_time')
            
            # 评估当前段的质量
            metrics = self._evaluate_single_segment(heart_rate_df, breath_rate_df, i)
            all_metrics.append(metrics)
        
        # 合并所有段的结果
        combined_metrics = {
            '总段数': len(segments),
            '段落详情': all_metrics
        }
        
        return combined_metrics

    def _evaluate_single_segment(self, heart_rate_df, breath_rate_df, segment_number):
        """评估单个数据段的质量"""
        quality_metrics = {}
        
        # 添加段落编号
        quality_metrics['段落编号'] = segment_number
        
        # 先对数据进行平滑处理
        heart_rate_df['corrected_heart_rate'] = self.correct_heart_rate(heart_rate_df['heart_rate'])
        breath_rate_df['corrected_breath_rate'] = self.correct_breath_rate(breath_rate_df['breath_rate'])
        
        # 计算相关性并过滤数据
        correlations = self.calculate_hr_br_correlation(heart_rate_df.reset_index(), breath_rate_df.reset_index())
        correlations = correlations.fillna(0)  # 填充NaN值
        
        # 根据相关性过滤数据
        valid_correlation = correlations >= self.min_correlation_threshold
        filtered_correlations = correlations[valid_correlation]
        
        if filtered_correlations.empty:
            # 如果过滤后没有数据，返回空的评估结果
            return {
                '段落编号': segment_number,
                '数据完整性': {
                    '总时长(分钟)': 0,
                    '心率数据点数': 0,
                    '呼吸率数据点数': 0,
                    '心率平均采样间隔(秒)': 0,
                    '呼吸率平均采样间隔(秒)': 0,
                    '心率数据完整度': 0,
                    '呼吸率数据完整度': 0
                },
                '数据有效性': {
                    '有效心率数据比例': 0,
                    '有效呼吸率数据比例': 0
                },
                '数据稳定性': {
                    '心率标准差': 0,
                    '呼吸率标准差': 0
                },
                '数据相关性': {
                    '心率-呼吸率相关性得分': 0,
                    '高相关性比例': 0,
                    '中相关性比例': 0,
                    '低相关性比例': 0
                }
            }
        
        # 过滤心率和呼吸率数据
        valid_times = correlations[valid_correlation].index
        heart_rate_df = heart_rate_df.loc[valid_times]
        breath_rate_df = breath_rate_df.loc[valid_times]
        
        # 1. 数据完整性检查
        total_duration = abs((heart_rate_df.index[-1] - heart_rate_df.index[0]).total_seconds() / 60)
        heart_rate_count = len(heart_rate_df)
        breath_rate_count = len(breath_rate_df)
        
        # 计算预期数据点数：假设每5秒一个数据点
        expected_points_per_minute = 12
        expected_count = max(1, total_duration * expected_points_per_minute)  # 避免除以0
        
        # 计算实际的平均采样间隔（秒）
        hr_time_diffs = pd.Series(heart_rate_df.index).diff().dt.total_seconds()
        br_time_diffs = pd.Series(breath_rate_df.index).diff().dt.total_seconds()
        
        hr_avg_interval = abs(hr_time_diffs.mean()) if not pd.isna(hr_time_diffs.mean()) else 0
        br_avg_interval = abs(br_time_diffs.mean()) if not pd.isna(br_time_diffs.mean()) else 0
        
        quality_metrics['数据完整性'] = {
            '总时长(分钟)': round(total_duration, 2),
            '心率数据点数': int(heart_rate_count),
            '呼吸率数据点数': int(breath_rate_count),
            '心率平均采样间隔(秒)': round(hr_avg_interval, 2),
            '呼吸率平均采样间隔(秒)': round(br_avg_interval, 2),
            '心率数据完整度': min(round((heart_rate_count / expected_count) * 100, 2), 100),
            '呼吸率数据完整度': min(round((breath_rate_count / expected_count) * 100, 2), 100)
        }
        
        # 2. 数据有效性检查
        heart_rate_valid = heart_rate_df[
            (heart_rate_df['heart_rate'] >= 40) & 
            (heart_rate_df['heart_rate'] <= 200)
        ]
        breath_rate_valid = breath_rate_df[
            (breath_rate_df['breath_rate'] >= 8) & 
            (breath_rate_df['breath_rate'] <= 30)
        ]
        
        quality_metrics['数据有效性'] = {
            '有效心率数据比例': round((len(heart_rate_valid) / max(1, len(heart_rate_df))) * 100, 2),
            '有效呼吸率数据比例': round((len(breath_rate_valid) / max(1, len(breath_rate_df))) * 100, 2)
        }
        
        # 3. 数据稳定性检查
        heart_rate_std = heart_rate_df['heart_rate'].std()
        breath_rate_std = breath_rate_df['breath_rate'].std()
        
        quality_metrics['数据稳定性'] = {
            '心率标准差': round(float(heart_rate_std if not pd.isna(heart_rate_std) else 0), 2),
            '呼吸率标准差': round(float(breath_rate_std if not pd.isna(breath_rate_std) else 0), 2)
        }
        
        # 计算每个点的动态相关性阈值
        dynamic_thresholds = heart_rate_df['heart_rate'].apply(self._calculate_dynamic_correlation_threshold)
        
        # 计算不同相关性水平的比例
        high_correlation = filtered_correlations >= dynamic_thresholds
        medium_correlation = (filtered_correlations >= self.min_correlation_threshold) & (filtered_correlations < dynamic_thresholds)
        low_correlation = filtered_correlations < self.min_correlation_threshold
        
        total_points = len(filtered_correlations)
        high_ratio = (high_correlation.sum() / total_points * 100) if total_points > 0 else 0
        medium_ratio = (medium_correlation.sum() / total_points * 100) if total_points > 0 else 0
        low_ratio = (low_correlation.sum() / total_points * 100) if total_points > 0 else 0
        
        quality_metrics['数据相关性'] = {
            '心率-呼吸率相关性得分': round(filtered_correlations.mean(), 2),
            '高相关性比例': round(high_ratio, 2),
            '中相关性比例': round(medium_ratio, 2),
            '低相关性比例': round(low_ratio, 2)
        }
        
        return quality_metrics

    def _calculate_dynamic_correlation_threshold(self, heart_rate):
        """计算动态相关性阈值
        
        基于心率与正常心率（68）的偏离程度来计算相关性阈值。
        心率越偏离正常值，要求的相关性阈值越高。
        
        Args:
            heart_rate: 当前心率值
            
        Returns:
            float: 动态计算的相关性阈值
        """
        # 计算与正常心率的偏差百分比
        deviation = abs(heart_rate - self.normal_heart_rate) / self.normal_heart_rate
        
        # 基础阈值为0.3，最大阈值为0.9
        base_threshold = 0.3
        max_threshold = 0.9
        
        # 使用幂函数计算阈值
        # 幂指数增加到8使曲线更陡峭
        # 缩放因子改为0.3，这样30%偏差时就接近最大值
        threshold = base_threshold + (max_threshold - base_threshold) * min(1, (deviation / 0.3) ** 11)
        
        # 确保阈值在合理范围内
        return min(max(threshold, base_threshold), max_threshold)

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore, pearsonr
from scipy.signal import correlate
from io import StringIO
import signal

class HeartRateAnalyzer:
    def __init__(self, window_size=15, reliability_threshold=70, correlation_half_window=30, correlation_threshold=0.4):
        self.window_size = window_size
        self.reliability_threshold = reliability_threshold
        self.correlation_half_window = correlation_half_window  # 相关性分析窗口的半径，实际窗口大小为 2*half_window+1
        self.correlation_threshold = correlation_threshold  # 相关性阈值
        self.max_time_gap = 60  # 最大允许的时间间隔（秒）
        self.min_correlation_threshold = 0.2  # 最低相关性阈值

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
        
        使用滑动窗口计算局部相关性。窗口大小为 2*correlation_half_window+1。
        使用pandas的rolling方法进行向量化计算，提高效率。
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
        
        # 计算窗口大小
        window_size = 2 * self.correlation_half_window + 1
        
        # 使用rolling计算相关性
        correlations = merged_df['heart_rate'].rolling(
            window=window_size,
            min_periods=3,
            center=True
        ).corr(merged_df['breath_rate'])
        
        # 处理常量数据（标准差为0的情况）
        std_hr = merged_df['heart_rate'].rolling(window=window_size, min_periods=3, center=True).std()
        std_br = merged_df['breath_rate'].rolling(window=window_size, min_periods=3, center=True).std()
        correlations = correlations.where((std_hr > 0) & (std_br > 0), 0.0)
        
        # 填充NaN值为0并取绝对值
        correlations = correlations.fillna(0.0).abs()
        
        # 创建一个与原始heart_rate_df索引相同的Series
        full_correlations = pd.Series(0.0, index=heart_rate_df.index)
        full_correlations.loc[correlations.index] = correlations
        
        return full_correlations

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
            print("\n=== 开始数据分析 ===")
            # 分离心率和呼吸率数据
            split_index = data.index('这是一段 呼吸率的记录')
            heart_rate_data = pd.read_csv(StringIO('\n'.join(data[:split_index])))
            breath_rate_data = pd.read_csv(StringIO('\n'.join(data[split_index + 1:])))
            
            print(f"原始心率数据行数: {len(heart_rate_data)}")
            print(f"原始呼吸率数据行数: {len(breath_rate_data)}")
            print("\n心率数据前5行:")
            print(heart_rate_data.head())
            print("\n呼吸率数据前5行:")
            print(breath_rate_data.head())
            
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
            print(f"\n过滤前心率数据中0值的数量: {len(heart_rate_data[heart_rate_data['value'] == 0])}")
            print(f"过滤前呼吸率数据中0值的数量: {len(breath_rate_data[breath_rate_data['value'] == 0])}")
            
            heart_rate_data = heart_rate_data[heart_rate_data['value'] > 0].copy()
            breath_rate_data = breath_rate_data[breath_rate_data['value'] > 0].copy()
            
            print(f"过滤0值后心率数据行数: {len(heart_rate_data)}")
            print(f"过滤0值后呼吸率数据行数: {len(breath_rate_data)}")
            
            if heart_rate_data.empty or breath_rate_data.empty:
                print("过滤0值后数据为空")
                return pd.DataFrame()
            
            # 设置时间索引
            print("\n=== 处理时间索引 ===")
            heart_rate_data = (heart_rate_data
                             .assign(device_time=lambda x: pd.to_datetime(x['device_time']))
                             .set_index('device_time')
                             .sort_index())
            
            print(f"设置时间索引后心率数据行数: {len(heart_rate_data)}")
            print("心率数据时间范围:", heart_rate_data.index.min(), "至", heart_rate_data.index.max())
            
            # 对齐呼吸率数据
            print("\n=== 开始对齐呼吸率数据 ===")
            aligned_breath_rate = self.align_breath_rate(heart_rate_data, breath_rate_data)
            print(f"对齐后的呼吸率数据点数: {len(aligned_breath_rate)}")
            
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
            
            # 基础过滤条件
            valid_basic = (
                (results['heart_rate'] > 0) & 
                (results['breath_rate'] > 0)
            )
            
            # 相关性过滤条件（使用更宽松的标准）
            results['correlation'] = results['correlation'].fillna(0)  # 填充NaN值
            high_correlation = results['correlation'] >= self.correlation_threshold
            medium_correlation = (results['correlation'] >= self.min_correlation_threshold)
            
            # 计算相关性的整体统计
            correlation_mean = results['correlation'].mean()
            correlation_std = results['correlation'].std()
            
            print(f"\n相关性统计:")
            print(f"平均相关性: {correlation_mean:.3f}")
            print(f"相关性标准差: {correlation_std:.3f}")
            print(f"高相关数据比例: {(high_correlation.sum() / len(results)) * 100:.2f}%")
            print(f"可用数据比例: {(medium_correlation.sum() / len(results)) * 100:.2f}%")
            
            # 使用更宽松的过滤条件
            valid = valid_basic & medium_correlation
            
            results = results[valid].copy()
            
            # 添加质量标记
            results['data_quality'] = 'high'
            results.loc[results['correlation'] < self.correlation_threshold, 'data_quality'] = 'medium'
            
            print(f"\n过滤后最终数据点数: {len(results)}")
            if not results.empty:
                print("最终数据时间范围:", results.index.min(), "至", results.index.max())
                print(f"高质量数据比例: {(results['data_quality'] == 'high').mean() * 100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"分析数据时出错: {str(e)}")
            import traceback
            print("错误详情:")
            print(traceback.format_exc())
            return pd.DataFrame()

    def align_breath_rate(self, heart_rate_df, breath_rate_df):
        """将呼吸率数据对齐到心率数据的时间点"""
        try:
            print("\n=== 呼吸率数据对齐详情 ===")
            # 检查输入数据是否为空
            if heart_rate_df.empty or breath_rate_df.empty:
                print("心率或呼吸率数据为空，返回空Series")
                return pd.Series(index=heart_rate_df.index, dtype=float)
            
            print(f"对齐前心率数据点数: {len(heart_rate_df)}")
            print(f"对齐前呼吸率数据点数: {len(breath_rate_df)}")
            
            # 准备呼吸率数据
            breath_rate_df['device_time'] = pd.to_datetime(breath_rate_df['device_time'])
            breath_rate_df.sort_values('device_time', inplace=True)
            
            # 过滤掉呼吸率为25的数据
            invalid_count = len(breath_rate_df[breath_rate_df['value'] == 25])
            print(f"过滤掉呼吸率为25的数据点数: {invalid_count}")
            breath_rate_df = breath_rate_df[breath_rate_df['value'] != 25].copy()
            
            # 对呼吸率数据应用移动平均
            print("应用指数加权移动平均...")
            breath_rate_df['value'] = breath_rate_df['value'].ewm(span=self.window_size * 2).mean()
            
            # 使用pandas的merge_asof进行最近邻匹配
            print(f"使用最大时间间隔 {self.max_time_gap} 秒进行数据对齐...")
            merged_df = pd.merge_asof(
                pd.DataFrame({'time': heart_rate_df.index}),
                pd.DataFrame({
                    'time': breath_rate_df['device_time'],
                    'breath_rate': breath_rate_df['value']
                }),
                on='time',
                tolerance=pd.Timedelta(f'{self.max_time_gap}s'),
                direction='nearest'
            )
            
            # 创建结果Series
            aligned_breath = pd.Series(
                merged_df['breath_rate'].values,
                index=heart_rate_df.index
            )
            
            # 过滤无效值
            valid_count_before = len(aligned_breath)
            aligned_breath = aligned_breath[aligned_breath > 0]
            valid_count_after = len(aligned_breath)
            print(f"对齐后过滤无效值: {valid_count_before} -> {valid_count_after}")
            
            # 最终平滑
            aligned_breath = self.apply_moving_average(aligned_breath, self.window_size)
            print(f"最终对齐的有效数据点数: {len(aligned_breath)}")
            
            return aligned_breath
            
        except Exception as e:
            print(f"对齐呼吸率数据时出错: {str(e)}")
            import traceback
            print("错误详情:")
            print(traceback.format_exc())
            return pd.Series(index=heart_rate_df.index, dtype=float)

    def correct_heart_rate(self, heart_rate):
        """修正心率数据，使用指数加权移动平均"""
        # 先使用简单移动平均去除极端值
        smoothed_heart_rate = self.apply_moving_average(heart_rate)
        
        # 再使用指数加权移动平均获得更平滑的结果
        final_heart_rate = smoothed_heart_rate.ewm(span=self.window_size).mean()
        
        return final_heart_rate

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

    def evaluate_data_quality(self, heart_rate_df, breath_rate_df):
        """评估心率和呼吸率数据的质量
        
        返回：
            dict: 包含各项质量指标的字典
        """
        quality_metrics = {}
        
        # 1. 数据完整性检查
        total_duration = (heart_rate_df.index[-1] - heart_rate_df.index[0]).total_seconds() / 60  # 转换为分钟
        heart_rate_count = len(heart_rate_df)
        breath_rate_count = len(breath_rate_df)
        
        # 计算预期数据点数：假设每5秒一个数据点
        expected_points_per_minute = 12  # 每分钟应该有12个数据点（每5秒一个）
        expected_count = total_duration * expected_points_per_minute
        
        # 计算实际的平均采样间隔（秒）
        hr_time_diffs = pd.Series(heart_rate_df.index).diff().dt.total_seconds()
        br_time_diffs = pd.Series(breath_rate_df.index).diff().dt.total_seconds()
        
        hr_avg_interval = hr_time_diffs.mean()
        br_avg_interval = br_time_diffs.mean()
        
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
            '有效心率数据比例': round((len(heart_rate_valid) / len(heart_rate_df)) * 100, 2),
            '有效呼吸率数据比例': round((len(breath_rate_valid) / len(breath_rate_df)) * 100, 2)
        }
        
        # 3. 数据稳定性检查
        heart_rate_std = heart_rate_df['heart_rate'].std()
        breath_rate_std = breath_rate_df['breath_rate'].std()
        
        quality_metrics['数据稳定性'] = {
            '心率标准差': round(float(heart_rate_std), 2),  # 确保是float
            '呼吸率标准差': round(float(breath_rate_std), 2)  # 确保是float
        }
        
        # 4. 数据相关性
        correlations = self.calculate_hr_br_correlation(heart_rate_df, breath_rate_df)
        mean_correlation = float(correlations.mean())  # 计算平均相关性并转换为float
        quality_metrics['数据相关性'] = {
            '心率-呼吸率相关性得分': round(mean_correlation, 2)
        }
        
        return quality_metrics

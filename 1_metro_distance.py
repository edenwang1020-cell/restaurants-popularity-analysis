#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海餐厅地铁距离分析程序
功能：计算餐厅到最近地铁站的距离，进行数据筛选和分析

使用方法:
    python 课题1_optimized.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import logging
import warnings
from typing import Tuple, Optional
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metro_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    # 文件路径配置
    INPUT_RESTAURANTS: str = 'data/shanghai_restaurants_with_coordinates.csv'
    INPUT_METRO: str = 'data/shanghai_subway_stations.csv'
    OUTPUT_PROCESSED: str = 'data/课题数据1.csv'
    OUTPUT_FINAL: str = 'data/restaurants_with_metro_distance.csv'
    OUTPUT_FILTERED: str = 'data/课题数据_筛选后.csv'
    
    # 处理参数
    BATCH_SIZE: int = 1000  # 批处理大小
    PROGRESS_INTERVAL: int = 100  # 进度显示间隔
    MAX_DISTANCE_FILTER: float = 3000.0  # 距离筛选阈值（米）
    
    # 距离区间配置
    DISTANCE_BINS: list = None
    
    def __post_init__(self):
        if self.DISTANCE_BINS is None:
            self.DISTANCE_BINS = [0, 500, 1000, 1500, 2000, float('inf')]
        self.DISTANCE_LABELS = ['<500m', '500-1000m', '1000-1500m', '1500-2000m', '>2000m']

class MetroDistanceCalculator:
    """地铁距离计算器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("地铁距离计算器初始化完成")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        使用Haversine公式计算两个经纬度坐标之间的球面距离
        
        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度
            
        Returns:
            float: 两点之间的距离（米）
        """
        try:
            # 将十进制度数转换为弧度
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            
            # Haversine公式
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            r = 6371000  # 地球平均半径，单位为米
            return c * r
        except Exception as e:
            logger.error(f"距离计算失败: {e}")
            return float('inf')
    
    def find_nearest_metro_vectorized(self, restaurants_df: pd.DataFrame, metro_df: pd.DataFrame) -> pd.DataFrame:
        """
        使用向量化方法找到距离餐厅最近的地铁站及其距离
        
        Args:
            restaurants_df: 餐厅数据DataFrame
            metro_df: 地铁站数据DataFrame
            
        Returns:
            DataFrame: 添加了最近地铁站信息的餐厅数据
        """
        logger.info("开始计算餐厅到最近地铁站的距离...")
        start_time = time.time()
        
        # 创建结果列
        restaurants_df = restaurants_df.copy()
        restaurants_df['nearest_metro'] = None
        restaurants_df['distance_to_metro'] = None
        
        # 过滤有效的坐标数据
        valid_restaurants = restaurants_df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"有效坐标的餐厅数量: {len(valid_restaurants)}")
        
        if len(valid_restaurants) == 0:
            logger.warning("没有有效的餐厅坐标数据")
            return restaurants_df
        
        # 使用向量化计算提高性能
        distances_matrix = self._calculate_distance_matrix(valid_restaurants, metro_df)
        
        # 找到最近的地铁站
        min_distances_idx = distances_matrix.idxmin(axis=1)
        min_distances = distances_matrix.min(axis=1)
        
        # 更新结果
        restaurants_df.loc[valid_restaurants.index, 'nearest_metro'] = \
            metro_df.loc[min_distances_idx, 'station_name'].values
        restaurants_df.loc[valid_restaurants.index, 'distance_to_metro'] = min_distances.values
        
        elapsed_time = time.time() - start_time
        logger.info(f"距离计算完成，耗时: {elapsed_time:.2f}秒")
        
        return restaurants_df
    
    def _calculate_distance_matrix(self, restaurants: pd.DataFrame, metro_df: pd.DataFrame) -> pd.DataFrame:
        """计算餐厅到所有地铁站的距离矩阵"""
        logger.info("计算距离矩阵...")
        
        # 创建距离矩阵
        distances = pd.DataFrame(index=restaurants.index, columns=metro_df.index)
        
        # 分批处理以提高内存效率
        batch_size = self.config.BATCH_SIZE
        total_batches = (len(restaurants) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(restaurants))
            batch_restaurants = restaurants.iloc[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
            
            # 计算当前批次的距离
            for rest_idx, rest_row in batch_restaurants.iterrows():
                rest_lat, rest_lon = rest_row['latitude'], rest_row['longitude']
                
                # 向量化计算到所有地铁站的距离
                distances.loc[rest_idx] = metro_df.apply(
                    lambda metro_row: self.haversine_distance(
                        rest_lat, rest_lon, metro_row['latitude'], metro_row['longitude']
                    ), 
                    axis=1
                )
        
        return distances
    
    def find_nearest_metro_simple(self, restaurant_row: pd.Series, metro_df: pd.DataFrame) -> Tuple[str, float]:
        """
        找到距离餐厅最近的地铁站及其距离（简单版本，用于小数据集）
        
        Args:
            restaurant_row: 餐厅数据的行
            metro_df: 地铁站数据DataFrame
            
        Returns:
            tuple: (最近地铁站名称, 距离(米))
        """
        try:
            rest_lat = restaurant_row['latitude']
            rest_lon = restaurant_row['longitude']
            
            if pd.isna(rest_lat) or pd.isna(rest_lon):
                return None, None
            
            # 计算餐厅到每个地铁站的距离
            distances = metro_df.apply(
                lambda metro_row: self.haversine_distance(
                    rest_lat, rest_lon, metro_row['latitude'], metro_row['longitude']
                ), 
                axis=1
            )
            
            # 找到最小距离的索引
            min_idx = distances.idxmin()
            
            # 返回最近的地铁站名称和距离
            return metro_df.loc[min_idx, 'station_name'], distances[min_idx]
            
        except Exception as e:
            logger.error(f"计算最近地铁站失败: {e}")
            return None, None

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("数据处理器初始化完成")
    
    def load_and_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载和清理数据"""
        logger.info("开始加载数据...")
        
        try:
            # 加载餐厅数据
            if Path(self.config.INPUT_RESTAURANTS).exists():
                restaurants_df = pd.read_csv(self.config.INPUT_RESTAURANTS)
                logger.info(f"成功加载餐厅数据: {restaurants_df.shape}")
            else:
                logger.error(f"餐厅数据文件不存在: {self.config.INPUT_RESTAURANTS}")
                raise FileNotFoundError(f"餐厅数据文件不存在: {self.config.INPUT_RESTAURANTS}")
            
            # 加载地铁站数据
            if Path(self.config.INPUT_METRO).exists():
                metro_df = pd.read_csv(self.config.INPUT_METRO)
                logger.info(f"成功加载地铁站数据: {metro_df.shape}")
            else:
                logger.warning(f"地铁站数据文件不存在: {self.config.INPUT_METRO}")
                logger.info("创建示例地铁站数据...")
                metro_df = self._create_sample_metro_data()
                logger.info(f"创建了 {len(metro_df)} 个示例地铁站")
            
            # 验证数据列
            self._validate_data_columns(restaurants_df, metro_df)
            
            # 清理餐厅数据
            restaurants_df = self._clean_restaurant_data(restaurants_df)
            
            # 保存清理后的数据
            restaurants_df.to_csv(self.config.OUTPUT_PROCESSED, index=False, encoding='utf-8')
            logger.info(f"清理后的数据已保存到: {self.config.OUTPUT_PROCESSED}")
            
            return restaurants_df, metro_df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _create_sample_metro_data(self) -> pd.DataFrame:
        """创建示例地铁站数据"""
        # 上海主要地铁站坐标（示例数据）
        sample_stations = [
            {'station_name': '人民广场', 'latitude': 31.2337, 'longitude': 121.4765},
            {'station_name': '南京东路', 'latitude': 31.2376, 'longitude': 121.4806},
            {'station_name': '陆家嘴', 'latitude': 31.2397, 'longitude': 121.4998},
            {'station_name': '静安寺', 'latitude': 31.2234, 'longitude': 121.4456},
            {'station_name': '徐家汇', 'latitude': 31.1945, 'longitude': 121.4367},
            {'station_name': '中山公园', 'latitude': 31.2189, 'longitude': 121.4167},
            {'station_name': '虹桥火车站', 'latitude': 31.1945, 'longitude': 121.4201},
            {'station_name': '世纪大道', 'latitude': 31.2301, 'longitude': 121.5278},
            {'station_name': '五角场', 'latitude': 31.3034, 'longitude': 121.5145},
            {'station_name': '外滩', 'latitude': 31.2345, 'longitude': 121.4901},
            {'station_name': '新天地', 'latitude': 31.2201, 'longitude': 121.4756},
            {'station_name': '淮海中路', 'latitude': 31.2189, 'longitude': 121.4698},
            {'station_name': '陕西南路', 'latitude': 31.2167, 'longitude': 121.4567},
            {'station_name': '打浦桥', 'latitude': 31.2078, 'longitude': 121.4689},
            {'station_name': '日月光', 'latitude': 31.2089, 'longitude': 121.4678},
            {'station_name': '田子坊', 'latitude': 31.2098, 'longitude': 121.4667},
            {'station_name': '豫园', 'latitude': 31.2278, 'longitude': 121.4923},
            {'station_name': '南京西路', 'latitude': 31.2301, 'longitude': 121.4567},
            {'station_name': '静安嘉里中心', 'latitude': 31.2245, 'longitude': 121.4456},
            {'station_name': '虹桥天地', 'latitude': 31.1945, 'longitude': 121.4201},
        ]
        
        metro_df = pd.DataFrame(sample_stations)
        
        # 保存示例地铁站数据
        sample_metro_file = 'data/sample_metro_stations.csv'
        Path('data').mkdir(exist_ok=True)
        metro_df.to_csv(sample_metro_file, index=False, encoding='utf-8')
        logger.info(f"示例地铁站数据已保存到: {sample_metro_file}")
        
        return metro_df
    
    def _validate_data_columns(self, restaurants_df: pd.DataFrame, metro_df: pd.DataFrame):
        """验证数据列"""
        required_restaurant_cols = ['latitude', 'longitude']
        required_metro_cols = ['latitude', 'longitude', 'station_name']
        
        missing_restaurant = [col for col in required_restaurant_cols if col not in restaurants_df.columns]
        missing_metro = [col for col in required_metro_cols if col not in metro_df.columns]
        
        if missing_restaurant:
            raise ValueError(f"餐厅数据缺少必要的列: {missing_restaurant}")
        if missing_metro:
            raise ValueError(f"地铁站数据缺少必要的列: {missing_metro}")
        
        logger.info("数据列验证通过")
    
    def _clean_restaurant_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理餐厅数据"""
        logger.info("开始清理餐厅数据...")
        original_shape = df.shape
        
        # 去掉坐标列的空值
        df = df.dropna(subset=['longitude', 'latitude'])
        logger.info(f"删除坐标空值后: {df.shape}")
        
        # 确保坐标数据类型正确
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        
        # 再次删除无效坐标
        df = df.dropna(subset=['longitude', 'latitude'])
        logger.info(f"数据类型转换后: {df.shape}")
        
        final_shape = df.shape
        logger.info(f"数据清理完成: {original_shape} -> {final_shape}")
        
        return df
    
    def filter_by_distance(self, df: pd.DataFrame, max_distance: float = None) -> pd.DataFrame:
        """根据距离筛选数据"""
        if max_distance is None:
            max_distance = self.config.MAX_DISTANCE_FILTER
        
        logger.info(f"根据距离筛选数据，最大距离: {max_distance}米")
        
        # 检查是否有距离列
        if 'distance_to_metro' not in df.columns:
            logger.warning("数据中没有距离列，无法进行距离筛选")
            return df
        
        # 筛选数据
        filtered_df = df[df['distance_to_metro'] <= max_distance].copy()
        logger.info(f"筛选前: {len(df)} 条记录")
        logger.info(f"筛选后: {len(filtered_df)} 条记录")
        
        # 保存筛选后的数据
        filtered_df.to_csv(self.config.OUTPUT_FILTERED, index=False, encoding='utf-8-sig')
        logger.info(f"筛选后的数据已保存到: {self.config.OUTPUT_FILTERED}")
        
        return filtered_df

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_plot_style()
        logger.info("数据分析器初始化完成")
    
    def _setup_plot_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
    
    def analyze_distances(self, df: pd.DataFrame):
        """分析距离数据"""
        logger.info("开始分析距离数据...")
        
        # 基本统计信息
        self._show_distance_stats(df)
        
        # 距离分布可视化
        self._create_distance_plots(df)
        
        # 距离区间分析
        self._analyze_distance_groups(df)
        
        logger.info("距离数据分析完成")
    
    def _show_distance_stats(self, df: pd.DataFrame):
        """显示距离统计信息"""
        if 'distance_to_metro' not in df.columns:
            logger.warning("数据中没有距离列")
            return
        
        valid_distances = df['distance_to_metro'].dropna()
        if len(valid_distances) == 0:
            logger.warning("没有有效的距离数据")
            return
        
        logger.info(f"\n距离统计（单位：米）:")
        logger.info(f"平均距离: {valid_distances.mean():.2f}")
        logger.info(f"中位数距离: {valid_distances.median():.2f}")
        logger.info(f"最小距离: {valid_distances.min():.2f}")
        logger.info(f"最大距离: {valid_distances.max():.2f}")
        logger.info(f"标准差: {valid_distances.std():.2f}")
    
    def _create_distance_plots(self, df: pd.DataFrame):
        """创建距离分布图"""
        if 'distance_to_metro' not in df.columns:
            return
        
        valid_distances = df['distance_to_metro'].dropna()
        if len(valid_distances) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('餐厅到地铁站距离分析', fontsize=16)
        
        # 距离分布直方图
        axes[0, 0].hist(valid_distances, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('距离分布直方图')
        axes[0, 0].set_xlabel('距离 (米)')
        axes[0, 0].set_ylabel('餐厅数量')
        
        # 距离分布箱线图
        axes[0, 1].boxplot(valid_distances)
        axes[0, 1].set_title('距离分布箱线图')
        axes[0, 1].set_ylabel('距离 (米)')
        
        # 距离区间分布
        distance_groups = pd.cut(valid_distances, bins=self.config.DISTANCE_BINS, labels=self.config.DISTANCE_LABELS)
        distance_counts = distance_groups.value_counts().sort_index()
        axes[1, 0].bar(range(len(distance_counts)), distance_counts.values)
        axes[1, 0].set_title('距离区间分布')
        axes[1, 0].set_xlabel('距离区间')
        axes[1, 0].set_ylabel('餐厅数量')
        axes[1, 0].set_xticks(range(len(distance_counts)))
        axes[1, 0].set_xticklabels(distance_counts.index, rotation=45)
        
        # 累积分布图
        sorted_distances = np.sort(valid_distances)
        cumulative_prob = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        axes[1, 1].plot(sorted_distances, cumulative_prob)
        axes[1, 1].set_title('距离累积分布')
        axes[1, 1].set_xlabel('距离 (米)')
        axes[1, 1].set_ylabel('累积概率')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metro_distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("距离分析图已保存为 metro_distance_analysis.png")
    
    def _analyze_distance_groups(self, df: pd.DataFrame):
        """分析距离区间分布"""
        if 'distance_to_metro' not in df.columns:
            return
        
        valid_distances = df['distance_to_metro'].dropna()
        if len(valid_distances) == 0:
            return
        
        # 统计不同距离区间的餐厅数量
        distance_groups = pd.cut(valid_distances, bins=self.config.DISTANCE_BINS, labels=self.config.DISTANCE_LABELS)
        distance_counts = distance_groups.value_counts().sort_index()
        
        logger.info("\n距离区间分布:")
        for group, count in distance_counts.items():
            percentage = (count / len(valid_distances)) * 100
            logger.info(f"{group}: {count} 家餐厅 ({percentage:.1f}%)")

def main():
    """主函数"""
    logger.info("=== 上海餐厅地铁距离分析程序启动 ===")
    
    # 初始化配置
    config = Config()
    
    try:
        # 1. 数据加载和清理
        processor = DataProcessor(config)
        restaurants_df, metro_df = processor.load_and_clean_data()
        
        # 2. 计算地铁距离
        calculator = MetroDistanceCalculator(config)
        restaurants_with_metro = calculator.find_nearest_metro_vectorized(restaurants_df, metro_df)
        
        # 3. 保存结果
        restaurants_with_metro.to_csv(config.OUTPUT_FINAL, index=False, encoding='utf-8')
        logger.info(f"处理完成！结果已保存到 {config.OUTPUT_FINAL}")
        
        # 4. 数据分析
        analyzer = DataAnalyzer(config)
        analyzer.analyze_distances(restaurants_with_metro)
        
        # 5. 距离筛选
        filtered_df = processor.filter_by_distance(restaurants_with_metro)
        
        # 6. 显示最终统计
        success_count = restaurants_with_metro['distance_to_metro'].notna().sum()
        total_count = len(restaurants_with_metro)
        logger.info(f"\n最终统计:")
        logger.info(f"总餐厅数量: {total_count}")
        logger.info(f"成功计算距离: {success_count}")
        logger.info(f"成功率: {(success_count/total_count)*100:.1f}%")
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()

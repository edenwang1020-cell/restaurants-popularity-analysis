#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海餐厅密度分析程序
功能：计算餐厅周围指定半径内的餐厅总数和同类型餐厅数量，分析竞争环境

使用方法:
    python 课题3_optimized.py
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.neighbors import BallTree
import time
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/restaurant_density_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    # 文件路径配置
    INPUT_FILE: str = 'data/课题数据_筛选后.csv'
    OUTPUT_FILE: str = 'data/restaurants_with_density.csv'
    OUTPUT_DIR: str = 'data/density_analysis_outputs'
    
    # 分析参数
    SEARCH_RADIUS: float = 2000.0  # 搜索半径（米）
    BATCH_SIZE: int = 500  # 批处理大小
    PROGRESS_INTERVAL: int = 100  # 进度显示间隔
    
    # 性能配置
    USE_VECTORIZATION: bool = True  # 是否使用向量化计算
    EARTH_RADIUS: float = 6371000.0  # 地球半径（米）

class SpatialCalculator:
    """空间计算器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("空间计算器初始化完成")
    
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
            return c * self.config.EARTH_RADIUS
        except Exception as e:
            logger.error(f"距离计算失败: {e}")
            return float('inf')
    
    def haversine_distance_vectorized(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        向量化计算Haversine距离
        
        Args:
            coords1: 第一组坐标 (N, 2) - [lat, lon]
            coords2: 第二组坐标 (M, 2) - [lat, lon]
            
        Returns:
            np.ndarray: 距离矩阵 (N, M)
        """
        try:
            # 转换为弧度
            coords1_rad = np.deg2rad(coords1)
            coords2_rad = np.deg2rad(coords2)
            
            # 提取经纬度
            lat1 = coords1_rad[:, 0:1]
            lon1 = coords1_rad[:, 1:2]
            lat2 = coords2_rad[:, 0:1]
            lon2 = coords2_rad[:, 1:2]
            
            # 计算差值
            dlat = lat2.T - lat1
            dlon = lon2.T - lon1
            
            # Haversine公式
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2.T) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            return c * self.config.EARTH_RADIUS
            
        except Exception as e:
            logger.error(f"向量化距离计算失败: {e}")
            return np.full((coords1.shape[0], coords2.shape[0]), float('inf'))

class DensityAnalyzer:
    """密度分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.spatial_calculator = SpatialCalculator(config)
        logger.info("密度分析器初始化完成")
    
    def calculate_restaurant_density_balltree(self, restaurants_df: pd.DataFrame) -> pd.DataFrame:
        """
        使用BallTree计算餐厅密度（推荐方法）
        
        Args:
            restaurants_df: 餐厅数据DataFrame
            
        Returns:
            DataFrame: 添加了密度信息的餐厅数据
        """
        logger.info(f"开始使用BallTree计算 {len(restaurants_df)} 家餐厅的密度...")
        start_time = time.time()
        
        # 创建结果DataFrame的副本
        result_df = restaurants_df.copy()
        result_df['total_nearby'] = 0
        result_df['same_type_nearby'] = 0
        
        # 创建餐厅位置的球面坐标数组（转换为弧度）
        coords = np.deg2rad(restaurants_df[['latitude', 'longitude']].values)
        
        # 创建BallTree进行高效的空间查询
        tree = BallTree(coords, metric='haversine')
        
        # 查询半径（转换为弧度，除以地球半径）
        radius_rad = self.config.SEARCH_RADIUS / self.config.EARTH_RADIUS
        
        # 批量查询以提高性能
        all_indices = tree.query_radius(coords, r=radius_rad)
        
        # 处理查询结果
        for idx, indices in enumerate(all_indices):
            # 排除自身
            indices = [i for i in indices if i != idx]
            
            # 计算周围餐厅总数
            total_count = len(indices)
            result_df.at[idx, 'total_nearby'] = total_count
            
            # 计算同类型餐厅数量
            if total_count > 0:
                current_type = restaurants_df.iloc[idx]['类型']
                same_type_count = sum(1 for i in indices if restaurants_df.iloc[i]['类型'] == current_type)
                result_df.at[idx, 'same_type_nearby'] = same_type_count
            
            # 显示进度
            if (idx + 1) % self.config.PROGRESS_INTERVAL == 0:
                logger.info(f"已处理 {idx + 1}/{len(restaurants_df)} 家餐厅")
        
        elapsed_time = time.time() - start_time
        logger.info(f"BallTree密度计算完成，耗时: {elapsed_time:.2f}秒")
        
        return result_df
    
    def calculate_restaurant_density_vectorized(self, restaurants_df: pd.DataFrame) -> pd.DataFrame:
        """
        使用向量化方法计算餐厅密度（适用于小数据集）
        
        Args:
            restaurants_df: 餐厅数据DataFrame
            
        Returns:
            DataFrame: 添加了密度信息的餐厅数据
        """
        logger.info(f"开始使用向量化方法计算 {len(restaurants_df)} 家餐厅的密度...")
        start_time = time.time()
        
        # 创建结果DataFrame的副本
        result_df = restaurants_df.copy()
        result_df['total_nearby'] = 0
        result_df['same_type_nearby'] = 0
        
        # 获取坐标
        coords = restaurants_df[['latitude', 'longitude']].values
        
        # 分批处理以提高内存效率
        batch_size = self.config.BATCH_SIZE
        total_batches = (len(restaurants_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(restaurants_df))
            batch_coords = coords[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
            
            # 计算当前批次到所有餐厅的距离
            distances = self.spatial_calculator.haversine_distance_vectorized(batch_coords, coords)
            
            # 统计每个餐厅周围的餐厅数量
            for i, (batch_i, global_i) in enumerate(range(start_idx, end_idx)):
                # 找到指定半径内的餐厅
                nearby_mask = distances[i] <= self.config.SEARCH_RADIUS
                nearby_indices = np.where(nearby_mask)[0]
                
                # 排除自身
                nearby_indices = nearby_indices[nearby_indices != global_i]
                
                # 计算周围餐厅总数
                total_count = len(nearby_indices)
                result_df.at[global_i, 'total_nearby'] = total_count
                
                # 计算同类型餐厅数量
                if total_count > 0:
                    current_type = restaurants_df.iloc[global_i]['类型']
                    same_type_mask = restaurants_df.iloc[nearby_indices]['类型'] == current_type
                    same_type_count = np.sum(same_type_mask)
                    result_df.at[global_i, 'same_type_nearby'] = same_type_count
        
        elapsed_time = time.time() - start_time
        logger.info(f"向量化密度计算完成，耗时: {elapsed_time:.2f}秒")
        
        return result_df
    
    def calculate_restaurant_density(self, restaurants_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算餐厅密度（自动选择最优方法）
        
        Args:
            restaurants_df: 餐厅数据DataFrame
            
        Returns:
            DataFrame: 添加了密度信息的餐厅数据
        """
        # 根据数据量自动选择计算方法
        if len(restaurants_df) > 1000 and self.config.USE_VECTORIZATION:
            logger.info("数据量较大，使用BallTree方法")
            return self.calculate_restaurant_density_balltree(restaurants_df)
        else:
            logger.info("数据量较小，使用向量化方法")
            return self.calculate_restaurant_density_vectorized(restaurants_df)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_output_dir()
        logger.info("数据处理器初始化完成")
    
    def _setup_output_dir(self):
        """设置输出目录"""
        Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """加载和验证数据"""
        try:
            if not Path(self.config.INPUT_FILE).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.config.INPUT_FILE}")
            
            df = pd.read_csv(self.config.INPUT_FILE)
            logger.info(f"成功加载数据: {df.shape}")
            
            # 验证必要的列
            required_columns = ['商铺', 'latitude', 'longitude', '类型']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"数据中缺少必要的列: {missing_columns}")
            
            # 处理缺失值
            original_count = len(df)
            df = df.dropna(subset=['latitude', 'longitude', '类型'])
            cleaned_count = len(df)
            
            if cleaned_count < original_count:
                logger.warning(f"清理缺失值后，数据从 {original_count} 条减少到 {cleaned_count} 条")
            
            logger.info(f"数据验证完成，有效数据: {cleaned_count} 条")
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def save_results(self, df: pd.DataFrame) -> None:
        """保存结果"""
        try:
            # 保存主要结果
            df.to_csv(self.config.OUTPUT_FILE, index=False, encoding='utf-8')
            logger.info(f"主要结果已保存到: {self.config.OUTPUT_FILE}")
            
            # 保存密度统计报告
            self._save_density_report(df)
            
        except Exception as e:
            logger.error(f"结果保存失败: {e}")
            raise
    
    def _save_density_report(self, df: pd.DataFrame) -> None:
        """保存密度分析报告"""
        report_file = Path(self.config.OUTPUT_DIR) / 'density_analysis_report.txt'
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("上海餐厅密度分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"分析参数:\n")
                f.write(f"搜索半径: {self.config.SEARCH_RADIUS} 米\n")
                f.write(f"总餐厅数量: {len(df)}\n\n")
                
                f.write(f"密度统计:\n")
                f.write(f"平均每家餐厅周围餐厅总数: {df['total_nearby'].mean():.2f}\n")
                f.write(f"中位数: {df['total_nearby'].median():.2f}\n")
                f.write(f"标准差: {df['total_nearby'].std():.2f}\n")
                f.write(f"最小值: {df['total_nearby'].min()}\n")
                f.write(f"最大值: {df['total_nearby'].max()}\n\n")
                
                f.write(f"同类型竞争统计:\n")
                f.write(f"平均每家餐厅周围同类型餐厅数: {df['same_type_nearby'].mean():.2f}\n")
                f.write(f"中位数: {df['same_type_nearby'].median():.2f}\n")
                f.write(f"标准差: {df['same_type_nearby'].std():.2f}\n")
                f.write(f"最小值: {df['same_type_nearby'].min()}\n")
                f.write(f"最大值: {df['same_type_nearby'].max()}\n\n")
                
                # 竞争最激烈的区域
                max_total_idx = df['total_nearby'].idxmax()
                max_total_restaurant = df.loc[max_total_idx]
                f.write(f"竞争最激烈的区域:\n")
                f.write(f"餐厅名称: {max_total_restaurant['商铺']}\n")
                f.write(f"餐厅类型: {max_total_restaurant['类型']}\n")
                f.write(f"周围餐厅总数: {max_total_restaurant['total_nearby']}\n")
                f.write(f"同类型餐厅数量: {max_total_restaurant['same_type_nearby']}\n")
                f.write(f"坐标: ({max_total_restaurant['latitude']:.6f}, {max_total_restaurant['longitude']:.6f})\n\n")
                
                # 同类型竞争最激烈的餐厅
                max_same_type_idx = df['same_type_nearby'].idxmax()
                max_same_type_restaurant = df.loc[max_same_type_idx]
                f.write(f"同类型竞争最激烈的餐厅:\n")
                f.write(f"餐厅名称: {max_same_type_restaurant['商铺']}\n")
                f.write(f"餐厅类型: {max_same_type_restaurant['类型']}\n")
                f.write(f"周围同类型餐厅数量: {max_same_type_restaurant['same_type_nearby']}\n")
                f.write(f"周围餐厅总数: {max_same_type_restaurant['total_nearby']}\n")
                f.write(f"坐标: ({max_same_type_restaurant['latitude']:.6f}, {max_same_type_restaurant['longitude']:.6f})\n\n")
                
                # 按类型统计竞争情况
                f.write(f"各类型餐厅的竞争情况:\n")
                type_stats = df.groupby('类型').agg({
                    'total_nearby': ['mean', 'median', 'std'],
                    'same_type_nearby': ['mean', 'median', 'std']
                }).round(2)
                f.write(f"{type_stats.to_string()}\n")
            
            logger.info(f"密度分析报告已保存: {report_file}")
            
        except Exception as e:
            logger.error(f"报告保存失败: {e}")

class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("结果分析器初始化完成")
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """分析结果并显示统计信息"""
        logger.info("开始分析结果...")
        
        # 基本统计信息
        logger.info(f"\n密度分析结果统计:")
        logger.info(f"搜索半径: {self.config.SEARCH_RADIUS} 米")
        logger.info(f"总餐厅数量: {len(df)}")
        logger.info(f"平均每家餐厅周围餐厅总数: {df['total_nearby'].mean():.2f}")
        logger.info(f"平均每家餐厅周围同类型餐厅数量: {df['same_type_nearby'].mean():.2f}")
        
        # 竞争最激烈的区域
        max_total_idx = df['total_nearby'].idxmax()
        max_total_restaurant = df.loc[max_total_idx]
        logger.info(f"\n🏆 竞争最激烈的区域:")
        logger.info(f"  餐厅名称: {max_total_restaurant['商铺']}")
        logger.info(f"  餐厅类型: {max_total_restaurant['类型']}")
        logger.info(f"  周围餐厅总数: {max_total_restaurant['total_nearby']}")
        logger.info(f"  同类型餐厅数量: {max_total_restaurant['same_type_nearby']}")
        
        # 同类型竞争最激烈的餐厅
        max_same_type_idx = df['same_type_nearby'].idxmax()
        max_same_type_restaurant = df.loc[max_same_type_idx]
        logger.info(f"\n🔥 同类型竞争最激烈的餐厅:")
        logger.info(f"  餐厅名称: {max_same_type_restaurant['商铺']}")
        logger.info(f"  餐厅类型: {max_same_type_restaurant['类型']}")
        logger.info(f"  周围同类型餐厅数量: {max_same_type_restaurant['same_type_nearby']}")
        logger.info(f"  周围餐厅总数: {max_same_type_restaurant['total_nearby']}")
        
        # 竞争环境分布
        logger.info(f"\n📊 竞争环境分布:")
        total_nearby_ranges = [
            (0, 10, "低竞争"),
            (11, 30, "中等竞争"),
            (31, 60, "高竞争"),
            (61, float('inf'), "极高竞争")
        ]
        
        for min_val, max_val, label in total_nearby_ranges:
            if max_val == float('inf'):
                count = len(df[df['total_nearby'] >= min_val])
            else:
                count = len(df[(df['total_nearby'] >= min_val) & (df['total_nearby'] <= max_val)])
            percentage = (count / len(df)) * 100
            logger.info(f"  {label}: {count}家餐厅 ({percentage:.1f}%)")

def main():
    """主函数"""
    logger.info("=== 上海餐厅密度分析程序启动 ===")
    
    try:
        # 1. 初始化配置
        config = Config()
        
        # 2. 加载和验证数据
        processor = DataProcessor(config)
        restaurants_df = processor.load_and_validate_data()
        
        # 3. 计算餐厅密度
        analyzer = DensityAnalyzer(config)
        start_time = time.time()
        
        restaurants_with_density = analyzer.calculate_restaurant_density(restaurants_df)
        
        total_time = time.time() - start_time
        logger.info(f"密度计算完成，总耗时: {total_time:.2f}秒")
        
        # 4. 保存结果
        processor.save_results(restaurants_with_density)
        
        # 5. 分析结果
        result_analyzer = ResultAnalyzer(config)
        result_analyzer.analyze_results(restaurants_with_density)
        
        logger.info("程序执行完成！")
        logger.info(f"主要结果保存在: {config.OUTPUT_FILE}")
        logger.info(f"详细报告保存在: {config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()

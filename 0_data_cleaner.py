#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海餐厅数据地理编码程序
功能：将餐厅地址转换为经纬度坐标，支持批量处理和进度恢复

使用方法:
    python 课题_optimized.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import json
import time
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    # API配置
    AMAP_API_KEY: str = '<replace_with_your_apikey>'
    API_BASE_URL: str = "https://restapi.amap.com/v3/geocode/geo"
    
    # 文件路径
    INPUT_FILE: str = 'data/geocode_progress.csv'
    OUTPUT_FILE: str = 'data/shanghai_restaurants_with_coordinates.csv'
    PROGRESS_FILE: str = 'geocode_progress.csv'
    CLEANED_FILE: str = 'cleaned_shanghai_restaurants.csv'
    
    # 处理参数
    BATCH_SIZE: int = 100
    SAVE_INTERVAL: int = 10
    REQUEST_DELAY: float = 0.1  # API请求间隔
    MAX_RETRIES: int = 3
    
    # 数据清洗参数
    MIN_PRICE: float = 20.0
    MAX_PRICE: float = 1000.0
    IQR_MULTIPLIER: float = 1.5

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("数据清洗器初始化完成")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
        try:
            if file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)
            logger.info(f"成功加载数据: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        logger.info("开始数据清洗...")
        original_shape = df.shape
        
        # 1. 处理缺失值
        logger.info("处理缺失值...")
        missing_info = df.isnull().sum()
        logger.info(f"缺失值情况:\n{missing_info}")
        
        # 删除关键字段缺失的记录
        df = df.dropna(subset=['评论数', '均价', '口味', '环境', '服务'])
        logger.info(f"删除缺失值后: {df.shape}")
        
        # 2. 处理重复值
        df = df.drop_duplicates(subset='商铺')
        logger.info(f"删除重复值后: {df.shape}")
        
        # 3. 数据类型转换
        logger.info("转换数据类型...")
        df = self._convert_data_types(df)
        
        # 4. 创建总评分
        df['总评分'] = df[['口味', '环境', '服务']].mean(axis=1).round(2)
        
        # 5. 删除不需要的列
        if '星级' in df.columns:
            df = df.drop(columns=['星级'])
        
        # 6. 价格范围过滤
        df = df[(df['均价'] > self.config.MIN_PRICE) & (df['均价'] < self.config.MAX_PRICE)]
        logger.info(f"价格过滤后: {df.shape}")
        
        # 7. 评论数异常值处理
        df = self._handle_outliers(df, '评论数')
        logger.info(f"异常值处理后: {df.shape}")
        
        # 保存清洗后的数据
        df.to_csv(self.config.CLEANED_FILE, index=False, encoding='utf-8')
        logger.info(f"清洗后数据已保存到: {self.config.CLEANED_FILE}")
        
        final_shape = df.shape
        logger.info(f"数据清洗完成: {original_shape} -> {final_shape}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        # 处理均价
        df['均价'] = df['均价'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
        df['均价'] = pd.to_numeric(df['均价'], errors='coerce')
        
        # 转换评分字段
        numeric_cols = ['口味', '环境', '服务']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """处理异常值"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config.IQR_MULTIPLIER * IQR
        upper_bound = Q3 + self.config.IQR_MULTIPLIER * IQR
        
        # 记录异常值信息
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        logger.info(f"{column} 异常值数量: {len(outliers)}")
        
        # 过滤异常值
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df_filtered

class Geocoder:
    """地理编码器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        logger.info("地理编码器初始化完成")
    
    def geocode_address(self, address: str, city: str = "上海") -> Tuple[Optional[float], Optional[float]]:
        """
        使用高德地图地理编码API将地址转换为经纬度
        
        Args:
            address: 需要转换的地址
            city: 地址所在城市，默认为上海
            
        Returns:
            (经度, 纬度) 或 (None, None) 如果转换失败
        """
        params = {
            'key': self.config.AMAP_API_KEY,
            'address': address,
            'city': city,
            'output': 'json'
        }
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.get(self.config.API_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == '1' and data['count'] != '0':
                    location = data['geocodes'][0]['location']
                    lng, lat = location.split(',')
                    return float(lng), float(lat)
                else:
                    logger.warning(f"地址解析失败: {address}, 响应: {data}")
                    return None, None
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.config.MAX_RETRIES}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
            except Exception as e:
                logger.error(f"未知错误: {e}, 地址: {address}")
                return None, None
        
        logger.error(f"地址 {address} 在 {self.config.MAX_RETRIES} 次尝试后仍然失败")
        return None, None
    
    def batch_geocode(self, df: pd.DataFrame, address_col: str = '地址') -> pd.DataFrame:
        """
        批量处理地址转换
        
        Args:
            df: 包含地址的DataFrame
            address_col: 地址列的名称
            
        Returns:
            添加了经度和纬度列的DataFrame
        """
        # 创建新列存储经纬度
        if 'longitude' not in df.columns:
            df['longitude'] = None
        if 'latitude' not in df.columns:
            df['latitude'] = None
        
        # 检查是否有保存的进度
        if os.path.exists(self.config.PROGRESS_FILE):
            df = self._load_progress(df)
        
        # 主处理循环
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"开始第 {iteration} 轮处理...")
            
            # 获取需要处理的记录
            to_process = df[df['longitude'].isna()].index
            if len(to_process) == 0:
                logger.info("所有记录都已经处理完成")
                break
            
            logger.info(f"本轮需要处理 {len(to_process)} 条记录")
            
            # 处理记录
            success_count, failed_count = self._process_batch(df, to_process, address_col)
            
            # 保存进度
            self._save_progress(df)
            
            # 检查是否应该继续
            if success_count == 0:
                logger.warning(f"本轮没有成功处理任何记录，剩余 {len(to_process)} 条记录暂时无法处理")
                break
            
            logger.info(f"第 {iteration} 轮完成: 成功 {success_count}, 失败 {failed_count}")
            
            # 避免过于频繁的API调用
            time.sleep(1)
        
        return df
    
    def _load_progress(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载进度"""
        try:
            df_progress = pd.read_csv(self.config.PROGRESS_FILE)
            # 合并进度
            df.update(df_progress[['longitude', 'latitude']])
            completed_count = df['longitude'].notna().sum()
            logger.info(f"已加载进度，已完成 {completed_count} 条记录")
        except Exception as e:
            logger.warning(f"加载进度失败: {e}")
        
        return df
    
    def _save_progress(self, df: pd.DataFrame):
        """保存进度"""
        try:
            df.to_csv(self.config.PROGRESS_FILE, index=False, encoding='utf-8')
            logger.debug("进度已保存")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def _process_batch(self, df: pd.DataFrame, to_process: pd.Index, address_col: str) -> Tuple[int, int]:
        """处理一批记录"""
        success_count = 0
        failed_count = 0
        
        for i, idx in enumerate(to_process):
            address = df.loc[idx, address_col]
            lng, lat = self.geocode_address(address)
            
            if lng is not None and lat is not None:
                df.loc[idx, 'longitude'] = lng
                df.loc[idx, 'latitude'] = lat
                success_count += 1
            else:
                failed_count += 1
            
            # 显示进度
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i + 1}/{len(to_process)} 条记录")
            
            # API请求间隔
            time.sleep(self.config.REQUEST_DELAY)
        
        return success_count, failed_count

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
    
    def analyze_data(self, df: pd.DataFrame):
        """分析数据"""
        logger.info("开始数据分析...")
        
        # 基本统计信息
        self._show_basic_stats(df)
        
        # 数据分布可视化
        self._create_distribution_plots(df)
        
        # 相关性分析
        self._analyze_correlations(df)
        
        logger.info("数据分析完成")
    
    def _show_basic_stats(self, df: pd.DataFrame):
        """显示基本统计信息"""
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"数据信息:\n{df.info()}")
        logger.info(f"描述性统计:\n{df.describe()}")
    
    def _create_distribution_plots(self, df: pd.DataFrame):
        """创建分布图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('餐厅数据分布分析', fontsize=16)
        
        # 评论数分布
        axes[0, 0].hist(df['评论数'], bins=50, alpha=0.7)
        axes[0, 0].set_title('评论数分布')
        axes[0, 0].set_xlabel('评论数')
        axes[0, 0].set_ylabel('频次')
        
        # 均价分布
        axes[0, 1].hist(df['均价'], bins=50, alpha=0.7)
        axes[0, 1].set_title('均价分布')
        axes[0, 1].set_xlabel('均价')
        axes[0, 1].set_ylabel('频次')
        
        # 总评分分布
        axes[0, 2].hist(df['总评分'], bins=30, alpha=0.7)
        axes[0, 2].set_title('总评分分布')
        axes[0, 2].set_xlabel('总评分')
        axes[0, 2].set_ylabel('频次')
        
        # 经纬度散点图
        if 'longitude' in df.columns and 'latitude' in df.columns:
            valid_coords = df.dropna(subset=['longitude', 'latitude'])
            if len(valid_coords) > 0:
                axes[1, 0].scatter(valid_coords['longitude'], valid_coords['latitude'], alpha=0.6)
                axes[1, 0].set_title('餐厅地理位置分布')
                axes[1, 0].set_xlabel('经度')
                axes[1, 0].set_ylabel('纬度')
        
        # 评分关系图
        axes[1, 1].scatter(df['均价'], df['总评分'], alpha=0.6)
        axes[1, 1].set_title('价格与评分关系')
        axes[1, 1].set_xlabel('均价')
        axes[1, 1].set_ylabel('总评分')
        
        # 评论数与评分关系
        axes[1, 2].scatter(df['评论数'], df['总评分'], alpha=0.6)
        axes[1, 2].set_title('评论数与评分关系')
        axes[1, 2].set_xlabel('评论数')
        axes[1, 2].set_ylabel('总评分')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("分布图已保存为 data_analysis.png")
    
    def _analyze_correlations(self, df: pd.DataFrame):
        """分析相关性"""
        numeric_cols = ['评论数', '均价', '口味', '环境', '服务', '总评分']
        if 'longitude' in df.columns and 'latitude' in df.columns:
            numeric_cols.extend(['longitude', 'latitude'])
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("相关性热力图已保存为 correlation_heatmap.png")

def main():
    """主函数"""
    logger.info("=== 上海餐厅数据地理编码程序启动 ===")
    
    # 初始化配置
    config = Config()
    
    try:
        # 1. 数据清洗
        cleaner = DataCleaner(config)
        
        # 检查是否需要清洗数据
        if not os.path.exists(config.CLEANED_FILE):
            logger.info("开始数据清洗...")
            # 这里需要根据实际情况选择输入文件
            input_file = 'data/shanghai_restaurants.json'  # 或者使用其他文件
            if os.path.exists(input_file):
                df = cleaner.load_data(input_file)
                df = cleaner.clean_data(df)
            else:
                logger.warning(f"输入文件 {input_file} 不存在，跳过数据清洗")
                df = None
        else:
            logger.info("使用已清洗的数据文件")
            df = pd.read_csv(config.CLEANED_FILE)
        
        # 2. 地理编码
        if df is not None:
            geocoder = Geocoder(config)
            df_with_coords = geocoder.batch_geocode(df, address_col='地址')
            
            # 统计成功转换的数量
            success_count = df_with_coords['longitude'].notna().sum()
            total_count = len(df_with_coords)
            logger.info(f"地理编码完成: 成功 {success_count}/{total_count} 条记录")
            
            # 保存最终结果
            df_with_coords.to_csv(config.OUTPUT_FILE, index=False, encoding='utf-8')
            logger.info(f"最终结果已保存到: {config.OUTPUT_FILE}")
            
            # 3. 数据分析
            analyzer = DataAnalyzer(config)
            analyzer.analyze_data(df_with_coords)
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()

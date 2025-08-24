#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海餐厅数据分析程序
功能：分析餐厅热度与各种因素的关系，生成可视化图表

使用方法:
    python 课题2_optimized.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import os

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/restaurant_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    # 文件路径配置
    INPUT_FILE: str = 'data/restaurants_with_density.csv'
    OUTPUT_DIR: str = 'data/visualization_outputs'
    
    # 图表配置
    FIGURE_SIZE: Tuple[int, int] = (12, 8)
    DPI: int = 300
    FONT_SIZE: int = 12
    
    # 分析参数
    TOP_N_TYPES: int = 10  # 显示前N个餐厅类型
    TOP_N_DISTRICTS: int = 10  # 显示前N个商区
    ALPHA: float = 0.6  # 散点图透明度
    
    # 颜色配置
    COLOR_PALETTE: str = 'Set2'
    HEATMAP_CMAP: str = 'coolwarm'

class ChineseFontManager:
    """中文字体管理器"""
    
    @staticmethod
    def setup_chinese_fonts():
        """设置中文字体"""
        try:
            # 尝试设置中文字体
            plt.rcParams['font.sans-serif'] = [
                'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 
                'STHeiti', 'PingFang SC', 'DejaVu Sans'
            ]
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.size'] = 12
            
            # 设置Seaborn样式
            sns.set_style("whitegrid")
            sns.set_palette("Set2")
            
            logger.info("中文字体配置完成")
            return True
        except Exception as e:
            logger.error(f"字体配置失败: {e}")
            return False

class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("数据加载器初始化完成")
    
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            if not Path(self.config.INPUT_FILE).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.config.INPUT_FILE}")
            
            df = pd.read_csv(self.config.INPUT_FILE)
            logger.info(f"成功加载数据: {df.shape}")
            
            # 验证必要的列
            required_columns = ['评论数', '均价', '总评分', 'distance_to_metro', '类型', '商区']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"缺少列: {missing_columns}")
            
            return df
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

class RestaurantAnalyzer:
    """餐厅数据分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_output_dir()
        logger.info("餐厅数据分析器初始化完成")
    
    def _setup_output_dir(self):
        """设置输出目录"""
        Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def analyze_popularity_distribution(self, df: pd.DataFrame) -> None:
        """分析餐厅热度分布"""
        logger.info("分析餐厅热度分布...")
        
        plt.figure(figsize=self.config.FIGURE_SIZE)
        
        # 使用对数变换处理长尾分布
        log_comments = np.log1p(df['评论数'])
        
        sns.histplot(log_comments, kde=True, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Log(评论数 + 1)', fontsize=self.config.FONT_SIZE)
        plt.ylabel('频次', fontsize=self.config.FONT_SIZE)
        plt.title('餐厅热度分布 (对数尺度)', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_log = log_comments.mean()
        median_log = log_comments.median()
        plt.axvline(mean_log, color='red', linestyle='--', alpha=0.8, label=f'平均值: {mean_log:.2f}')
        plt.axvline(median_log, color='orange', linestyle='--', alpha=0.8, label=f'中位数: {median_log:.2f}')
        plt.legend()
        
        self._save_and_show_plot('popularity_distribution.png')
    
    def analyze_price_vs_popularity(self, df: pd.DataFrame) -> None:
        """分析价格与热度的关系"""
        logger.info("分析价格与热度关系...")
        
        plt.figure(figsize=self.config.FIGURE_SIZE)
        
        # 过滤有效数据
        valid_data = df.dropna(subset=['均价', '评论数'])
        
        sns.scatterplot(
            x='均价', 
            y=np.log1p(valid_data['评论数']), 
            data=valid_data, 
            alpha=self.config.ALPHA,
            s=50
        )
        plt.xlabel('均价 (元)', fontsize=self.config.FONT_SIZE)
        plt.ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
        plt.title('价格 vs 热度', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(valid_data['均价'], np.log1p(valid_data['评论数']), 1)
        p = np.poly1d(z)
        plt.plot(valid_data['均价'], p(valid_data['均价']), "r--", alpha=0.8)
        
        self._save_and_show_plot('price_vs_popularity.png')
    
    def analyze_rating_vs_popularity(self, df: pd.DataFrame) -> None:
        """分析评分与热度的关系"""
        logger.info("分析评分与热度关系...")
        
        plt.figure(figsize=self.config.FIGURE_SIZE)
        
        # 过滤有效数据
        valid_data = df.dropna(subset=['总评分', '评论数'])
        
        sns.scatterplot(
            x='总评分', 
            y=np.log1p(valid_data['评论数']), 
            data=valid_data, 
            alpha=self.config.ALPHA,
            s=50
        )
        plt.xlabel('综合评分', fontsize=self.config.FONT_SIZE)
        plt.ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
        plt.title('评分 vs 热度', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(valid_data['总评分'], np.log1p(valid_data['评论数']), 1)
        p = np.poly1d(z)
        plt.plot(valid_data['总评分'], p(valid_data['总评分']), "r--", alpha=0.8)
        
        self._save_and_show_plot('rating_vs_popularity.png')
    
    def analyze_metro_distance_vs_popularity(self, df: pd.DataFrame) -> None:
        """分析地铁距离与热度的关系"""
        logger.info("分析地铁距离与热度关系...")
        
        plt.figure(figsize=self.config.FIGURE_SIZE)
        
        # 过滤有效数据
        valid_data = df.dropna(subset=['distance_to_metro', '评论数'])
        
        sns.scatterplot(
            x='distance_to_metro', 
            y=np.log1p(valid_data['评论数']), 
            data=valid_data, 
            alpha=self.config.ALPHA,
            s=50
        )
        plt.xlabel('到最近地铁站距离 (米)', fontsize=self.config.FONT_SIZE)
        plt.ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
        plt.title('交通便利性 vs 热度', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(valid_data['distance_to_metro'], np.log1p(valid_data['评论数']), 1)
        p = np.poly1d(z)
        plt.plot(valid_data['distance_to_metro'], p(valid_data['distance_to_metro']), "r--", alpha=0.8)
        
        self._save_and_show_plot('metro_distance_vs_popularity.png')
    
    def analyze_cuisine_types_popularity(self, df: pd.DataFrame) -> None:
        """分析不同餐厅类型的平均热度"""
        logger.info("分析不同餐厅类型的平均热度...")
        
        plt.figure(figsize=(14, 8))
        
        # 选取评论数最多的前N个类型
        top_types = df['类型'].value_counts().nlargest(self.config.TOP_N_TYPES).index
        top_type_data = df[df['类型'].isin(top_types)]
        
        # 按中位数排序
        order = top_type_data.groupby('类型')['评论数'].median().sort_values(ascending=False).index
        
        # 创建箱线图
        sns.boxplot(
            x='类型', 
            y=np.log1p(top_type_data['评论数']), 
            data=top_type_data, 
            order=order
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('餐厅类型', fontsize=self.config.FONT_SIZE)
        plt.ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
        plt.title(f'不同餐厅类型的平均热度 (前{self.config.TOP_N_TYPES}名)', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        self._save_and_show_plot('cuisine_types_popularity.png')
    
    def analyze_business_districts_popularity(self, df: pd.DataFrame) -> None:
        """分析不同商区的平均热度"""
        logger.info("分析不同商区的平均热度...")
        
        plt.figure(figsize=(14, 8))
        
        # 选取评论数最多的前N个商区
        top_districts = df['商区'].value_counts().nlargest(self.config.TOP_N_DISTRICTS).index
        top_district_data = df[df['商区'].isin(top_districts)]
        
        # 按中位数排序
        order_district = top_district_data.groupby('商区')['评论数'].median().sort_values(ascending=False).index
        
        # 创建箱线图
        sns.boxplot(
            x='商区', 
            y=np.log1p(top_district_data['评论数']), 
            data=top_district_data, 
            order=order_district
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('商区', fontsize=self.config.FONT_SIZE)
        plt.ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
        plt.title(f'不同商区的平均热度 (前{self.config.TOP_N_DISTRICTS}名)', fontsize=self.config.FONT_SIZE + 2)
        plt.grid(True, alpha=0.3)
        
        self._save_and_show_plot('business_districts_popularity.png')
    
    def analyze_competition_vs_popularity(self, df: pd.DataFrame) -> None:
        """分析竞争环境与热度的关系"""
        logger.info("分析竞争环境与热度关系...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 周围2km内所有餐厅数量 vs 热度
        if 'total_nearby' in df.columns:
            valid_data = df.dropna(subset=['total_nearby', '评论数'])
            sns.scatterplot(
                x='total_nearby', 
                y=np.log1p(valid_data['评论数']), 
                data=valid_data, 
                alpha=self.config.ALPHA,
                ax=ax1
            )
            ax1.set_xlabel('2km内餐厅总数', fontsize=self.config.FONT_SIZE)
            ax1.set_ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
            ax1.set_title('总竞争环境 vs 热度', fontsize=self.config.FONT_SIZE)
            ax1.grid(True, alpha=0.3)
        
        # 周围2km内同类型餐厅数量 vs 热度
        if 'same_type_nearby' in df.columns:
            valid_data = df.dropna(subset=['same_type_nearby', '评论数'])
            sns.scatterplot(
                x='same_type_nearby', 
                y=np.log1p(valid_data['评论数']), 
                data=valid_data, 
                alpha=self.config.ALPHA,
                ax=ax2
            )
            ax2.set_xlabel('2km内同类型餐厅数', fontsize=self.config.FONT_SIZE)
            ax2.set_ylabel('Log(评论数)', fontsize=self.config.FONT_SIZE)
            ax2.set_title('同类型竞争 vs 热度', fontsize=self.config.FONT_SIZE)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_and_show_plot('competition_vs_popularity.png')
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """创建相关性热力图"""
        logger.info("创建相关性热力图...")
        
        # 选择数值变量
        numeric_columns = ['评论数', '均价', '总评分']
        if 'distance_to_metro' in df.columns:
            numeric_columns.append('distance_to_metro')
        if 'total_nearby' in df.columns:
            numeric_columns.append('total_nearby')
        if 'same_type_nearby' in df.columns:
            numeric_columns.append('same_type_nearby')
        
        numeric_df = df[numeric_columns].copy()
        
        # 计算相关性矩阵
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap=self.config.HEATMAP_CMAP, 
            center=0,
            fmt='.3f',
            square=True,
            linewidths=0.5
        )
        plt.title('数值变量相关性热力图', fontsize=self.config.FONT_SIZE + 2)
        plt.tight_layout()
        
        self._save_and_show_plot('correlation_heatmap.png')
    
    def _save_and_show_plot(self, filename: str) -> None:
        """保存并显示图表"""
        filepath = Path(self.config.OUTPUT_DIR) / filename
        plt.savefig(filepath, dpi=self.config.DPI, bbox_inches='tight')
        logger.info(f"图表已保存: {filepath}")
        plt.show()
        plt.close()  # 关闭图表释放内存
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> None:
        """生成汇总统计信息"""
        logger.info("生成汇总统计信息...")
        
        summary_file = Path(self.config.OUTPUT_DIR) / 'summary_statistics.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("上海餐厅数据分析汇总报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"数据概览:\n")
            f.write(f"总餐厅数量: {len(df)}\n")
            f.write(f"数据列数: {len(df.columns)}\n\n")
            
            f.write(f"评论数统计:\n")
            f.write(f"平均值: {df['评论数'].mean():.2f}\n")
            f.write(f"中位数: {df['评论数'].median():.2f}\n")
            f.write(f"标准差: {df['评论数'].std():.2f}\n")
            f.write(f"最小值: {df['评论数'].min()}\n")
            f.write(f"最大值: {df['评论数'].max()}\n\n")
            
            f.write(f"价格统计:\n")
            f.write(f"平均价格: {df['均价'].mean():.2f}元\n")
            f.write(f"中位数价格: {df['均价'].median():.2f}元\n\n")
            
            f.write(f"评分统计:\n")
            f.write(f"平均评分: {df['总评分'].mean():.2f}\n")
            f.write(f"中位数评分: {df['总评分'].median():.2f}\n\n")
            
            if 'distance_to_metro' in df.columns:
                f.write(f"地铁距离统计:\n")
                f.write(f"平均距离: {df['distance_to_metro'].mean():.2f}米\n")
                f.write(f"中位数距离: {df['distance_to_metro'].median():.2f}米\n\n")
            
            f.write(f"餐厅类型分布 (前10名):\n")
            type_counts = df['类型'].value_counts().head(10)
            for cuisine_type, count in type_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{cuisine_type}: {count}家 ({percentage:.1f}%)\n")
            
            f.write(f"\n商区分布 (前10名):\n")
            district_counts = df['商区'].value_counts().head(10)
            for district, count in district_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{district}: {count}家 ({percentage:.1f}%)\n")
        
        logger.info(f"汇总统计已保存: {summary_file}")

def main():
    """主函数"""
    logger.info("=== 上海餐厅数据分析程序启动 ===")
    
    try:
        # 1. 初始化配置
        config = Config()
        
        # 2. 设置中文字体
        ChineseFontManager.setup_chinese_fonts()
        
        # 3. 加载数据
        data_loader = DataLoader(config)
        df = data_loader.load_data()
        
        # 4. 创建分析器
        analyzer = RestaurantAnalyzer(config)
        
        # 5. 执行各项分析
        analyzer.analyze_popularity_distribution(df)
        analyzer.analyze_price_vs_popularity(df)
        analyzer.analyze_rating_vs_popularity(df)
        analyzer.analyze_metro_distance_vs_popularity(df)
        analyzer.analyze_cuisine_types_popularity(df)
        analyzer.analyze_business_districts_popularity(df)
        analyzer.analyze_competition_vs_popularity(df)
        analyzer.create_correlation_heatmap(df)
        
        # 6. 生成汇总统计
        analyzer.generate_summary_statistics(df)
        
        logger.info("所有分析完成！")
        logger.info(f"输出文件保存在: {config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()

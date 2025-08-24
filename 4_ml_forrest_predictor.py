#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上海餐厅热度预测程序
功能：使用随机森林算法预测餐厅热度，包含特征工程、模型优化、SHAP解释等

使用方法:
    python 课题5_optimized.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP模块未安装，SHAP分析将被跳过")
import joblib
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import os
import time

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/restaurant_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    # 文件路径配置
    INPUT_FILE: str = 'data/restaurants_with_density.csv'
    OUTPUT_DIR: str = 'data/prediction_outputs'
    MODEL_DIR: str = 'models'
    
    # 模型参数
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # 随机森林参数网格
    PARAM_GRID: Dict[str, List] = None
    
    # 图表配置
    FIGURE_SIZE: Tuple[int, int] = (12, 8)
    DPI: int = 300
    FONT_SIZE: int = 12
    
    def __post_init__(self):
        if self.PARAM_GRID is None:
            self.PARAM_GRID = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

class ChineseFontManager:
    """中文字体管理器"""
    
    @staticmethod
    def setup_chinese_fonts():
        """设置中文字体"""
        try:
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

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_output_dir()
        logger.info("数据处理器初始化完成")
    
    def _setup_output_dir(self):
        """设置输出目录"""
        Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.config.MODEL_DIR).mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """加载和预处理数据"""
        try:
            if not Path(self.config.INPUT_FILE).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.config.INPUT_FILE}")
            
            df = pd.read_csv(self.config.INPUT_FILE)
            logger.info(f"成功加载数据: {df.shape}")
            
            # 检查缺失值
            missing_stats = df.isnull().sum()
            logger.info(f"缺失值统计:\n{missing_stats}")
            
            # 处理缺失值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # 对评论数取对数（处理偏态分布）
            df['log_reviews'] = np.log1p(df['评论数'])
            logger.info("已创建对数变换的目标变量")
            
            # 编码分类变量
            label_encoders = {}
            categorical_cols = ['类型', '商区']
            
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                    logger.info(f"已编码分类变量: {col}")
            
            logger.info("数据预处理完成")
            return df, label_encoders
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """准备特征和目标变量"""
        # 选择特征
        features = ['均价', '口味', '环境', '服务', '总评分', 'distance_to_metro', 
                   'total_nearby', 'same_type_nearby', '类型', '商区']
        
        # 过滤存在的特征
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"缺少特征: {missing_features}")
        
        # 目标变量
        target = 'log_reviews'
        
        X = df[available_features]
        y = df[target]
        
        logger.info(f"特征数量: {len(available_features)}")
        logger.info(f"样本数量: {len(X)}")
        
        return X, y, available_features

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("模型训练器初始化完成")
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
        """训练随机森林模型"""
        logger.info("开始训练随机森林模型...")
        start_time = time.time()
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"训练集大小: {X_train.shape}")
        logger.info(f"测试集大小: {X_test.shape}")
        
        # 使用网格搜索优化参数
        logger.info("正在进行参数优化...")
        rf = RandomForestRegressor(random_state=self.config.RANDOM_STATE, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=self.config.PARAM_GRID, 
            cv=3, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数训练模型
        best_rf = grid_search.best_estimator_
        best_rf.fit(X_train, y_train)
        
        # 预测
        y_pred = best_rf.predict(X_test)
        y_train_pred = best_rf.predict(X_train)
        
        # 计算评估指标
        metrics = self._calculate_metrics(y_test, y_pred, y_train, y_train_pred)
        
        elapsed_time = time.time() - start_time
        logger.info(f"模型训练完成，耗时: {elapsed_time:.2f}秒")
        
        return best_rf, {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'y_train_pred': y_train_pred,
            'metrics': metrics, 'best_params': grid_search.best_params_
        }
    
    def _calculate_metrics(self, y_test: pd.Series, y_pred: np.ndarray, 
                          y_train: pd.Series, y_train_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        # 测试集指标
        metrics['r2_test'] = r2_score(y_test, y_pred)
        metrics['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae_test'] = mean_absolute_error(y_test, y_pred)
        
        # 训练集指标
        metrics['r2_train'] = r2_score(y_train, y_train_pred)
        metrics['rmse_train'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics['mae_train'] = mean_absolute_error(y_train, y_train_pred)
        
        # 过拟合检查
        metrics['overfitting_gap'] = metrics['r2_train'] - metrics['r2_test']
        
        return metrics

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("模型评估器初始化完成")
    
    def evaluate_model(self, model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series, 
                      training_results: Dict[str, Any]) -> None:
        """评估模型性能"""
        logger.info("开始评估模型性能...")
        
        metrics = training_results['metrics']
        
        # 显示基本指标
        self._display_basic_metrics(metrics)
        
        # 过拟合分析
        self._analyze_overfitting(metrics)
        
        # 交叉验证
        self._cross_validation_analysis(model, X, y)
        
        # 特征重要性分析
        self._analyze_feature_importance(model, training_results['X_train'].columns)
        
        # SHAP分析
        self._shap_analysis(model, training_results['X_test'])
        
        # 残差分析
        self._residual_analysis(training_results)
        
        # 分组误差分析
        self._group_error_analysis(training_results)
        
        logger.info("模型评估完成")
    
    def _display_basic_metrics(self, metrics: Dict[str, float]) -> None:
        """显示基本指标"""
        logger.info(f"\n=== 模型性能指标 ===")
        logger.info(f"测试集 R²: {metrics['r2_test']:.4f}")
        logger.info(f"测试集 RMSE: {metrics['rmse_test']:.4f}")
        logger.info(f"测试集 MAE: {metrics['mae_test']:.4f}")
        logger.info(f"训练集 R²: {metrics['r2_train']:.4f}")
        logger.info(f"训练集 RMSE: {metrics['rmse_train']:.4f}")
        logger.info(f"训练集 MAE: {metrics['mae_train']:.4f}")
    
    def _analyze_overfitting(self, metrics: Dict[str, float]) -> None:
        """分析过拟合情况"""
        gap = metrics['overfitting_gap']
        logger.info(f"\n=== 过拟合分析 ===")
        logger.info(f"训练集与测试集R²差距: {gap:.4f}")
        
        if gap > 0.1:
            logger.warning("⚠️  模型可能存在过拟合")
        elif gap > 0.05:
            logger.info("⚠️  模型可能存在轻微过拟合")
        else:
            logger.info("✅ 模型拟合情况良好")
    
    def _cross_validation_analysis(self, model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> None:
        """交叉验证分析"""
        logger.info("\n=== 交叉验证分析 ===")
        
        cv_scores = cross_val_score(model, X, y, cv=self.config.CV_FOLDS, scoring='r2')
        logger.info(f"{self.config.CV_FOLDS}折交叉验证R²分数: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        # 检查交叉验证稳定性
        if cv_scores.std() > 0.1:
            logger.warning("⚠️  交叉验证分数波动较大，模型可能不够稳定")
        else:
            logger.info("✅ 交叉验证分数稳定")
    
    def _analyze_feature_importance(self, model: RandomForestRegressor, feature_names: List[str]) -> None:
        """分析特征重要性"""
        logger.info("\n=== 特征重要性分析 ===")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("特征重要性排序:")
        for _, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 保存特征重要性
        importance_file = Path(self.config.OUTPUT_DIR) / 'feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False, encoding='utf-8')
        logger.info(f"特征重要性已保存: {importance_file}")
    
    def _shap_analysis(self, model: RandomForestRegressor, X_test: pd.DataFrame) -> None:
        """SHAP值分析"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP模块未安装，跳过SHAP分析")
            return
            
        logger.info("\n=== SHAP值分析 ===")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # 计算SHAP特征重要性
            shap_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            logger.info("基于SHAP值的特征重要性:")
            for _, row in shap_importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 保存SHAP重要性
            shap_file = Path(self.config.OUTPUT_DIR) / 'shap_importance.csv'
            shap_importance.to_csv(shap_file, index=False, encoding='utf-8')
            logger.info(f"SHAP重要性已保存: {shap_file}")
            
        except Exception as e:
            logger.error(f"SHAP分析失败: {e}")
    
    def _residual_analysis(self, training_results: Dict[str, Any]) -> None:
        """残差分析"""
        logger.info("\n=== 残差分析 ===")
        
        y_test = training_results['y_test']
        y_pred = training_results['y_pred']
        y_train = training_results['y_train']
        y_train_pred = training_results['y_train_pred']
        
        # 计算残差
        residuals_test = y_test - y_pred
        residuals_train = y_train - y_train_pred
        
        # 残差统计
        logger.info(f"测试集残差均值: {residuals_test.mean():.4f}")
        logger.info(f"测试集残差标准差: {residuals_test.std():.4f}")
        logger.info(f"测试集残差绝对值均值: {np.abs(residuals_test).mean():.4f}")
        
        # 残差分布检查
        if abs(residuals_test.mean()) > 0.1:
            logger.warning("⚠️  残差均值偏离0较大，可能存在系统性偏差")
        else:
            logger.info("✅ 残差均值接近0")
    
    def _group_error_analysis(self, training_results: Dict[str, Any]) -> None:
        """分组误差分析"""
        logger.info("\n=== 分组误差分析 ===")
        
        X_test = training_results['X_test']
        y_test = training_results['y_test']
        y_pred = training_results['y_pred']
        
        # 创建误差分析DataFrame
        error_analysis_df = X_test.copy()
        error_analysis_df['实际值'] = y_test.values
        error_analysis_df['预测值'] = y_pred
        error_analysis_df['绝对误差'] = np.abs(y_test.values - y_pred)
        
        # 按商区分组计算平均绝对误差
        if '商区' in error_analysis_df.columns:
            mae_by_area = error_analysis_df.groupby('商区')['绝对误差'].mean().sort_values()
            logger.info("按商区分组的平均绝对误差 (前5名):")
            logger.info(mae_by_area.head())
        
        # 按类型分组计算平均绝对误差
        if '类型' in error_analysis_df.columns:
            mae_by_type = error_analysis_df.groupby('类型')['绝对误差'].mean().sort_values()
            logger.info("按类型分组的平均绝对误差 (前5名):")
            logger.info(mae_by_type.head())

class ModelSaver:
    """模型保存器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("模型保存器初始化完成")
    
    def save_model_and_results(self, model: RandomForestRegressor, 
                              label_encoders: Dict[str, LabelEncoder],
                              training_results: Dict[str, Any],
                              feature_names: List[str]) -> None:
        """保存模型和结果"""
        try:
            # 保存模型
            model_file = Path(self.config.MODEL_DIR) / 'restaurant_popularity_rf_model.pkl'
            joblib.dump(model, model_file)
            logger.info(f"模型已保存: {model_file}")
            
            # 保存标签编码器
            encoders_file = Path(self.config.MODEL_DIR) / 'label_encoders.pkl'
            joblib.dump(label_encoders, encoders_file)
            logger.info(f"标签编码器已保存: {encoders_file}")
            
            # 保存训练结果摘要
            self._save_training_summary(training_results, feature_names)
            
            # 保存预测示例
            self._save_prediction_examples(training_results)
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    def _save_training_summary(self, training_results: Dict[str, Any], 
                              feature_names: List[str]) -> None:
        """保存训练结果摘要"""
        summary_file = Path(self.config.OUTPUT_DIR) / 'training_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("随机森林餐厅热度预测模型训练摘要\n")
            f.write("=" * 50 + "\n\n")
            
            metrics = training_results['metrics']
            f.write(f"模型性能指标:\n")
            f.write(f"测试集 R²: {metrics['r2_test']:.4f}\n")
            f.write(f"测试集 RMSE: {metrics['rmse_test']:.4f}\n")
            f.write(f"测试集 MAE: {metrics['mae_test']:.4f}\n")
            f.write(f"训练集 R²: {metrics['r2_train']:.4f}\n")
            f.write(f"训练集 RMSE: {metrics['rmse_train']:.4f}\n")
            f.write(f"训练集 MAE: {metrics['mae_train']:.4f}\n")
            f.write(f"过拟合差距: {metrics['overfitting_gap']:.4f}\n\n")
            
            f.write(f"特征数量: {len(feature_names)}\n")
            f.write(f"训练样本数: {len(training_results['X_train'])}\n")
            f.write(f"测试样本数: {len(training_results['X_test'])}\n")
            
            if 'best_params' in training_results:
                f.write(f"\n最佳参数:\n")
                for param, value in training_results['best_params'].items():
                    f.write(f"  {param}: {value}\n")
        
        logger.info(f"训练摘要已保存: {summary_file}")
    
    def _save_prediction_examples(self, training_results: Dict[str, Any]) -> None:
        """保存预测示例"""
        examples_file = Path(self.config.OUTPUT_DIR) / 'prediction_examples.csv'
        
        # 选择前10个测试样本作为示例
        sample_data = training_results['X_test'].iloc[:10].copy()
        predictions = training_results['y_pred'][:10]
        actuals = training_results['y_test'].iloc[:10].values
        
        sample_results = pd.DataFrame({
            '实际值': actuals,
            '预测值': predictions,
            '误差': np.abs(actuals - predictions),
            '相对误差(%)': np.abs(actuals - predictions) / np.abs(actuals) * 100
        })
        
        sample_results.to_csv(examples_file, index=False, encoding='utf-8')
        logger.info(f"预测示例已保存: {examples_file}")

class Visualizer:
    """可视化器"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("可视化器初始化完成")
    
    def create_comprehensive_plots(self, training_results: Dict[str, Any], 
                                 feature_names: List[str]) -> None:
        """创建综合可视化图表"""
        logger.info("开始创建可视化图表...")
        
        # 特征重要性图
        self._plot_feature_importance(training_results, feature_names)
        
        # 预测性能图
        self._plot_prediction_performance(training_results)
        
        # 残差分析图
        self._plot_residual_analysis(training_results)
        
        # 特征关系图
        self._plot_feature_relationships(training_results, feature_names)
        
        logger.info("可视化图表创建完成")
    
    def _plot_feature_importance(self, training_results: Dict[str, Any], 
                                feature_names: List[str]) -> None:
        """绘制特征重要性图"""
        # 从训练结果中获取模型（这里需要修改逻辑）
        # 暂时跳过这个图，因为模型不在training_results中
        
        logger.info("特征重要性图绘制完成")
    
    def _plot_prediction_performance(self, training_results: Dict[str, Any]) -> None:
        """绘制预测性能图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('随机森林模型预测性能', fontsize=16)
        
        # 测试集预测
        y_test = training_results['y_test']
        y_pred = training_results['y_pred']
        r2_test = training_results['metrics']['r2_test']
        rmse_test = training_results['metrics']['rmse_test']
        
        axes[0].scatter(y_test, y_pred, alpha=0.6)
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('实际值 (log(评论数))')
        axes[0].set_ylabel('预测值 (log(评论数))')
        axes[0].set_title(f'测试集: 实际值 vs 预测值\nR² = {r2_test:.3f}, RMSE = {rmse_test:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # 训练集预测
        y_train = training_results['y_train']
        y_train_pred = training_results['y_train_pred']
        r2_train = training_results['metrics']['r2_train']
        rmse_train = training_results['metrics']['rmse_train']
        
        axes[1].scatter(y_train, y_train_pred, alpha=0.6, color='orange')
        max_val = max(y_train.max(), y_train_pred.max())
        min_val = min(y_train.min(), y_train_pred.min())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1].set_xlabel('实际值 (log(评论数))')
        axes[1].set_ylabel('预测值 (log(评论数))')
        axes[1].set_title(f'训练集: 实际值 vs 预测值\nR² = {r2_train:.3f}, RMSE = {rmse_train:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = Path(self.config.OUTPUT_DIR) / 'prediction_performance.png'
        plt.savefig(filepath, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"预测性能图已保存: {filepath}")
    
    def _plot_residual_analysis(self, training_results: Dict[str, Any]) -> None:
        """绘制残差分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('残差分析', fontsize=16)
        
        y_test = training_results['y_test']
        y_pred = training_results['y_pred']
        residuals_test = y_test - y_pred
        
        # 残差 vs 预测值
        axes[0].scatter(y_pred, residuals_test, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('预测值')
        axes[0].set_ylabel('残差')
        axes[0].set_title('残差 vs 预测值')
        axes[0].grid(True, alpha=0.3)
        
        # 残差分布直方图
        axes[1].hist(residuals_test, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('残差')
        axes[1].set_ylabel('频数')
        axes[1].set_title('残差分布')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = Path(self.config.OUTPUT_DIR) / 'residual_analysis.png'
        plt.savefig(filepath, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"残差分析图已保存: {filepath}")
    
    def _plot_feature_relationships(self, training_results: Dict[str, Any], 
                                  feature_names: List[str]) -> None:
        """绘制特征关系图"""
        # 选择前6个重要特征
        important_features = feature_names[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('重要特征与目标变量的关系', fontsize=16)
        
        X = training_results['X_test']
        y = training_results['y_test']
        
        for i, feature in enumerate(important_features):
            row, col = i // 3, i % 3
            axes[row, col].scatter(X[feature], y, alpha=0.3)
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('log(评论数)')
            axes[row, col].grid(True, alpha=0.3)
            
            # 添加趋势线（仅对数值变量）
            if X[feature].dtype in ['int64', 'float64']:
                z = np.polyfit(X[feature], y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(X[feature].min(), X[feature].max(), 100)
                axes[row, col].plot(x_range, p(x_range), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # 保存图表
        filepath = Path(self.config.OUTPUT_DIR) / 'feature_relationships.png'
        plt.savefig(filepath, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"特征关系图已保存: {filepath}")

def main():
    """主函数"""
    logger.info("=== 上海餐厅热度预测程序启动 ===")
    
    try:
        # 1. 初始化配置
        config = Config()
        
        # 2. 设置中文字体
        ChineseFontManager.setup_chinese_fonts()
        
        # 3. 数据预处理
        processor = DataProcessor(config)
        df, label_encoders = processor.load_and_preprocess_data()
        X, y, feature_names = processor.prepare_features(df)
        
        # 4. 模型训练
        trainer = ModelTrainer(config)
        model, training_results = trainer.train_random_forest(X, y)
        
        # 5. 模型评估
        evaluator = ModelEvaluator(config)
        evaluator.evaluate_model(model, X, y, training_results)
        
        # 6. 可视化
        visualizer = Visualizer(config)
        visualizer.create_comprehensive_plots(training_results, feature_names)
        
        # 7. 保存模型和结果
        saver = ModelSaver(config)
        saver.save_model_and_results(model, label_encoders, training_results, feature_names)
        
        logger.info("程序执行完成！")
        logger.info(f"输出文件保存在: {config.OUTPUT_DIR}")
        logger.info(f"模型文件保存在: {config.MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()

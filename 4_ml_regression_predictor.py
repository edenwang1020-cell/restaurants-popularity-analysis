#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
餐厅热度影响因素分析程序
功能：使用机器学习方法分析影响餐厅热度的各种因素，包括价格、评分、地理位置等

使用方法:
    python 课题4_fixed.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
import platform
import os

warnings.filterwarnings('ignore')

# 导入matplotlib相关库
import matplotlib
# 注释掉Agg后端，允许图表显示
# matplotlib.use('Agg')  # 强制使用Agg后端

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # 处理分类特征的缺失值
    categorical_cols = ['类型', '商区']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('未知')
    
    # 对评论数取对数（处理偏态分布）
    df['log_reviews'] = np.log1p(df['评论数'])
    
    print(f"数据预处理完成，处理后形状: {df.shape}")
    return df

def prepare_features(df):
    """准备特征和目标变量"""
    # 数值特征
    numeric_features = ['均价', '口味', '环境', '服务', '总评分', 'distance_to_metro', 'total_nearby', 'same_type_nearby']
    
    # 分类特征
    categorical_features = ['类型', '商区']
    
    # 目标变量
    target = 'log_reviews'
    
    # 检查特征是否存在
    missing_features = [f for f in numeric_features + categorical_features if f not in df.columns]
    if missing_features:
        print(f"警告: 以下特征在数据中不存在: {missing_features}")
        return None, None, None
    
    # 检查分类特征的基数（唯一值数量）
    for col in categorical_features:
        unique_count = df[col].nunique()
        print(f"{col} 的唯一值数量: {unique_count}")
        if unique_count > 50:  # 如果分类特征过多，进行分组
            print(f"警告: {col} 的分类过多，将进行分组处理")
            df = group_rare_categories(df, col, threshold=0.01)
    
    return numeric_features, categorical_features, target

def group_rare_categories(df, column, threshold=0.01):
    """将稀有分类合并为'其他'"""
    value_counts = df[column].value_counts(normalize=True)
    rare_categories = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare_categories, '其他')
    print(f"将 {len(rare_categories)} 个稀有分类合并为'其他'")
    return df

def create_preprocessing_pipeline(numeric_features, categorical_features):
    """创建预处理管道"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.01))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_and_evaluate_model(X, y, preprocessor, test_size=0.2, random_state=42):
    """训练和评估模型"""
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # 尝试不同的模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01)
    }
    
    best_model = None
    best_score = -np.inf
    best_model_name = None
    
    print("正在训练和比较不同模型...")
    
    for name, model in models.items():
        # 创建管道
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # 使用交叉验证评估
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            mean_cv_score = cv_scores.mean()
            print(f"{name} - 交叉验证R²: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_model = pipeline
                best_model_name = name
        except Exception as e:
            print(f"{name} 训练失败: {e}")
            continue
    
    if best_model is None:
        print("所有模型都训练失败，使用默认的LinearRegression")
        best_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        best_model_name = 'LinearRegression'
    
    print(f"\n选择最佳模型: {best_model_name}")
    
    # 训练最佳模型
    best_model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # 计算评估指标
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n模型评估结果:")
    print(f"训练集 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
    print(f"测试集 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
    
    return best_model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def analyze_feature_importance(model, numeric_features, categorical_features):
    """分析特征重要性"""
    try:
        # 获取所有特征名称和系数
        numeric_feature_names = numeric_features
        
        # 获取分类特征名称（独热编码后的）
        cat_encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        
        # 所有特征名称
        all_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])
        
        # 获取系数
        coefficients = model.named_steps['regressor'].coef_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n所有特征系数（按绝对值排序，前20个）:")
        print(feature_importance.head(20))
        
        return feature_importance, categorical_feature_names
    except Exception as e:
        print(f"特征重要性分析失败: {e}")
        return None, None

def visualize_feature_importance(feature_importance, numeric_features, categorical_features):
    """可视化特征重要性"""
    if feature_importance is None:
        print("无法可视化特征重要性")
        return
    
    # 数值特征重要性分析
    numeric_importance = feature_importance[feature_importance['feature'].isin(numeric_features)].copy()
    numeric_importance = numeric_importance.sort_values('abs_coefficient', ascending=False)
    
    print("\n数值特征重要性排序:")
    print(numeric_importance[['feature', 'coefficient', 'abs_coefficient']])
    
    # 可视化数值特征系数
    plt.figure(figsize=(10, 6))
    colors = ['red' if coef < 0 else 'blue' for coef in numeric_importance['coefficient']]
    plt.barh(numeric_importance['feature'], numeric_importance['coefficient'], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('系数值')
    plt.title('数值特征对餐厅热度的影响')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # 确保图表显示
    
    # 分类特征内部差异分析
    categorical_importance = feature_importance[~feature_importance['feature'].isin(numeric_features)].copy()
    if len(categorical_importance) > 0:
        categorical_importance['original_feature'] = categorical_importance['feature'].str.split('_').str[0]
        
        # 计算每个原始分类特征的标准差
        category_std = categorical_importance.groupby('original_feature')['coefficient'].std().reset_index()
        category_std.columns = ['feature', 'std_coefficient']
        category_std = category_std.sort_values('std_coefficient', ascending=False)
        
        print("\n分类特征内部差异程度（标准差）:")
        print(category_std)
        
        # 可视化分类特征内部差异
        plt.figure(figsize=(10, 6))
        plt.barh(category_std['feature'], category_std['std_coefficient'])
        plt.xlabel('系数标准差')
        plt.title('分类特征内部差异程度')
        plt.tight_layout()
        plt.show()
        
        # 每个分类特征下具体类别的系数分布（只显示前20个）
        for cat_feature in categorical_features:
            cat_df = categorical_importance[categorical_importance['original_feature'] == cat_feature].copy()
            cat_df = cat_df.sort_values('coefficient', ascending=False)
            
            # 只显示前20个类别
            if len(cat_df) > 20:
                cat_df = cat_df.head(20)
                print(f"\n{cat_feature} 只显示前20个类别（共{len(categorical_importance[categorical_importance['original_feature'] == cat_feature])}个）")
            
            plt.figure(figsize=(12, max(6, len(cat_df) * 0.3)))
            colors = ['red' if coef < 0 else 'blue' for coef in cat_df['coefficient']]
            plt.barh(cat_df['feature'], cat_df['coefficient'], color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('系数值')
            plt.title(f'{cat_feature}各类别对餐厅热度的影响')
            plt.tight_layout()
            plt.show()

def create_comprehensive_visualizations(y_train, y_test, y_train_pred, y_test_pred, X_test, 
                                      numeric_features, residuals_test, residuals_train):
    """创建综合的可视化图表"""
    print("\n创建综合可视化图表...")
    
    # 计算评估指标
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型预测性能与残差分析', fontsize=16)
    
    # 实际值 vs 预测值（测试集）
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6)
    max_val = max(y_test.max(), y_test_pred.max())
    min_val = min(y_test.min(), y_test_pred.min())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际值 (log(评论数))')
    axes[0, 0].set_ylabel('预测值 (log(评论数))')
    axes[0, 0].set_title(f'测试集: 实际值 vs 预测值\nR² = {r2_test:.3f}, RMSE = {rmse_test:.3f}')
    
    # 实际值 vs 预测值（训练集）
    axes[0, 1].scatter(y_train, y_train_pred, alpha=0.6, color='orange')
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值 (log(评论数))')
    axes[0, 1].set_ylabel('预测值 (log(评论数))')
    axes[0, 1].set_title(f'训练集: 实际值 vs 预测值\nR² = {r2_train:.3f}, RMSE = {rmse_train:.3f}')
    
    # 残差 vs 预测值（测试集）
    axes[0, 2].scatter(y_test_pred, residuals_test, alpha=0.6)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('预测值')
    axes[0, 2].set_ylabel('残差')
    axes[0, 2].set_title('测试集: 残差 vs 预测值')
    
    # 残差 vs 预测值（训练集）
    axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, color='orange')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('训练集: 残差 vs 预测值')
    
    # 残差分布直方图（测试集）
    axes[1, 1].hist(residuals_test, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('测试集: 残差分布')
    
    # Q-Q图（测试集残差的正态性检验）
    try:
        stats.probplot(residuals_test, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('测试集: 残差Q-Q图')
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f'Q-Q图生成失败:\n{e}', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('测试集: 残差Q-Q图')
    
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # 确保图表显示
    
    return r2_test, rmse_test, mae_test, r2_train, rmse_train

def analyze_residuals(X_test, residuals_test, y_test_pred, numeric_features):
    """分析残差与特征的关联"""
    print("\n分析残差与特征的关联...")
    
    # 残差统计分析
    print("\n残差统计分析:")
    print(f"测试集残差均值: {residuals_test.mean():.4f}")
    print(f"测试集残差标准差: {residuals_test.std():.4f}")
    print(f"测试集残差绝对值均值: {np.abs(residuals_test).mean():.4f}")
    
    # 分析数值特征与残差的关系
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('数值特征与残差的关系', fontsize=16)
    
    for i, feature in enumerate(numeric_features):
        row, col = i // 4, i % 4
        axes[row, col].scatter(X_test[feature], residuals_test, alpha=0.6)
        axes[row, col].axhline(y=0, color='r', linestyle='--')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('残差')
        # 添加趋势线
        try:
            z = np.polyfit(X_test[feature], residuals_test, 1)
            p = np.poly1d(z)
            axes[row, col].plot(X_test[feature], p(X_test[feature]), "r--", alpha=0.8)
        except:
            pass
    
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # 确保图表显示

def analyze_prediction_errors(X_test, y_test, y_test_pred):
    """分析预测误差（按商区和类型分组）"""
    print("\n分析预测误差...")
    
    # 将预测误差按商区和类型分组计算平均误差
    error_analysis_df = X_test.copy()
    error_analysis_df['实际值'] = y_test.values
    error_analysis_df['预测值'] = y_test_pred
    error_analysis_df['绝对误差'] = np.abs(y_test.values - y_test_pred)
    
    # 按商区分组计算平均绝对误差
    mae_by_area = error_analysis_df.groupby('商区')['绝对误差'].mean().sort_values()
    print("\n按商区分组的平均绝对误差（前10个和后10个）:")
    print("前10个（误差最小）:")
    print(mae_by_area.head(10))
    print("\n后10个（误差最大）:")
    print(mae_by_area.tail(10))
    
    # 按类型分组计算平均绝对误差
    mae_by_type = error_analysis_df.groupby('类型')['绝对误差'].mean().sort_values()
    print("\n按类型分组的平均绝对误差（前10个和后10个）:")
    print("前10个（误差最小）:")
    print(mae_by_type.head(10))
    print("\n后10个（误差最大）:")
    print(mae_by_type.tail(10))
    
    # 可视化商区误差（只显示前20个和后20个）
    plt.figure(figsize=(12, 8))
    mae_by_area_plot = pd.concat([mae_by_area.head(20), mae_by_area.tail(20)])
    mae_by_area_plot.plot(kind='barh')
    plt.xlabel('平均绝对误差')
    plt.title('各商区预测平均绝对误差（前20个和后20个）')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # 确保图表显示
    
    # 可视化类型误差（只显示前20个和后20个）
    plt.figure(figsize=(12, 8))
    mae_by_type_plot = pd.concat([mae_by_type.head(20), mae_by_type.tail(20)])
    mae_by_type_plot.plot(kind='barh')
    plt.xlabel('平均绝对误差')
    plt.title('各类型餐厅预测平均绝对误差（前20个和后20个）')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)  # 确保图表显示

def main():
    """主函数"""
    print("=== 餐厅热度影响因素分析 ===")
    
    # 1. 加载和预处理数据
    df = load_and_preprocess_data(r'data/课题\restaurants_with_density.csv')
    
    # 2. 准备特征
    result = prepare_features(df)
    if result is None:
        print("特征准备失败，程序退出")
        return
    
    numeric_features, categorical_features, target = result
    
    # 3. 创建预处理管道
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # 4. 准备数据
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    # 5. 训练和评估模型
    model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_and_evaluate_model(
        X, y, preprocessor
    )
    
    # 6. 分析特征重要性
    feature_importance, categorical_feature_names = analyze_feature_importance(
        model, numeric_features, categorical_features
    )
    
    # 7. 可视化特征重要性
    visualize_feature_importance(feature_importance, numeric_features, categorical_features)
    
    # 8. 计算残差
    residuals_test = y_test - y_test_pred
    residuals_train = y_train - y_train_pred
    
    # 9. 创建综合可视化
    r2_test, rmse_test, mae_test, r2_train, rmse_train = create_comprehensive_visualizations(
        y_train, y_test, y_train_pred, y_test_pred, X_test, numeric_features, 
        residuals_test, residuals_train
    )
    
    # 10. 残差分析
    analyze_residuals(X_test, residuals_test, y_test_pred, numeric_features)
    
    # 11. 预测误差分析
    analyze_prediction_errors(X_test, y_test, y_test_pred)
    
    # 12. 最终模型性能总结
    print("\n=== 模型性能总结 ===")
    print(f"测试集R²: {r2_test:.4f}")
    print(f"测试集RMSE: {rmse_test:.4f}")
    print(f"测试集MAE: {mae_test:.4f}")
    print(f"训练集R²: {r2_train:.4f}")
    print(f"训练集RMSE: {rmse_train:.4f}")
    
    # 检查过拟合情况
    if r2_train - r2_test > 0.1:
        print("警告: 模型可能存在过拟合（训练集R²远高于测试集R²）")
        print("建议:")
        print("1. 使用正则化模型（Ridge或Lasso）")
        print("2. 减少特征数量")
        print("3. 增加训练数据")
        print("4. 使用交叉验证选择最佳参数")
    else:
        print("模型拟合情况良好")
    
    # 检查残差均值是否接近0
    if abs(residuals_test.mean()) > 0.1:
        print("警告: 残差均值偏离0较多，模型可能存在系统性偏差")
        print("建议检查数据质量和特征工程")
    else:
        print("残差均值接近0，模型无显著系统性偏差")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()

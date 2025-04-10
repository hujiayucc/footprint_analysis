# src/evaluation.py
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ================== 初始化配置 ==================
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ================== 路径配置 ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "footprint_model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "footprint_data.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FONT_PATH = os.path.join(BASE_DIR, "fonts", "MiSansVF.ttf")

# 创建必要目录
os.makedirs(REPORT_DIR, exist_ok=True)

# ================== 字体配置 ==================
def configure_font():
    """配置自定义字体并验证"""
    try:
        # 验证字体文件是否存在
        if not os.path.exists(FONT_PATH):
            raise FileNotFoundError(f"字体文件未找到: {FONT_PATH}")

        # 注册字体到matplotlib
        fm.fontManager.addfont(FONT_PATH)
        custom_font = fm.FontProperties(fname=FONT_PATH)
        
        # 设置全局字体
        plt.rcParams['font.sans-serif'] = [custom_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        
        logging.info("✅ 成功加载自定义字体")
        return custom_font
    except Exception as e:
        logging.warning(f"⚠️ 字体配置失败: {str(e)}")
        # 回退到系统字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
        plt.rcParams['axes.unicode_minus'] = False
        return None

# ================== 可视化函数 ==================
def generate_visualizations(y_true, y_pred_reg, y_cls_true, y_cls_pred, font_prop):
    """生成可视化分析图表"""
    try:
        plt.figure(figsize=(18, 12))
        font_args = {'fontproperties': font_prop} if font_prop else {}

        # 身高预测分析
        plt.subplot(2, 3, 1)
        sns.scatterplot(x=y_true[:, 0], y=y_pred_reg[:, 0], alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title('身高预测 vs 真实值', **font_args)
        plt.xlabel('真实身高 (cm)', **font_args)
        plt.ylabel('预测身高 (cm)', **font_args)

        # 体重预测分析
        plt.subplot(2, 3, 2)
        sns.scatterplot(x=y_true[:, 1], y=y_pred_reg[:, 1], alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title('体重预测 vs 真实值', **font_args)
        plt.xlabel('真实体重 (kg)', **font_args)
        plt.ylabel('预测体重 (kg)', **font_args)

        # 身高残差分析
        plt.subplot(2, 3, 3)
        residuals = y_true[:, 0] - y_pred_reg[:, 0]
        sns.histplot(residuals, kde=True, bins=20)
        plt.title('身高预测残差分布', **font_args)
        plt.xlabel('残差值 (cm)', **font_args)

        # 体重残差分析
        plt.subplot(2, 3, 4)
        residuals = y_true[:, 1] - y_pred_reg[:, 1]
        sns.histplot(residuals, kde=True, bins=20)
        plt.title('体重预测残差分布', **font_args)
        plt.xlabel('残差值 (kg)', **font_args)

        # 分类混淆矩阵
        plt.subplot(2, 3, 5)
        cm = confusion_matrix(y_cls_true, np.argmax(y_cls_pred, axis=1))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['正常', 'O型腿', 'X型腿'],
            yticklabels=['正常', 'O型腿', 'X型腿'],
            annot_kws=font_args
        )
        plt.title('腿型分类混淆矩阵', **font_args)
        plt.xlabel('预测标签', **font_args)
        plt.ylabel('真实标签', **font_args)

        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, 'model_performance.png'))
        plt.close()
        logging.info("📈 可视化报告已保存至 reports 目录")
    except Exception as e:
        logging.error(f"⚠️ 可视化生成失败: {str(e)}", exc_info=True)

# ================== 评估函数 ==================
def evaluate():
    try:
        # 初始化字体配置
        font_prop = configure_font()

        # 1. 加载资源
        logging.info("🔧 加载模型和预处理器...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"预处理器文件未找到: {PREPROCESSOR_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # 2. 加载数据
        logging.info(f"📂 加载数据: {DATA_PATH}")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        # 验证必要列是否存在
        required_columns = ["height", "weight", "leg_type"]
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            raise ValueError(f"数据文件缺少必要列: {missing}")

        X = df.drop(required_columns, axis=1)
        y_reg = df[["height", "weight"]].values
        y_cls = df["leg_type"].values

        # 3. 预处理
        logging.info("⚙ 执行数据预处理...")
        X_processed = preprocessor.transform(X)
        
        # 验证预处理维度
        if X_processed.shape[1] != 13:
            raise ValueError(f"预处理维度异常: 预期13列，实际{X_processed.shape[1]}列")

        # 4. 分割特征
        X_cat = X_processed[:, :5].astype(np.float32)
        X_num = X_processed[:, 5:].astype(np.float32)

        # 5. 预测
        logging.info("🔮 执行预测...")
        reg_pred, cls_pred = model.predict([X_cat, X_num], verbose=0)

        # 6. 计算指标
        height_mae = mean_absolute_error(y_reg[:, 0], reg_pred[:, 0])
        weight_mae = mean_absolute_error(y_reg[:, 1], reg_pred[:, 1])
        cls_accuracy = accuracy_score(y_cls, np.argmax(cls_pred, axis=1))

        # 7. 输出结果
        logging.info("\n📊 评估结果:")
        logging.info(f"  - 身高MAE: {height_mae:.2f} cm")
        logging.info(f"  - 体重MAE: {weight_mae:.2f} kg")
        logging.info(f"  - 分类准确率: {cls_accuracy:.2%}")

        # 8. 生成可视化
        logging.info("🖌 生成可视化图表...")
        generate_visualizations(y_reg, reg_pred, y_cls, cls_pred, font_prop)

    except Exception as e:
        logging.error("‼️ 评估流程失败", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    evaluate()
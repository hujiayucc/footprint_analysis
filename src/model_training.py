# src/model_training.py
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ================== 配置区 ==================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "footprint_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================== 核心函数定义 ==================
def load_data():
    """加载数据集并验证完整性"""
    try:
        logging.info(f"🔍 正在加载数据: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        required_columns = {
            "foot_length", "foot_width", "arch_height", 
            "pressure_avg", "ground_type", "humidity",
            "depth", "depth_diff", "pressure_offset_x",
            "height", "weight", "leg_type"
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"缺失关键字段: {missing}")
            
        logging.info(f"✅ 数据加载成功 | 样本数: {len(df)}")
        return df
    except Exception as e:
        logging.error("‼️ 数据加载失败", exc_info=True)
        sys.exit(1)

def build_preprocessor():
    """构建预处理管道"""
    try:
        logging.info("🛠️ 构建预处理管道...")
        return ColumnTransformer([
            ('cat', OneHotEncoder(), ['ground_type']),  # 生成5列
            ('num', StandardScaler(), [
                'foot_length', 'foot_width', 'arch_height',
                'pressure_avg', 'humidity', 'depth',
                'depth_diff', 'pressure_offset_x'
            ])  # 8个数值特征
        ])
    except Exception as e:
        logging.error("‼️ 预处理器构建失败", exc_info=True)
        sys.exit(1)

# ================== 动态模型配置 ==================
def calculate_model_capacity(data_size):
    """根据数据量动态计算模型参数"""
    base_units = 64
    scaling_factor = min(1.0, np.log10(data_size / 5000 + 1))  # 计算缩放因子
    
    return {
        "scaling_factor": scaling_factor,  # 必须包含此项
        "hidden_units": int(base_units * (1 + scaling_factor)),
        "num_layers": max(1, int(2 * scaling_factor)),
        "dropout_rate": max(0.1, 0.5 - 0.1 * scaling_factor)
    }

# ================== 优化后的模型构建 ==================
def build_dynamic_model(input_shape_cat, input_shape_num, model_cfg):
    """构建动态调整的神经网络模型"""
    try:
        logging.info("🧠 构建动态模型...")
        
        # 输入层
        input_cat = tf.keras.Input(shape=input_shape_cat, name="categorical_input")
        input_num = tf.keras.Input(shape=input_shape_num, name="numerical_input")
        
        # 特征融合
        merged = tf.keras.layers.concatenate([input_cat, input_num])
        
        # 动态隐藏层
        x = merged
        for _ in range(model_cfg["num_layers"]):
            x = tf.keras.layers.Dense(model_cfg["hidden_units"], activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(model_cfg["dropout_rate"])(x)
        
        # 多任务输出
        reg_output = tf.keras.layers.Dense(2, name="regression")(x)
        cls_output = tf.keras.layers.Dense(3, activation="softmax", name="classification")(x)
        
        model = tf.keras.Model(
            inputs=[input_cat, input_num],
            outputs=[reg_output, cls_output]
        )
        
        # 混合精度优化
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "regression": "mse",
                "classification": "sparse_categorical_crossentropy"
            },
            metrics={
                "regression": ["mae"],
                "classification": ["accuracy"]
            }
        )
        logging.info(f"✅ 动态模型构建成功 | 隐藏层: {model_cfg['num_layers']}x{model_cfg['hidden_units']}")
        return model
    except Exception as e:
        logging.error("‼️ 模型构建失败", exc_info=True)
        sys.exit(1)

# ================== 训练流程优化 ==================
def train_model():
    """训练主流程"""
    try:
        # ================== 数据准备阶段 ==================
        # 加载数据
        df = load_data()
        data_size = len(df)
        
        # 动态计算模型参数
        model_cfg = calculate_model_capacity(data_size)
        logging.info(f"📊 模型动态配置 | 隐藏层数: {model_cfg['num_layers']} | 神经元数: {model_cfg['hidden_units']} | 缩放因子: {model_cfg['scaling_factor']:.2f}")

        # 构建预处理器
        preprocessor = build_preprocessor()
        X = df.drop(["height", "weight", "leg_type"], axis=1)
        y_reg = df[["height", "weight"]].values.astype(np.float32)
        y_cls = df["leg_type"].values.astype(np.int32)

        # ================== 数据分割 ==================
        X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
            X, y_reg, y_cls, 
            test_size=0.2, 
            stratify=df["leg_type"],
            random_state=42
        )

        # ================== 特征工程 ==================
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # 输入维度验证
        input_shape_cat = (X_train_processed[:, :5].shape[1],)
        input_shape_num = (X_train_processed[:, 5:].shape[1],)
        if input_shape_cat[0] != 5 or input_shape_num[0] != 8:
            raise ValueError(f"输入维度异常 | 预期: 分类5维+数值8维，实际: {input_shape_cat}+{input_shape_num}")

        # ================== 模型构建 ==================
        strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()
        with strategy.scope():
            model = build_dynamic_model(input_shape_cat, input_shape_num, model_cfg)

        # ================== 训练参数 ==================
        batch_size = min(256, int(32 * model_cfg["scaling_factor"]))
        epochs = int(50 * (1 + model_cfg["scaling_factor"]))
        logging.info(f"⚙️ 训练参数 | 批次大小: {batch_size} | 训练轮次: {epochs}")

        # ================== 训练配置 ==================
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=max(5, int(5 * model_cfg["scaling_factor"])),
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, "footprint_model.h5"),
                monitor="val_classification_accuracy",
                mode="max",
                save_best_only=True,
                save_format="h5"
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(MODEL_DIR, "training_log.csv")
            )
        ]

        # ================== 开始训练 ==================
        history = model.fit(
            [X_train_processed[:, :5].astype(np.float32), X_train_processed[:, 5:].astype(np.float32)],
            [y_reg_train, y_cls_train],
            validation_data=(
                [X_val_processed[:, :5].astype(np.float32), X_val_processed[:, 5:].astype(np.float32)],
                [y_reg_val, y_cls_val]
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        )

        # ================== 保存结果 ==================
        # 分别保存预处理器和元数据
        joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))  # 单独保存预处理器
        joblib.dump(
            {
                "model_config": model_cfg,
                "training_history": history.history,
                "input_shapes": {
                    "categorical": input_shape_cat,
                    "numerical": input_shape_num
                }
            },
            os.path.join(MODEL_DIR, "training_metadata.pkl")  # 元数据另存新文件
        )

        # 模型量化压缩
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(os.path.join(MODEL_DIR, "footprint_model.tflite"), "wb") as f:
            f.write(tflite_model)

        # ================== 训练报告 ==================
        best_epoch = np.argmax(history.history["val_classification_accuracy"])
        logging.info("🏆 最终训练报告:")
        logging.info(f"最佳验证准确率: {history.history['val_classification_accuracy'][best_epoch]:.2%}")
        logging.info(f"回归任务MAE: {history.history['regression_mae'][best_epoch]:.2f}")
        logging.info(f"模型文件: {os.path.join(MODEL_DIR, 'footprint_model.h5')}")
        logging.info(f"预处理器文件: {os.path.join(MODEL_DIR, 'preprocessor.pkl')}")
        logging.info(f"训练元数据: {os.path.join(MODEL_DIR, 'training_metadata.pkl')}")

    except Exception as e:
        logging.error("‼️ 训练流程异常终止", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    train_model()

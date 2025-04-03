# src/app.py
import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# 初始化应用
app = Flask(__name__)

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建项目根目录路径
project_root = os.path.dirname(current_dir)
# 模型文件路径
model_path = os.path.join(project_root, "models", "footprint_model.h5")
preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")

# 加载模型和预处理器
model = tf.keras.models.load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求数据
        data = request.json
        
        # 构建DataFrame
        input_df = pd.DataFrame([data])
        
        # 预处理
        processed = preprocessor.transform(input_df)
        
        # 分割特征
        X_cat = processed[:, :5].astype(np.float32)
        X_num = processed[:, 5:].astype(np.float32)
        
        # 预测
        reg_pred, cls_pred = model.predict([X_cat, X_num])
        
        # 构造响应
        return jsonify({
            "height": float(reg_pred[0][0]),
            "weight": float(reg_pred[0][1]),
            "leg_type_probs": cls_pred[0].tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
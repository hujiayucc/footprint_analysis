# app.py
import os
from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

# 初始化应用
app = Flask(__name__)

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 模型文件路径
model_path = os.path.join(current_dir, "models", "footprint_model.keras")
preprocessor_path = os.path.join(current_dir, "models", "preprocessor.pkl")

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

@app.route('/')
def home_page():
    api_url = url_for('predict', _external=True)
    return render_template('index.html', api_url=api_url)

if __name__ == '__main__':
    debug = True
    host = '0.0.0.0' if debug else '127.0.0.1'
    app.run(debug=debug, host=host, port=5000)
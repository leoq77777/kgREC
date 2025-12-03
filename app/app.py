from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys
import logging
from pathlib import Path
import dotenv

dotenv.load_dotenv()  # 加载环境变量

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from models.kg_inference_engine import KnowledgeGraphInferenceEngine

app = Flask(__name__)
CORS(app)  # 启用跨域资源共享

# 配置日志
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# 初始化推理引擎
inference_engine = None

def init_engine():
    global inference_engine
    if not inference_engine:
        app.logger.info("初始化知识图谱推理引擎...")
        inference_engine = KnowledgeGraphInferenceEngine(
            embeddings_dir=os.getenv('EMBEDDINGS_DIR'),
            index_dir=os.getenv('INDEX_DIR')
        )
        inference_engine.load_graph(os.getenv('KG_GRAPH_PATH'))

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "kg-recommendation-api"})

@app.route('/recommend/<user_id>', methods=['GET'])
def recommend_items(user_id):
    try:
        init_engine()
        recommendations = inference_engine.recommend_items(user_id, top_k=10)
        return jsonify({
            "user_id": user_id,
            "recommendations": [{
                "item_id": item,
                "score": float(score)
            } for item, score in recommendations]
        })
    except Exception as e:
        app.logger.error(f"推荐服务错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
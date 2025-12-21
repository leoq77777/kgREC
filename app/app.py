from flask import Flask, jsonify, request
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
CORS(app, resources={r"/*": {"origins": "*"}})  # 启用跨域资源共享

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

@app.route('/user-recommend/<user_id>', methods=['GET'])
def recommend_items(user_id):
    print(f"Received request for user_id: {user_id}")
    try:
        app.logger.info(f"接收到推荐请求 - 用户ID: {user_id}, 请求方法: {request.method}, 请求参数: {request.args}")
        init_engine()
        recommendations = inference_engine.recommend_items(user_id, top_k=10)
        app.logger.info(f"推荐成功 - 用户ID: {user_id}, 推荐数量: {len(recommendations)}")
        return jsonify({
            "user_id": user_id,
            "recommendations": [{
                "item_id": item,
                "score": float(score)
            } for item, score in recommendations]
        })
    except Exception as e:
        app.logger.error(f"推荐接口异常: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    try:
        init_engine()
        item_id = int(request.args.get('item_id'))
        top_k = int(request.args.get('top_k', 10))

        # 获取物品嵌入
        item_embedding = inference_engine.get_item_embedding(item_id)
        if item_embedding is None:
            return jsonify({'error': 'Item not found'}), 404

        # 搜索相似物品
        distances, indices = inference_engine.search_similar_items(item_embedding, top_k)

        return jsonify({
            'item_id': item_id,
            'recommendations': [{
                'item_index': int(idx),
                'distance': float(dist)
            } for idx, dist in zip(indices, distances)]
        })
    except Exception as e:
        app.logger.error(f"推荐服务错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
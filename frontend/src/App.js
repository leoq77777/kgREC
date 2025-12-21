import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        // 调用后端推荐API (使用用户ID=1作为示例)
        const response = await fetch('http://localhost:5000/user-recommend/1');
        console.log('Response status:', response.status);
        if (!response.ok) {
          const errorDetails = await response.text();
          throw new Error(`Failed to fetch recommendations: ${response.status} - ${errorDetails}`);
        }
        const data = await response.json();
        console.log('Received recommendations:', data);
        setRecommendations(data.recommendations);
      } catch (err) {
        console.error('Fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>电影推荐系统</h1>
        {loading ? (
          <p>加载推荐中...</p>
        ) : error ? (
          <p className="error">错误: {error}</p>
        ) : (
          <div className="recommendations">
            <h2>为您推荐</h2>
            <ul>
              {recommendations.map((item, index) => (
                <li key={index} className="recommendation-item">
                  <span>电影ID: {item.item_id}</span>
                  <span>推荐分数: {item.score.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;

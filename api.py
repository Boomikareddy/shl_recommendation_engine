from flask import Flask, request, jsonify
from recommendation import get_recommendations  # Now this works!

app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        results = get_recommendations(query)
        return jsonify({'query': query, 'recommendations': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

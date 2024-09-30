# app.py
from flask import Flask, request, jsonify
from recommendation_system import RecommendationSystem
import joblib
import os

app = Flask(__name__)

rs = RecommendationSystem(data={}) 
model_filepath = 'recommendation_model.joblib'

if os.path.exists(model_filepath):
    rs.load_model(model_filepath)
else:
    raise FileNotFoundError(f"Model file {model_filepath} not found. Please train the model first.")

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API endpoint to get question recommendations for a user.
    Example: /recommend?user_id=1&n=5
    """
    try:
        user_id = int(request.args.get('user_id', 1))
        num_recs = int(request.args.get('n', 5))
    except ValueError:
        return jsonify({"error": "Invalid input parameters."}), 400

    recommended_questions = rs.recommend_questions(user_id, N=num_recs)
    precision, recall = rs.evaluate_recommendations(user_id, recommended_questions)
    question_details = rs.get_question_details(user_id, recommended_questions)

    details = [
        {
            "question_id": qd.question_id,
            "topic": qd.topic,
            "difficulty_level": qd.difficulty_level,
            "avg_score": qd.avg_score,
            "avg_time": qd.avg_time,
            "similar_users": qd.similar_users
        }
        for qd in question_details
    ]

    response = {
        "user_id": user_id,
        "recommended_questions": recommended_questions,
        "precision": precision,
        "recall": recall,
        "question_details": details
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

# train_model.py
from recommendation_system import RecommendationSystem

def train_and_save_model():
    data = {
        'student_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'question_id': [101, 102, 101, 103, 102, 104, 105, 106],
        'score': [0.8, 0.9, 0.7, 0.6, 0.85, 0.9, 0.75, 0.8],
        'answer_time': [5, 7, 6, 8, 5, 6, 7, 5],
        'difficulty_level': [0.5, 0.6, 0.5, 0.7, 0.6, 0.8, 0.65, 0.7],
        'course': ['Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 
                   'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics'],
        'topic': ['Algebra', 'Geometry', 'Algebra', 'Calculus', 
                  'Geometry', 'Statistics', 'Trigonometry', 'Probability']
    }

    rs = RecommendationSystem(data)

    course_name = 'Mathematics'
    rs.train(course_name)

    model_filepath = 'recommendation_model.joblib'
    rs.save_model(model_filepath)

if __name__ == "__main__":
    train_and_save_model()

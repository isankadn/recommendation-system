import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuestionDetails:
    question_id: int
    avg_score: float
    avg_time: float
    similar_users: List[int]
    difficulty_level: float
    course: str
    topic: str

class RecommendationSystem:
    def __init__(self, data: Dict[str, List], score_weight: float = 0.7, time_weight: float = 0.3):
        self.df = pd.DataFrame(data)
        self.score_weight = score_weight
        self.time_weight = time_weight
        self.user_item_matrices = {}
        self.user_similarities = {}
        self.svd_models = {}
        self.course = None

    def _preprocess_data(self, course: str):
        course_df = self.df[self.df['course'] == course].copy()

        scaler = MinMaxScaler()
        course_df['normalized_time'] = 1 - scaler.fit_transform(course_df[['answer_time']])
        course_df['combined_score'] = (
            self.score_weight * course_df['score'] + 
            self.time_weight * course_df['normalized_time']
        )
        
        self.df.loc[self.df['course'] == course, 'combined_score'] = course_df['combined_score']
        
        user_item_matrix = course_df.pivot(
            index='student_id', 
            columns='question_id', 
            values='combined_score'
        ).fillna(0)

        sparse_matrix = csr_matrix(user_item_matrix.values)

        n_components = min(50, sparse_matrix.shape[1] - 1)
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_features = svd_model.fit_transform(sparse_matrix)

        user_similarity = cosine_similarity(user_features)

        self.user_item_matrices[course] = user_item_matrix
        self.user_similarities[course] = user_similarity
        self.svd_models[course] = svd_model

    def train(self, course: str):
        logger.info(f"Training the recommendation system for {course}...")
        self._preprocess_data(course)
        self.course = course
        logger.info("Training completed.")

    @lru_cache(maxsize=128)
    def get_similar_users(self, user_id: int, N: int = 5) -> List[int]:
        if user_id not in self.user_item_matrices[self.course].index:
            return []
        user_index = self.user_item_matrices[self.course].index.get_loc(user_id)
        similar_indices = self.user_similarities[self.course][user_index].argsort()[::-1][1:N+1]
        return self.user_item_matrices[self.course].index[similar_indices].tolist()

    def recommend_questions(self, user_id: int, N: int = 5) -> List[int]:
        if user_id not in self.user_item_matrices[self.course].index:
            return []

        similar_users = self.get_similar_users(user_id, N)
        user_questions = set(self.user_item_matrices[self.course].loc[user_id][self.user_item_matrices[self.course].loc[user_id] > 0].index)
        recommendations = set()

        for similar_user in similar_users:
            similar_user_questions = set(self.user_item_matrices[self.course].loc[similar_user][self.user_item_matrices[self.course].loc[similar_user] > 0].index)
            recommendations.update(similar_user_questions - user_questions)

        sorted_recommendations = sorted(
            recommendations,
            key=lambda q: self._predict_score(user_id, q),
            reverse=True
        )

        return sorted_recommendations[:N]

    def _predict_score(self, user_id: int, question_id: int) -> float:
        similar_users = self.get_similar_users(user_id)
        course_df = self.df[self.df['course'] == self.course]
        similar_scores = course_df[
            (course_df['student_id'].isin(similar_users)) & 
            (course_df['question_id'] == question_id)
        ]['combined_score']
        return similar_scores.mean() if not similar_scores.empty else 0

    def evaluate_recommendations(self, user_id: int, recommended_questions: List[int]) -> Tuple[float, float]:
        actual_questions = set(self.df[(self.df['student_id'] == user_id) & (self.df['course'] == self.course)]['question_id'])
        recommended_questions = set(recommended_questions)
        correct_recommendations = actual_questions.intersection(recommended_questions)

        precision = len(correct_recommendations) / len(recommended_questions) if recommended_questions else 0
        recall = len(correct_recommendations) / len(actual_questions) if actual_questions else 0

        return precision, recall

    def get_question_details(self, user_id: int, recommended_questions: List[int]) -> List[QuestionDetails]:
        course_df = self.df[self.df['course'] == self.course]
        recommended_data = course_df[course_df['question_id'].isin(recommended_questions)]

        question_details = []
        for question in recommended_questions:
            question_data = recommended_data[recommended_data['question_id'] == question]
            similar_users = question_data['student_id'].tolist()
            avg_score = question_data['score'].mean()
            avg_time = question_data['answer_time'].mean()
            difficulty_level = question_data['difficulty_level'].mean()
            topic = question_data['topic'].mode().iloc[0] if not question_data['topic'].empty else ""

            question_details.append(QuestionDetails(
                question_id=question,
                avg_score=avg_score,
                avg_time=avg_time,
                similar_users=similar_users,
                difficulty_level=difficulty_level,
                course=self.course,
                topic=topic
            ))

        return question_details

def main():
    # Sample data
    data = {
        'student_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'question_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3, 4],
        'score': [0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.7, 0.8, 0.7, 0.8, 0.9],
        'answer_time': [5, 8, 6, 10, 7, 9, 6, 8, 7, 8, 7, 5],
        'difficulty_level': [0.3, 0.5, 0.7, 0.3, 0.5, 0.6, 0.5, 0.7, 0.6, 0.3, 0.7, 0.6],
        'course': ['Mathematics', 'English', 'Mathematics', 'Mathematics', 'English', 'Mathematics', 'English', 'Mathematics', 'English', 'Mathematics', 'English', 'Mathematics'],
        'topic': ['Algebra', 'Short Stories', 'Geometry', 'Algebra', 'Novels', 'Trigonometry', 'Grammar', 'Statistics', 'Poetry', 'Calculus', 'Essays', 'Probability']
    }

    rs = RecommendationSystem(data)

    for course in ['Mathematics', 'English']:
        rs.train(course)
        user_id = 1
        recommended_questions = rs.recommend_questions(user_id)

        logger.info(f"Recommended {course} questions for user {user_id}: {recommended_questions}")

        precision, recall = rs.evaluate_recommendations(user_id, recommended_questions)
        logger.info(f"Precision: {precision:.2f}")
        logger.info(f"Recall: {recall:.2f}")

        logger.info(f"Details of recommended {course} questions:")
        question_details = rs.get_question_details(user_id, recommended_questions)
        for details in question_details:
            logger.info(f"Question {details.question_id}:")
            logger.info(f"  Topic: {details.topic}")
            logger.info(f"  Difficulty level: {details.difficulty_level:.2f}")
            logger.info(f"  Average score of similar users: {details.avg_score:.2f}")
            logger.info(f"  Average answer time of similar users: {details.avg_time:.2f} minutes")
            logger.info(f"  Answered by users: {details.similar_users}")

if __name__ == "__main__":
    main()
# recommendation_system.py
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
import joblib  
import os


logging.basicConfig(level=logging.INFO)
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
        """
        Initializes the Recommendation System with the provided data.
        """
        self.df = pd.DataFrame(data)
        self.score_weight = score_weight
        self.time_weight = time_weight
        self.user_item_matrices = {}
        self.user_similarities = {}
        self.svd_models = {}
        self.course = None

    def _preprocess_data(self, course: str):
        """
        Preprocesses data for the given course.
        """
        course_df = self.df[self.df['course'] == course].copy()

        scaler = MinMaxScaler()
        course_df['normalized_time'] = 1 - scaler.fit_transform(course_df[['answer_time']])

        course_df['combined_score'] = (
            self.score_weight * course_df['score'] +
            self.time_weight * course_df['normalized_time']
        )

        self.df.loc[self.df['course'] == course, 'combined_score'] = course_df['combined_score']


        user_item_matrix = course_df.pivot_table(
            index='student_id',
            columns='question_id',
            values='combined_score',
            fill_value=0
        )

        sparse_matrix = csr_matrix(user_item_matrix.values)


        n_components = min(50, sparse_matrix.shape[1] - 1) or 1
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_features = svd_model.fit_transform(sparse_matrix)

        user_similarity = cosine_similarity(user_features)

        self.user_item_matrices[course] = user_item_matrix
        self.user_similarities[course] = user_similarity
        self.svd_models[course] = svd_model

    def train(self, course: str):
        """
        Trains the recommendation system for a specific course.
        """
        logger.info(f"Training the recommendation system for {course}...")
        self._preprocess_data(course)
        self.course = course
        logger.info("Training completed.")

    @lru_cache(maxsize=128)
    def get_similar_users(self, user_id: int, N: int = 5) -> List[int]:
        """
        Retrieves a list of similar users to the given user.
        """
        try:
            user_index = self.user_item_matrices[self.course].index.get_loc(user_id)
        except KeyError:
            logger.warning(f"User ID {user_id} not found in course {self.course}.")
            return []

        similarity_scores = self.user_similarities[self.course][user_index]
        similar_indices = np.argsort(similarity_scores)[::-1][1:N+1]
        similar_users = self.user_item_matrices[self.course].index[similar_indices].tolist()
        return similar_users

    def recommend_questions(self, user_id: int, N: int = 5) -> List[int]:
        """
        Recommends questions to the user based on similar users' interactions.
        """
        if user_id not in self.user_item_matrices[self.course].index:
            logger.warning(f"User ID {user_id} not found in course {self.course}. No recommendations can be made.")
            return []

        similar_users = self.get_similar_users(user_id, N)
        user_questions = set(self.user_item_matrices[self.course].loc[user_id][
            self.user_item_matrices[self.course].loc[user_id] > 0
        ].index)

        recommendations = set()

        for similar_user in similar_users:
            similar_user_questions = set(self.user_item_matrices[self.course].loc[similar_user][
                self.user_item_matrices[self.course].loc[similar_user] > 0
            ].index)
            recommendations.update(similar_user_questions - user_questions)

        sorted_recommendations = sorted(
            recommendations,
            key=lambda q: self._predict_score(user_id, q),
            reverse=True
        )

        return sorted_recommendations[:N]

    def _predict_score(self, user_id: int, question_id: int) -> float:
        """
        Predicts the score a user might give to a question.
        """
        similar_users = self.get_similar_users(user_id)
        course_df = self.df[self.df['course'] == self.course]
        similar_scores = course_df[
            (course_df['student_id'].isin(similar_users)) &
            (course_df['question_id'] == question_id)
        ]['combined_score']
        return similar_scores.mean() if not similar_scores.empty else 0

    def evaluate_recommendations(self, user_id: int, recommended_questions: List[int]) -> Tuple[float, float]:
        """
        Evaluates the recommendations using precision and recall metrics.
        """
        actual_questions = set(self.df[
            (self.df['student_id'] == user_id) & (self.df['course'] == self.course)
        ]['question_id'])
        recommended_questions_set = set(recommended_questions)
        correct_recommendations = actual_questions.intersection(recommended_questions_set)

        precision = len(correct_recommendations) / len(recommended_questions_set) if recommended_questions_set else 0
        recall = len(correct_recommendations) / len(actual_questions) if actual_questions else 0

        return precision, recall

    def get_question_details(self, user_id: int, recommended_questions: List[int]) -> List[QuestionDetails]:
        """
        Retrieves detailed information for the recommended questions.
        """
        course_df = self.df[self.df['course'] == self.course]
        recommended_data = course_df[course_df['question_id'].isin(recommended_questions)]

        question_details_list = []
        for question_id in recommended_questions:
            question_data = recommended_data[recommended_data['question_id'] == question_id]
            similar_users = question_data['student_id'].tolist()
            avg_score = question_data['score'].mean()
            avg_time = question_data['answer_time'].mean()
            difficulty_level = question_data['difficulty_level'].mean()
            topic_series = question_data['topic']
            topic = topic_series.mode().iloc[0] if not topic_series.empty else "Unknown"

            question_details = QuestionDetails(
                question_id=question_id,
                avg_score=avg_score,
                avg_time=avg_time,
                similar_users=similar_users,
                difficulty_level=difficulty_level,
                course=self.course,
                topic=topic
            )
            question_details_list.append(question_details)

        return question_details_list

    def save_model(self, filepath: str):
        """
        Saves the trained models and data structures to a binary file.
        """
        logger.info(f"Saving the model to {filepath}...")
        model_data = {
            'df': self.df,
            'score_weight': self.score_weight,
            'time_weight': self.time_weight,
            'user_item_matrices': self.user_item_matrices,
            'user_similarities': self.user_similarities,
            'svd_models': self.svd_models,
            'course': self.course
        }
        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully.")

    def load_model(self, filepath: str):
        """
        Loads the trained models and data structures from a binary file.
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} does not exist.")
            raise FileNotFoundError(f"Model file {filepath} not found.")

        logger.info(f"Loading the model from {filepath}...")
        model_data = joblib.load(filepath)
        self.df = model_data['df']
        self.score_weight = model_data['score_weight']
        self.time_weight = model_data['time_weight']
        self.user_item_matrices = model_data['user_item_matrices']
        self.user_similarities = model_data['user_similarities']
        self.svd_models = model_data['svd_models']
        self.course = model_data['course']
        logger.info("Model loaded successfully.")

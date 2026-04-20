"""
Hybrid Recommendation Engine for E-Learning System
Combines Content-Based Filtering and Collaborative Filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os


class ContentBasedRecommender:
    
    def __init__(self, courses):
        """
        Initialize with course data.
        
        Args:
            courses: DataFrame with course metadata
        """
        self.courses = courses
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.course_indices = None
        self.vectorizer = None
    
    def fit(self):
        """
        Build the TF-IDF matrix and compute cosine similarities.
        
        TF-IDF (Term Frequency-Inverse Document Frequency):
        - TF: How often a term appears in a document
        - IDF: How important a term is across all documents
        - TF-IDF = TF × IDF
        
        Cosine Similarity:
        - Measures the cosine of the angle between two vectors
        - Range: 0 (orthogonal) to 1 (identical direction)
        - Formula: cos(θ) = (A·B) / (||A|| × ||B||)
        """
        print("\n[Content-Based Filtering] Building TF-IDF matrix...")
        
        # Create content field by combining relevant text features
        self.courses['content'] = (
            self.courses['title'] + ' ' +
            self.courses['category'] + ' ' +
            self.courses['subcategory'] + ' ' +
            self.courses['description'] + ' ' +
            self.courses['tags'] + ' ' +
            self.courses['difficulty']
        )
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=5000
        )
        
        # Fit and transform the content
        self.tfidf_matrix = self.vectorizer.fit_transform(self.courses['content'])
        
        print(f"  ✓ TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
        print(f"  ✓ Vocabulary Size: {len(self.vectorizer.vocabulary_)}")
        
        # Compute cosine similarity matrix
        print("[Content-Based Filtering] Computing cosine similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create reverse mapping of course_id to index
        self.course_indices = pd.Series(
            self.courses.index,
            index=self.courses['course_id']
        )
        
        print(f"  ✓ Cosine Similarity Matrix: {self.cosine_sim.shape}")
        
        return self
    
    def get_similar_courses(self, course_id, n=10):
        """
        Get top N similar courses to a given course.
        
        Args:
            course_id: ID of the reference course
            n: Number of recommendations
            
        Returns:
            list: List of (course_id, similarity_score) tuples
        """
        if course_id not in self.course_indices:
            return []
        
        # Get the index of the course
        idx = self.course_indices[course_id]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort by similarity (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N (excluding itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get course IDs and scores
        recommendations = [
            (self.courses.iloc[i[0]]['course_id'], i[1])
            for i in sim_scores
        ]
        
        return recommendations
    
    def recommend_for_user(self, user_courses, n=10, exclude_completed=True):
        """
        Recommend courses for a user based on courses they've engaged with.
        
        Args:
            user_courses: List of course_ids the user has interacted with
            n: Number of recommendations
            exclude_completed: Whether to exclude already taken courses
            
        Returns:
            list: List of recommended (course_id, score) tuples
        """
        if not user_courses:
            return []
        
        # Aggregate similarity scores for all user's courses
        sim_scores = defaultdict(float)
        
        for course_id in user_courses:
            if course_id in self.course_indices:
                idx = self.course_indices[course_id]
                for i, score in enumerate(self.cosine_sim[idx]):
                    other_course_id = self.courses.iloc[i]['course_id']
                    if exclude_completed and other_course_id in user_courses:
                        continue
                    sim_scores[other_course_id] += score
        
        # Sort by aggregated score
        recommendations = sorted(
            sim_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return recommendations


class CollaborativeRecommender:
    """
    Collaborative Filtering using User-Item Matrix and Similarity.
    Recommends courses based on similar users' preferences.
    """
    
    def __init__(self, ratings):
        """
        Initialize with ratings data.
        
        Args:
            ratings: DataFrame with user-course ratings
        """
        self.ratings = ratings
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
    
    def fit(self):
        """
        Build the User-Item matrix and compute similarities.
        
        User-Item Matrix:
        - Rows: Users
        - Columns: Items (Courses)
        - Values: Ratings (0 if not rated)
        
        Similarity Calculation:
        - User-User: Find users with similar rating patterns
        - Item-Item: Find items rated similarly by users
        """
        print("\n[Collaborative Filtering] Building User-Item matrix...")
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index='student_id',
            columns='course_id',
            values='rating',
            fill_value=0
        )
        
        print(f"  ✓ User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"  ✓ Sparsity: {(self.user_item_matrix == 0).sum().sum() / self.user_item_matrix.size:.2%}")
        
        # Compute user-user similarity
        print("[Collaborative Filtering] Computing user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Compute item-item similarity
        print("[Collaborative Filtering] Computing item similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print(f"  ✓ User Similarity Matrix: {self.user_similarity.shape}")
        print(f"  ✓ Item Similarity Matrix: {self.item_similarity.shape}")
        
        return self
    
    def get_similar_users(self, student_id, n=5):
        """
        Find top N similar users.
        
        Args:
            student_id: ID of the target user
            n: Number of similar users to find
            
        Returns:
            list: List of (user_id, similarity_score) tuples
        """
        if student_id not in self.user_similarity.index:
            return []
        
        similarities = self.user_similarity[student_id].sort_values(ascending=False)
        # Exclude self
        return list(similarities[1:n+1].items())
    
    def predict_rating(self, student_id, course_id):
        """
        Predict rating for a user-course pair using user-based CF.
        
        Formula:
        predicted_rating = Σ(similarity × rating) / Σ(|similarity|)
        
        Args:
            student_id: Target user ID
            course_id: Target course ID
            
        Returns:
            float: Predicted rating
        """
        if student_id not in self.user_similarity.index:
            return 0
        if course_id not in self.user_item_matrix.columns:
            return 0
        
        # Get similar users who rated this course
        similar_users = self.user_similarity[student_id]
        course_ratings = self.user_item_matrix[course_id]
        
        # Filter to users who have rated the course
        rated_mask = course_ratings > 0
        
        if rated_mask.sum() == 0:
            return 0
        
        # Weighted average
        numerator = (similar_users[rated_mask] * course_ratings[rated_mask]).sum()
        denominator = similar_users[rated_mask].abs().sum()
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def recommend_for_user(self, student_id, n=10):
        """
        Recommend top N courses for a user.
        
        Args:
            student_id: Target user ID
            n: Number of recommendations
            
        Returns:
            list: List of (course_id, predicted_rating) tuples
        """
        if student_id not in self.user_item_matrix.index:
            return []
        
        # Get courses user hasn't rated
        user_ratings = self.user_item_matrix.loc[student_id]
        unrated_courses = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings for unrated courses
        predictions = []
        for course_id in unrated_courses:
            pred = self.predict_rating(student_id, course_id)
            if pred > 0:
                predictions.append((course_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


class HybridRecommender:
    """
    Hybrid Recommendation System combining Content-Based and Collaborative Filtering.
    
    Hybrid Approach:
    - Weighted combination of both methods
    - Addresses limitations of each individual method
    - Provides more robust recommendations
    """
    
    def __init__(self, courses, ratings, activities,
                 enrollments=None, quizzes=None,
                 content_weight=0.4, collaborative_weight=0.6):
        """
        Initialize with all required data.
        
        Args:
            courses: Course metadata DataFrame
            ratings: User ratings DataFrame
            activities: Learning activity logs DataFrame
            enrollments: Optional enrollment history DataFrame
            quizzes: Optional quiz results DataFrame
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative recommendations
        """
        self.courses = courses
        self.ratings = ratings
        self.activities = activities
        self.enrollments = enrollments if enrollments is not None else pd.DataFrame()
        self.quizzes = quizzes if quizzes is not None else pd.DataFrame()
        
        self.content_recommender = ContentBasedRecommender(courses)
        self.collaborative_recommender = CollaborativeRecommender(ratings)
        
        # Hybrid weights
        total_weight = content_weight + collaborative_weight
        if total_weight <= 0:
            raise ValueError("Hybrid weights must sum to a positive value.")
        self.content_weight = content_weight / total_weight
        self.collaborative_weight = collaborative_weight / total_weight
    
    def fit(self):
        """Train both recommendation models."""
        print("\n" + "=" * 70)
        print("  HYBRID RECOMMENDATION SYSTEM - MODEL TRAINING")
        print("=" * 70)
        
        self.content_recommender.fit()
        self.collaborative_recommender.fit()
        
        print("\n" + "=" * 70)
        print("  MODEL TRAINING COMPLETED")
        print("=" * 70)
        
        return self
    
    def get_user_courses(self, student_id):
        """Get list of courses a user has interacted with or already studied."""
        seen_courses = set(
            self.activities[
                self.activities['student_id'] == student_id
            ]['course_id'].unique().tolist()
        )
        
        if not self.enrollments.empty:
            enrollment_mask = (
                (self.enrollments['student_id'] == student_id) &
                (self.enrollments['status'].isin(['active', 'completed']))
            )
            seen_courses.update(
                self.enrollments[enrollment_mask]['course_id'].unique().tolist()
            )
        
        if not self.quizzes.empty:
            seen_courses.update(
                self.quizzes[
                    self.quizzes['student_id'] == student_id
                ]['course_id'].unique().tolist()
            )
        
        return list(seen_courses)
    
    def recommend(self, student_id, n=10, difficulty_filter=None):
        """
        Generate hybrid recommendations for a user.
        
        Args:
            student_id: Target user ID
            n: Number of recommendations
            difficulty_filter: Optional difficulty level to filter by
            
        Returns:
            DataFrame: Recommended courses with scores
        """
        user_courses = self.get_user_courses(student_id)
        
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend_for_user(
            user_courses, n=n*2
        )
        
        # Get collaborative recommendations
        collab_recs = self.collaborative_recommender.recommend_for_user(
            student_id, n=n*2
        )
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        # Normalize and add content-based scores
        if content_recs:
            max_content_score = max(r[1] for r in content_recs)
            for course_id, score in content_recs:
                normalized_score = score / max_content_score if max_content_score > 0 else 0
                combined_scores[course_id] += self.content_weight * normalized_score
        
        # Normalize and add collaborative scores
        if collab_recs:
            max_collab_score = max(r[1] for r in collab_recs) if collab_recs else 1
            for course_id, score in collab_recs:
                normalized_score = score / max_collab_score if max_collab_score > 0 else 0
                combined_scores[course_id] += self.collaborative_weight * normalized_score
        
        # Remove anything the learner has already seen/studied before ranking.
        recommendations = [
            (course_id, score)
            for course_id, score in combined_scores.items()
            if course_id not in user_courses
        ]
        
        # Sort by combined score
        recommendations = sorted(
            recommendations,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by difficulty if specified
        if difficulty_filter:
            difficulty_courses = self.courses[
                self.courses['difficulty'] == difficulty_filter
            ]['course_id'].tolist()
            recommendations = [
                r for r in recommendations 
                if r[0] in difficulty_courses
            ]
        
        # Get top N
        top_recommendations = recommendations[:n]
        
        # Create result DataFrame with course details
        if not top_recommendations:
            return pd.DataFrame()
        
        result = pd.DataFrame(top_recommendations, columns=['course_id', 'hybrid_score'])
        result = result.merge(
            self.courses[['course_id', 'title', 'category', 'difficulty', 
                         'duration_hours', 'rating']],
            on='course_id'
        )
        
        return result
    
    def build_learning_path(self, student_id, target_category, n_courses=5):
        """
        Build a personalized learning path for a student.
        Orders courses from beginner to advanced within a category.
        
        Args:
            student_id: Target student ID
            target_category: Category to build path for
            n_courses: Number of courses in the path
            
        Returns:
            DataFrame: Ordered learning path
        """
        # Get recommendations filtered by category
        category_courses = self.courses[
            self.courses['category'] == target_category
        ]['course_id'].tolist()
        
        # Get user's completed courses
        user_courses = self.get_user_courses(student_id)
        
        # Filter out completed courses
        available_courses = [c for c in category_courses if c not in user_courses]
        
        if not available_courses:
            return pd.DataFrame()
        
        # Get course details and sort by difficulty
        path_df = self.courses[
            self.courses['course_id'].isin(available_courses)
        ].copy()
        
        # Map difficulty to numeric for sorting
        difficulty_order = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        path_df['difficulty_order'] = path_df['difficulty'].map(difficulty_order)
        
        # Sort by difficulty, then by rating
        path_df = path_df.sort_values(
            by=['difficulty_order', 'rating'],
            ascending=[True, False]
        ).head(n_courses)
        
        path_df['path_order'] = range(1, len(path_df) + 1)
        
        return path_df[['path_order', 'course_id', 'title', 'difficulty', 
                       'duration_hours', 'rating', 'prerequisites']]


def run_recommendation_demo(data_dir):
    """
    Run a demonstration of the recommendation system.
    
    Args:
        data_dir: Path to the data directory
    """
    print("\n" + "=" * 70)
    print("  E-LEARNING HYBRID RECOMMENDATION SYSTEM DEMO")
    print("=" * 70)
    
    # Load data
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    activities = pd.read_csv(os.path.join(data_dir, 'activity_logs.csv'))
    
    # Initialize and train the hybrid recommender
    recommender = HybridRecommender(courses, ratings, activities)
    recommender.fit()
    
    # Demo: Get recommendations for a sample user
    sample_student = 'S001'
    
    print("\n" + "=" * 70)
    print(f"  RECOMMENDATIONS FOR STUDENT: {sample_student}")
    print("=" * 70)
    
    # Get user's current courses
    user_courses = recommender.get_user_courses(sample_student)
    print(f"\nCurrent enrolled courses: {user_courses}")
    
    # Get hybrid recommendations
    print("\n--- Top 10 Recommended Courses ---")
    recommendations = recommender.recommend(sample_student, n=10)
    if not recommendations.empty:
        print(recommendations.to_string(index=False))
    
    # Build learning path
    print("\n--- Personalized Learning Path for Data Science ---")
    learning_path = recommender.build_learning_path(
        sample_student, 
        target_category='Data Science',
        n_courses=5
    )
    if not learning_path.empty:
        print(learning_path.to_string(index=False))
    
    # Show similar courses demo
    print("\n--- Courses Similar to 'Machine Learning Fundamentals' (C004) ---")
    similar = recommender.content_recommender.get_similar_courses('C004', n=5)
    for course_id, score in similar:
        course_title = courses[courses['course_id'] == course_id]['title'].values[0]
        print(f"  {course_id}: {course_title} (similarity: {score:.3f})")
    
    return recommender


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    recommender = run_recommendation_demo(data_dir)

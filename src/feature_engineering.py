"""
Feature Engineering Module for E-Learning Recommendation System
This module extracts meaningful features from raw learning data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class FeatureEngineer:
    """
    Extracts and engineers features from student learning data
    for the recommendation system.
    """
    
    def __init__(self, students, courses, activities, quizzes, ratings):
        """
        Initialize with all required datasets.
        
        Args:
            students: Student profiles DataFrame
            courses: Course metadata DataFrame
            activities: Learning activity logs DataFrame
            quizzes: Quiz results DataFrame
            ratings: Course ratings DataFrame
        """
        self.students = students
        self.courses = courses
        self.activities = activities
        self.quizzes = quizzes
        self.ratings = ratings
        
        # Feature storage
        self.student_features = None
        self.course_features = None
        self.interaction_features = None
    
    # =========================================================================
    # TOPIC INTEREST FEATURES
    # =========================================================================
    
    def extract_topic_interest(self):
        """
        Extract topic interest scores for each student based on their
        activity patterns and engagement with different course categories.
        
        Topic Interest Formula:
        interest_score = (time_weight * normalized_time) + 
                        (activity_weight * normalized_activities) +
                        (rating_weight * normalized_ratings)
        
        Returns:
            DataFrame: Student topic interest matrix
        """
        print("\n[Feature Engineering] Extracting Topic Interest...")
        
        # Merge activities with courses to get categories
        activity_categories = self.activities.merge(
            self.courses[['course_id', 'category', 'subcategory']],
            on='course_id'
        )
        
        # Calculate time spent per category per student
        time_by_category = activity_categories.groupby(
            ['student_id', 'category']
        )['time_spent_minutes'].sum().reset_index()
        
        # Calculate activity count per category
        activity_count = activity_categories.groupby(
            ['student_id', 'category']
        ).size().reset_index(name='activity_count')
        
        # Merge with ratings
        rating_categories = self.ratings.merge(
            self.courses[['course_id', 'category']],
            on='course_id'
        )
        avg_ratings = rating_categories.groupby(
            ['student_id', 'category']
        )['rating'].mean().reset_index()
        
        # Combine all metrics
        interest_df = time_by_category.merge(
            activity_count, on=['student_id', 'category'], how='outer'
        ).merge(
            avg_ratings, on=['student_id', 'category'], how='outer'
        ).fillna(0)
        
        # Normalize and calculate weighted interest score
        for col in ['time_spent_minutes', 'activity_count', 'rating']:
            max_val = interest_df[col].max()
            if max_val > 0:
                interest_df[f'{col}_norm'] = interest_df[col] / max_val
            else:
                interest_df[f'{col}_norm'] = 0
        
        # Weighted interest score (time: 0.4, activity: 0.3, rating: 0.3)
        interest_df['interest_score'] = (
            0.4 * interest_df['time_spent_minutes_norm'] +
            0.3 * interest_df['activity_count_norm'] +
            0.3 * interest_df['rating_norm']
        )
        
        # Create pivot table for student-category interest matrix
        topic_interest_matrix = interest_df.pivot_table(
            index='student_id',
            columns='category',
            values='interest_score',
            fill_value=0
        )
        
        print(f"  ✓ Topic interest matrix created: {topic_interest_matrix.shape}")
        return topic_interest_matrix
    
    # =========================================================================
    # COMPLETION RATE FEATURES
    # =========================================================================
    
    def extract_completion_rate(self):
        """
        Extract course completion rates for each student-course pair.
        
        Completion Rate Formula:
        completion_rate = max_progress_percentage / 100
        
        Weighted completion considers:
        - Progress percentage
        - Modules completed vs total modules
        - Quiz attempts and passes
        
        Returns:
            DataFrame: Student-course completion rates
        """
        print("[Feature Engineering] Extracting Completion Rates...")
        
        # Get max progress per student-course from activities
        progress_df = self.activities.groupby(['student_id', 'course_id']).agg({
            'progress_percentage': 'max',
            'module_completed': 'max'
        }).reset_index()
        
        # Merge with course info for total modules (estimated from duration)
        progress_df = progress_df.merge(
            self.courses[['course_id', 'duration_hours', 'completion_rate']],
            on='course_id'
        )
        
        # Estimate total modules (1 module per 5 hours of content)
        progress_df['estimated_modules'] = (progress_df['duration_hours'] / 5).astype(int).clip(lower=1)
        
        # Calculate module completion rate
        progress_df['module_completion_rate'] = (
            progress_df['module_completed'] / progress_df['estimated_modules']
        ).clip(0, 1)
        
        # Get quiz completion info
        quiz_completion = self.quizzes.groupby(['student_id', 'course_id']).agg({
            'passed': 'sum',
            'quiz_id': 'count'
        }).reset_index()
        quiz_completion.columns = ['student_id', 'course_id', 'quizzes_passed', 'total_quizzes']
        quiz_completion['quiz_pass_rate'] = (
            quiz_completion['quizzes_passed'] / quiz_completion['total_quizzes']
        )
        
        # Merge all completion metrics
        completion_df = progress_df.merge(
            quiz_completion[['student_id', 'course_id', 'quiz_pass_rate']],
            on=['student_id', 'course_id'],
            how='left'
        ).fillna(0)
        
        # Calculate weighted completion rate
        # Progress: 50%, Module: 30%, Quiz: 20%
        completion_df['weighted_completion'] = (
            0.5 * (completion_df['progress_percentage'] / 100) +
            0.3 * completion_df['module_completion_rate'] +
            0.2 * completion_df['quiz_pass_rate']
        )
        
        print(f"  ✓ Completion rates calculated: {len(completion_df)} records")
        return completion_df[['student_id', 'course_id', 'progress_percentage', 
                             'module_completion_rate', 'quiz_pass_rate', 'weighted_completion']]
    
    # =========================================================================
    # QUIZ SCORE FEATURES
    # =========================================================================
    
    def extract_quiz_performance(self):
        """
        Extract quiz performance features for each student.
        
        Features include:
        - Average quiz score per category
        - Score improvement trend
        - Difficulty-adjusted performance
        
        Returns:
            DataFrame: Student quiz performance features
        """
        print("[Feature Engineering] Extracting Quiz Performance...")
        
        # Merge quizzes with courses
        quiz_courses = self.quizzes.merge(
            self.courses[['course_id', 'category', 'difficulty']],
            on='course_id'
        )
        
        # Map difficulty to numeric
        difficulty_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        quiz_courses['difficulty_level'] = quiz_courses['difficulty'].map(difficulty_map)
        
        # Calculate average score per student per category
        category_scores = quiz_courses.groupby(['student_id', 'category']).agg({
            'score_percentage': ['mean', 'std', 'count'],
            'difficulty_level': 'mean'
        }).reset_index()
        category_scores.columns = [
            'student_id', 'category', 'avg_score', 'score_std', 
            'quiz_count', 'avg_difficulty'
        ]
        category_scores['score_std'] = category_scores['score_std'].fillna(0)
        
        # Difficulty-adjusted score (bonus for harder courses)
        category_scores['adjusted_score'] = (
            category_scores['avg_score'] * 
            (1 + 0.1 * (category_scores['avg_difficulty'] - 1))
        ).clip(0, 100)
        
        # Calculate overall student performance
        student_performance = quiz_courses.groupby('student_id').agg({
            'score_percentage': ['mean', 'std', 'min', 'max'],
            'passed': 'mean',  # Pass rate
            'attempts': 'mean'  # Average attempts
        }).reset_index()
        student_performance.columns = [
            'student_id', 'overall_avg_score', 'score_volatility',
            'min_score', 'max_score', 'pass_rate', 'avg_attempts'
        ]
        student_performance['score_volatility'] = student_performance['score_volatility'].fillna(0)
        
        print(f"  ✓ Quiz performance features: {len(student_performance)} students")
        return student_performance, category_scores
    
    # =========================================================================
    # TIME SPENT FEATURES
    # =========================================================================
    
    def extract_time_features(self):
        """
        Extract time-based features from learning activities.
        
        Features include:
        - Total time spent per course
        - Average session duration
        - Learning consistency (std of daily time)
        - Preferred learning times
        
        Returns:
            DataFrame: Time-based features
        """
        print("[Feature Engineering] Extracting Time Features...")
        
        # Ensure activity_date is datetime
        activities = self.activities.copy()
        activities['activity_date'] = pd.to_datetime(activities['activity_date'])
        
        # Time per student-course
        time_per_course = activities.groupby(['student_id', 'course_id']).agg({
            'time_spent_minutes': ['sum', 'mean', 'count']
        }).reset_index()
        time_per_course.columns = [
            'student_id', 'course_id', 'total_time', 
            'avg_session_time', 'session_count'
        ]
        
        # Merge with course duration for comparison
        time_per_course = time_per_course.merge(
            self.courses[['course_id', 'duration_hours']],
            on='course_id'
        )
        
        # Calculate time efficiency (time spent / expected time)
        time_per_course['expected_time_minutes'] = time_per_course['duration_hours'] * 60
        time_per_course['time_efficiency'] = (
            time_per_course['total_time'] / time_per_course['expected_time_minutes']
        ).clip(0, 2)  # Cap at 200% of expected time
        
        # Daily learning pattern
        daily_time = activities.groupby(['student_id', 'activity_date']).agg({
            'time_spent_minutes': 'sum'
        }).reset_index()
        
        learning_consistency = daily_time.groupby('student_id').agg({
            'time_spent_minutes': ['mean', 'std', 'count']
        }).reset_index()
        learning_consistency.columns = [
            'student_id', 'avg_daily_time', 'daily_time_std', 'active_days'
        ]
        learning_consistency['daily_time_std'] = learning_consistency['daily_time_std'].fillna(0)
        
        # Consistency score (lower std = more consistent)
        max_std = learning_consistency['daily_time_std'].max()
        if max_std > 0:
            learning_consistency['consistency_score'] = (
                1 - (learning_consistency['daily_time_std'] / max_std)
            )
        else:
            learning_consistency['consistency_score'] = 1
        
        print(f"  ✓ Time features: {len(time_per_course)} student-course pairs")
        return time_per_course, learning_consistency
    
    # =========================================================================
    # DIFFICULTY LEVEL FEATURES
    # =========================================================================
    
    def extract_difficulty_preference(self):
        """
        Determine each student's optimal difficulty level based on their
        performance across different difficulty courses.
        
        Returns:
            DataFrame: Student difficulty preferences and performance
        """
        print("[Feature Engineering] Extracting Difficulty Preferences...")
        
        # Merge quizzes with course difficulty
        quiz_difficulty = self.quizzes.merge(
            self.courses[['course_id', 'difficulty']],
            on='course_id'
        )
        
        # Calculate performance per difficulty level
        difficulty_performance = quiz_difficulty.groupby(
            ['student_id', 'difficulty']
        ).agg({
            'score_percentage': 'mean',
            'passed': 'mean',
            'quiz_id': 'count'
        }).reset_index()
        difficulty_performance.columns = [
            'student_id', 'difficulty', 'avg_score', 'pass_rate', 'quiz_count'
        ]
        
        # Create pivot for difficulty performance
        difficulty_pivot = difficulty_performance.pivot_table(
            index='student_id',
            columns='difficulty',
            values='avg_score',
            fill_value=0
        )
        
        # Determine recommended difficulty for each student
        def get_recommended_difficulty(row):
            """Recommend difficulty based on performance pattern."""
            scores = {
                'Beginner': row.get('Beginner', 0),
                'Intermediate': row.get('Intermediate', 0),
                'Advanced': row.get('Advanced', 0)
            }
            
            # If scoring well on current level, recommend higher
            if scores['Advanced'] >= 80:
                return 'Advanced'
            elif scores['Intermediate'] >= 75 or scores['Advanced'] >= 65:
                return 'Advanced'
            elif scores['Beginner'] >= 75 or scores['Intermediate'] >= 65:
                return 'Intermediate'
            else:
                return 'Beginner'
        
        difficulty_pivot['recommended_difficulty'] = difficulty_pivot.apply(
            get_recommended_difficulty, axis=1
        )
        
        print(f"  ✓ Difficulty preferences: {len(difficulty_pivot)} students")
        return difficulty_pivot.reset_index()
    
    # =========================================================================
    # COMBINED FEATURE EXTRACTION
    # =========================================================================
    
    def extract_all_features(self):
        """
        Extract all features and create comprehensive feature matrices.
        
        Returns:
            dict: Dictionary containing all feature DataFrames
        """
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Extract all feature types
        topic_interest = self.extract_topic_interest()
        completion_rates = self.extract_completion_rate()
        quiz_perf, category_scores = self.extract_quiz_performance()
        time_features, learning_consistency = self.extract_time_features()
        difficulty_prefs = self.extract_difficulty_preference()
        
        # Create student feature matrix
        student_features = self.students[['student_id', 'learning_style', 
                                          'preferred_difficulty', 'avg_quiz_score']].copy()
        
        # Merge with extracted features
        student_features = student_features.merge(quiz_perf, on='student_id', how='left')
        student_features = student_features.merge(learning_consistency, on='student_id', how='left')
        student_features = student_features.merge(
            difficulty_prefs[['student_id', 'recommended_difficulty']], 
            on='student_id', how='left'
        )
        
        # Fill NaN values
        student_features = student_features.fillna(0)
        
        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING COMPLETED")
        print("=" * 60)
        print(f"\n  Summary:")
        print(f"  - Student features: {student_features.shape}")
        print(f"  - Topic interest matrix: {topic_interest.shape}")
        print(f"  - Completion rates: {len(completion_rates)} records")
        print(f"  - Time features: {len(time_features)} records")
        
        return {
            'student_features': student_features,
            'topic_interest': topic_interest,
            'category_scores': category_scores,
            'completion_rates': completion_rates,
            'time_features': time_features,
            'difficulty_prefs': difficulty_prefs
        }


def run_feature_engineering(data_dir):
    """
    Run the complete feature engineering pipeline.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: All extracted features
    """
    import os
    
    # Load data
    students = pd.read_csv(os.path.join(data_dir, 'students.csv'))
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    activities = pd.read_csv(os.path.join(data_dir, 'activity_logs.csv'))
    quizzes = pd.read_csv(os.path.join(data_dir, 'quiz_results.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    
    # Initialize feature engineer
    engineer = FeatureEngineer(students, courses, activities, quizzes, ratings)
    
    # Extract all features
    features = engineer.extract_all_features()
    
    return features


if __name__ == "__main__":
    import os
    
    # Run feature engineering
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    features = run_feature_engineering(data_dir)
    
    # Display sample outputs
    print("\n\n" + "=" * 70)
    print("  SAMPLE OUTPUT: Extracted Features")
    print("=" * 70)
    
    print("\n--- Student Features (first 5 rows) ---")
    print(features['student_features'].head())
    
    print("\n--- Topic Interest Matrix (first 5 rows) ---")
    print(features['topic_interest'].head())
    
    print("\n--- Difficulty Preferences (first 5 rows) ---")
    print(features['difficulty_prefs'].head())

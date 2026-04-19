"""
Data Preprocessing Module for E-Learning Recommendation System
This module handles ETL (Extract, Transform, Load) operations for the data warehouse.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataPreprocessor:
    """
    Handles data extraction, cleaning, transformation, and loading
    for the e-learning recommendation system.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the preprocessor with the data directory path.
        
        Args:
            data_dir: Path to the directory containing raw data files
        """
        self.data_dir = data_dir
        self.students = None
        self.courses = None
        self.activity_logs = None
        self.quiz_results = None
        self.ratings = None
    
    # =========================================================================
    # EXTRACTION PHASE
    # =========================================================================
    
    def extract_data(self):
        """
        Extract data from CSV files (Data Collection phase of ETL).
        
        Returns:
            dict: Dictionary containing all loaded DataFrames
        """
        print("=" * 60)
        print("EXTRACTION PHASE: Loading data from source files...")
        print("=" * 60)
        
        # Load all datasets
        self.students = pd.read_csv(os.path.join(self.data_dir, 'students.csv'))
        self.courses = pd.read_csv(os.path.join(self.data_dir, 'courses.csv'))
        self.activity_logs = pd.read_csv(os.path.join(self.data_dir, 'activity_logs.csv'))
        self.quiz_results = pd.read_csv(os.path.join(self.data_dir, 'quiz_results.csv'))
        self.ratings = pd.read_csv(os.path.join(self.data_dir, 'ratings.csv'))
        
        print(f"✓ Loaded students.csv: {len(self.students)} records")
        print(f"✓ Loaded courses.csv: {len(self.courses)} records")
        print(f"✓ Loaded activity_logs.csv: {len(self.activity_logs)} records")
        print(f"✓ Loaded quiz_results.csv: {len(self.quiz_results)} records")
        print(f"✓ Loaded ratings.csv: {len(self.ratings)} records")
        
        return {
            'students': self.students,
            'courses': self.courses,
            'activity_logs': self.activity_logs,
            'quiz_results': self.quiz_results,
            'ratings': self.ratings
        }
    
    # =========================================================================
    # CLEANING PHASE
    # =========================================================================
    
    def clean_data(self):
        """
        Clean the extracted data (Data Cleaning phase of ETL).
        - Handle missing values
        - Remove duplicates
        - Fix data types
        - Validate data ranges
        """
        print("\n" + "=" * 60)
        print("CLEANING PHASE: Cleaning and validating data...")
        print("=" * 60)
        
        # Clean Students Data
        print("\n[1] Cleaning Students Data...")
        self.students = self._clean_students(self.students)
        
        # Clean Courses Data
        print("[2] Cleaning Courses Data...")
        self.courses = self._clean_courses(self.courses)
        
        # Clean Activity Logs
        print("[3] Cleaning Activity Logs...")
        self.activity_logs = self._clean_activity_logs(self.activity_logs)
        
        # Clean Quiz Results
        print("[4] Cleaning Quiz Results...")
        self.quiz_results = self._clean_quiz_results(self.quiz_results)
        
        # Clean Ratings
        print("[5] Cleaning Ratings...")
        self.ratings = self._clean_ratings(self.ratings)
        
        print("\n✓ Data cleaning completed successfully!")
    
    def _clean_students(self, df):
        """Clean student data."""
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['student_id'])
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  - Removed {removed} duplicate student records")
        
        # Handle missing values
        df['avg_quiz_score'] = df['avg_quiz_score'].fillna(0)
        df['total_time_spent_hours'] = df['total_time_spent_hours'].fillna(0)
        
        # Convert date column
        df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
        
        # Validate age range (16-60)
        df['age'] = df['age'].clip(16, 60)
        
        print(f"  ✓ Students cleaned: {len(df)} valid records")
        return df
    
    def _clean_courses(self, df):
        """Clean course data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['course_id'])
        
        # Handle missing values
        df['prerequisites'] = df['prerequisites'].fillna('None')
        df['rating'] = df['rating'].fillna(df['rating'].mean())
        
        # Validate rating range (0-5)
        df['rating'] = df['rating'].clip(0, 5)
        
        # Validate completion rate (0-100)
        df['completion_rate'] = df['completion_rate'].clip(0, 100)
        
        print(f"  ✓ Courses cleaned: {len(df)} valid records")
        return df
    
    def _clean_activity_logs(self, df):
        """Clean activity log data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['log_id'])
        
        # Convert date column
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        
        # Validate time spent (positive values only)
        df['time_spent_minutes'] = df['time_spent_minutes'].clip(lower=0)
        
        # Validate progress percentage (0-100)
        df['progress_percentage'] = df['progress_percentage'].clip(0, 100)
        
        # Convert boolean columns
        bool_cols = ['video_watched', 'notes_taken', 'resources_downloaded']
        for col in bool_cols:
            df[col] = df[col].astype(bool)
        
        print(f"  ✓ Activity logs cleaned: {len(df)} valid records")
        return df
    
    def _clean_quiz_results(self, df):
        """Clean quiz results data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['quiz_id'])
        
        # Convert date column
        df['quiz_date'] = pd.to_datetime(df['quiz_date'])
        
        # Validate score percentage (0-100)
        df['score_percentage'] = df['score_percentage'].clip(0, 100)
        
        # Ensure correct_answers <= total_questions
        df['correct_answers'] = df.apply(
            lambda x: min(x['correct_answers'], x['total_questions']), axis=1
        )
        
        print(f"  ✓ Quiz results cleaned: {len(df)} valid records")
        return df
    
    def _clean_ratings(self, df):
        """Clean ratings data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['rating_id'])
        
        # Convert date column
        df['rating_date'] = pd.to_datetime(df['rating_date'])
        
        # Validate ratings (1-5)
        rating_cols = ['rating', 'difficulty_rating', 'content_quality', 'instructor_rating']
        for col in rating_cols:
            df[col] = df[col].clip(1, 5)
        
        print(f"  ✓ Ratings cleaned: {len(df)} valid records")
        return df
    
    # =========================================================================
    # TRANSFORMATION PHASE
    # =========================================================================
    
    def transform_data(self):
        """
        Transform data for the data warehouse (Data Transformation phase of ETL).
        - Create dimension tables
        - Create fact tables
        - Add derived features
        """
        print("\n" + "=" * 60)
        print("TRANSFORMATION PHASE: Transforming data for warehouse...")
        print("=" * 60)
        
        # Create Time Dimension
        print("\n[1] Creating Time Dimension...")
        dim_time = self._create_time_dimension()
        
        # Create User Dimension
        print("[2] Creating User Dimension...")
        dim_user = self._create_user_dimension()
        
        # Create Course Dimension
        print("[3] Creating Course Dimension...")
        dim_course = self._create_course_dimension()
        
        # Create Fact Tables
        print("[4] Creating Fact Tables...")
        fact_learning = self._create_learning_fact()
        fact_quiz = self._create_quiz_fact()
        fact_rating = self._create_rating_fact()
        
        print("\n✓ Data transformation completed successfully!")
        
        return {
            'dim_time': dim_time,
            'dim_user': dim_user,
            'dim_course': dim_course,
            'fact_learning': fact_learning,
            'fact_quiz': fact_quiz,
            'fact_rating': fact_rating
        }
    
    def _create_time_dimension(self):
        """Create time dimension table."""
        # Get all unique dates from activity logs
        dates = pd.concat([
            self.activity_logs['activity_date'],
            self.quiz_results['quiz_date'],
            self.ratings['rating_date']
        ]).unique()
        
        dim_time = pd.DataFrame({'date': pd.to_datetime(dates)})
        dim_time['time_key'] = range(1, len(dim_time) + 1)
        dim_time['day'] = dim_time['date'].dt.day
        dim_time['month'] = dim_time['date'].dt.month
        dim_time['year'] = dim_time['date'].dt.year
        dim_time['quarter'] = dim_time['date'].dt.quarter
        dim_time['day_of_week'] = dim_time['date'].dt.dayofweek
        dim_time['week_of_year'] = dim_time['date'].dt.isocalendar().week
        dim_time['is_weekend'] = dim_time['day_of_week'].isin([5, 6])
        
        print(f"  ✓ Time dimension created: {len(dim_time)} records")
        return dim_time
    
    def _create_user_dimension(self):
        """Create user dimension table."""
        dim_user = self.students.copy()
        dim_user['user_key'] = range(1, len(dim_user) + 1)
        
        # Add derived attributes
        dim_user['experience_level'] = dim_user['total_courses_completed'].apply(
            lambda x: 'Beginner' if x < 5 else ('Intermediate' if x < 15 else 'Advanced')
        )
        
        dim_user['performance_tier'] = dim_user['avg_quiz_score'].apply(
            lambda x: 'Low' if x < 60 else ('Medium' if x < 80 else 'High')
        )
        
        print(f"  ✓ User dimension created: {len(dim_user)} records")
        return dim_user
    
    def _create_course_dimension(self):
        """Create course dimension table."""
        dim_course = self.courses.copy()
        dim_course['course_key'] = range(1, len(dim_course) + 1)
        
        # Map difficulty to numeric values
        difficulty_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        dim_course['difficulty_level'] = dim_course['difficulty'].map(difficulty_map)
        
        # Add popularity tier based on enrollments
        dim_course['popularity_tier'] = pd.qcut(
            dim_course['total_enrollments'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"  ✓ Course dimension created: {len(dim_course)} records")
        return dim_course
    
    def _create_learning_fact(self):
        """Create learning activity fact table."""
        fact_learning = self.activity_logs.copy()
        
        # Aggregate by student-course
        fact_agg = fact_learning.groupby(['student_id', 'course_id']).agg({
            'time_spent_minutes': 'sum',
            'progress_percentage': 'max',
            'module_completed': 'max',
            'video_watched': 'sum',
            'notes_taken': 'sum',
            'resources_downloaded': 'sum'
        }).reset_index()
        
        fact_agg.columns = [
            'student_id', 'course_id', 'total_time_minutes', 
            'max_progress', 'modules_completed', 'videos_watched',
            'notes_count', 'resources_count'
        ]
        
        print(f"  ✓ Learning fact table created: {len(fact_agg)} records")
        return fact_agg
    
    def _create_quiz_fact(self):
        """Create quiz performance fact table."""
        fact_quiz = self.quiz_results.groupby(['student_id', 'course_id']).agg({
            'score_percentage': 'mean',
            'time_taken_minutes': 'sum',
            'attempts': 'sum',
            'passed': 'sum',
            'quiz_id': 'count'
        }).reset_index()
        
        fact_quiz.columns = [
            'student_id', 'course_id', 'avg_score', 
            'total_time', 'total_attempts', 'quizzes_passed', 'total_quizzes'
        ]
        
        print(f"  ✓ Quiz fact table created: {len(fact_quiz)} records")
        return fact_quiz
    
    def _create_rating_fact(self):
        """Create rating fact table."""
        fact_rating = self.ratings.groupby(['student_id', 'course_id']).agg({
            'rating': 'mean',
            'difficulty_rating': 'mean',
            'content_quality': 'mean',
            'instructor_rating': 'mean',
            'would_recommend': 'max'
        }).reset_index()
        
        print(f"  ✓ Rating fact table created: {len(fact_rating)} records")
        return fact_rating
    
    # =========================================================================
    # LOADING PHASE
    # =========================================================================
    
    def get_processed_data(self):
        """
        Return all processed datasets ready for the ML engine.
        
        Returns:
            dict: Dictionary containing all processed DataFrames
        """
        return {
            'students': self.students,
            'courses': self.courses,
            'activity_logs': self.activity_logs,
            'quiz_results': self.quiz_results,
            'ratings': self.ratings
        }


def run_etl_pipeline(data_dir):
    """
    Run the complete ETL pipeline.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        tuple: (preprocessor, warehouse_tables)
    """
    print("\n" + "=" * 70)
    print("  E-LEARNING RECOMMENDATION SYSTEM - ETL PIPELINE")
    print("=" * 70)
    
    preprocessor = DataPreprocessor(data_dir)
    
    # Extract
    preprocessor.extract_data()
    
    # Clean
    preprocessor.clean_data()
    
    # Transform
    warehouse_tables = preprocessor.transform_data()
    
    print("\n" + "=" * 70)
    print("  ETL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return preprocessor, warehouse_tables


if __name__ == "__main__":
    # Run ETL pipeline
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    preprocessor, warehouse = run_etl_pipeline(data_dir)
    
    # Display sample outputs
    print("\n\n" + "=" * 70)
    print("  SAMPLE OUTPUT: Dimension and Fact Tables")
    print("=" * 70)
    
    print("\n--- Time Dimension (first 5 rows) ---")
    print(warehouse['dim_time'].head())
    
    print("\n--- User Dimension (first 5 rows) ---")
    print(warehouse['dim_user'][['student_id', 'name', 'experience_level', 'performance_tier']].head())
    
    print("\n--- Course Dimension (first 5 rows) ---")
    print(warehouse['dim_course'][['course_id', 'title', 'difficulty', 'popularity_tier']].head())
    
    print("\n--- Learning Fact Table (first 5 rows) ---")
    print(warehouse['fact_learning'].head())

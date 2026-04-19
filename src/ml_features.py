"""
Advanced ML Features for E-Learning Recommendation System
- Grade Predictor
- Student Clustering
- Recommendation Explainer
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class GradePredictor:
    """
    Predicts expected quiz score based on student's past performance,
    course difficulty, and learning behavior.
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
    
    def prepare_features(self, students_df, activities_df, quizzes_df, courses_df):
        """Prepare features for training."""
        features_list = []
        targets = []
        
        for _, quiz in quizzes_df.iterrows():
            student_id = quiz['student_id']
            course_id = quiz['course_id']
            
            # Get student info
            student = students_df[students_df['student_id'] == student_id]
            if student.empty:
                continue
            student = student.iloc[0]
            
            # Get course info
            course = courses_df[courses_df['course_id'] == course_id]
            if course.empty:
                continue
            course = course.iloc[0]
            
            # Get student's activity on this course
            activity = activities_df[
                (activities_df['student_id'] == student_id) & 
                (activities_df['course_id'] == course_id)
            ]
            
            # Build feature vector
            features = {
                'avg_quiz_score': student['avg_quiz_score'],
                'total_courses_completed': student['total_courses_completed'],
                'total_time_spent_hours': student['total_time_spent_hours'],
                'course_difficulty': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}.get(course['difficulty'], 2),
                'course_duration': course['duration_hours'],
                'course_rating': course['rating'],
                'time_spent_on_course': activity['time_spent_minutes'].sum() if not activity.empty else 0,
                'progress_percentage': activity['progress_percentage'].max() if not activity.empty else 0,
                'videos_watched': activity['video_watched'].sum() if not activity.empty else 0,
                'notes_taken': activity['notes_taken'].sum() if not activity.empty else 0,
            }
            
            features_list.append(features)
            targets.append(quiz['score_percentage'])
        
        if not features_list:
            return None, None
            
        X = pd.DataFrame(features_list)
        y = np.array(targets)
        self.feature_names = list(X.columns)
        
        return X, y
    
    def fit(self, students_df, activities_df, quizzes_df, courses_df):
        """Train the grade predictor."""
        X, y = self.prepare_features(students_df, activities_df, quizzes_df, courses_df)
        
        if X is None or len(X) < 10:
            print("[GradePredictor] Not enough data to train")
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"[GradePredictor] Trained on {len(X)} samples")
        return True
    
    def predict(self, student_id, course_id, students_df, activities_df, courses_df):
        """Predict expected quiz score for a student on a course."""
        if not self.is_fitted:
            return {'predicted_score': 70.0, 'confidence': 'Low', 'factors': []}
        
        # Get student info
        student = students_df[students_df['student_id'] == student_id]
        if student.empty:
            return {'predicted_score': 70.0, 'confidence': 'Low', 'factors': []}
        student = student.iloc[0]
        
        # Get course info
        course = courses_df[courses_df['course_id'] == course_id]
        if course.empty:
            return {'predicted_score': 70.0, 'confidence': 'Low', 'factors': []}
        course = course.iloc[0]
        
        # Get activity
        activity = activities_df[
            (activities_df['student_id'] == student_id) & 
            (activities_df['course_id'] == course_id)
        ]
        
        # Build feature vector
        features = {
            'avg_quiz_score': student['avg_quiz_score'],
            'total_courses_completed': student['total_courses_completed'],
            'total_time_spent_hours': student['total_time_spent_hours'],
            'course_difficulty': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}.get(course['difficulty'], 2),
            'course_duration': course['duration_hours'],
            'course_rating': course['rating'],
            'time_spent_on_course': activity['time_spent_minutes'].sum() if not activity.empty else 0,
            'progress_percentage': activity['progress_percentage'].max() if not activity.empty else 0,
            'videos_watched': activity['video_watched'].sum() if not activity.empty else 0,
            'notes_taken': activity['notes_taken'].sum() if not activity.empty else 0,
        }
        
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        predicted_score = self.model.predict(X_scaled)[0]
        predicted_score = max(0, min(100, predicted_score))  # Clamp to 0-100
        
        # Get feature importances
        importances = self.model.feature_importances_
        factors = []
        for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1])[:3]:
            if imp > 0.1:
                factors.append({'feature': name, 'importance': round(imp * 100, 1)})
        
        # Determine confidence based on data availability
        confidence = 'High' if not activity.empty and features['progress_percentage'] > 50 else 'Medium'
        if activity.empty:
            confidence = 'Low'
        
        return {
            'predicted_score': round(predicted_score, 1),
            'confidence': confidence,
            'factors': factors,
            'course_difficulty': course['difficulty'],
            'student_avg': student['avg_quiz_score']
        }


class StudentClusterer:
    """
    Clusters students into learning profiles based on behavior patterns.
    """
    
    CLUSTER_PROFILES = {
        0: {
            'name': 'Visual Learner',
            'icon': 'bi-eye',
            'color': 'primary',
            'description': 'Prefers video content and visual materials'
        },
        1: {
            'name': 'Fast Achiever',
            'icon': 'bi-lightning',
            'color': 'success',
            'description': 'Completes courses quickly with high scores'
        },
        2: {
            'name': 'Steady Learner',
            'icon': 'bi-clock',
            'color': 'info',
            'description': 'Takes time but achieves consistent results'
        },
        3: {
            'name': 'Active Participant',
            'icon': 'bi-pencil',
            'color': 'warning',
            'description': 'Takes notes and engages deeply with content'
        }
    }
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.cluster_centers = None
    
    def prepare_features(self, students_df, activities_df, quizzes_df):
        """Prepare student features for clustering."""
        features_list = []
        student_ids = []
        
        for _, student in students_df.iterrows():
            sid = student['student_id']
            
            # Get activity stats
            activities = activities_df[activities_df['student_id'] == sid]
            quizzes = quizzes_df[quizzes_df['student_id'] == sid]
            
            features = {
                'avg_quiz_score': student['avg_quiz_score'],
                'total_courses': student['total_courses_completed'],
                'total_time': student['total_time_spent_hours'],
                'avg_time_per_course': student['total_time_spent_hours'] / max(1, student['total_courses_completed']),
                'videos_watched': activities['video_watched'].sum() if not activities.empty else 0,
                'notes_taken': activities['notes_taken'].sum() if not activities.empty else 0,
                'avg_progress': activities['progress_percentage'].mean() if not activities.empty else 0,
                'quiz_attempts': len(quizzes),
                'pass_rate': quizzes['passed'].mean() * 100 if not quizzes.empty else 0,
            }
            
            features_list.append(features)
            student_ids.append(sid)
        
        X = pd.DataFrame(features_list)
        return X, student_ids
    
    def fit(self, students_df, activities_df, quizzes_df):
        """Fit the clustering model."""
        X, _ = self.prepare_features(students_df, activities_df, quizzes_df)
        
        if len(X) < self.n_clusters:
            print("[StudentClusterer] Not enough students to cluster")
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.cluster_centers = self.model.cluster_centers_
        self.is_fitted = True
        
        print(f"[StudentClusterer] Clustered {len(X)} students into {self.n_clusters} groups")
        return True
    
    def get_cluster(self, student_id, students_df, activities_df, quizzes_df):
        """Get cluster assignment for a student."""
        if not self.is_fitted:
            return self.CLUSTER_PROFILES[0]
        
        # Get student row
        student = students_df[students_df['student_id'] == student_id]
        if student.empty:
            return self.CLUSTER_PROFILES[0]
        
        student = student.iloc[0]
        activities = activities_df[activities_df['student_id'] == student_id]
        quizzes = quizzes_df[quizzes_df['student_id'] == student_id]
        
        features = {
            'avg_quiz_score': student['avg_quiz_score'],
            'total_courses': student['total_courses_completed'],
            'total_time': student['total_time_spent_hours'],
            'avg_time_per_course': student['total_time_spent_hours'] / max(1, student['total_courses_completed']),
            'videos_watched': activities['video_watched'].sum() if not activities.empty else 0,
            'notes_taken': activities['notes_taken'].sum() if not activities.empty else 0,
            'avg_progress': activities['progress_percentage'].mean() if not activities.empty else 0,
            'quiz_attempts': len(quizzes),
            'pass_rate': quizzes['passed'].mean() * 100 if not quizzes.empty else 0,
        }
        
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        cluster_id = self.model.predict(X_scaled)[0]
        
        profile = self.CLUSTER_PROFILES.get(cluster_id, self.CLUSTER_PROFILES[0]).copy()
        profile['cluster_id'] = int(cluster_id)
        
        return profile
    
    def get_all_clusters(self, students_df, activities_df, quizzes_df):
        """Get cluster distribution."""
        if not self.is_fitted:
            return {}
        
        X, student_ids = self.prepare_features(students_df, activities_df, quizzes_df)
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        distribution = {}
        for i in range(self.n_clusters):
            count = (labels == i).sum()
            profile = self.CLUSTER_PROFILES.get(i, {'name': f'Cluster {i}'})
            distribution[i] = {
                'name': profile['name'],
                'count': int(count),
                'percentage': round(count / len(labels) * 100, 1)
            }
        
        return distribution


class RecommendationExplainer:
    """
    Explains why specific courses are recommended.
    """
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=100)
        self.is_fitted = False
        self.course_keywords = {}
    
    def fit(self, courses_df):
        """Extract keywords from courses."""
        texts = courses_df['title'] + ' ' + courses_df['description'] + ' ' + courses_df['tags']
        self.tfidf.fit(texts)
        
        # Extract top keywords per course
        tfidf_matrix = self.tfidf.transform(texts)
        feature_names = self.tfidf.get_feature_names_out()
        
        for idx, row in courses_df.iterrows():
            course_id = row['course_id']
            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            top_indices = tfidf_scores.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            self.course_keywords[course_id] = keywords
        
        self.is_fitted = True
        print(f"[RecommendationExplainer] Extracted keywords for {len(courses_df)} courses")
    
    def explain(self, student_id, course_id, students_df, activities_df, ratings_df, courses_df):
        """Generate explanation for why a course is recommended."""
        explanations = []
        
        # Get student info
        student = students_df[students_df['student_id'] == student_id]
        if student.empty:
            return explanations
        student = student.iloc[0]
        
        # Get course info
        course = courses_df[courses_df['course_id'] == course_id]
        if course.empty:
            return explanations
        course = course.iloc[0]
        
        # Reason 1: Category match with past interests
        student_activities = activities_df[activities_df['student_id'] == student_id]
        if not student_activities.empty:
            past_courses = student_activities['course_id'].unique()
            past_categories = courses_df[courses_df['course_id'].isin(past_courses)]['category'].unique()
            
            if course['category'] in past_categories:
                explanations.append({
                    'type': 'category',
                    'icon': 'bi-bookmark-check',
                    'text': f"Matches your interest in {course['category']}"
                })
        
        # Reason 2: Difficulty alignment
        pref_difficulty = student['preferred_difficulty']
        if course['difficulty'] == pref_difficulty:
            explanations.append({
                'type': 'difficulty',
                'icon': 'bi-bullseye',
                'text': f"Matches your preferred difficulty level ({pref_difficulty})"
            })
        elif course['difficulty'] == 'Beginner' and student['avg_quiz_score'] < 60:
            explanations.append({
                'type': 'difficulty',
                'icon': 'bi-arrow-down-circle',
                'text': "Easier course to build foundational skills"
            })
        
        # Reason 3: High rating
        if course['rating'] >= 4.5:
            explanations.append({
                'type': 'rating',
                'icon': 'bi-star-fill',
                'text': f"Highly rated course ({course['rating']}/5.0)"
            })
        
        # Reason 4: Similar students liked it
        student_ratings = ratings_df[ratings_df['course_id'] == course_id]
        if len(student_ratings) > 5:
            avg_rating = student_ratings['rating'].mean()
            if avg_rating >= 4.0:
                explanations.append({
                    'type': 'collaborative',
                    'icon': 'bi-people',
                    'text': f"Popular among similar students (avg {avg_rating:.1f}/5)"
                })
        
        # Reason 5: Content keywords
        if self.is_fitted and course_id in self.course_keywords:
            keywords = self.course_keywords[course_id][:3]
            if keywords:
                explanations.append({
                    'type': 'content',
                    'icon': 'bi-tags',
                    'text': f"Covers: {', '.join(keywords)}"
                })
        
        # Reason 6: Prerequisite for advanced courses
        if course['difficulty'] == 'Beginner':
            advanced_in_category = courses_df[
                (courses_df['category'] == course['category']) & 
                (courses_df['difficulty'] == 'Advanced')
            ]
            if len(advanced_in_category) > 0:
                explanations.append({
                    'type': 'path',
                    'icon': 'bi-signpost-split',
                    'text': f"Foundation for {len(advanced_in_category)} advanced courses"
                })
        
        return explanations[:4]  # Return top 4 explanations


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == '__main__':
    import os
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    print("\n" + "=" * 60)
    print("  ADVANCED ML FEATURES - TEST")
    print("=" * 60)
    
    # Load data
    students = pd.read_csv(os.path.join(DATA_DIR, 'students.csv'))
    courses = pd.read_csv(os.path.join(DATA_DIR, 'courses.csv'))
    activities = pd.read_csv(os.path.join(DATA_DIR, 'activity_logs.csv'))
    quizzes = pd.read_csv(os.path.join(DATA_DIR, 'quiz_results.csv'))
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
    
    print("\n--- Grade Predictor ---")
    predictor = GradePredictor()
    predictor.fit(students, activities, quizzes, courses)
    prediction = predictor.predict('S001', 'C001', students, activities, courses)
    print(f"Predicted score for S001 on C001: {prediction}")
    
    print("\n--- Student Clustering ---")
    clusterer = StudentClusterer()
    clusterer.fit(students, activities, quizzes)
    profile = clusterer.get_cluster('S001', students, activities, quizzes)
    print(f"Student S001 profile: {profile}")
    distribution = clusterer.get_all_clusters(students, activities, quizzes)
    print(f"Cluster distribution: {distribution}")
    
    print("\n--- Recommendation Explainer ---")
    explainer = RecommendationExplainer()
    explainer.fit(courses)
    explanations = explainer.explain('S001', 'C001', students, activities, ratings, courses)
    print(f"Explanations for S001 -> C001: {explanations}")
    
    print("\n" + "=" * 60)

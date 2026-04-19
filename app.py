
import os
import sys
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from datetime import datetime

# Add src to path for importing ML modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from recommendation_engine import HybridRecommender, ContentBasedRecommender
from feature_engineering import FeatureEngineer
from ml_features import GradePredictor, StudentClusterer, RecommendationExplainer

app = Flask(__name__)
app.secret_key = 'elearning_secret_key_2024'

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Global data cache and ML models
_data_cache = {}
_recommender = None
_grade_predictor = None
_clusterer = None
_explainer = None



def load_data():
    """Load all datasets into cache."""
    global _data_cache
    if not _data_cache:
        _data_cache = {
            'students': pd.read_csv(os.path.join(DATA_DIR, 'students.csv')),
            'courses': pd.read_csv(os.path.join(DATA_DIR, 'courses.csv')),
            'activities': pd.read_csv(os.path.join(DATA_DIR, 'activity_logs.csv')),
            'quizzes': pd.read_csv(os.path.join(DATA_DIR, 'quiz_results.csv')),
            'ratings': pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv')),
            'enrollments': pd.read_csv(os.path.join(DATA_DIR, 'enrollments.csv')),
            'course_modules': pd.read_csv(os.path.join(DATA_DIR, 'course_modules.csv')),
            'quiz_questions': pd.read_csv(os.path.join(DATA_DIR, 'quiz_questions.csv')),
            'module_progress': pd.read_csv(os.path.join(DATA_DIR, 'module_progress.csv'))
        }
    return _data_cache


def get_recommender():
    """Get or create the hybrid recommender."""
    global _recommender
    if _recommender is None:
        data = load_data()
        _recommender = HybridRecommender(
            data['courses'], 
            data['ratings'], 
            data['activities'],
            enrollments=data['enrollments'],
            quizzes=data['quizzes']
        )
        _recommender.fit()
    return _recommender


def get_grade_predictor():
    """Get or create the grade predictor."""
    global _grade_predictor
    if _grade_predictor is None:
        data = load_data()
        _grade_predictor = GradePredictor()
        _grade_predictor.fit(data['students'], data['activities'], data['quizzes'], data['courses'])
    return _grade_predictor


def get_clusterer():
    """Get or create the student clusterer."""
    global _clusterer
    if _clusterer is None:
        data = load_data()
        _clusterer = StudentClusterer()
        _clusterer.fit(data['students'], data['activities'], data['quizzes'])
    return _clusterer


def get_explainer():
    """Get or create the recommendation explainer."""
    global _explainer
    if _explainer is None:
        data = load_data()
        _explainer = RecommendationExplainer()
        _explainer.fit(data['courses'])
    return _explainer


def reload_data():
    """Reload data from CSV files (for after registration)."""
    global _data_cache, _recommender, _grade_predictor, _clusterer, _explainer
    _data_cache = {}
    _recommender = None
    _grade_predictor = None
    _clusterer = None
    _explainer = None
    return load_data()


def get_student(student_id):
    """Get student details by ID."""
    data = load_data()
    student = data['students'][data['students']['student_id'] == student_id]
    if student.empty:
        return None
    return student.iloc[0].to_dict()


def get_student_courses(student_id):
    """Get courses a student is enrolled in."""
    data = load_data()
    enrollments = data['enrollments'][data['enrollments']['student_id'] == student_id]
    
    if enrollments.empty:
        return []
    
    course_ids = enrollments['course_id'].unique()
    courses = data['courses'][data['courses']['course_id'].isin(course_ids)].copy()
    
    # Merge with enrollment data to get progress
    courses = courses.merge(
        enrollments[['course_id', 'progress_percentage', 'status', 'enrollment_date', 'last_accessed']],
        on='course_id',
        how='left'
    )
    
    # Rename for consistency
    courses.rename(columns={'progress_percentage': 'progress'}, inplace=True)
    
    return courses.to_dict('records')


def get_seen_course_ids(student_id, data):
    """Get course ids the student has already studied or meaningfully opened."""
    seen_ids = set(
        data['activities'][
            data['activities']['student_id'] == student_id
        ]['course_id'].unique().tolist()
    )
    
    enrollment_mask = (
        (data['enrollments']['student_id'] == student_id) &
        (data['enrollments']['status'].isin(['active', 'completed']))
    )
    seen_ids.update(
        data['enrollments'][enrollment_mask]['course_id'].unique().tolist()
    )
    seen_ids.update(
        data['quizzes'][
            data['quizzes']['student_id'] == student_id
        ]['course_id'].unique().tolist()
    )
    return seen_ids


def apply_recommendation_business_rules(student_id, recs, data, limit=10):
    """
    Apply app-level business rules so the UI/API never shows already-seen
    courses or duplicate items.
    """
    if recs.empty:
        return recs
    
    seen_ids = get_seen_course_ids(student_id, data)
    filtered = recs[~recs['course_id'].isin(seen_ids)].copy()
    filtered = filtered.drop_duplicates(subset=['course_id'])
    return filtered.head(limit)


def get_student_performance(student_id):
    """Analyze student strengths and weaknesses."""
    data = load_data()
    quizzes = data['quizzes'][data['quizzes']['student_id'] == student_id]
    
    if quizzes.empty:
        return {'strengths': [], 'weaknesses': [], 'scores': {}}
    
    # Merge with courses to get categories
    quiz_courses = quizzes.merge(
        data['courses'][['course_id', 'category']],
        on='course_id'
    )
    
    # Calculate average per category
    category_perf = quiz_courses.groupby('category')['score_percentage'].mean()
    
    strengths = category_perf[category_perf >= 75].index.tolist()
    weaknesses = category_perf[category_perf < 60].index.tolist()
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'scores': category_perf.to_dict()
    }


def get_quiz_history(student_id):
    """Get quiz score history for charts."""
    data = load_data()
    quizzes = data['quizzes'][data['quizzes']['student_id'] == student_id].copy()
    
    if quizzes.empty:
        return []
    
    quizzes['quiz_date'] = pd.to_datetime(quizzes['quiz_date'])
    quizzes = quizzes.sort_values('quiz_date')
    
    return quizzes[['quiz_date', 'score_percentage', 'quiz_title']].to_dict('records')


# =============================================================================
# ENROLLMENT & LEARNING HELPER FUNCTIONS
# =============================================================================

def is_enrolled(student_id, course_id):
    """Check if student is enrolled in a course."""
    data = load_data()
    enrollment = data['enrollments'][
        (data['enrollments']['student_id'] == student_id) &
        (data['enrollments']['course_id'] == course_id) &
        (data['enrollments']['status'].isin(['active', 'completed']))
    ]
    return not enrollment.empty


def enroll_student(student_id, course_id):
    """Enroll a student in a course."""
    if is_enrolled(student_id, course_id):
        return False, "Already enrolled"
    
    data = load_data()
    
    # Generate new enrollment ID
    existing_ids = data['enrollments']['enrollment_id'].tolist()
    max_num = max([int(eid[1:]) for eid in existing_ids if eid.startswith('E')])
    new_id = f'E{max_num +1:03d}'
    
    # Create enrollment record
    new_enrollment = {
        'enrollment_id': new_id,
        'student_id': student_id,
        'course_id': course_id,
        'enrollment_date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'active',
        'progress_percentage': 0,
        'last_accessed': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Append to CSV
    new_df = pd.DataFrame([new_enrollment])
    csv_path = os.path.join(DATA_DIR, 'enrollments.csv')
    new_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Reload data
    reload_data()
    
    return True, "Enrollment successful"


def unenroll_student(student_id, course_id):
    """Unenroll (drop) a student from a course."""
    data = load_data()
    enrollments = data['enrollments'].copy()
    
    # Update status to dropped
    mask = (enrollments['student_id'] == student_id) & (enrollments['course_id'] == course_id)
    if not enrollments[mask].empty:
        enrollments.loc[mask, 'status'] = 'dropped'
        
        # Save back to CSV
        csv_path = os.path.join(DATA_DIR, 'enrollments.csv')
        enrollments.to_csv(csv_path, index=False)
        
        reload_data()
        return True, "Unenrolled successfully"
    
    return False, "Not enrolled in this course"


def get_course_modules(course_id):
    """Get all modules for a course."""
    data = load_data()
    modules = data['course_modules'][data['course_modules']['course_id'] == course_id].copy()
    modules = modules.sort_values('order')
    return modules.to_dict('records')


def get_module_progress_for_student(student_id, course_id):
    """Get module completion status for a student in a course."""
    data = load_data()
    modules = data['course_modules'][data['course_modules']['course_id'] == course_id].copy()
    progress = data['module_progress'][data['module_progress']['student_id'] == student_id]
    
    # Merge to get completion status
    modules = modules.merge(
        progress[['module_id', 'completed', 'completion_date', 'score']],
        on='module_id',
        how='left'
    )
    
    # Fill NaN values
    modules['completed'] = modules['completed'].fillna(False)
    
    return modules.to_dict('records')


def get_quiz_questions(course_id, quiz_title):
    """Get quiz questions for a specific quiz."""
    data = load_data()
    questions = data['quiz_questions'][
        (data['quiz_questions']['course_id'] == course_id) &
        (data['quiz_questions']['quiz_title'] == quiz_title)
    ].copy()
    
    questions = questions.sort_values('question_number')
    return questions.to_dict('records')


def calculate_quiz_score(answers, course_id, quiz_title):
    """Calculate quiz score based on submitted answers."""
    questions = get_quiz_questions(course_id, quiz_title)
    
    if not questions:
        return 0, 0, 0
    
    correct = 0
    total = len(questions)
    total_points = sum(q['points'] for q in questions)
    earned_points = 0
    
    for q in questions:
        question_id = str(q['question_id'])
        if question_id in answers and answers[question_id] == q['correct_answer']:
            correct += 1
            earned_points += q['points']
    
    percentage = (correct / total * 100) if total > 0 else 0
    
    return correct, total, round(percentage, 1)


def save_quiz_result(student_id, course_id, quiz_title, correct, total, percentage, time_taken):
    """Save quiz result to database."""
    data = load_data()
    
    # Generate new quiz ID
    existing_ids = data['quizzes']['quiz_id'].tolist()
    max_num = max([int(qid[1:]) for qid in existing_ids if qid.startswith('Q')])
    new_id = f'Q{max_num + 1:03d}'
    
    # Count attempts
    previous_attempts = data['quizzes'][
        (data['quizzes']['student_id'] == student_id) &
        (data['quizzes']['course_id'] == course_id) &
        (data['quizzes']['quiz_title'] == quiz_title)
    ]
    attempts = len(previous_attempts) + 1
    
    # Create quiz result record
    new_quiz = {
        'quiz_id': new_id,
        'student_id': student_id,
        'course_id': course_id,
        'quiz_date': datetime.now().strftime('%Y-%m-%d'),
        'quiz_title': quiz_title,
        'total_questions': total,
        'correct_answers': correct,
        'score_percentage': percentage,
        'time_taken_minutes': time_taken,
        'attempts': attempts,
        'passed': percentage >= 60
    }
    
    # Append to CSV
    new_df = pd.DataFrame([new_quiz])
    csv_path = os.path.join(DATA_DIR, 'quiz_results.csv')
    new_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    reload_data()
    
    return new_id


def complete_module(student_id, module_id, score=None):
    """Mark a module as completed."""
    data = load_data()
    
    # Check if already completed
    existing = data['module_progress'][
        (data['module_progress']['student_id'] == student_id) &
        (data['module_progress']['module_id'] == module_id)
    ]
    
    if not existing.empty:
        return False, "Module already completed"
    
    # Generate new progress ID
    existing_ids = data['module_progress']['progress_id'].tolist()
    max_num = max([int(pid[1:]) for pid in existing_ids if pid.startswith('P')])
    new_id = f'P{max_num + 1:03d}'
    
    # Get module info
    module = data['course_modules'][data['course_modules']['module_id'] == module_id]
    if module.empty:
        return False, "Module not found"
    
    module_data = module.iloc[0]
    
    # Create progress record
    new_progress = {
        'progress_id': new_id,
        'student_id': student_id,
        'module_id': module_id,
        'completed': True,
        'completion_date': datetime.now().strftime('%Y-%m-%d'),
        'time_spent_minutes': module_data['duration_minutes'],
        'score': score if score is not None else ''
    }
    
    # Append to CSV
    new_df = pd.DataFrame([new_progress])
    csv_path = os.path.join(DATA_DIR, 'module_progress.csv')
    new_df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Update course progress
    update_course_progress(student_id, module_data['course_id'])
    
    reload_data()
    
    return True, "Module completed"


def update_course_progress(student_id, course_id):
    """Update overall course progress percentage."""
    data = load_data()
    
    # Get all modules for course
    all_modules = data['course_modules'][data['course_modules']['course_id'] == course_id]
    total_modules = len(all_modules)
    
    if total_modules == 0:
        return
    
    # Get completed modules
    module_ids = all_modules['module_id'].tolist()
    completed = data['module_progress'][
        (data['module_progress']['student_id'] == student_id) &
        (data['module_progress']['module_id'].isin(module_ids)) &
        (data['module_progress']['completed'] == True)
    ]
    
    completed_count = len(completed)
    progress = round((completed_count / total_modules) * 100, 1)
    
    # Update enrollment progress
    enrollments = data['enrollments'].copy()
    mask = (enrollments['student_id'] == student_id) & (enrollments['course_id'] == course_id)
    enrollments.loc[mask, 'progress_percentage'] = progress
    enrollments.loc[mask, 'last_accessed'] = datetime.now().strftime('%Y-%m-%d')
    
    # Mark as completed if 100%
    if progress >= 100:
        enrollments.loc[mask, 'status'] = 'completed'
    
    # Save back to CSV
    csv_path = os.path.join(DATA_DIR, 'enrollments.csv')
    enrollments.to_csv(csv_path, index=False)
    
    reload_data()



# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Redirect to login or dashboard."""
    if 'student_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip().upper()
        
        # Validate student ID
        student = get_student(student_id)
        if student:
            session['student_id'] = student_id
            session['student_name'] = student['name']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Student ID. Please try again.', 'error')
    
    # Get list of valid student IDs for demo
    data = load_data()
    sample_students = data['students'][['student_id', 'name']].head(5).to_dict('records')
    
    return render_template('login.html', sample_students=sample_students)


@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.clear()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new student."""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        age = request.form.get('age', 20)
        education = request.form.get('education', 'Undergraduate')
        major = request.form.get('major', 'Computer Science')
        learning_style = request.form.get('learning_style', 'Visual')
        preferred_difficulty = request.form.get('preferred_difficulty', 'Intermediate')
        
        if not name or not email:
            flash('Name and Email are required.', 'error')
            return render_template('register.html')
        
        # Generate new student ID
        data = load_data()
        existing_ids = data['students']['student_id'].tolist()
        max_num = max([int(sid[1:]) for sid in existing_ids if sid.startswith('S')])
        new_id = f'S{max_num + 1:03d}'
        
        # Create new student record
        new_student = {
            'student_id': new_id,
            'name': name,
            'email': email,
            'age': int(age),
            'education_level': education,
            'major': major,
            'learning_style': learning_style,
            'preferred_difficulty': preferred_difficulty,
            'enrollment_date': datetime.now().strftime('%Y-%m-%d'),
            'total_courses_completed': 0,
            'avg_quiz_score': 0.0,
            'total_time_spent_hours': 0.0
        }
        
        # Append to CSV
        new_df = pd.DataFrame([new_student])
        csv_path = os.path.join(DATA_DIR, 'students.csv')
        new_df.to_csv(csv_path, mode='a', header=False, index=False)
        
        # Reload data
        reload_data()
        
        # Auto-login
        session['student_id'] = new_id
        session['student_name'] = name
        
        flash(f'Registration successful! Your Student ID is {new_id}', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    """Student dashboard."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    student = get_student(student_id)
    courses = get_student_courses(student_id)
    performance = get_student_performance(student_id)
    
    return render_template('dashboard.html',
                         student=student,
                         courses=courses,
                         performance=performance)


@app.route('/recommendations')
def recommendations():
    """Recommendations page with explanations."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    data = load_data()
    recommender = get_recommender()
    explainer = get_explainer()
    
    # Get recommendations
    recs = recommender.recommend(student_id, n=10)
    recs = apply_recommendation_business_rules(student_id, recs, data, limit=10)
    recommendations_list = recs.to_dict('records') if not recs.empty else []
    
    # Add explanations for each recommendation
    for rec in recommendations_list:
        explanations = explainer.explain(
            student_id, rec['course_id'],
            data['students'], data['activities'], data['ratings'], data['courses']
        )
        rec['explanations'] = explanations
    
    return render_template('recommendations.html',
                         recommendations=recommendations_list)


@app.route('/learning-path')
def learning_path():
    """Personalized learning path."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    category = request.args.get('category', 'Data Science')
    
    recommender = get_recommender()
    data = load_data()
    
    # Build learning path
    path = recommender.build_learning_path(student_id, category, n_courses=6)
    path_list = path.to_dict('records') if not path.empty else []
    
    # Get all unique categories
    categories = data['courses']['category'].unique().tolist()
    
    return render_template('learning_path.html',
                         path=path_list,
                         category=category,
                         categories=categories)


@app.route('/similar-courses')
def similar_courses():
    """Similar courses page."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    data = load_data()
    course_id = request.args.get('course_id', 'C004')
    
    recommender = get_recommender()
    
    # Get similar courses
    similar = recommender.content_recommender.get_similar_courses(course_id, n=10)
    
    # Get course details
    similar_list = []
    for cid, score in similar:
        course = data['courses'][data['courses']['course_id'] == cid]
        if not course.empty:
            course_dict = course.iloc[0].to_dict()
            course_dict['similarity'] = round(score * 100, 2)
            similar_list.append(course_dict)
    
    # Get selected course info
    selected_course = data['courses'][data['courses']['course_id'] == course_id]
    selected = selected_course.iloc[0].to_dict() if not selected_course.empty else None
    
    # All courses for dropdown
    all_courses = data['courses'][['course_id', 'title']].to_dict('records')
    
    return render_template('similar_courses.html',
                         similar=similar_list,
                         selected=selected,
                         all_courses=all_courses,
                         course_id=course_id)


@app.route('/analytics')
def analytics():
    """Analytics page with charts."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    data = load_data()
    
    # Quiz history
    quiz_history = get_quiz_history(student_id)
    
    # Category performance
    performance = get_student_performance(student_id)
    
    # Activity stats
    activities = data['activities'][data['activities']['student_id'] == student_id]
    total_time = activities['time_spent_minutes'].sum() / 60  # hours
    total_videos = activities['video_watched'].sum()
    
    # Completion rates
    courses = get_student_courses(student_id)
    
    return render_template('analytics.html',
                         quiz_history=quiz_history,
                         performance=performance,
                         total_time=round(total_time, 1),
                         total_videos=int(total_videos),
                         courses=courses)


@app.route('/admin')
def admin():
    """Admin data tables page."""
    data = load_data()
    
    return render_template('admin.html',
                         students=data['students'].head(20).to_dict('records'),
                         courses=data['courses'].to_dict('records'),
                         activities=data['activities'].head(50).to_dict('records'),
                         quizzes=data['quizzes'].head(50).to_dict('records'),
                         ratings=data['ratings'].head(50).to_dict('records'))


@app.route('/grade-predictor')
def grade_predictor():
    """Grade predictor page."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    data = load_data()
    course_id = request.args.get('course_id', None)
    
    # Get all courses for dropdown
    all_courses = data['courses'][['course_id', 'title', 'difficulty']].to_dict('records')
    
    prediction = None
    selected_course = None
    
    if course_id:
        predictor = get_grade_predictor()
        prediction = predictor.predict(student_id, course_id, data['students'], data['activities'], data['courses'])
        
        course = data['courses'][data['courses']['course_id'] == course_id]
        if not course.empty:
            selected_course = course.iloc[0].to_dict()
    
    return render_template('grade_predictor.html',
                         all_courses=all_courses,
                         prediction=prediction,
                         selected_course=selected_course,
                         course_id=course_id)


@app.route('/student-profile')
def student_profile():
    """Student profile with ML clustering."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    data = load_data()
    
    # Get student details
    student = get_student(student_id)
    
    # Get cluster profile
    clusterer = get_clusterer()
    profile = clusterer.get_cluster(student_id, data['students'], data['activities'], data['quizzes'])
    
    # Get cluster distribution
    distribution = clusterer.get_all_clusters(data['students'], data['activities'], data['quizzes'])
    
    # Get performance
    performance = get_student_performance(student_id)
    
    return render_template('student_profile.html',
                         student=student,
                         profile=profile,
                         distribution=distribution,
                         performance=performance)


@app.route('/courses')
def courses():
    """Course catalog page."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    data = load_data()
    student_id = session['student_id']
    
    # Get all courses
    all_courses = data['courses'].copy()
    
    # Get enrolled course IDs
    enrolled = data['enrollments'][(data['enrollments']['student_id'] == student_id) &
                                    (data['enrollments']['status'].isin(['active', 'completed']))]
    enrolled_ids = enrolled['course_id'].tolist()
    
    # Mark enrolled courses
    all_courses['enrolled'] = all_courses['course_id'].isin(enrolled_ids)
    
    # Filter by category if provided
    category = request.args.get('category', '')
    difficulty = request.args.get('difficulty', '')
    search = request.args.get('search', '')
    
    if category:
        all_courses = all_courses[all_courses['category'] == category]
    if difficulty:
        all_courses = all_courses[all_courses['difficulty'] == difficulty]
    if search:
        all_courses = all_courses[
            all_courses['title'].str.contains(search, case=False,na=False) |
            all_courses['description'].str.contains(search, case=False, na=False)
        ]
    
    #Get unique categories and difficulties
    categories = data['courses']['category'].unique().tolist()
    difficulties = data['courses']['difficulty'].unique().tolist()
    
    return render_template('courses.html',
                          courses=all_courses.to_dict('records'),
                          categories=categories,
                          difficulties=difficulties,
                          selected_category=category,
                          selected_difficulty=difficulty,
                          search_query=search)


@app.route('/courses/<course_id>')
def course_detail(course_id):
    """Course detail page."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    data = load_data()
    student_id = session['student_id']
    
    # Get course details
    course = data['courses'][data['courses']['course_id'] == course_id]
    if course.empty:
        flash('Course not found', 'error')
        return redirect(url_for('courses'))
    
    course_data = course.iloc[0].to_dict()
    
    # Check enrollment status
    enrolled = is_enrolled(student_id, course_id)
    
    # Get course modules/syllabus
    modules = get_course_modules(course_id)
    
    # Get ratings
    ratings = data['ratings'][data['ratings']['course_id'] == course_id]
    rating_list = ratings.to_dict('records') if not ratings.empty else []
    
    return render_template('course_detail.html',
                          course=course_data,
                          enrolled=enrolled,
                          modules=modules,
                          ratings=rating_list)


@app.route('/courses/<course_id>/learn')
def course_learn(course_id):
    """Course learning interface."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    
    # Check enrollment
    if not is_enrolled(student_id, course_id):
        flash('You must enroll in this course first', 'error')
        return redirect(url_for('course_detail', course_id=course_id))
    
    data = load_data()
    
    # Get course details
    course = data['courses'][data['courses']['course_id'] == course_id]
    if course.empty:
        flash('Course not found', 'error')
        return redirect(url_for('courses'))
    
    course_data = course.iloc[0].to_dict()
    
    # Get modules with progress
    modules = get_module_progress_for_student(student_id, course_id)
    
    # Get current module (from query param or first incomplete)
    module_id = request.args.get('module_id', '')
    current_module = None
    
    if module_id:
        current_module = next((m for m in modules if m['module_id'] == module_id), None)
    
    if not current_module and modules:
        # Find first incomplete module
        current_module = next((m for m in modules if not m['completed']), modules[0])
    
    return render_template('course_learn.html',
                          course=course_data,
                          modules=modules,
                          current_module=current_module)


@app.route('/courses/<course_id>/quiz/<quiz_title>',methods=['GET', 'POST'])
def quiz_take(course_id, quiz_title):
    """Quiz taking interface."""
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    student_id = session['student_id']
    
    # Check enrollment
    if not is_enrolled(student_id, course_id):
        flash('You must enroll in this course first', 'error')
        return redirect(url_for('course_detail', course_id=course_id))
    
    if request.method == 'POST':
        # Process quiz submission
        start_time = request.form.get('start_time', '')
        answers = {}
        
        # Collect answers
        for key in request.form.keys():
            if key.startswith('question_'):
                question_id = key.replace('question_', '')
                answers[question_id] = request.form.get(key)
        
        # Calculate score
        correct, total, percentage = calculate_quiz_score(answers, course_id, quiz_title)
        
        # Calculate time taken (simplified - in minutes)
        time_taken = 15  # Default 15 minutes for now
        if start_time:
            try:
                start = datetime.fromisoformat(start_time)
                time_taken = int((datetime.now() - start).total_seconds() / 60)
            except:
                pass
        
        # Save result
        quiz_id = save_quiz_result(student_id, course_id, quiz_title, correct, total, percentage, time_taken)
        
        # Get questions with correct answers for review
        questions = get_quiz_questions(course_id, quiz_title)
        
        return render_template('quiz_result.html',
                              quiz_title=quiz_title,
                              course_id=course_id,
                              correct=correct,
                              total=total,
                              percentage=percentage,
                              passed=percentage >= 60,
                              questions=questions,
                              answers=answers)
    
    # GET request - show quiz
    questions = get_quiz_questions(course_id, quiz_title)
    
    if not questions:
        flash('Quiz questions not available', 'error')
        return redirect(url_for('course_learn', course_id=course_id))
    
    data = load_data()
    course = data['courses'][data['courses']['course_id'] == course_id]
    course_data = course.iloc[0].to_dict() if not course.empty else {}
    
    return render_template('quiz_take.html',
                          quiz_title=quiz_title,
                          course=course_data,
                          questions=questions,
                          start_time=datetime.now().isoformat())


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/recommend/<student_id>')
def api_recommend(student_id):
    """API: Get recommendations for a student."""
    data = load_data()
    recommender = get_recommender()
    recs = recommender.recommend(student_id, n=10)
    recs = apply_recommendation_business_rules(student_id, recs, data, limit=10)
    
    if recs.empty:
        return jsonify({'error': 'No recommendations found', 'data': []})
    
    return jsonify({
        'student_id': student_id,
        'recommendations': recs.to_dict('records')
    })


@app.route('/api/learning-path/<student_id>')
def api_learning_path(student_id):
    """API: Get learning path for a student."""
    category = request.args.get('category', 'Data Science')
    
    recommender = get_recommender()
    path = recommender.build_learning_path(student_id, category, n_courses=6)
    
    if path.empty:
        return jsonify({'error': 'No learning path generated', 'data': []})
    
    return jsonify({
        'student_id': student_id,
        'category': category,
        'learning_path': path.to_dict('records')
    })


@app.route('/api/similar/<course_id>')
def api_similar(course_id):
    """API: Get similar courses."""
    data = load_data()
    recommender = get_recommender()
    
    similar = recommender.content_recommender.get_similar_courses(course_id, n=10)
    
    similar_list = []
    for cid, score in similar:
        course = data['courses'][data['courses']['course_id'] == cid]
        if not course.empty:
            course_dict = course.iloc[0].to_dict()
            course_dict['similarity'] = round(score, 4)
            similar_list.append(course_dict)
    
    return jsonify({
        'course_id': course_id,
        'similar_courses': similar_list
    })


@app.route('/api/student/<student_id>')
def api_student(student_id):
    """API: Get student details and stats."""
    student = get_student(student_id)
    
    if not student:
        return jsonify({'error': 'Student not found'}), 404
    
    courses = get_student_courses(student_id)
    performance = get_student_performance(student_id)
    
    return jsonify({
        'student': student,
        'enrolled_courses': courses,
        'performance': performance
    })


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """API: Enroll a student in a course."""
    if 'student_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    student_id = session['student_id']
    course_id = request.json.get('course_id', '')
    
    if not course_id:
        return jsonify({'error': 'Course ID required'}), 400
    
    success, message = enroll_student(student_id, course_id)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400


@app.route('/api/unenroll', methods=['POST'])
def api_unenroll():
    """API: Unenroll a student from a course."""
    if 'student_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    student_id = session['student_id']
    course_id = request.json.get('course_id', '')
    
    if not course_id:
        return jsonify({'error': 'Course ID required'}), 400
    
    success, message = unenroll_student(student_id, course_id)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400


@app.route('/api/complete-module', methods=['POST'])
def api_complete_module():
    """API: Mark a module as completed."""
    if 'student_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    student_id = session['student_id']
    module_id = request.json.get('module_id', '')
    score = request.json.get('score', None)
    
    if not module_id:
        return jsonify({'error': 'Module ID required'}), 400
    
    success, message = complete_module(student_id, module_id, score)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DEHA- E-LEARNING RECOMMENDATION SYSTEM - WEB APPLICATION")
    print("=" * 60)
    print("\n  Starting server at: http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    
    app.run(debug=True, port=5000)

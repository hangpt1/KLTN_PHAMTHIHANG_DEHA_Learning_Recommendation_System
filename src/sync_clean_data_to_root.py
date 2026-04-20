

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


STUDENTS_COLUMNS = [
    'student_id', 'name', 'email', 'age', 'education_level', 'major',
    'enrollment_date', 'learning_style', 'preferred_difficulty',
    'total_courses_completed', 'avg_quiz_score', 'total_time_spent_hours'
]
RATINGS_COLUMNS = [
    'rating_id', 'student_id', 'course_id', 'rating_date', 'rating',
    'review_text', 'difficulty_rating', 'content_quality',
    'instructor_rating', 'would_recommend'
]
ENROLLMENTS_COLUMNS = [
    'enrollment_id', 'student_id', 'course_id', 'enrollment_date',
    'status', 'progress_percentage', 'last_accessed'
]
QUIZ_COLUMNS = [
    'quiz_id', 'student_id', 'course_id', 'quiz_date', 'quiz_title',
    'total_questions', 'correct_answers', 'score_percentage',
    'time_taken_minutes', 'attempts', 'passed'
]
ACTIVITY_COLUMNS = [
    'log_id', 'student_id', 'course_id', 'activity_type', 'activity_date',
    'time_spent_minutes', 'progress_percentage', 'module_completed',
    'video_watched', 'notes_taken', 'resources_downloaded'
]


def stable_fraction(*parts: object) -> float:
    key = '|'.join(str(part) for part in parts)
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def spread_dates(start: pd.Timestamp, end: pd.Timestamp, count: int,
                 *parts: object) -> list[pd.Timestamp]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        end = start
    if count <= 1:
        return [start]
    span = max((end - start).days, 0)
    points = []
    for idx in range(count):
        base = idx / (count - 1)
        jitter = (stable_fraction(*parts, idx) - 0.5) * 0.15
        ratio = min(max(base + jitter, 0), 1)
        points.append(start + pd.Timedelta(days=int(round(span * ratio))))
    return sorted(points)


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d')


def build_dense_activity_logs(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    log_num = 1

    for row in interactions.itertuples(index=False):
        start = pd.to_datetime(row.first_event_date)
        end = pd.to_datetime(row.last_event_date)
        if pd.isna(start) and pd.isna(end):
            continue
        if pd.isna(start):
            start = end
        if pd.isna(end):
            end = start
        if end < start:
            end = start

        target_progress = max(
            float(getattr(row, 'enrollment_progress', 0) or 0),
            float(getattr(row, 'activity_progress', 0) or 0)
        )
        if row.signal_tier == 'strong' and target_progress < 45:
            target_progress = 45
        elif row.signal_tier == 'medium' and target_progress < 25:
            target_progress = 25
        elif target_progress == 0 and row.positive_label:
            target_progress = 20

        if row.signal_tier == 'strong':
            log_count = 4
        elif row.signal_tier == 'medium':
            log_count = 3
        else:
            log_count = 2 if (row.has_enrollment or row.has_quiz or row.has_activity) else 1

        if row.has_quiz and log_count < 3:
            log_count = 3

        dates = spread_dates(start, end, log_count, row.student_id, row.course_id, row.signal_tier)
        total_time = float(getattr(row, 'total_time_spent', 0) or 0)
        if total_time <= 0:
            total_time = 20 + 80 * stable_fraction(row.student_id, row.course_id, 'time')
        time_weights = np.linspace(1.0, 0.7, log_count)
        time_weights = time_weights / time_weights.sum()
        progress_steps = np.linspace(
            max(5, min(target_progress, 12)) if target_progress > 0 else 5,
            max(target_progress, 5),
            log_count
        )
        progress_steps = np.round(progress_steps).astype(int)

        for idx in range(log_count):
            if idx == log_count - 1 and row.has_quiz:
                activity_type = 'quiz_attempt'
            elif idx == log_count - 2 and row.signal_tier in ('strong', 'medium'):
                activity_type = 'assignment'
            else:
                activity_type = 'video_watch'

            progress = int(progress_steps[idx])
            module_completed = max(1, int(np.ceil(progress / 25))) if progress > 0 else 1
            time_spent = max(10, int(round(total_time * time_weights[idx])))

            rows.append({
                'log_id': f'L{log_num:04d}',
                'student_id': row.student_id,
                'course_id': row.course_id,
                'activity_type': activity_type,
                'activity_date': dates[idx].strftime('%Y-%m-%d'),
                'time_spent_minutes': time_spent,
                'progress_percentage': min(progress, 100),
                'module_completed': module_completed,
                'video_watched': activity_type == 'video_watch',
                'notes_taken': activity_type == 'assignment',
                'resources_downloaded': activity_type in ('assignment', 'quiz_attempt')
            })
            log_num += 1

    activity_df = pd.DataFrame(rows, columns=ACTIVITY_COLUMNS)
    activity_df = activity_df.sort_values(['student_id', 'course_id', 'activity_date', 'log_id']).reset_index(drop=True)
    return activity_df


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    clean_dir = data_dir / 'cleaned'

    students = pd.read_csv(clean_dir / 'students_clean.csv')
    ratings = pd.read_csv(clean_dir / 'ratings_clean.csv')
    enrollments = pd.read_csv(clean_dir / 'enrollments_clean.csv')
    quizzes = pd.read_csv(clean_dir / 'quiz_results_clean.csv')
    interactions = pd.read_csv(clean_dir / 'interaction_quality.csv')

    students_out = students[STUDENTS_COLUMNS].copy()
    students_out['enrollment_date'] = normalize_date(students['student_enrollment_date_clean'])

    ratings_out = ratings[RATINGS_COLUMNS].copy()
    ratings_out['rating_date'] = normalize_date(ratings['rating_date_clean'])

    enrollments_out = enrollments[ENROLLMENTS_COLUMNS].copy()
    enrollments_out['enrollment_date'] = normalize_date(enrollments['enrollment_date_clean'])
    enrollments_out['last_accessed'] = normalize_date(enrollments['last_accessed_clean'])
    enrollments_out = enrollments_out.sort_values(
        ['student_id', 'course_id', 'enrollment_date', 'enrollment_id']
    ).reset_index(drop=True)

    quizzes_out = quizzes[QUIZ_COLUMNS].copy()
    quizzes_out['quiz_date'] = normalize_date(quizzes['quiz_date_clean'])
    quizzes_out = quizzes_out.sort_values(
        ['student_id', 'course_id', 'quiz_date', 'quiz_id']
    ).reset_index(drop=True)

    activities_out = build_dense_activity_logs(interactions)

    students_out.to_csv(data_dir / 'students.csv', index=False)
    ratings_out.to_csv(data_dir / 'ratings.csv', index=False)
    enrollments_out.to_csv(data_dir / 'enrollments.csv', index=False)
    quizzes_out.to_csv(data_dir / 'quiz_results.csv', index=False)
    activities_out.to_csv(data_dir / 'activity_logs.csv', index=False)

    print({
        'students_rows': len(students_out),
        'ratings_rows': len(ratings_out),
        'enrollments_rows': len(enrollments_out),
        'quiz_rows': len(quizzes_out),
        'activity_rows': len(activities_out),
        'activity_students': activities_out['student_id'].nunique(),
        'activity_pairs': activities_out[['student_id', 'course_id']].drop_duplicates().shape[0],
    })


if __name__ == '__main__':
    main()

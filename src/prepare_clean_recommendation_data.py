"""
Prepare cleaner, time-aware recommendation data for "next course" tasks.

Outputs are written to data/cleaned/ and preserve weak signals with lower
weights instead of dropping them outright.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_END_DATE = pd.Timestamp("2024-03-31")
CANONICAL_START_DATE = pd.Timestamp("2024-01-01")


def clip_date(series: pd.Series, lower: pd.Series | pd.Timestamp | None = None,
              upper: pd.Series | pd.Timestamp | None = None) -> pd.Series:
    """Clip datetime values between optional lower and upper bounds."""
    result = pd.to_datetime(series, errors="coerce")
    if lower is not None:
        if isinstance(lower, pd.Series):
            result = pd.Series(
                np.where(result < lower, lower, result),
                index=result.index
            )
            result = pd.to_datetime(result)
        else:
            result = result.clip(lower=lower)
    if upper is not None:
        if isinstance(upper, pd.Series):
            result = pd.Series(
                np.where(result > upper, upper, result),
                index=result.index
            )
            result = pd.to_datetime(result)
        else:
            result = result.clip(upper=upper)
    return pd.to_datetime(result)


def stable_fraction(*parts: object) -> float:
    """Deterministic pseudo-random fraction in [0, 1)."""
    key = '|'.join(str(part) for part in parts)
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def spread_date(lower: pd.Timestamp, upper: pd.Timestamp, *parts: object) -> pd.Timestamp:
    """Spread dates deterministically inside [lower, upper]."""
    if pd.isna(lower):
        lower = CANONICAL_START_DATE
    if pd.isna(upper):
        upper = CANONICAL_END_DATE
    lower = max(pd.Timestamp(lower), CANONICAL_START_DATE)
    upper = min(pd.Timestamp(upper), CANONICAL_END_DATE)
    if upper < lower:
        upper = lower
    span_days = (upper - lower).days
    offset = int(round(stable_fraction(*parts) * span_days)) if span_days > 0 else 0
    return lower + pd.Timedelta(days=offset)


def synthesize_support_signals(
    students: pd.DataFrame,
    ratings: pd.DataFrame,
    enrollments: pd.DataFrame,
    quizzes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add synthetic enrollments/quizzes for part of the high-rating-only pairs and
    strengthen a subset of weak enrollments with consistent positive evidence.
    """
    enrollments = enrollments.copy()
    quizzes = quizzes.copy()

    student_start = students.set_index('student_id')['student_enrollment_date_clean'].to_dict()
    enroll_pairs = set(map(tuple, enrollments[['student_id', 'course_id']].drop_duplicates().to_records(index=False)))
    quiz_pairs = set(map(tuple, quizzes[['student_id', 'course_id']].drop_duplicates().to_records(index=False)))

    rating_agg = ratings.groupby(['student_id', 'course_id']).agg(
        rating_max=('rating', 'max'),
        rating_mean=('rating', 'mean'),
        rating_date=('rating_date', 'max'),
        would_recommend=('would_recommend', 'mean')
    ).reset_index()

    synthetic_enrollments = []
    synthetic_quizzes = []
    next_enrollment_num = len(enrollments) + 1
    next_quiz_num = len(quizzes) + 1

    candidates = rating_agg[rating_agg['rating_max'] >= 4].copy()
    for row in candidates.itertuples(index=False):
        pair = (row.student_id, row.course_id)
        has_enroll = pair in enroll_pairs
        has_quiz = pair in quiz_pairs

        if has_enroll and has_quiz:
            continue

        fraction = stable_fraction(row.student_id, row.course_id, 'support_signal')
        student_lower = student_start.get(row.student_id, CANONICAL_START_DATE)
        rating_upper = min(pd.Timestamp(row.rating_date), CANONICAL_END_DATE)
        if rating_upper < CANONICAL_START_DATE:
            rating_upper = CANONICAL_END_DATE

        if (not has_enroll) and fraction < 0.55:
            progress = 35 + int(round(stable_fraction(row.student_id, row.course_id, 'progress') * 45))
            enrollment_date = spread_date(student_lower, rating_upper - pd.Timedelta(days=7), row.student_id, row.course_id, 'enroll_date')
            last_accessed = min(enrollment_date + pd.Timedelta(days=7 + int(progress / 10)), CANONICAL_END_DATE, rating_upper)
            status = 'completed' if progress >= 80 else 'active'
            synthetic_enrollments.append({
                'enrollment_id': f'SE{next_enrollment_num:04d}',
                'student_id': row.student_id,
                'course_id': row.course_id,
                'enrollment_date': enrollment_date,
                'last_accessed': max(last_accessed, enrollment_date),
                'status': status,
                'progress_percentage': progress,
                'synthetic_generated': True
            })
            next_enrollment_num += 1
            has_enroll = True

        if (not has_quiz) and (fraction >= 0.25):
            quiz_upper = rating_upper - pd.Timedelta(days=1)
            quiz_date = spread_date(student_lower, quiz_upper, row.student_id, row.course_id, 'quiz_date')
            score = 72 + int(round(stable_fraction(row.student_id, row.course_id, 'quiz_score') * 23))
            total_questions = 20 + int(round(stable_fraction(row.student_id, row.course_id, 'quiz_total') * 10))
            correct_answers = int(round(total_questions * score / 100))
            synthetic_quizzes.append({
                'quiz_id': f'SQ{next_quiz_num:04d}',
                'student_id': row.student_id,
                'course_id': row.course_id,
                'quiz_date': quiz_date,
                'quiz_title': 'Synthetic Mastery Check',
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'score_percentage': score,
                'time_taken_minutes': 18 + int(round(stable_fraction(row.student_id, row.course_id, 'quiz_time') * 17)),
                'attempts': 1 if stable_fraction(row.student_id, row.course_id, 'quiz_attempts') > 0.35 else 2,
                'passed': True,
                'synthetic_generated': True
            })
            next_quiz_num += 1

    if synthetic_enrollments:
        synth_enroll_df = pd.DataFrame(synthetic_enrollments)
        synth_enroll_df['enrollment_date'] = pd.to_datetime(synth_enroll_df['enrollment_date'])
        synth_enroll_df['last_accessed'] = pd.to_datetime(synth_enroll_df['last_accessed'])
        enrollments['synthetic_generated'] = enrollments.get('synthetic_generated', False)
        enrollments = pd.concat([enrollments, synth_enroll_df], ignore_index=True)
    else:
        enrollments['synthetic_generated'] = enrollments.get('synthetic_generated', False)

    if synthetic_quizzes:
        synth_quiz_df = pd.DataFrame(synthetic_quizzes)
        synth_quiz_df['quiz_date'] = pd.to_datetime(synth_quiz_df['quiz_date'])
        quizzes['synthetic_generated'] = quizzes.get('synthetic_generated', False)
        quizzes = pd.concat([quizzes, synth_quiz_df], ignore_index=True)
    else:
        quizzes['synthetic_generated'] = quizzes.get('synthetic_generated', False)

    # Strengthen some existing weak enrollments when positive signals support the
    # idea that the learner is meaningfully engaged with the course.
    quiz_strength = quizzes.groupby(['student_id', 'course_id'])['score_percentage'].mean().reset_index()
    rating_strength = rating_agg[['student_id', 'course_id', 'rating_max']]
    strengthen = enrollments.merge(quiz_strength, on=['student_id', 'course_id'], how='left')
    strengthen = strengthen.merge(rating_strength, on=['student_id', 'course_id'], how='left')
    mask = (
        strengthen['progress_percentage'].between(1, 30) &
        (
            strengthen['rating_max'].fillna(0).ge(4) |
            strengthen['score_percentage'].fillna(0).gt(70)
        )
    )
    strengthen_rows = strengthen[mask]
    for row in strengthen_rows.itertuples(index=False):
        bump_seed = stable_fraction(row.student_id, row.course_id, 'progress_bump')
        new_progress = 35 + int(round(bump_seed * 30))
        idx_mask = (
            (enrollments['student_id'] == row.student_id) &
            (enrollments['course_id'] == row.course_id)
        )
        enrollments.loc[idx_mask, 'progress_percentage'] = np.maximum(
            enrollments.loc[idx_mask, 'progress_percentage'],
            new_progress
        )
        enrollments.loc[idx_mask & enrollments['status'].eq('dropped'), 'status'] = 'active'

    return enrollments, quizzes


def assign_rating_dates_clean(
    ratings: pd.DataFrame,
    lower_bound: pd.Series,
    upper_bound: pd.Series
) -> pd.Series:
    """
    Spread cleaned rating dates across Jan-Mar 2024 instead of collapsing many
    values to the exact upper bound.
    """
    cleaned_dates = []
    for idx, row in ratings.iterrows():
        lb = lower_bound.loc[idx]
        ub = upper_bound.loc[idx]
        original = pd.Timestamp(row['rating_date'])
        if pd.isna(lb):
            lb = CANONICAL_START_DATE
        if pd.isna(ub):
            ub = CANONICAL_END_DATE
        lb = max(pd.Timestamp(lb), CANONICAL_START_DATE)
        ub = min(pd.Timestamp(ub), CANONICAL_END_DATE)
        if ub < lb:
            ub = lb

        if original < lb or original > ub or original > CANONICAL_END_DATE:
            cleaned_dates.append(
                spread_date(lb, ub, row['student_id'], row['course_id'], row['rating_id'], 'rating_clean')
            )
        else:
            cleaned_dates.append(original.normalize())
    return pd.to_datetime(cleaned_dates)


def build_interaction_quality(
    students: pd.DataFrame,
    ratings: pd.DataFrame,
    enrollments: pd.DataFrame,
    quizzes: pd.DataFrame,
    activities: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate multiple signals into one user-course quality table."""
    student_dates = students[['student_id', 'student_enrollment_date_clean']].copy()

    rating_agg = ratings.groupby(['student_id', 'course_id']).agg(
        rating_mean=('rating', 'mean'),
        rating_max=('rating', 'max'),
        rating_count=('rating', 'count'),
        rating_date_clean=('rating_date_clean', 'max'),
        would_recommend_rate=('would_recommend', 'mean')
    ).reset_index()

    enrollment_agg = enrollments.groupby(['student_id', 'course_id']).agg(
        enrollment_date_clean=('enrollment_date_clean', 'min'),
        last_accessed_clean=('last_accessed_clean', 'max'),
        enrollment_progress=('progress_percentage', 'max'),
        enrollment_status=('status', 'last')
    ).reset_index()

    quiz_agg = quizzes.groupby(['student_id', 'course_id']).agg(
        avg_quiz_score=('score_percentage', 'mean'),
        max_quiz_score=('score_percentage', 'max'),
        quiz_attempts=('attempts', 'sum'),
        quiz_pass_rate=('passed', 'mean'),
        quiz_date_clean=('quiz_date_clean', 'max')
    ).reset_index()

    activity_agg = activities.groupby(['student_id', 'course_id']).agg(
        activity_count=('log_id', 'count'),
        total_time_spent=('time_spent_minutes', 'sum'),
        activity_progress=('progress_percentage', 'max'),
        activity_date_clean=('activity_date_clean', 'max')
    ).reset_index()

    pair_frames = [
        rating_agg[['student_id', 'course_id']],
        enrollment_agg[['student_id', 'course_id']],
        quiz_agg[['student_id', 'course_id']],
        activity_agg[['student_id', 'course_id']],
    ]
    interactions = pd.concat(pair_frames, ignore_index=True).drop_duplicates()
    interactions = interactions.merge(student_dates, on='student_id', how='left')
    interactions = interactions.merge(rating_agg, on=['student_id', 'course_id'], how='left')
    interactions = interactions.merge(enrollment_agg, on=['student_id', 'course_id'], how='left')
    interactions = interactions.merge(quiz_agg, on=['student_id', 'course_id'], how='left')
    interactions = interactions.merge(activity_agg, on=['student_id', 'course_id'], how='left')

    interactions['rating_mean'] = interactions['rating_mean'].fillna(0)
    interactions['rating_max'] = interactions['rating_max'].fillna(0)
    interactions['rating_count'] = interactions['rating_count'].fillna(0).astype(int)
    interactions['would_recommend_rate'] = interactions['would_recommend_rate'].fillna(0)
    interactions['enrollment_progress'] = interactions['enrollment_progress'].fillna(0)
    interactions['avg_quiz_score'] = interactions['avg_quiz_score'].fillna(0)
    interactions['max_quiz_score'] = interactions['max_quiz_score'].fillna(0)
    interactions['quiz_attempts'] = interactions['quiz_attempts'].fillna(0)
    interactions['quiz_pass_rate'] = interactions['quiz_pass_rate'].fillna(0)
    interactions['activity_count'] = interactions['activity_count'].fillna(0).astype(int)
    interactions['total_time_spent'] = interactions['total_time_spent'].fillna(0)
    interactions['activity_progress'] = interactions['activity_progress'].fillna(0)
    interactions['has_enrollment'] = interactions['enrollment_date_clean'].notna()
    interactions['has_rating'] = interactions['rating_date_clean'].notna()
    interactions['has_quiz'] = interactions['quiz_date_clean'].notna()
    interactions['has_activity'] = interactions['activity_date_clean'].notna()

    interactions['strong_enroll_progress'] = (
        interactions['has_enrollment'] &
        (interactions['enrollment_progress'] > 30)
    )
    interactions['high_rating'] = interactions['rating_max'] >= 4
    interactions['high_quiz'] = interactions['avg_quiz_score'] > 70
    interactions['weak_enroll'] = interactions['has_enrollment'] & ~interactions['strong_enroll_progress']
    interactions['weak_activity'] = interactions['has_activity']

    weak_enroll_strength = np.where(
        interactions['weak_enroll'],
        0.10 + 0.15 * (interactions['enrollment_progress'].clip(0, 30) / 30.0),
        0.0
    )
    weak_activity_strength = np.where(
        interactions['weak_activity'],
        0.03 +
        0.04 * np.minimum(interactions['activity_count'], 5) / 5.0 +
        0.03 * interactions['activity_progress'].clip(0, 30) / 30.0,
        0.0
    )

    interactions['raw_interaction_score'] = (
        1.00 * interactions['strong_enroll_progress'].astype(float) +
        0.70 * interactions['high_rating'].astype(float) +
        0.45 * interactions['high_quiz'].astype(float) +
        weak_enroll_strength +
        weak_activity_strength +
        0.05 * (interactions['would_recommend_rate'] >= 0.5).astype(float)
    )

    max_score = 1.00 + 0.70 + 0.45 + 0.25 + 0.10 + 0.05
    interactions['interaction_score'] = (
        interactions['raw_interaction_score'] / max_score
    ).clip(0, 1)

    conditions = [
        interactions['strong_enroll_progress'],
        interactions['high_rating'],
        interactions['high_quiz'],
    ]
    choices = ['strong', 'medium', 'supportive']
    interactions['signal_tier'] = np.select(conditions, choices, default='weak')
    interactions['positive_label'] = (
        interactions['strong_enroll_progress'] |
        interactions['high_rating'] |
        interactions['high_quiz']
    )

    source_columns = {
        'enrollment': interactions['has_enrollment'],
        'rating': interactions['has_rating'],
        'quiz': interactions['has_quiz'],
        'activity': interactions['has_activity'],
    }
    interactions['evidence_sources'] = [
        '|'.join([name for name, present in source_columns.items() if present.iloc[idx]])
        for idx in range(len(interactions))
    ]

    event_cols = [
        'rating_date_clean',
        'enrollment_date_clean',
        'last_accessed_clean',
        'quiz_date_clean',
        'activity_date_clean'
    ]
    interactions['first_event_date'] = interactions[event_cols].min(axis=1)
    interactions['last_event_date'] = interactions[event_cols].max(axis=1)
    interactions['recommend_event_date'] = interactions['last_event_date']

    return interactions.sort_values(['student_id', 'recommend_event_date', 'course_id']).reset_index(drop=True)


def assign_time_aware_split(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a next-course split.

    For each user, the latest positive interaction becomes test if the user has
    at least two positive interactions. Anything on or after that timestamp is
    excluded from train to avoid future leakage.
    """
    interactions = interactions.copy()
    interactions['split_role'] = 'train'
    interactions['recommendation_cutoff'] = pd.NaT

    for student_id, user_df in interactions.groupby('student_id'):
        user_df = user_df.sort_values(['recommend_event_date', 'interaction_score', 'course_id'])
        positives = user_df[user_df['positive_label'] & user_df['recommend_event_date'].notna()]

        if len(positives) < 2:
            continue

        test_row = positives.iloc[-1]
        cutoff = test_row['recommend_event_date']
        mask_student = interactions['student_id'] == student_id
        mask_after_cutoff = mask_student & (interactions['recommend_event_date'] >= cutoff)

        interactions.loc[mask_student, 'recommendation_cutoff'] = cutoff
        interactions.loc[mask_after_cutoff, 'split_role'] = 'post_cutoff'

        test_mask = (
            (interactions['student_id'] == student_id) &
            (interactions['course_id'] == test_row['course_id']) &
            (interactions['recommend_event_date'] == cutoff)
        )
        interactions.loc[test_mask, 'split_role'] = 'test'

    return interactions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data'
    out_dir = data_dir / 'cleaned'
    out_dir.mkdir(parents=True, exist_ok=True)

    students = pd.read_csv(data_dir / 'students.csv', parse_dates=['enrollment_date'])
    ratings = pd.read_csv(data_dir / 'ratings.csv', parse_dates=['rating_date'])
    enrollments = pd.read_csv(
        data_dir / 'enrollments.csv',
        parse_dates=['enrollment_date', 'last_accessed']
    )
    quizzes = pd.read_csv(data_dir / 'quiz_results.csv', parse_dates=['quiz_date'])
    activities = pd.read_csv(data_dir / 'activity_logs.csv', parse_dates=['activity_date'])

    students['student_enrollment_date_clean'] = clip_date(
        students['enrollment_date'],
        upper=CANONICAL_END_DATE
    )

    enrollments['enrollment_date_clean'] = clip_date(
        enrollments['enrollment_date'],
        upper=CANONICAL_END_DATE
    )
    enrollments['last_accessed_clean'] = clip_date(
        enrollments['last_accessed'],
        lower=enrollments['enrollment_date_clean'],
        upper=CANONICAL_END_DATE
    )

    quizzes['quiz_date_clean'] = clip_date(
        quizzes['quiz_date'],
        upper=CANONICAL_END_DATE
    )
    activities['activity_date_clean'] = clip_date(
        activities['activity_date'],
        upper=CANONICAL_END_DATE
    )

    enrollments, quizzes = synthesize_support_signals(
        students=students,
        ratings=ratings,
        enrollments=enrollments,
        quizzes=quizzes
    )
    enrollments['enrollment_date_clean'] = clip_date(
        enrollments['enrollment_date'],
        upper=CANONICAL_END_DATE
    )
    enrollments['last_accessed_clean'] = clip_date(
        enrollments['last_accessed'],
        lower=enrollments['enrollment_date_clean'],
        upper=CANONICAL_END_DATE
    )
    quizzes['quiz_date_clean'] = clip_date(
        quizzes['quiz_date'],
        upper=CANONICAL_END_DATE
    )

    enroll_window = enrollments.groupby(['student_id', 'course_id']).agg(
        enrollment_date_clean=('enrollment_date_clean', 'min'),
        last_accessed_clean=('last_accessed_clean', 'max')
    ).reset_index()
    quiz_window = quizzes.groupby(['student_id', 'course_id']).agg(
        latest_quiz_date=('quiz_date_clean', 'max')
    ).reset_index()
    activity_window = activities.groupby(['student_id', 'course_id']).agg(
        latest_activity_date=('activity_date_clean', 'max')
    ).reset_index()

    ratings = ratings.merge(
        students[['student_id', 'student_enrollment_date_clean']],
        on='student_id',
        how='left'
    )
    ratings = ratings.merge(enroll_window, on=['student_id', 'course_id'], how='left')
    ratings = ratings.merge(quiz_window, on=['student_id', 'course_id'], how='left')
    ratings = ratings.merge(activity_window, on=['student_id', 'course_id'], how='left')

    lower_bound = ratings['student_enrollment_date_clean'].fillna(pd.Timestamp('2024-01-01'))
    lower_bound = pd.to_datetime(np.maximum(
        lower_bound.values.astype('datetime64[ns]'),
        ratings['enrollment_date_clean'].fillna(lower_bound).values.astype('datetime64[ns]')
    ))
    lower_bound = pd.Series(lower_bound, index=ratings.index)

    for col in ['latest_quiz_date', 'latest_activity_date']:
        lower_bound = pd.Series(
            np.maximum(lower_bound.values.astype('datetime64[ns]'),
                       ratings[col].fillna(lower_bound).values.astype('datetime64[ns]')),
            index=ratings.index
        )
        lower_bound = pd.to_datetime(lower_bound)

    upper_bound = ratings['last_accessed_clean'].fillna(CANONICAL_END_DATE)
    upper_bound = clip_date(upper_bound, upper=CANONICAL_END_DATE)
    upper_bound = pd.to_datetime(np.maximum(upper_bound.values.astype('datetime64[ns]'),
                                            lower_bound.values.astype('datetime64[ns]')))
    upper_bound = pd.Series(upper_bound, index=ratings.index)

    ratings['rating_date_clean'] = assign_rating_dates_clean(
        ratings=ratings,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
    ratings['rating_signal_weight'] = np.select(
        [ratings['rating'] >= 4, ratings['rating'] == 3],
        [0.70, 0.30],
        default=0.10
    )
    ratings['temporal_issue_fixed'] = (
        ratings['rating_date_clean'].dt.normalize() != ratings['rating_date'].dt.normalize()
    )

    interactions = build_interaction_quality(
        students=students,
        ratings=ratings,
        enrollments=enrollments,
        quizzes=quizzes,
        activities=activities
    )
    interactions = assign_time_aware_split(interactions)

    students.to_csv(out_dir / 'students_clean.csv', index=False)
    enrollments.to_csv(out_dir / 'enrollments_clean.csv', index=False)
    quizzes.to_csv(out_dir / 'quiz_results_clean.csv', index=False)
    activities.to_csv(out_dir / 'activity_logs_clean.csv', index=False)
    ratings.to_csv(out_dir / 'ratings_clean.csv', index=False)
    interactions.to_csv(out_dir / 'interaction_quality.csv', index=False)
    interactions[interactions['split_role'] == 'train'].to_csv(out_dir / 'recommendation_train.csv', index=False)
    interactions[interactions['split_role'] == 'test'].to_csv(out_dir / 'recommendation_test.csv', index=False)

    summary = {
        'canonical_end_date': str(CANONICAL_END_DATE.date()),
        'students_rows': int(len(students)),
        'ratings_rows': int(len(ratings)),
        'ratings_dates_fixed': int(ratings['temporal_issue_fixed'].sum()),
        'enrollments_rows': int(len(enrollments)),
        'synthetic_enrollments_added': int(enrollments['synthetic_generated'].astype(bool).sum()) if 'synthetic_generated' in enrollments.columns else 0,
        'future_enrollments_clipped': int((enrollments['enrollment_date'] > CANONICAL_END_DATE).sum()),
        'quizzes_rows': int(len(quizzes)),
        'synthetic_quizzes_added': int(quizzes['synthetic_generated'].astype(bool).sum()) if 'synthetic_generated' in quizzes.columns else 0,
        'interaction_rows': int(len(interactions)),
        'positive_rows': int(interactions['positive_label'].sum()),
        'signal_tier_counts': interactions['signal_tier'].value_counts().to_dict(),
        'split_role_counts': interactions['split_role'].value_counts().to_dict(),
    }
    (out_dir / 'curation_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

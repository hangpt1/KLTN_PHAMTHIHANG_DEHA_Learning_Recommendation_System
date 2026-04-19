"""
Evaluate next-course recommendation models on the cleaned interaction dataset.

Models:
- Content-based baseline
- User-based collaborative filtering baseline
- Item-based collaborative filtering baseline
- Learning-feature hybrid using progress + quiz + difficulty + category
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

try:
    from recommendation_engine import ContentBasedRecommender, CollaborativeRecommender
except ModuleNotFoundError:
    from src.recommendation_engine import ContentBasedRecommender, CollaborativeRecommender


DIFFICULTY_MAP = {'Beginner': 1.0, 'Intermediate': 2.0, 'Advanced': 3.0}


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


@dataclass
class DataBundle:
    courses: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


class ItemBasedCFRecommender:
    """Item-item CF on interaction-quality-derived preference values."""

    def __init__(self, interactions: pd.DataFrame):
        self.interactions = interactions.copy()
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_means = None

    def fit(self):
        print("\n[Item-Based CF] Building user-item matrix...")
        value_col = 'preference_score' if 'preference_score' in self.interactions.columns else 'rating'
        self.user_item_matrix = self.interactions.pivot_table(
            index='student_id',
            columns='course_id',
            values=value_col,
            fill_value=0
        )
        self.item_means = self.user_item_matrix.replace(0, np.nan).mean(axis=0).fillna(0)
        print(f"  ✓ User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"  ✓ Sparsity: {(self.user_item_matrix == 0).sum().sum() / self.user_item_matrix.size:.2%}")

        print("[Item-Based CF] Computing item similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        np.fill_diagonal(self.item_similarity.values, 0)
        print(f"  ✓ Item Similarity Matrix: {self.item_similarity.shape}")
        return self

    def predict_score(self, student_id: str, course_id: str, top_n: int = 20) -> float:
        if student_id not in self.user_item_matrix.index:
            return float(self.item_means.get(course_id, 0))
        if course_id not in self.user_item_matrix.columns:
            return 0.0

        user_vector = self.user_item_matrix.loc[student_id]
        seen = user_vector[user_vector > 0]
        if seen.empty:
            return float(self.item_means.get(course_id, 0))

        sims = self.item_similarity.loc[course_id, seen.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(top_n)
        if sims.empty:
            return float(self.item_means.get(course_id, 0))

        vals = seen.loc[sims.index]
        denom = sims.abs().sum()
        if denom == 0:
            return float(self.item_means.get(course_id, 0))
        return float((sims * vals).sum() / denom)

    def recommend_for_user(self, student_id: str, n: int = 10) -> list[tuple[str, float]]:
        if student_id not in self.user_item_matrix.index:
            return []
        seen_courses = set(self.user_item_matrix.loc[student_id][lambda s: s > 0].index.tolist())
        predictions = []
        for course_id in self.user_item_matrix.columns:
            if course_id in seen_courses:
                continue
            score = self.predict_score(student_id, course_id)
            if score > 0:
                predictions.append((course_id, score))
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class LearningFeatureHybridRecommender:
    """Hybrid recommender trained on user-course learning features."""

    def __init__(self, courses: pd.DataFrame, train_df: pd.DataFrame,
                 content_model: ContentBasedRecommender,
                 user_cf_model: CollaborativeRecommender,
                 item_cf_model: ItemBasedCFRecommender):
        self.courses = courses.copy()
        self.train_df = train_df.copy()
        self.content_model = content_model
        self.user_cf_model = user_cf_model
        self.item_cf_model = item_cf_model
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight='balanced_subsample',
            min_samples_leaf=2
        )
        self.course_info = None
        self.user_histories = None
        self.user_category_affinity = None
        self.user_category_progress = None
        self.user_category_quiz = None
        self.user_difficulty_pref = None
        self.user_positive_courses = None
        self.course_popularity = None
        self.is_fitted = False

    def _prepare_context(self):
        self.courses['difficulty_level'] = self.courses['difficulty'].map(DIFFICULTY_MAP).fillna(2.0)
        self.course_info = self.courses.set_index('course_id')
        observed = self.train_df[self.train_df['preference_score'] > 0].copy()
        observed = observed.merge(
            self.courses[['course_id', 'category', 'difficulty_level', 'difficulty']],
            on='course_id',
            how='left'
        )

        self.user_histories = observed.groupby('student_id')['course_id'].apply(list).to_dict()
        self.user_positive_courses = observed[observed['positive_label']].groupby('student_id')['course_id'].apply(list).to_dict()
        self.user_category_affinity = observed.groupby(['student_id', 'category'])['preference_score'].mean().to_dict()
        self.user_category_progress = (observed.groupby(['student_id', 'category'])['enrollment_progress']
                                       .mean().fillna(0).div(100).to_dict())
        self.user_category_quiz = (observed.groupby(['student_id', 'category'])['avg_quiz_score']
                                   .mean().fillna(0).div(100).to_dict())
        weighted_difficulty = observed.copy()
        weighted_difficulty['weighted_difficulty'] = (
            weighted_difficulty['difficulty_level'] * weighted_difficulty['preference_score']
        )
        user_difficulty = weighted_difficulty.groupby('student_id').agg(
            weighted_sum=('weighted_difficulty', 'sum'),
            weight=('preference_score', 'sum')
        )
        user_difficulty['difficulty_pref'] = user_difficulty['weighted_sum'] / user_difficulty['weight'].clip(lower=1e-8)
        self.user_difficulty_pref = user_difficulty['difficulty_pref'].to_dict()
        course_pop = observed.groupby('course_id')['positive_label'].mean()
        max_pop = max(course_pop.max(), 1e-8)
        self.course_popularity = (course_pop / max_pop).to_dict()

    def _content_score(self, student_id: str, course_id: str) -> float:
        positives = self.user_positive_courses.get(student_id, [])
        if not positives or course_id not in self.content_model.course_indices:
            return 0.0
        target_idx = self.content_model.course_indices[course_id]
        sims = []
        for seen_course in positives:
            if seen_course not in self.content_model.course_indices:
                continue
            seen_idx = self.content_model.course_indices[seen_course]
            sims.append(float(self.content_model.cosine_sim[target_idx, seen_idx]))
        return max(sims) if sims else 0.0

    def _user_cf_score(self, student_id: str, course_id: str) -> float:
        score = self.user_cf_model.predict_rating(student_id, course_id)
        return float(score / 5.0) if score > 0 else 0.0

    def _item_cf_score(self, student_id: str, course_id: str) -> float:
        score = self.item_cf_model.predict_score(student_id, course_id)
        return float(score / 5.0) if score > 0 else 0.0

    def _feature_row(self, student_id: str, course_id: str) -> dict[str, float]:
        course = self.course_info.loc[course_id]
        category = course['category']
        difficulty_level = float(course['difficulty_level'])
        difficulty_pref = float(self.user_difficulty_pref.get(student_id, 2.0))
        difficulty_gap = abs(difficulty_pref - difficulty_level)
        difficulty_match = max(0.0, 1.0 - (difficulty_gap / 2.0))
        history_count = len(self.user_histories.get(student_id, []))
        positive_count = len(self.user_positive_courses.get(student_id, []))
        return {
            'content_score': self._content_score(student_id, course_id),
            'user_cf_score': self._user_cf_score(student_id, course_id),
            'item_cf_score': self._item_cf_score(student_id, course_id),
            'category_affinity': float(self.user_category_affinity.get((student_id, category), 0.0)),
            'category_progress': float(self.user_category_progress.get((student_id, category), 0.0)),
            'category_quiz': float(self.user_category_quiz.get((student_id, category), 0.0)),
            'difficulty_level': difficulty_level / 3.0,
            'difficulty_match': difficulty_match,
            'course_popularity': float(self.course_popularity.get(course_id, 0.0)),
            'history_size': min(history_count / 20.0, 1.0),
            'positive_history_size': min(positive_count / 10.0, 1.0),
        }

    def fit(self):
        print("\n[Learning Hybrid] Training feature model...")
        self._prepare_context()
        training_rows = []
        labels = []
        rng = np.random.default_rng(42)
        for row in self.train_df.itertuples(index=False):
            training_rows.append(self._feature_row(row.student_id, row.course_id))
            labels.append(int(row.positive_label))

        # Add unseen negatives so the model learns to rank candidate courses,
        # not only to separate observed weak vs strong interactions.
        all_courses = list(self.course_info.index)
        for student_id, seen_courses in self.user_histories.items():
            seen_set = set(seen_courses)
            unseen = [course_id for course_id in all_courses if course_id not in seen_set]
            if not unseen:
                continue
            positive_count = max(1, len(self.user_positive_courses.get(student_id, [])))
            sample_size = min(len(unseen), positive_count * 2)
            negative_courses = rng.choice(unseen, size=sample_size, replace=False)
            for course_id in negative_courses:
                training_rows.append(self._feature_row(student_id, course_id))
                labels.append(0)
        X = pd.DataFrame(training_rows).fillna(0)
        y = np.array(labels)
        self.model.fit(X, y)
        self.feature_columns = X.columns.tolist()
        self.is_fitted = True
        print(f"  ✓ Trained on {len(X)} user-course pairs")
        return self

    def recommend_for_user(self, student_id: str, n: int = 10) -> list[tuple[str, float]]:
        if not self.is_fitted:
            return []
        seen_courses = set(self.user_histories.get(student_id, []))
        candidates = [course_id for course_id in self.course_info.index if course_id not in seen_courses]
        if not candidates:
            return []
        X = pd.DataFrame([self._feature_row(student_id, course_id) for course_id in candidates])[self.feature_columns].fillna(0)
        scores = self.model.predict_proba(X)[:, 1]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n]


def load_clean_bundle(root: Path) -> DataBundle:
    return DataBundle(
        courses=pd.read_csv(root / 'data' / 'courses.csv'),
        train=pd.read_csv(root / 'data' / 'cleaned' / 'recommendation_train.csv'),
        test=pd.read_csv(root / 'data' / 'cleaned' / 'recommendation_test.csv')
    )


def build_train_inputs(train_df: pd.DataFrame):
    observed = train_df[train_df['preference_score'] > 0].copy()
    ratings_train = observed[['student_id', 'course_id', 'preference_score']].rename(
        columns={'preference_score': 'rating'}
    )
    ratings_train['rating'] = (1 + 4 * ratings_train['rating']).clip(1, 5)

    activities_train = observed[['student_id', 'course_id', 'recommend_event_date', 'activity_progress']].drop_duplicates().copy()
    activities_train.rename(
        columns={
            'recommend_event_date': 'activity_date',
            'activity_progress': 'progress_percentage'
        },
        inplace=True
    )
    activities_train['activity_type'] = 'clean_interaction'
    activities_train['time_spent_minutes'] = 0
    activities_train['module_completed'] = 0
    activities_train['video_watched'] = False
    activities_train['notes_taken'] = False
    activities_train['resources_downloaded'] = False
    return ratings_train, activities_train


def evaluate_model(model_name: str, recommend_fn, test_df: pd.DataFrame, course_count: int) -> dict[str, float]:
    precision_scores = {5: [], 10: []}
    recall_scores = {5: [], 10: []}
    all_recommendations = []
    evaluated_users = 0

    for student_id, user_test in test_df.groupby('student_id'):
        relevant = set(user_test[user_test['positive_label']]['course_id'].tolist())
        if not relevant:
            continue
        recs = recommend_fn(student_id, 10)
        recommended_items = [course_id for course_id, _ in recs] if recs and isinstance(recs[0], tuple) else recs
        evaluated_users += 1
        for k in [5, 10]:
            p = precision_at_k(recommended_items, relevant, k)
            r = recall_at_k(recommended_items, relevant, k)
            precision_scores[k].append(p)
            recall_scores[k].append(r)
        all_recommendations.extend(recommended_items)

    result = {'Model': model_name, 'EvaluatedUsers': evaluated_users}
    for k in [5, 10]:
        precision = float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
        recall = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
        result[f'Precision@{k}'] = precision
        result[f'Recall@{k}'] = recall
        result[f'F1@{k}'] = f1_score(precision, recall)
    result['Coverage'] = len(set(all_recommendations)) / course_count if all_recommendations else 0.0
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    data = load_clean_bundle(root)
    data.train['preference_score'] = data.train['interaction_score'].clip(0, 1)

    ratings_train, activities_train = build_train_inputs(data.train)

    content_model = ContentBasedRecommender(data.courses.copy()).fit()
    user_cf_model = CollaborativeRecommender(ratings_train.copy()).fit()
    item_cf_model = ItemBasedCFRecommender(ratings_train.copy()).fit()
    hybrid_model = LearningFeatureHybridRecommender(
        data.courses.copy(),
        data.train.copy(),
        content_model,
        user_cf_model,
        item_cf_model
    ).fit()

    train_history = data.train[data.train['preference_score'] > 0].groupby('student_id')['course_id'].apply(list).to_dict()

    results = []
    results.append(evaluate_model(
        'Content-Based',
        lambda uid, n: content_model.recommend_for_user(train_history.get(uid, []), n=n),
        data.test,
        data.courses['course_id'].nunique()
    ))
    results.append(evaluate_model(
        'User-Based CF',
        lambda uid, n: user_cf_model.recommend_for_user(uid, n=n),
        data.test,
        data.courses['course_id'].nunique()
    ))
    results.append(evaluate_model(
        'Item-Based CF',
        lambda uid, n: item_cf_model.recommend_for_user(uid, n=n),
        data.test,
        data.courses['course_id'].nunique()
    ))
    results.append(evaluate_model(
        'Learning Hybrid',
        lambda uid, n: hybrid_model.recommend_for_user(uid, n=n),
        data.test,
        data.courses['course_id'].nunique()
    ))

    result_df = pd.DataFrame(results).sort_values(by='F1@10', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 78)
    print("NEXT-COURSE EVALUATION ON CLEANED DATA")
    print("=" * 78)
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()

"""
Model Evaluation Module for E-Learning Recommendation System
Calculates Precision, Recall, RMSE and other evaluation metrics.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RecommendationEvaluator:
    """
    Evaluates recommendation system performance using standard metrics:
    - Precision@K
    - Recall@K
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - Coverage
    - Novelty
    """
    
    def __init__(self, ratings, courses):
        """
        Initialize evaluator with data.
        
        Args:
            ratings: DataFrame with user-course ratings
            courses: DataFrame with course metadata
        """
        self.ratings = ratings
        self.courses = courses
        self.train_data = None
        self.test_data = None
    
    def split_data(self, test_size=0.2, random_state=42, min_train_items=1):
        """
        Split ratings data into train and test sets per user.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            min_train_items: Minimum number of interactions to keep in train
        """
        rng = np.random.default_rng(random_state)
        train_parts = []
        test_parts = []
        
        for _, user_ratings in self.ratings.groupby('student_id'):
            user_ratings = user_ratings.sample(
                frac=1.0, random_state=random_state
            ).reset_index(drop=True)
            
            if len(user_ratings) <= min_train_items:
                train_parts.append(user_ratings)
                continue
            
            tentative_test = max(1, int(round(len(user_ratings) * test_size)))
            max_test = len(user_ratings) - min_train_items
            test_count = min(tentative_test, max_test)
            
            if test_count <= 0:
                train_parts.append(user_ratings)
                continue
            
            test_indices = rng.choice(
                len(user_ratings), size=test_count, replace=False
            )
            mask = np.zeros(len(user_ratings), dtype=bool)
            mask[test_indices] = True
            
            test_parts.append(user_ratings[mask])
            train_parts.append(user_ratings[~mask])
        
        self.train_data = pd.concat(train_parts, ignore_index=True)
        self.test_data = pd.concat(test_parts, ignore_index=True)
        print(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")
        return self.train_data, self.test_data
    
    # =========================================================================
    # PRECISION@K METRIC
    # =========================================================================
    
    def precision_at_k(self, recommended, relevant, k):
        """
        Calculate Precision@K.
        
        Precision@K = |Relevant ∩ Recommended@K| / K
        
        Measures what fraction of top-K recommendations are relevant.
        
        Args:
            recommended: List of recommended item IDs (ordered by rank)
            relevant: Set of relevant item IDs (ground truth)
            k: Number of top recommendations to consider
            
        Returns:
            float: Precision@K score (0 to 1)
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        precision = hits / k
        
        return precision
    
    # =========================================================================
    # RECALL@K METRIC
    # =========================================================================
    
    def recall_at_k(self, recommended, relevant, k):
        """
        Calculate Recall@K.
        
        Recall@K = |Relevant ∩ Recommended@K| / |Relevant|
        
        Measures what fraction of relevant items appear in top-K recommendations.
        
        Args:
            recommended: List of recommended item IDs (ordered by rank)
            relevant: Set of relevant item IDs (ground truth)
            k: Number of top recommendations to consider
            
        Returns:
            float: Recall@K score (0 to 1)
        """
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_k if item in relevant_set)
        recall = hits / len(relevant_set)
        
        return recall
    
    # =========================================================================
    # RMSE METRIC
    # =========================================================================
    
    def rmse(self, predicted_ratings, actual_ratings):
        """
        Calculate Root Mean Square Error (RMSE).
        
        RMSE = √(Σ(predicted - actual)² / n)
        
        Measures average prediction error magnitude.
        Lower RMSE indicates better predictions.
        
        Args:
            predicted_ratings: Array of predicted ratings
            actual_ratings: Array of actual ratings
            
        Returns:
            float: RMSE score
        """
        predicted = np.array(predicted_ratings)
        actual = np.array(actual_ratings)
        
        mse = np.mean((predicted - actual) ** 2)
        rmse_score = np.sqrt(mse)
        
        return rmse_score
    
    # =========================================================================
    # MAE METRIC
    # =========================================================================
    
    def mae(self, predicted_ratings, actual_ratings):
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE = Σ|predicted - actual| / n
        
        Args:
            predicted_ratings: Array of predicted ratings
            actual_ratings: Array of actual ratings
            
        Returns:
            float: MAE score
        """
        predicted = np.array(predicted_ratings)
        actual = np.array(actual_ratings)
        
        return np.mean(np.abs(predicted - actual))
    
    # =========================================================================
    # F1 SCORE
    # =========================================================================
    
    def f1_score(self, precision, recall):
        """
        Calculate F1 Score (harmonic mean of precision and recall).
        
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            float: F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # =========================================================================
    # COVERAGE METRIC
    # =========================================================================
    
    def coverage(self, all_recommendations, all_items):
        """
        Calculate catalog coverage.
        
        Coverage = |Unique Recommended Items| / |All Items|
        
        Measures what fraction of items can be recommended.
        
        Args:
            all_recommendations: List of all recommended item IDs
            all_items: Set of all available item IDs
            
        Returns:
            float: Coverage score (0 to 1)
        """
        unique_recommended = set(all_recommendations)
        return len(unique_recommended) / len(all_items)
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    def _build_interaction_history(self, ratings_df):
        """Build user interaction history from ratings."""
        history = (
            ratings_df.groupby('student_id')['course_id']
            .apply(lambda x: list(dict.fromkeys(x.tolist())))
            .to_dict()
        )
        return history
    
    def _evaluate_model(self, model_name, recommend_fn, predict_fn=None,
                        k_values=[5, 10], relevance_threshold=4.0):
        """Evaluate one recommendation model on the held-out test set."""
        test_users = self.test_data['student_id'].unique()
        
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        all_recommendations = []
        predicted_ratings = []
        actual_ratings = []
        evaluated_users = 0
        
        print(f"\n--- {model_name} ---")
        
        for user_id in test_users:
            user_test = self.test_data[self.test_data['student_id'] == user_id]
            relevant_items = set(
                user_test[user_test['rating'] >= relevance_threshold]['course_id'].tolist()
            )
            
            if not relevant_items:
                continue
            
            recommended_items = recommend_fn(user_id, max(k_values))
            if not recommended_items:
                continue
            
            evaluated_users += 1
            
            for k in k_values:
                precision_scores[k].append(
                    self.precision_at_k(recommended_items, relevant_items, k)
                )
                recall_scores[k].append(
                    self.recall_at_k(recommended_items, relevant_items, k)
                )
            
            all_recommendations.extend(recommended_items)
            
            if predict_fn is not None:
                for _, row in user_test.iterrows():
                    predicted = predict_fn(user_id, row['course_id'])
                    if predicted is None:
                        continue
                    predicted_ratings.append(predicted)
                    actual_ratings.append(row['rating'])
        
        results = {'EvaluatedUsers': evaluated_users}
        
        for k in k_values:
            avg_precision = np.mean(precision_scores[k]) if precision_scores[k] else 0
            avg_recall = np.mean(recall_scores[k]) if recall_scores[k] else 0
            f1 = self.f1_score(avg_precision, avg_recall)
            
            results[f'Precision@{k}'] = avg_precision
            results[f'Recall@{k}'] = avg_recall
            results[f'F1@{k}'] = f1
            
            print(f"  Precision@{k}: {avg_precision:.4f}")
            print(f"  Recall@{k}:    {avg_recall:.4f}")
            print(f"  F1@{k}:        {f1:.4f}")
        
        if predicted_ratings and actual_ratings:
            rmse_score = self.rmse(predicted_ratings, actual_ratings)
            mae_score = self.mae(predicted_ratings, actual_ratings)
            results['RMSE'] = rmse_score
            results['MAE'] = mae_score
            print(f"  RMSE:          {rmse_score:.4f}")
            print(f"  MAE:           {mae_score:.4f}")
        else:
            results['RMSE'] = None
            results['MAE'] = None
            print("  RMSE:          N/A")
            print("  MAE:           N/A")
        
        all_courses = set(self.courses['course_id'].tolist())
        coverage_score = self.coverage(all_recommendations, all_courses) if all_recommendations else 0
        results['Coverage'] = coverage_score
        print(f"  Coverage:      {coverage_score:.2%}")
        
        return results
    
    def evaluate_models(self, k_values=[5, 10], relevance_threshold=4.0,
                        random_state=42, hybrid_weights=(0.4, 0.6)):
        """
        Run comprehensive evaluation on content-based, collaborative,
        and hybrid recommenders using train/test split.
        
        Args:
            k_values: List of K values for Precision@K and Recall@K
            relevance_threshold: Minimum rating to consider an item relevant
            random_state: Random seed for reproducibility
            hybrid_weights: Tuple of (content_weight, collaborative_weight)
            
        Returns:
            dict: Metrics for each model
        """
        try:
            from recommendation_engine import (
                ContentBasedRecommender,
                CollaborativeRecommender,
                HybridRecommender
            )
        except ModuleNotFoundError:
            from src.recommendation_engine import (
                ContentBasedRecommender,
                CollaborativeRecommender,
                HybridRecommender
            )
        
        print("\n" + "=" * 70)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 70)
        
        if self.train_data is None:
            self.split_data(random_state=random_state)
        
        train_history = self._build_interaction_history(self.train_data)
        synthetic_activities = self.train_data[['student_id', 'course_id']].drop_duplicates().copy()
        
        content_model = ContentBasedRecommender(self.courses.copy()).fit()
        cf_model = CollaborativeRecommender(self.train_data.copy()).fit()
        hybrid_model = HybridRecommender(
            self.courses.copy(),
            self.train_data.copy(),
            synthetic_activities,
            content_weight=hybrid_weights[0],
            collaborative_weight=hybrid_weights[1]
        ).fit()
        
        results = {}
        
        def content_recommend(user_id, n):
            user_courses = train_history.get(user_id, [])
            recs = content_model.recommend_for_user(user_courses, n=n)
            return [course_id for course_id, _ in recs]
        
        def cf_recommend(user_id, n):
            recs = cf_model.recommend_for_user(user_id, n=n)
            return [course_id for course_id, _ in recs]
        
        def hybrid_recommend(user_id, n):
            recs = hybrid_model.recommend(user_id, n=n)
            if recs.empty:
                return []
            return recs['course_id'].tolist()
        
        def cf_predict(user_id, course_id):
            pred = cf_model.predict_rating(user_id, course_id)
            return pred if pred > 0 else None
        
        results['Content-Based'] = self._evaluate_model(
            'Content-Based',
            content_recommend,
            predict_fn=None,
            k_values=k_values,
            relevance_threshold=relevance_threshold
        )
        results['Collaborative Filtering'] = self._evaluate_model(
            'Collaborative Filtering',
            cf_recommend,
            predict_fn=cf_predict,
            k_values=k_values,
            relevance_threshold=relevance_threshold
        )
        results['Hybrid'] = self._evaluate_model(
            'Hybrid',
            hybrid_recommend,
            predict_fn=None,
            k_values=k_values,
            relevance_threshold=relevance_threshold
        )
        
        return results
    
    def evaluate_recommenders_kfold(self, k_folds=5, k_values=[5, 10],
                                    relevance_threshold=4.0, hybrid_weights=(0.4, 0.6)):
        """
        Run K-Fold Cross-Validation evaluation on all recommenders.
        
        Args:
            k_folds: Number of folds for cross-validation
            k_values: List of K values for metrics (e.g., [5, 10])
            relevance_threshold: Minimum rating to consider an item relevant
            hybrid_weights: Tuple of (content_weight, collaborative_weight)
            
        Returns:
            dict: Average metrics across all folds + detailed per-fold results
        """
        print("\n" + "=" * 70)
        print(f"  K-FOLD CROSS-VALIDATION ({k_folds} FOLDS)")
        print("=" * 70)
        
        fold_results = {
            'Content-Based': [],
            'Collaborative Filtering': [],
            'Hybrid': []
        }
        
        for fold in range(k_folds):
            print(f"\n{'='*70}")
            print(f"  FOLD {fold + 1}/{k_folds} (random_state={fold})")
            print(f"{'='*70}")
            
            # Reset data để tránh side effects
            self.train_data = None
            self.test_data = None
            
            # Chia dữ liệu với random_state khác nhau cho mỗi fold
            self.split_data(random_state=fold, min_train_items=1)
            
            # Chạy evaluation trên fold này
            fold_metrics = self.evaluate_models(
                k_values=k_values,
                relevance_threshold=relevance_threshold,
                random_state=fold,
                hybrid_weights=hybrid_weights
            )
            
            # Lưu kết quả
            for model_name, metrics in fold_metrics.items():
                fold_results[model_name].append(metrics)
        
        # Tính trung bình kết quả từ tất cả folds
        print("\n" + "=" * 70)
        print("  K-FOLD CROSS-VALIDATION RESULTS (AVERAGED)")
        print("=" * 70)
        
        averaged_results = {}
        
        for model_name, metrics_list in fold_results.items():
            print(f"\n{model_name}:")
            print("-" * 70)
            
            # Lấy các key của metrics từ fold đầu tiên
            if metrics_list:
                metric_keys = metrics_list[0].keys()
                averaged_metrics = {}
                
                for key in metric_keys:
                    # Lấy giá trị từ tất cả folds
                    values = [m[key] for m in metrics_list]
                    
                    # Tính trung bình
                    if isinstance(values[0], (int, float)):
                        avg_value = np.mean(values)
                        std_value = np.std(values)
                        averaged_metrics[key] = {
                            'mean': avg_value,
                            'std': std_value,
                            'values': values
                        }
                        print(f"  {key}: {avg_value:.4f} ± {std_value:.4f}")
                    else:
                        # Nếu không phải số, chỉ lấy giá trị từ fold 1
                        averaged_metrics[key] = values[0]
                        print(f"  {key}: {values[0]}")
                
                averaged_results[model_name] = averaged_metrics
        
        # Tạo DataFrame tổng hợp kết quả
        summary_data = []
        
        for model_name, metrics in averaged_results.items():
            row = {'Model': model_name}
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    row[f"{metric_name}_mean"] = metric_value['mean']
                    row[f"{metric_name}_std"] = metric_value['std']
                else:
                    row[metric_name] = metric_value
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "=" * 70)
        print("  SUMMARY TABLE")
        print("=" * 70)
        print(summary_df.to_string())
        
        return {
            'summary': summary_df,
            'fold_results': fold_results,
            'detailed_results': averaged_results
        }
    
    def find_best_hybrid_weights(self, weight_grid, k_values=[5, 10],
                                 relevance_threshold=4.0, random_state=42,
                                 target_metric='F1@10'):
        """
        Evaluate multiple hybrid weight settings and rank them.
        
        Args:
            weight_grid: Iterable of (content_weight, collaborative_weight)
            k_values: Ranking cutoffs to evaluate
            relevance_threshold: Rating threshold for relevant items
            random_state: Random seed for reproducibility
            target_metric: Metric used to sort configurations
        
        Returns:
            DataFrame: Ranked hybrid weight search results
        """
        results = []
        
        # Reuse the same split across all weight settings for fair comparison.
        self.train_data = None
        self.test_data = None
        self.split_data(random_state=random_state)
        
        for content_weight, collaborative_weight in weight_grid:
            print("\n" + "-" * 70)
            print(
                f"Testing hybrid weights: content={content_weight:.2f}, "
                f"collaborative={collaborative_weight:.2f}"
            )
            print("-" * 70)
            
            metrics = self.evaluate_models(
                k_values=k_values,
                relevance_threshold=relevance_threshold,
                random_state=random_state,
                hybrid_weights=(content_weight, collaborative_weight)
            )['Hybrid']
            
            row = {
                'content_weight': content_weight,
                'collaborative_weight': collaborative_weight,
                **metrics
            }
            results.append(row)
        
        result_df = pd.DataFrame(results)
        if target_metric in result_df.columns:
            result_df = result_df.sort_values(
                by=target_metric,
                ascending=False
            ).reset_index(drop=True)
        
        return result_df


def run_evaluation_demo(data_dir):
    """
    Run evaluation demonstration.
    
    Args:
        data_dir: Path to data directory
    """
    print("\n" + "=" * 70)
    print("  E-LEARNING RECOMMENDATION SYSTEM - EVALUATION DEMO")
    print("=" * 70)
    
    # Load data
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    activities = pd.read_csv(os.path.join(data_dir, 'activity_logs.csv'))
    
    print("\n[Step 1] Initializing Evaluator...")
    evaluator = RecommendationEvaluator(ratings, courses)
    
    print("\n[Step 2] Running Evaluation...")
    results = evaluator.evaluate_models(
        k_values=[5, 10],
        relevance_threshold=4.0
    )
    
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        print(f"\n  {model_name}")
        print("  " + "-" * len(model_name))
        for metric, value in metrics.items():
            if value is None:
                print(f"  {metric:<16}: N/A")
            elif metric == 'EvaluatedUsers':
                print(f"  {metric:<16}: {int(value)}")
            elif metric == 'Coverage':
                print(f"  {metric:<16}: {value:.2%}")
            else:
                print(f"  {metric:<16}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if os.path.exists(os.path.join(data_dir, 'ratings.csv')):
        try:
            run_evaluation_demo(data_dir)
        except Exception as e:
            print(f"\nEvaluation failed: {e}")

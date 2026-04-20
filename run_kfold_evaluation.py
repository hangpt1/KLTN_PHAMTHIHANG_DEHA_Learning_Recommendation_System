#!/usr/bin/env python3
"""
Run K-Fold Cross-Validation Evaluation for Recommendation System
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluation import RecommendationEvaluator

def main():
    """Run K-Fold evaluation"""
    
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print("\n" + "=" * 70)
    print("  K-FOLD CROSS-VALIDATION FOR RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    # Load data
    print("\n[Step 1] Loading data...")
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    
    print(f"  - Loaded {len(courses)} courses")
    print(f"  - Loaded {len(ratings)} ratings")
    
    # Initialize evaluator
    print("\n[Step 2] Initializing evaluator...")
    evaluator = RecommendationEvaluator(ratings, courses)
    
    # Run K-Fold evaluation
    print("\n[Step 3] Running K-Fold Cross-Validation (10 folds)...")
    kfold_results = evaluator.evaluate_recommenders_kfold(
        k_folds=10,
        k_values=[5, 10],
        relevance_threshold=4.0,
        hybrid_weights=(0.4, 0.6)
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("  K-FOLD EVALUATION SUMMARY")
    print("=" * 70)
    print("\n" + kfold_results['summary'].to_string())
    
    # Save results to CSV
    output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'kfold_summary.csv')
    kfold_results['summary'].to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, 'kfold_detailed_results.txt')
    with open(detailed_path, 'w') as f:
        f.write("K-FOLD CROSS-VALIDATION DETAILED RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        for model_name, metrics in kfold_results['detailed_results'].items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 70 + "\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, dict):
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    Mean: {metric_value['mean']:.4f}\n")
                    f.write(f"    Std:  {metric_value['std']:.4f}\n")
                    f.write(f"    Values: {metric_value['values']}\n")
                else:
                    f.write(f"  {metric_name}: {metric_value}\n")
    
    print(f"✓ Detailed results saved to: {detailed_path}")
    
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()

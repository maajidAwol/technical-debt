"""
Main Pipeline Script for High-Risk Technical Debt Prediction

This script runs the complete ML pipeline:
1. Load data from Technical Debt Dataset
2. Create high-risk labels
3. Extract static and historical features
4. Train and evaluate ML models
5. Run cross-project validation
6. Generate visualizations and results

Usage:
    python run_pipeline.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TD_DATASET_PATH, PROCESSED_DATA_DIR, RESULTS_DIR


def check_dataset():
    """Check if the Technical Debt Dataset is downloaded."""
    if not TD_DATASET_PATH.exists():
        print("=" * 60)
        print("ERROR: Technical Debt Dataset not found!")
        print("=" * 60)
        print(f"\nExpected location: {TD_DATASET_PATH}")
        print("\nPlease download the dataset:")
        print("  1. Go to: https://github.com/clowee/The-Technical-Debt-Dataset/releases")
        print("  2. Download 'technical_debt_dataset.db' (Release 2.0)")
        print(f"  3. Place it in: {TD_DATASET_PATH.parent}")
        print("\nAlternatively, rename the downloaded file to match the expected name.")
        return False
    return True


def run_step(step_name: str, func, *args, **kwargs):
    """Run a pipeline step with status output."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print('='*60)
    try:
        result = func(*args, **kwargs)
        print(f"\n[SUCCESS] {step_name} completed!")
        return result
    except Exception as e:
        print(f"\n[ERROR] {step_name} failed: {e}")
        raise


def main():
    """Run the complete ML pipeline."""
    print("=" * 60)
    print("HIGH-RISK TECHNICAL DEBT PREDICTION PIPELINE")
    print("=" * 60)
    
    # Step 0: Check dataset
    if not check_dataset():
        return
    
    # Import modules
    from src.data.load_data import get_connection, get_severity_distribution
    from src.data.labeling import create_high_risk_labels, get_label_statistics, save_labels
    from src.features.static_features import extract_all_static_features, save_features as save_static
    from src.features.historical_features import extract_all_historical_features, save_features as save_historical
    
    # Step 1: Connect to database
    conn = run_step("Connecting to database", get_connection)
    
    # Step 2: Explore severity distribution
    print("\n--- Severity Distribution ---")
    severity_dist = get_severity_distribution(conn)
    print(severity_dist)
    
    # Step 3: Create labels
    labels_df = run_step("Creating high-risk TD labels", create_high_risk_labels, conn)
    stats = get_label_statistics(labels_df)
    print("\n--- Label Statistics ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    save_labels(labels_df)
    
    # Step 4: Extract static features
    static_features = run_step("Extracting static code metrics", 
                               extract_all_static_features, conn)
    save_static(static_features)
    
    # Step 5: Extract historical features
    historical_features = run_step("Extracting historical change metrics",
                                   extract_all_historical_features, conn)
    save_historical(historical_features)
    
    conn.close()
    
    # Step 6: Train models
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    from src.models.train import (
        load_dataset, get_models, train_and_evaluate, 
        train_final_model, get_feature_importance, save_results
    )
    
    X, y, project_ids, feature_names = load_dataset()
    print(f"\nDataset loaded: {len(y)} samples, {X.shape[1]} features")
    print(f"High-Risk: {y.sum()} ({y.mean()*100:.1f}%)")
    
    models = get_models()
    results_df = train_and_evaluate(X, y, models)
    save_results(results_df)
    
    # Feature importance
    best_model_name = results_df.loc[results_df['F1_mean'].idxmax(), 'Model']
    model, scaler = train_final_model(X, y, best_model_name)
    importance_df = get_feature_importance(model, feature_names)
    
    if importance_df is not None:
        importance_path = RESULTS_DIR / "tables" / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"\nFeature importance saved to: {importance_path}")
    
    # Step 7: Cross-project validation
    print("\n" + "=" * 60)
    print("CROSS-PROJECT VALIDATION")
    print("=" * 60)
    
    from src.models.cross_project import (
        cross_project_validation, summarize_cross_project_results, 
        save_results as save_cross_results
    )
    
    cross_results = cross_project_validation(X, y, project_ids)
    summary = summarize_cross_project_results(cross_results)
    save_cross_results(cross_results, summary)
    
    # Step 8: Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    from src.visualization.plots import (
        plot_class_distribution, plot_model_comparison,
        plot_feature_importance, plot_cross_project_results
    )
    
    plot_class_distribution(y)
    plot_model_comparison(results_df)
    if importance_df is not None:
        plot_feature_importance(importance_df)
    plot_cross_project_results(cross_results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - tables/model_comparison.csv")
    print(f"  - tables/feature_importance.csv")
    print(f"  - tables/cross_project_results.csv")
    print(f"  - figures/class_distribution.png")
    print(f"  - figures/model_comparison.png")
    print(f"  - figures/feature_importance.png")
    print(f"  - figures/cross_project_results.png")
    
    print(f"\n--- Best Model: {best_model_name} ---")
    best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
    print(f"  F1-Score: {best_row['F1-Score']}")
    print(f"  AUC-ROC:  {best_row['AUC-ROC']}")
    
    print(f"\n--- Cross-Project Performance ---")
    print(f"  Avg F1-Score: {summary['f1_score_mean']:.3f} ± {summary['f1_score_std']:.3f}")
    print(f"  Avg AUC-ROC:  {summary['auc_roc_mean']:.3f} ± {summary['auc_roc_std']:.3f}")


if __name__ == "__main__":
    main()

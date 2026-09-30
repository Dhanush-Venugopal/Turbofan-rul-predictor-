"""
STEP 4: Performance Metrics - NASA Evaluation Framework
========================================================
Evaluates RUL predictions using the asymmetric scoring function from the 
NASA paper (Saxena et al., 2008).

Key Concept:
-----------
NASA's metric heavily penalizes LATE predictions (predicting failure when 
it's already happened) more than EARLY predictions.

Asymmetric Scoring Function (Equation 11):
    - If d < 0 (early prediction):    score = Σ(exp(d/a1) - 1)
    - If d ≥ 0 (late prediction):     score = Σ(exp(d/a2) - 1)
    
    where:
        d = predicted_RUL - true_RUL (prediction error)
        a1 = 10 (early penalty factor)
        a2 = 13 (late penalty factor)

Why asymmetric?
    - Early warning allows time for maintenance → acceptable, low penalty
    - Late prediction means failure already happened → critical, high penalty
    - "It's better to be wrong early than wrong late"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error


# ==================== NASA SCORING METRICS ====================

class NASAMetrics:
    """
    Implements NASA's evaluation metrics for prognostics.
    Reference: Saxena et al., "Damage Propagation Modeling..." (2008)
    """
    
    # Asymmetric scoring parameters (from paper, Section VII)
    A1 = 10.0   # Early prediction penalty factor
    A2 = 13.0   # Late prediction penalty factor
    
    @staticmethod
    def asymmetric_score(y_true, y_pred):
        """
        Calculate NASA's asymmetric scoring metric.
        
        Lower score is better (0 = perfect prediction)
        
        Args:
            y_true: actual RUL values
            y_pred: predicted RUL values
        
        Returns:
            total_score: sum of exponential penalties
        """
        
        # Calculate prediction errors
        errors = y_pred - y_true  # positive = late (bad), negative = early (ok)
        
        # Apply asymmetric scoring
        scores = np.zeros_like(errors, dtype=float)
        
        for i, d in enumerate(errors):
            if d < 0:  # Early prediction (desired)
                scores[i] = np.exp(d / NASAMetrics.A1) - 1.0
            else:  # Late prediction (undesired)
                scores[i] = np.exp(d / NASAMetrics.A2) - 1.0
        
        return np.sum(scores)
    
    @staticmethod
    def score_by_engine(y_true, y_pred, engine_ids):
        """Calculate score for each engine separately"""
        unique_engines = np.unique(engine_ids)
        scores = {}
        
        for engine_id in unique_engines:
            mask = engine_ids == engine_id
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            score = NASAMetrics.asymmetric_score(y_t, y_p)
            scores[engine_id] = score
        
        return scores


class RULEvaluator:
    """Comprehensive RUL prediction evaluator"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_all_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate all performance metrics for a model"""
        
        errors = y_pred - y_true
        
        metrics = {
            'Model': model_name,
            'NASA_Score': NASAMetrics.asymmetric_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MedAE': median_absolute_error(y_true, y_pred),
            'Mean_Error': np.mean(errors),
            'Std_Error': np.std(errors),
            'Min_Error': np.min(errors),
            'Max_Error': np.max(errors),
            'Early_Predictions': np.sum(errors < 0),
            'Late_Predictions': np.sum(errors >= 0),
            'Perfect_Predictions': np.sum(np.abs(errors) <= 1),
        }
        
        self.results[model_name] = metrics
        return metrics


def evaluate_models(predictions_file):
    """
    Evaluate all trained models using NASA metrics.
    
    Args:
        predictions_file: path to CSV with actual and predicted RUL values
    
    Returns:
        DataFrame with evaluation results
    """
    
    print("Loading predictions...")
    data = pd.read_csv(predictions_file)
    
    y_true = data['RUL'].values
    
    evaluator = RULEvaluator()
    
    # Evaluate each model
    models = ['LR_Pred', 'RF_Pred', 'GB_Pred']
    model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    
    for model_col, model_name in zip(models, model_names):
        y_pred = data[model_col].values
        
        print(f"\nEvaluating {model_name}...")
        metrics = evaluator.calculate_all_metrics(y_true, y_pred, model_name)
        
        # Print metrics
        print(f"  NASA Score: {metrics['NASA_Score']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f} cycles")
        print(f"  MAE: {metrics['MAE']:.2f} cycles")
        print(f"  Early predictions: {metrics['Early_Predictions']}")
        print(f"  Late predictions: {metrics['Late_Predictions']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(list(evaluator.results.values()))
    
    return results_df, evaluator, data


# ==================== ADVANCED ANALYSIS ====================

def analyze_prediction_errors(y_true, y_pred, model_name="Model"):
    """Detailed error analysis"""
    
    errors = y_pred - y_true
    early_errors = errors[errors < 0]
    late_errors = errors[errors >= 0]
    
    analysis = {
        'Model': model_name,
        'Total_Errors': len(errors),
        'Early_Pct': len(early_errors) / len(errors) * 100,
        'Late_Pct': len(late_errors) / len(errors) * 100,
        'Early_MAE': np.mean(np.abs(early_errors)) if len(early_errors) > 0 else 0,
        'Late_MAE': np.mean(np.abs(late_errors)) if len(late_errors) > 0 else 0,
        'Worst_Early_Error': np.min(early_errors) if len(early_errors) > 0 else 0,
        'Worst_Late_Error': np.max(late_errors) if len(late_errors) > 0 else 0,
    }
    
    return analysis


# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 70)
    print("STEP 4: PERFORMANCE EVALUATION - NASA METRICS")
    print("=" * 70)
    
    # Load predictions
    input_file = Path("data/model_predictions.csv")
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        print("Please run 03_rul_predictor.py first.")
        return None
    
    print(f"\nLoading predictions from: {input_file}")
    
    # Evaluate models
    results_df, evaluator, predictions = evaluate_models(str(input_file))
    
    # ==================== DISPLAY RESULTS ====================
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    # Main metrics
    display_cols = ['Model', 'NASA_Score', 'RMSE', 'MAE', 'Early_Predictions', 'Late_Predictions']
    print("\n" + results_df[display_cols].to_string(index=False))
    
    # Detailed metrics
    print("\n" + "=" * 70)
    print("DETAILED METRICS")
    print("=" * 70)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  NASA Score: {row['NASA_Score']:.2f}")
        print(f"  RMSE: {row['RMSE']:.2f} cycles")
        print(f"  MAE: {row['MAE']:.2f} cycles")
        print(f"  Median AE: {row['MedAE']:.2f} cycles")
        print(f"  Error range: [{row['Min_Error']:.0f}, {row['Max_Error']:.0f}] cycles")
        print(f"  Early predictions: {row['Early_Predictions']} ({row['Early_Predictions']/len(predictions)*100:.1f}%)")
        print(f"  Late predictions: {row['Late_Predictions']} ({row['Late_Predictions']/len(predictions)*100:.1f}%)")
    
    # Save results
    results_df.to_csv('data/evaluation_results.csv', index=False)
    print(f"\n✓ Results saved to: data/evaluation_results.csv")
    
    # ==================== VISUALIZATION ====================
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Asymmetric Score Comparison
    ax = axes[0, 0]
    models = results_df['Model'].values
    scores = results_df['NASA_Score'].values
    colors = ['green', 'orange', 'blue']
    bars = ax.bar(models, scores, alpha=0.8, color=colors)
    ax.set_ylabel('NASA Asymmetric Score (lower is better)')
    ax.set_title('Model Comparison - NASA Scoring')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    # Plot 2: RMSE Comparison
    ax = axes[0, 1]
    rmse = results_df['RMSE'].values
    ax.bar(models, rmse, alpha=0.8, color=colors)
    ax.set_ylabel('RMSE (cycles)')
    ax.set_title('Root Mean Squared Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Early vs Late Predictions
    ax = axes[1, 0]
    x = np.arange(len(models))
    width = 0.35
    early = results_df['Early_Predictions'].values
    late = results_df['Late_Predictions'].values
    
    ax.bar(x - width/2, early, width, label='Early Predictions', alpha=0.8, color='green')
    ax.bar(x + width/2, late, width, label='Late Predictions', alpha=0.8, color='red')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Direction Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error Distribution (violin plot)
    ax = axes[1, 1]
    errors_data = []
    labels = []
    
    y_true = predictions['RUL'].values
    
    for model_col, model_name in zip(['LR_Pred', 'RF_Pred', 'GB_Pred'], 
                                     ['Linear Reg', 'Random Forest', 'Gradient Boost']):
        y_pred = predictions[model_col].values
        errors = y_pred - y_true
        errors_data.append(errors)
        labels.append(model_name)
    
    parts = ax.violinplot(errors_data, positions=range(len(labels)), 
                          showmeans=True, showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Prediction Error (cycles)')
    ax.set_title('Error Distribution by Model')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect prediction')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig('data/evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: data/evaluation_metrics.png")
    
    # ==================== SCORING CURVE VISUALIZATION ====================
    print("\nGenerating NASA scoring curve...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scoring function
    ax = axes[0]
    errors_range = np.linspace(-50, 50, 100)
    scores = []
    
    for d in errors_range:
        if d < 0:
            score = np.exp(d / NASAMetrics.A1) - 1.0
        else:
            score = np.exp(d / NASAMetrics.A2) - 1.0
        scores.append(score)
    
    ax.plot(errors_range, scores, linewidth=2)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Perfect prediction')
    ax.fill_between(errors_range[errors_range < 0], 0, 
                    np.array(scores)[errors_range < 0], alpha=0.2, color='green', 
                    label='Early (preferred)')
    ax.fill_between(errors_range[errors_range >= 0], 0, 
                    np.array(scores)[errors_range >= 0], alpha=0.2, color='red', 
                    label='Late (penalized)')
    ax.set_xlabel('Prediction Error: Predicted RUL - Actual RUL (cycles)')
    ax.set_ylabel('Penalty Score')
    ax.set_title('NASA Asymmetric Scoring Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative score vs prediction error threshold
    ax = axes[1]
    
    for model_col, model_name in zip(['LR_Pred', 'RF_Pred', 'GB_Pred'], 
                                     ['Linear Regression', 'Random Forest', 'Gradient Boosting']):
        y_pred = predictions[model_col].values
        errors = y_pred - y_true
        
        # Calculate cumulative score vs error threshold
        error_thresholds = np.linspace(-50, 50, 50)
        cumulative_scores = []
        
        for threshold in error_thresholds:
            mask = errors <= threshold
            if np.sum(mask) > 0:
                score = NASAMetrics.asymmetric_score(y_true[mask], y_pred[mask])
            else:
                score = 0
            cumulative_scores.append(score)
        
        ax.plot(error_thresholds, cumulative_scores, linewidth=2, marker='o', 
               markersize=4, label=model_name, alpha=0.7)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error Threshold (cycles)')
    ax.set_ylabel('Cumulative NASA Score')
    ax.set_title('Cumulative Score vs Error Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('data/nasa_scoring_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Scoring analysis saved to: data/nasa_scoring_analysis.png")
    
    # ==================== DETAILED ERROR ANALYSIS ====================
    print("\n" + "=" * 70)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 70)
    
    for model_col, model_name in zip(['LR_Pred', 'RF_Pred', 'GB_Pred'], 
                                     ['Linear Regression', 'Random Forest', 'Gradient Boosting']):
        y_pred = predictions[model_col].values
        analysis = analyze_prediction_errors(y_true, y_pred, model_name)
        
        print(f"\n{model_name}:")
        print(f"  Early predictions: {analysis['Early_Pct']:.1f}%")
        print(f"  Late predictions: {analysis['Late_Pct']:.1f}%")
        if analysis['Early_Pct'] > 0:
            print(f"  Avg error (early): {analysis['Early_MAE']:.2f} cycles")
            print(f"  Worst case (early): {analysis['Worst_Early_Error']:.2f} cycles (too early)")
        if analysis['Late_Pct'] > 0:
            print(f"  Avg error (late): {analysis['Late_MAE']:.2f} cycles")
            print(f"  Worst case (late): {analysis['Worst_Late_Error']:.2f} cycles (too late)")
    
    return results_df, predictions


if __name__ == "__main__":
    results, predictions = main()
    print("\n" + "=" * 70)
    print("Next step: Run 05_visualization.py for interactive dashboard")
    print("=" * 70)

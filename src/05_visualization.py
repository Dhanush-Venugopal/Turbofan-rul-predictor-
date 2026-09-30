"""
STEP 5: Visualization Dashboard
================================
Creates comprehensive visualizations of the entire RUL prediction pipeline.

Outputs:
- Degradation trajectories
- Health index evolution
- RUL predictions over time
- Model comparisons
- Error distributions
- Engine-specific analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class VisualizationDashboard:
    """Create comprehensive analysis visualizations"""
    
    def __init__(self, dataset_file, predictions_file):
        """Initialize with data files"""
        self.dataset = pd.read_csv(dataset_file)
        self.predictions = pd.read_csv(predictions_file)
    
    def plot_engine_trajectories_detailed(self):
        """Plot detailed engine health trajectories with multiple panels"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Select 9 diverse engines
        engine_ids = sorted(self.dataset['engine_id'].unique())
        selected_engines = [engine_ids[i] for i in np.linspace(0, len(engine_ids)-1, 9, dtype=int)]
        
        for idx, engine_id in enumerate(selected_engines):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            engine_data = self.dataset[self.dataset['engine_id'] == engine_id]
            
            # Plot health and RUL
            ax2 = ax.twinx()
            
            line1 = ax.plot(engine_data['cycle'], engine_data['health'], 
                          'b-', linewidth=2, label='Actual Health')
            line2 = ax2.plot(engine_data['cycle'], engine_data['calculated_health'], 
                           'g--', linewidth=2, label='Calculated Health')
            
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Actual Health', color='b')
            ax2.set_ylabel('Calculated Health', color='g')
            ax.set_title(f'Engine {engine_id}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            ax2.set_ylim([0, 1.05])
            
            # Highlight failure region
            ax.axhspan(0, 0.1, alpha=0.2, color='red', label='Failure zone')
        
        fig.suptitle('Engine Health Degradation Trajectories (9 Examples)', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_rul_predictions_over_time(self):
        """Show RUL predictions evolving over engine lifecycle"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Select 4 engines for detailed RUL tracking
        engine_ids = sorted(self.predictions['engine_id'].unique())[:4]
        
        for ax_idx, engine_id in enumerate(engine_ids):
            ax = axes[ax_idx // 2, ax_idx % 2]
            
            engine_pred = self.predictions[self.predictions['engine_id'] == engine_id]
            
            # Plot actual and predicted RUL over time
            ax.plot(engine_pred['cycle'], engine_pred['RUL'], 'k-', linewidth=2, label='Actual RUL')
            ax.plot(engine_pred['cycle'], engine_pred['LR_Pred'], 'o-', alpha=0.6, label='Linear Reg')
            ax.plot(engine_pred['cycle'], engine_pred['RF_Pred'], 's-', alpha=0.6, label='Random Forest')
            ax.plot(engine_pred['cycle'], engine_pred['GB_Pred'], '^-', alpha=0.6, label='Grad Boost')
            
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Flight Cycle')
            ax.set_ylabel('Remaining Useful Life (cycles)')
            ax.set_title(f'Engine {engine_id}: RUL Predictions Over Time')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('RUL Evolution: Actual vs Predicted', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_prediction_errors_timeline(self):
        """Plot prediction errors as functions of RUL"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        y_true = self.predictions['RUL'].values
        models = ['LR_Pred', 'RF_Pred', 'GB_Pred']
        model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
        colors = ['blue', 'orange', 'green']
        
        for ax_idx, (model_col, model_name, color) in enumerate(zip(models, model_names, colors)):
            ax = axes[ax_idx]
            
            y_pred = self.predictions[model_col].values
            errors = y_pred - y_true
            
            # Sort by RUL for better visualization
            sort_idx = np.argsort(y_true)
            sorted_rul = y_true[sort_idx]
            sorted_errors = errors[sort_idx]
            
            # Plot scatter with moving average
            ax.scatter(sorted_rul, sorted_errors, alpha=0.3, s=20, color=color)
            
            # Add moving average line
            window = 20
            if len(sorted_errors) > window:
                moving_avg = pd.Series(sorted_errors).rolling(window, center=True).mean()
                ax.plot(sorted_rul, moving_avg, color=color, linewidth=2, label='Moving avg')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect')
            ax.fill_between(sorted_rul, 0, sorted_errors, where=(sorted_errors<0), 
                          alpha=0.2, color='green', label='Early (good)')
            ax.fill_between(sorted_rul, 0, sorted_errors, where=(sorted_errors>=0), 
                          alpha=0.2, color='red', label='Late (bad)')
            
            ax.set_xlabel('Actual RUL (cycles)')
            ax.set_ylabel('Prediction Error (cycles)')
            ax.set_title(f'{model_name}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Prediction Error Distribution by Model', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_model_comparison_matrices(self):
        """Create comparison matrices for all models"""
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        y_true = self.predictions['RUL'].values
        models = ['LR_Pred', 'RF_Pred', 'GB_Pred']
        model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
        
        # Top row: Actual vs Predicted
        for idx, (model_col, model_name) in enumerate(zip(models, model_names)):
            ax = axes[0, idx]
            
            y_pred = self.predictions[model_col].values
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            ax.set_xlabel('Actual RUL')
            ax.set_ylabel('Predicted RUL')
            ax.set_title(f'{model_name}: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Residual plots
        for idx, (model_col, model_name) in enumerate(zip(models, model_names)):
            ax = axes[1, idx]
            
            y_pred = self.predictions[model_col].values
            residuals = y_pred - y_true
            
            # Residual histogram
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
            ax.axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2, 
                      label=f'Mean={np.mean(residuals):.1f}')
            
            ax.set_xlabel('Prediction Error (cycles)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model_name}: Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Comprehensive Model Comparison', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_sensor_degradation(self):
        """Show how key sensors degrade over engine life"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sensors = ['T30', 'Ps30', 'Nf', 'Nc']
        sensor_labels = ['HPC Outlet Temp (°R)', 'HPC Static Pressure (psia)', 
                        'Fan Speed (rpm)', 'Core Speed (rpm)']
        
        for ax_idx, (sensor, label) in enumerate(zip(sensors, sensor_labels)):
            ax = axes[ax_idx // 2, ax_idx % 2]
            
            # Plot for multiple engines
            engine_ids = sorted(self.dataset['engine_id'].unique())[:5]
            
            for engine_id in engine_ids:
                engine_data = self.dataset[self.dataset['engine_id'] == engine_id]
                # Normalize for comparison
                sensor_vals = engine_data[sensor].values
                normalized = (sensor_vals - sensor_vals.min()) / (sensor_vals.max() - sensor_vals.min())
                ax.plot(engine_data['cycle'], normalized, alpha=0.7, label=f'Engine {engine_id}')
            
            ax.set_xlabel('Cycle')
            ax.set_ylabel(f'Normalized {label}')
            ax.set_title(f'Degradation Pattern: {sensor}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Key Sensor Degradation Patterns', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_summary_dashboard(self):
        """Create a single comprehensive summary dashboard"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
        
        # 1. Health degradation (top left, spanning 2 cols)
        ax = fig.add_subplot(gs[0, :2])
        for engine_id in sorted(self.dataset['engine_id'].unique())[:5]:
            engine_data = self.dataset[self.dataset['engine_id'] == engine_id]
            ax.plot(engine_data['cycle'], engine_data['health'], 
                   alpha=0.6, label=f'E{engine_id}')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Health Index')
        ax.set_title('Engine Health Trajectories (5 Examples)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. RUL predictions (top right, spanning 2 cols)
        ax = fig.add_subplot(gs[0, 2:])
        y_true = self.predictions['RUL'].values
        ax.scatter(y_true, self.predictions['RF_Pred'].values, 
                  alpha=0.5, s=20, label='Random Forest')
        min_v, max_v = y_true.min(), y_true.max()
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual RUL')
        ax.set_ylabel('Predicted RUL')
        ax.set_title('Best Model: Prediction Accuracy', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3-5. Model comparisons (middle row)
        models = ['LR_Pred', 'RF_Pred', 'GB_Pred']
        model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
        
        for idx, (model_col, model_name) in enumerate(zip(models, model_names)):
            ax = fig.add_subplot(gs[1, idx])
            y_pred = self.predictions[model_col].values
            errors = y_pred - y_true
            
            ax.hist(errors, bins=25, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Error (cycles)')
            ax.set_ylabel('Count')
            ax.set_title(f'{model_name}')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Time-series prediction (middle right)
        ax = fig.add_subplot(gs[1, 3])
        engine_id = sorted(self.predictions['engine_id'].unique())[0]
        engine_pred = self.predictions[self.predictions['engine_id'] == engine_id]
        ax.plot(engine_pred['cycle'], engine_pred['RUL'], 'k-', linewidth=2)
        ax.plot(engine_pred['cycle'], engine_pred['RF_Pred'], 'o-', alpha=0.6)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('RUL (cycles)')
        ax.set_title(f'Engine {engine_id}: Prediction Timeline')
        ax.grid(True, alpha=0.3)
        
        # 7-9. Sensor trends (bottom row)
        sensors = ['T30', 'Nf', 'P30']
        sensor_titles = ['Temperature', 'Fan Speed', 'Pressure']
        
        for idx, (sensor, title) in enumerate(zip(sensors, sensor_titles)):
            ax = fig.add_subplot(gs[2, idx])
            
            for engine_id in sorted(self.dataset['engine_id'].unique())[:3]:
                engine_data = self.dataset[self.dataset['engine_id'] == engine_id]
                ax.plot(engine_data['cycle'], engine_data[sensor], alpha=0.7)
            
            ax.set_xlabel('Cycle')
            ax.set_ylabel(f'{title} Value')
            ax.set_title(f'{title} Degradation')
            ax.grid(True, alpha=0.3)
        
        # 10. Statistics summary (bottom right)
        ax = fig.add_subplot(gs[2, 3])
        ax.axis('off')
        
        # Calculate key statistics
        n_engines = self.dataset['engine_id'].nunique()
        n_samples = len(self.dataset)
        avg_health = self.dataset['health'].mean()
        avg_rul = self.predictions['RUL'].mean()
        
        stats_text = f"""
KEY STATISTICS

Engines Analyzed: {n_engines}
Total Measurements: {n_samples}

Average Health: {avg_health:.3f}
Average RUL: {avg_rul:.1f} cycles

RF Test Error:
  RMSE: {np.sqrt(np.mean((self.predictions['RF_Pred'].values - y_true)**2)):.2f}
  MAE: {np.mean(np.abs(self.predictions['RF_Pred'].values - y_true)):.2f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center')
        
        fig.suptitle('TURBOFAN RUL PREDICTION SYSTEM - COMPREHENSIVE DASHBOARD', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        
        print("Generating visualizations...")
        output_dir = Path("data")
        
        # 1. Engine trajectories
        print("  - Engine degradation trajectories...")
        fig = self.plot_engine_trajectories_detailed()
        fig.savefig(output_dir / 'viz_engine_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. RUL predictions over time
        print("  - RUL evolution charts...")
        fig = self.plot_rul_predictions_over_time()
        fig.savefig(output_dir / 'viz_rul_timeline.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Prediction errors
        print("  - Prediction error analysis...")
        fig = self.plot_prediction_errors_timeline()
        fig.savefig(output_dir / 'viz_prediction_errors.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 4. Model comparisons
        print("  - Model comparison matrices...")
        fig = self.plot_model_comparison_matrices()
        fig.savefig(output_dir / 'viz_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 5. Sensor degradation
        print("  - Sensor degradation patterns...")
        fig = self.plot_sensor_degradation()
        fig.savefig(output_dir / 'viz_sensor_degradation.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 6. Summary dashboard
        print("  - Summary dashboard...")
        fig = self.plot_summary_dashboard()
        fig.savefig(output_dir / 'viz_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ All visualizations complete!")
        return output_dir


# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 70)
    print("STEP 5: VISUALIZATION DASHBOARD")
    print("=" * 70)
    
    # Check required files
    dataset_file = Path("data/trajectories_with_health.csv")
    predictions_file = Path("data/model_predictions.csv")
    
    if not dataset_file.exists() or not predictions_file.exists():
        print(f"\n❌ Error: Required files not found!")
        print(f"  - {dataset_file}")
        print(f"  - {predictions_file}")
        print("Please run previous steps first.")
        return None
    
    print(f"\nLoading data...")
    print(f"  - Dataset: {dataset_file}")
    print(f"  - Predictions: {predictions_file}")
    
    # Create dashboard
    dashboard = VisualizationDashboard(str(dataset_file), str(predictions_file))
    
    # Generate all visualizations
    output_dir = dashboard.generate_all_visualizations()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print("""
Generated Files:
  1. viz_engine_trajectories.png     - Health degradation patterns
  2. viz_rul_timeline.png             - RUL evolution over time
  3. viz_prediction_errors.png        - Error distribution analysis
  4. viz_model_comparison.png         - Model performance matrices
  5. viz_sensor_degradation.png       - Key sensor trends
  6. viz_summary_dashboard.png        - Comprehensive overview

All visualizations are high-resolution (300 DPI) suitable for portfolio/reports.
    """)
    
    return dashboard


if __name__ == "__main__":
    dashboard = main()
    print("\n" + "=" * 70)
    print("PROJECT PIPELINE COMPLETE!")
    print("=" * 70)

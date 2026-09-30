"""
STEP 3: RUL Predictor
======================
Trains multiple ML models to predict Remaining Useful Life (RUL) from sensor data.

Models Trained:
1. Linear Regression (baseline)
2. Random Forest Regression
3. Neural Network (LSTM or Dense)

Approach:
---------
- Use first 60% of engine lifecycle as training data (healthy to mid-life)
- Test predictions on remaining 40% (mid-life to failure)
- Use sensor values + health indicator to predict RUL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Attempt to import deep learning (optional)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    HAS_KERAS = True
except:
    HAS_KERAS = False
    print("Warning: TensorFlow/Keras not available. Deep learning models will be skipped.")


# ==================== MODEL CONFIGURATION ====================

class RULPredictorConfig:
    """Configuration for RUL prediction models"""
    
    # Feature selection (sensors to use for prediction)
    SENSOR_FEATURES = [
    'T2', 'T24', 'T30', 'T50',           # Temperatures
    'P2', 'P15', 'P30', 'Ps30',          # Pressures
    'Nf', 'Nc', 'NRf', 'NRc',            # Speeds
    'epr', 'BPR', 'phi', 'farB',         # Ratios
    'W31', 'W32', 'health',              # Flows + health
    ]
    
    # Training split
    TRAIN_CYCLE_FRACTION = 0.6  # Use first 60% of lifecycle for training
    
    # Random seed
    RANDOM_STATE = 42
    
    # Hyperparameters
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    GB_N_ESTIMATORS = 100
    GB_LEARNING_RATE = 0.1
    
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    LSTM_LAYERS = [64, 32]  # Hidden units per layer


class RULPredictor:
    """Wrapper class for RUL prediction models"""
    
    def __init__(self, config=RULPredictorConfig()):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = config.SENSOR_FEATURES
        
    def prepare_features(self, data, fit=False):
        """Extract and scale features"""
        X = data[self.feature_names].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, X
    
    def split_training_data(self, data):
        """
        Split each engine's data into training/testing based on lifecycle fraction.
        
        Strategy:
        - For each engine, use first N cycles as training
        - Use remaining cycles as testing
        - This simulates predicting during mid-life on data trained on early-life
        """
        train_data = []
        test_data = []
        
        for engine_id in data['engine_id'].unique():
            engine = data[data['engine_id'] == engine_id].reset_index(drop=True)
            
            # Find split point
            max_cycle = engine['cycle'].max()
            split_cycle = int(max_cycle * self.config.TRAIN_CYCLE_FRACTION)
            
            # Split data
            train = engine[engine['cycle'] <= split_cycle].copy()
            test = engine[engine['cycle'] > split_cycle].copy()
            
            if len(train) > 0:
                train_data.append(train)
            if len(test) > 0:
                test_data.append(test)
        
        train_full = pd.concat(train_data, ignore_index=True)
        test_full = pd.concat(test_data, ignore_index=True)
        
        return train_full, test_full
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train linear regression baseline"""
        print("  Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        print(f"    RMSE (train): {train_rmse:.2f}, RMSE (test): {test_rmse:.2f}")
        print(f"    R² score: {test_r2:.4f}")
        
        return results
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest regression"""
        print("  Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'feature_importance': model.feature_importances_
        }
        
        print(f"    RMSE (train): {train_rmse:.2f}, RMSE (test): {test_rmse:.2f}")
        print(f"    R² score: {test_r2:.4f}")
        
        return results
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting regression"""
        print("  Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=self.config.GB_N_ESTIMATORS,
            learning_rate=self.config.GB_LEARNING_RATE,
            max_depth=5,
            random_state=self.config.RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'feature_importance': model.feature_importances_
        }
        
        print(f"    RMSE (train): {train_rmse:.2f}, RMSE (test): {test_rmse:.2f}")
        print(f"    R² score: {test_r2:.4f}")
        
        return results
    
    def train_all_models(self, data):
        """Train all available models"""
        print("\nPreparing data...")
        
        # Split data
        train_data, test_data = self.split_training_data(data)
        
        print(f"  Training set: {len(train_data)} samples")
        print(f"  Test set: {len(test_data)} samples")
        
        # Prepare features (fit scaler on training data)
        X_train_scaled, X_train = self.prepare_features(train_data, fit=True)
        X_test_scaled, X_test = self.prepare_features(test_data, fit=False)
        
        y_train = train_data['RUL'].values
        y_test = test_data['RUL'].values
        
        print("\nTraining models...")
        
        # Train models
        self.models['Linear_Regression'] = self.train_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        self.models['Random_Forest'] = self.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        self.models['Gradient_Boosting'] = self.train_gradient_boosting(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Store test data for later evaluation
        self.test_data = test_data.copy()
        self.y_test = y_test
        
        return self.models, test_data, y_test


# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 70)
    print("STEP 3: RUL PREDICTOR - MODEL TRAINING")
    print("=" * 70)
    
    # Load processed data
    input_file = Path("data/trajectories_with_health.csv")
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        print("Please run 02_health_index_calculator.py first.")
        return None
    
    print(f"\nLoading data from: {input_file}")
    data = pd.read_csv(input_file)
    print(f"  - Loaded {len(data)} records from {data['engine_id'].nunique()} engines")
    
    # Create predictor
    config = RULPredictorConfig()
    predictor = RULPredictor(config)
    
    print(f"\nFeatures used: {len(config.SENSOR_FEATURES)}")
    print(f"  - Sensors: {', '.join(config.SENSOR_FEATURES[:10])}...")
    print(f"  - Training/Test split: {config.TRAIN_CYCLE_FRACTION*100:.0f}% / {(1-config.TRAIN_CYCLE_FRACTION)*100:.0f}%")
    
    # Train models
    models, test_data, y_test = predictor.train_all_models(data)
    
    # ==================== RESULTS SUMMARY ====================
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame({
        'Model': list(models.keys()),
        'Train RMSE': [models[m]['train_rmse'] for m in models],
        'Test RMSE': [models[m]['test_rmse'] for m in models],
        'Test MAE': [models[m]['test_mae'] for m in models],
        'R² Score': [models[m]['test_r2'] for m in models],
    })
    
    print("\n" + results_df.to_string(index=False))
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_name, model_dict in models.items():
        model_file = models_dir / f"{model_name.lower()}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_dict['model'], f)
        print(f"\n✓ Saved: {model_file}")
    
    # Save scaler
    scaler_file = models_dir / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(predictor.scaler, f)
    print(f"✓ Saved: {scaler_file}")
    
    # ==================== VISUALIZATION ====================
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions vs Actual (Linear Regression)
    ax = axes[0, 0]
    y_pred_lr = models['Linear_Regression']['y_pred_test']
    ax.scatter(y_test, y_pred_lr, alpha=0.5, s=20)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Linear Regression: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual (Random Forest)
    ax = axes[0, 1]
    y_pred_rf = models['Random_Forest']['y_pred_test']
    ax.scatter(y_test, y_pred_rf, alpha=0.5, s=20, color='orange')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Random Forest: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs Actual (Gradient Boosting)
    ax = axes[1, 0]
    y_pred_gb = models['Gradient_Boosting']['y_pred_test']
    ax.scatter(y_test, y_pred_gb, alpha=0.5, s=20, color='green')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Gradient Boosting: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Model Comparison - RMSE
    ax = axes[1, 1]
    model_names = list(models.keys())
    train_rmse = [models[m]['train_rmse'] for m in model_names]
    test_rmse = [models[m]['test_rmse'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
    ax.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
    ax.set_ylabel('RMSE (cycles)')
    ax.set_title('Model Comparison - RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig('data/model_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: data/model_predictions.png")
    
    # ==================== FEATURE IMPORTANCE ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Random Forest importance
    ax = axes[0]
    rf_importance = models['Random_Forest']['feature_importance']
    top_n = 10
    top_indices = np.argsort(rf_importance)[-top_n:]
    top_features = [config.SENSOR_FEATURES[i] for i in top_indices]
    top_importance = rf_importance[top_indices]
    
    ax.barh(top_features, top_importance, alpha=0.8, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest - Top 10 Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Gradient Boosting importance
    ax = axes[1]
    gb_importance = models['Gradient_Boosting']['feature_importance']
    top_indices = np.argsort(gb_importance)[-top_n:]
    top_features = [config.SENSOR_FEATURES[i] for i in top_indices]
    top_importance = gb_importance[top_indices]
    
    ax.barh(top_features, top_importance, alpha=0.8, color='darkorange')
    ax.set_xlabel('Importance')
    ax.set_title('Gradient Boosting - Top 10 Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig('data/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved to: data/feature_importance.png")
    
    # Save test predictions for evaluation
    predictions_df = test_data[['engine_id', 'cycle', 'RUL', 'calculated_health']].copy()
    predictions_df['LR_Pred'] = models['Linear_Regression']['y_pred_test']
    predictions_df['RF_Pred'] = models['Random_Forest']['y_pred_test']
    predictions_df['GB_Pred'] = models['Gradient_Boosting']['y_pred_test']
    predictions_df.to_csv('data/model_predictions.csv', index=False)
    print(f"✓ Predictions saved to: data/model_predictions.csv")
    
    return models, predictor, test_data


if __name__ == "__main__":
    models, predictor, test_data = main()
    print("\n" + "=" * 70)
    print("Next step: Run 04_performance_metrics.py")
    print("=" * 70)

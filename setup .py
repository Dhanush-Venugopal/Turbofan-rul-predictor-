import os

# Create directories
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("✓ Folders created!")
print("\nNow run these commands in order:")
print("1. python src/01_degradation_simulator.py")
print("2. python src/02_health_index_calculator.py")
print("3. python src/03_rul_predictor.py")
print("4. python src/04_performance_metrics.py")
print("5. python src/05_visualization.py")
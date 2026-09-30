"""
STEP 1: Degradation Simulator - Generate synthetic turbofan engine data
FIXED VERSION - Includes all required sensors
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

class Config:
    N_ENGINES = 100
    A_MIN, A_MAX = 0.001, 0.003
    B_MIN, B_MAX = 1.4, 1.6
    INITIAL_DEGRADATION_MIN = 0.99
    INITIAL_DEGRADATION_MAX = 1.0
    SENSOR_NOISE_STD = 0.02
    MAX_CYCLES = 250

def health_function(t, a, b, initial_degradation=0.0):
    """Calculate health: h(t) = 1 - exp{-a*t^b}"""
    if t == 0:
        return 1.0 - initial_degradation
    h = 1.0 - initial_degradation - a * (t ** b)
    return max(0.0, h)

def generate_engine_trajectory(engine_id, config=Config()):
    """Generate one engine's run-to-failure data"""
    
    a = np.random.uniform(config.A_MIN, config.A_MAX)
    b = np.random.uniform(config.B_MIN, config.B_MAX)
    initial_deg = np.random.uniform(config.INITIAL_DEGRADATION_MIN, 
                                     config.INITIAL_DEGRADATION_MAX)
    
    trajectory = []
    cycle = 0
    
    while cycle < config.MAX_CYCLES:
        health = health_function(cycle, a, b, 1.0 - initial_deg)
        
        if health <= 0.0:
            break
        
        # Degradation factor (0 = healthy, 1 = failed)
        deg = 1.0 - health
        
        # Create ALL required sensor measurements
        row = {
            'engine_id': engine_id,
            'cycle': cycle,
            'health': health,
            # Temperatures
            'T2': 518.67 + deg * 50,
            'T24': 644.88 + deg * 80,
            'T30': 1040.50 + deg * 100,
            'T50': 928.45 + deg * 80,
            # Pressures
            'P2': 14.696,
            'P15': 24.670 * (1 - deg * 0.15),
            'P30': 43.820 * (1 - deg * 0.20),
            'Ps30': 27.850 * (1 - deg * 0.15),
            # Speeds
            'Nf': 2388.0 * (1 - deg * 0.08),
            'Nc': 9046.19 * (1 - deg * 0.08),
            'NRf': 100.0 * (1 - deg * 0.08),
            'NRc': 100.0 * (1 - deg * 0.08),
            # Ratios
            'epr': 2.98 * (1 - deg * 0.10),
            'BPR': 7.59 * (1 - deg * 0.05),
            'phi': 5.4 * (1 - deg * 0.08),
            'farB': 0.0185 * (1 + deg * 0.10),
            # Flow/Enthalpy
            'W31': 404.0 * (1 - deg * 0.10),
            'W32': 186.0 * (1 - deg * 0.10),
            'htBleed': 1269.0 * (1 - deg * 0.05),
            # Demands
            'Nf_dmd': 2388.0,
            'PCNfR_dmd': 100.0,
            # Degradation params
            'degradation_rate': a,
            'degradation_power': b,
        }
        
        trajectory.append(row)
        cycle += 1
    
    return pd.DataFrame(trajectory)

def generate_all_trajectories(config=Config()):
    """Generate all engines"""
    print(f"Generating {config.N_ENGINES} engine trajectories...")
    
    all_data = []
    for engine_id in range(1, config.N_ENGINES + 1):
        if engine_id % 20 == 0:
            print(f"  Engine {engine_id}/{config.N_ENGINES}")
        
        trajectory = generate_engine_trajectory(engine_id, config)
        all_data.append(trajectory)
    
    return pd.concat(all_data, ignore_index=True)

def main():
    config = Config()
    
    print("=" * 60)
    print("STEP 1: DEGRADATION SIMULATOR (FIXED)")
    print("=" * 60)
    
    dataset = generate_all_trajectories(config)
    
    print(f"\nGeneration complete!")
    print(f"  Total records: {len(dataset)}")
    print(f"  Engines: {dataset['engine_id'].nunique()}")
    print(f"  Columns: {len(dataset.columns)}")
    
    # Save
    Path("data").mkdir(exist_ok=True)
    dataset.to_csv("data/simulated_trajectories.csv", index=False)
    print(f"\n✓ Saved to: data/simulated_trajectories.csv")
    
    # Simple visualization
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for engine_id in range(1, 6):
        engine_data = dataset[dataset['engine_id'] == engine_id]
        ax.plot(engine_data['cycle'], engine_data['health'], 
                label=f'Engine {engine_id}', alpha=0.7)
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Health Index')
    ax.set_title('Engine Health Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    fig.savefig('data/degradation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: data/degradation_analysis.png")
    
    return dataset

if __name__ == "__main__":
    dataset = main()
    print("\n" + "=" * 60)
    print("✓ STEP 1 COMPLETE")
    print("=" * 60)

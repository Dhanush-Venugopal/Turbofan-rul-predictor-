"""
STEP 2: Health Index Calculator
================================
Calculates health indices and remaining useful life (RUL) based on sensor data.

Key Concepts:
-----------
The NASA paper uses operational margins to determine health:
    
    SmFan = Fan stall margin (%)
    SmHPC = High-Pressure Compressor stall margin (%)
    SmLPC = Low-Pressure Compressor stall margin (%)
    EGT_margin = Exhaust Gas Temperature margin (°R)

Overall health = minimum of all margins (most restrictive)
Failure = when any margin violates its threshold

This module:
1. Estimates margins from sensor readings
2. Calculates health index from margins
3. Computes RUL (remaining cycles to failure)
4. Identifies failure patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HealthCalculator:
    """
    Calculates health indices and RUL from simulated or real sensor data.
    
    Reference margins (from NASA C-MAPSS):
    - Stall margin thresholds: 15% loss
    - EGT margin threshold: 2% rise
    """
    
    # Healthy (nominal) sensor values
    NOMINAL_SENSORS = {
        'T2': 518.67,       # Inlet temp (°R)
        'T24': 644.88,      # LPC outlet temp
        'T30': 1040.50,     # HPC outlet temp
        'T50': 928.45,      # LPT outlet temp (EGT)
        'P2': 14.696,       # Inlet pressure (psia)
        'P15': 24.670,      # Bypass duct pressure
        'P30': 43.820,      # HPC outlet pressure
        'Ps30': 27.850,     # HPC static pressure
        'Nf': 2388.0,       # Fan speed (rpm)
        'Nc': 9046.19,      # Core speed
        'epr': 2.98,        # Engine pressure ratio
        'BPR': 7.59,        # Bypass ratio
    }
    
    # Failure thresholds (from paper Table 2 & Section V.D)
    STALL_MARGIN_THRESHOLD = 0.15    # 15% stall margin loss = failure
    EGT_MARGIN_THRESHOLD = 20.0      # °R above nominal
    
    @staticmethod
    def estimate_fan_stall_margin(nf_corrected, ps30, p2):
        """
        Estimate fan stall margin from corrected speeds and pressures.
        
        Based on: SmFan relates to pressure rise ability
        Healthy condition: SmFan ≈ 8-9%
        Failure when: SmFan < 0
        
        Simplified model:
        SmFan is proportional to corrected speed and pressure drop
        """
        # Nominal conditions
        nf_nominal = 100.0
        ps30_nominal = 27.85
        
        # Margin normalized to healthy state
        margin = 9.0 * (nf_corrected / nf_nominal) * (1.0 - 0.05 * (1 - ps30/ps30_nominal))
        
        return max(0.0, margin)
    
    @staticmethod
    def estimate_hpc_stall_margin(nc_corrected, p30, ps30):
        """
        Estimate HPC stall margin from core speed and pressures.
        
        Healthy condition: SmHPC ≈ 32%
        Failure when: SmHPC < 0 (actually, threshold is ~2-3%)
        
        The paper uses operational margins from response surfaces.
        We approximate based on thermodynamic relationships.
        """
        # Nominal conditions
        nc_nominal = 100.0
        p30_nominal = 43.82
        ps30_nominal = 27.85
        
        # Pressure ratio across HPC
        pressure_ratio = p30 / ps30
        
        # Margin based on speed and pressure
        margin = 32.0 * (nc_corrected / nc_nominal) * (1.0 - 0.04 * (pressure_ratio - 1.55))
        
        return max(0.0, margin)
    
    @staticmethod
    def estimate_lpc_stall_margin(nf_corrected, p15, p2):
        """
        Estimate LPC stall margin (Low-Pressure Compressor).
        
        Healthy condition: SmLPC ≈ 15%
        Failure threshold: SmLPC < 0
        """
        # Nominal conditions
        nf_nominal = 100.0
        p15_nominal = 24.67
        
        # Pressure ratio across LPC
        pressure_ratio = p15 / p2
        
        margin = 15.0 * (nf_corrected / nf_nominal) * (1.0 - 0.06 * (pressure_ratio - 1.68))
        
        return max(0.0, margin)
    
    @staticmethod
    def estimate_egt_margin(t50):
        """
        Estimate EGT (Exhaust Gas Temperature) margin.
        
        Healthy condition: T50 (LPT outlet) ≈ 928°R
        Maximum safe EGT: ~2450°R (maximum at takeoff)
        
        For cruise: safety margin ≈ 200-300°R from max
        
        Margin = how much room until we hit EGT limit
        """
        # Maximum allowable EGT
        egt_max = 2450.0
        
        # Current margin (how far below limit)
        margin = (egt_max - t50) / 10.0  # Normalized
        
        return max(0.0, margin)
    
    @classmethod
    def calculate_health_index(cls, sensor_row):
        """
        Calculate overall health index from sensor measurements.
        
        Health = minimum of all operational margins
        Range: 1.0 (healthy) to 0.0 (failed)
        """
        
        # Extract corrected speeds (already normalized in our simulator)
        nf_corr = sensor_row['NRf'] / 100.0 if 'NRf' in sensor_row else 1.0
        nc_corr = sensor_row['NRc'] / 100.0 if 'NRc' in sensor_row else 1.0
        
        # Calculate individual margins (normalized to 0-1 range)
        sm_fan = cls.estimate_fan_stall_margin(nf_corr, sensor_row['Ps30'], sensor_row['P2']) / 10.0
        sm_hpc = cls.estimate_hpc_stall_margin(nc_corr, sensor_row['P30'], sensor_row['Ps30']) / 35.0
        sm_lpc = cls.estimate_lpc_stall_margin(nf_corr, sensor_row['P15'], sensor_row['P2']) / 15.0
        egt_margin = cls.estimate_egt_margin(sensor_row['T50']) / 20.0
        
        # Clamp all margins to [0, 1]
        margins = {
            'SmFan': min(1.0, max(0.0, sm_fan)),
            'SmHPC': min(1.0, max(0.0, sm_hpc)),
            'SmLPC': min(1.0, max(0.0, sm_lpc)),
            'EGT_margin': min(1.0, max(0.0, egt_margin))
        }
        
        # Health index = minimum margin (most restrictive)
        health = min(margins.values())
        
        return health, margins
    
    @classmethod
    def calculate_rul(cls, engine_data):
        """
        Calculate Remaining Useful Life (RUL) for each measurement.
        
        RUL = cycles remaining until health reaches 0
        """
        # Calculate health for all measurements
        health_values = []
        for idx, row in engine_data.iterrows():
            health, _ = cls.calculate_health_index(row)
            health_values.append(health)
        
        engine_data = engine_data.copy()
        engine_data['calculated_health'] = health_values
        
        # Calculate RUL
        rul_values = []
        max_cycle = engine_data['cycle'].max()
        
        for idx, row in engine_data.iterrows():
            # RUL = cycles until health would reach 0
            health = row['calculated_health']
            current_cycle = row['cycle']
            
            # Estimate RUL based on current health and degradation trend
            if health > 0.01:
                # Assume monotonic degradation
                # Rough estimate: RUL proportional to current health
                rul = max_cycle - current_cycle
            else:
                rul = 0
            
            rul_values.append(rul)
        
        engine_data['RUL'] = rul_values
        
        return engine_data


def process_all_engines(dataset):
    """Process entire dataset to add health and RUL columns"""
    print("Calculating health indices and RUL...")
    
    calculator = HealthCalculator()
    processed_data = []
    
    for engine_id in dataset['engine_id'].unique():
        engine_data = dataset[dataset['engine_id'] == engine_id].reset_index(drop=True)
        
        # Calculate health for each measurement
        health_values = []
        sm_fan_values = []
        sm_hpc_values = []
        sm_lpc_values = []
        egt_values = []
        
        for idx, row in engine_data.iterrows():
            health, margins = calculator.calculate_health_index(row)
            health_values.append(health)
            sm_fan_values.append(margins['SmFan'])
            sm_hpc_values.append(margins['SmHPC'])
            sm_lpc_values.append(margins['SmLPC'])
            egt_values.append(margins['EGT_margin'])
        
        engine_data['calculated_health'] = health_values
        engine_data['SmFan'] = sm_fan_values
        engine_data['SmHPC'] = sm_hpc_values
        engine_data['SmLPC'] = sm_lpc_values
        engine_data['EGT_margin'] = egt_values
        
        # Calculate RUL
        rul_values = []
        n_measurements = len(engine_data)
        for idx, row in engine_data.iterrows():
            current_idx = idx - engine_data.index[0]
            rul = n_measurements - current_idx - 1  # Cycles remaining
            rul_values.append(rul)
        
        engine_data['RUL'] = rul_values
        processed_data.append(engine_data)
    
    return pd.concat(processed_data, ignore_index=True)


# ==================== MAIN EXECUTION ====================

def main():
    print("=" * 70)
    print("STEP 2: HEALTH INDEX CALCULATOR")
    print("=" * 70)
    
    # Load simulated data
    input_file = Path("data/simulated_trajectories.csv")
    if not input_file.exists():
        print(f"\n❌ Error: {input_file} not found!")
        print("Please run 01_degradation_simulator.py first.")
        return None
    
    print(f"\nLoading simulated data from: {input_file}")
    dataset = pd.read_csv(input_file)
    print(f"  - Loaded {len(dataset)} records from {dataset['engine_id'].nunique()} engines")
    
    # Process all engines
    processed = process_all_engines(dataset)
    
    print(f"\nHealth calculations complete!")
    print(f"\nHealth Index Summary:")
    print(processed[['calculated_health', 'RUL']].describe())
    
    # Save processed data
    output_file = Path("data/trajectories_with_health.csv")
    processed.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    # ==================== VISUALIZATION ====================
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Comparison of actual vs calculated health
    ax = axes[0, 0]
    for engine_id in range(1, 6):
        engine_data = processed[processed['engine_id'] == engine_id]
        ax.plot(engine_data['cycle'], engine_data['health'], 
                label=f'Actual (E{engine_id})', alpha=0.7, linestyle='--')
        ax.plot(engine_data['cycle'], engine_data['calculated_health'], 
                label=f'Calculated (E{engine_id})', alpha=0.7, linestyle='-')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Health Index')
    ax.set_title('Actual vs Calculated Health (First 5 Engines)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Operational margins breakdown (one engine)
    ax = axes[0, 1]
    engine1 = processed[processed['engine_id'] == 1]
    ax.plot(engine1['cycle'], engine1['SmFan'], label='Fan Margin', linewidth=2)
    ax.plot(engine1['cycle'], engine1['SmHPC'], label='HPC Margin', linewidth=2)
    ax.plot(engine1['cycle'], engine1['SmLPC'], label='LPC Margin', linewidth=2)
    ax.plot(engine1['cycle'], engine1['EGT_margin'], label='EGT Margin', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Margin Value')
    ax.set_title('Engine 1: Individual Operational Margins')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', label='Failure Threshold')
    
    # Plot 3: RUL vs Cycle (different engines)
    ax = axes[1, 0]
    for engine_id in range(1, 6):
        engine_data = processed[processed['engine_id'] == engine_id]
        ax.plot(engine_data['cycle'], engine_data['RUL'], 
                label=f'Engine {engine_id}', marker='o', markersize=3, alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title('RUL Degradation Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Health-RUL scatter (all engines)
    ax = axes[1, 1]
    sample = processed.sample(n=min(1000, len(processed)))
    scatter = ax.scatter(sample['calculated_health'], sample['RUL'], 
                        c=sample['engine_id'], cmap='viridis', 
                        alpha=0.5, s=20)
    ax.set_xlabel('Health Index')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('Health vs RUL (All Engines)')
    plt.colorbar(scatter, ax=ax, label='Engine ID')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('data/health_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: data/health_analysis.png")
    
    # ==================== STATISTICS ====================
    print("\n" + "=" * 70)
    print("Key Statistics:")
    print("=" * 70)
    
    final_health = processed.groupby('engine_id')['calculated_health'].min()
    print(f"\nFinal Health at Failure:")
    print(f"  - Mean: {final_health.mean():.4f}")
    print(f"  - Std:  {final_health.std():.4f}")
    print(f"  - Min:  {final_health.min():.4f}")
    print(f"  - Max:  {final_health.max():.4f}")
    
    ruls = processed.groupby('engine_id')['RUL'].max()
    print(f"\nTotal RUL Distribution:")
    print(f"  - Mean: {ruls.mean():.1f} cycles")
    print(f"  - Std:  {ruls.std():.1f} cycles")
    print(f"  - Min:  {ruls.min():.0f} cycles")
    print(f"  - Max:  {ruls.max():.0f} cycles")
    
    return processed


if __name__ == "__main__":
    processed_data = main()
    print("\n" + "=" * 70)
    print("Next step: Run 03_rul_predictor.py")
    print("=" * 70)

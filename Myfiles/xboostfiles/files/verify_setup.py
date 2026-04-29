#!/usr/bin/env python3
"""
Pre-Training Verification Script
=================================

Run this BEFORE training to verify everything is set up correctly.
"""

import sys
from pathlib import Path

def check_file(path: str, description: str) -> bool:
    """Check if file exists"""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ MISSING: {description}: {path}")
        return False

def check_module(module_name: str) -> bool:
    """Check if Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ Module installed: {module_name}")
        return True
    except ImportError:
        print(f"❌ Module NOT installed: {module_name}")
        print(f"   Install with: pip install {module_name}")
        return False

def verify_csv_columns(csv_path: str) -> bool:
    """Verify CSV has required columns"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=5)
        
        required = ['frame', 'object_type', 'team', 'x', 'y', 'dx', 'dy', 'speed']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"❌ CSV missing columns: {missing}")
            print(f"   File: {csv_path}")
            print(f"   Has: {list(df.columns)}")
            return False
        else:
            print(f"✅ CSV has all required columns")
            return True
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

def main():
    print("="*60)
    print("🔍 PRE-TRAINING VERIFICATION")
    print("="*60)
    
    all_ok = True
    
    # 1. Check Python modules
    print("\n📦 Checking Python dependencies:")
    for module in ['pandas', 'numpy', 'xgboost', 'sklearn']:
        if not check_module(module):
            all_ok = False
    
    # 2. Check training script
    print("\n📄 Checking training files:")
    if not check_file('train_xgboost_model.py', 'Training script'):
        all_ok = False
    
    # 3. Check feature extractor
    print("\n🔧 Checking feature extraction:")
    if not check_file('worker/rugby_detections/analytics/event_features.py', 'Feature extractor'):
        all_ok = False
    else:
        # Try importing
        sys.path.insert(0, '.')
        try:
            from worker.rugby_detections.analytics.event_features import EventFeatureExtractor
            print("✅ Feature extractor can be imported")
            
            # Check if it's the full version
            import inspect
            source = inspect.getsource(EventFeatureExtractor.extract_features)
            if 'rolling' in source.lower() and 'window' in source.lower():
                print("✅ Full production version (200+ features)")
            else:
                print("⚠️  Simplified version (40 features)")
        except Exception as e:
            print(f"❌ Cannot import feature extractor: {e}")
            all_ok = False
    
    # 4. Check training data
    print("\n📊 Checking training data:")
    data_dir = Path('training_data')
    
    if not data_dir.exists():
        print(f"❌ Training data directory not found: {data_dir}")
        all_ok = False
    else:
        xy_files = list(data_dir.glob('*_xy.csv'))
        events_files = list(data_dir.glob('*_events.csv'))
        
        print(f"✅ Found {len(xy_files)} XY files")
        print(f"✅ Found {len(events_files)} events files")
        
        if len(xy_files) == 0:
            print("❌ No *_xy.csv files found!")
            all_ok = False
        
        # Check first XY file for correct columns
        if xy_files:
            print("\n🔍 Verifying CSV format:")
            if not verify_csv_columns(str(xy_files[0])):
                all_ok = False
                print("\n⚠️  IMPORTANT: Your CSV is missing dx, dy, or speed columns!")
                print("   You need to re-run your pipeline with Phase 1 velocity computation enabled.")
    
    # 5. Check output directory
    print("\n📁 Checking output directory:")
    models_dir = Path('models')
    if not models_dir.exists():
        print(f"⚠️  Creating models directory: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"✅ Models directory exists: {models_dir}")
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*60)
        print("\nRun training with:")
        print("  python train_xgboost_model.py --data-dir training_data --output-model models/rugby_xgb.json")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

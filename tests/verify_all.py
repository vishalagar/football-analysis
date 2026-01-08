
import sys
import os

# Ensure backend acts as package
sys.path.append(os.path.abspath("."))

from backend.app.services.data import get_dataset_stats, detect_issues, CustomImageDataset
from backend.app.core.config import TRAIN_DIR, VAL_DIR
from backend.app.services.training import run_automated_training

def verify_system():
    print("--- 🔬 System Verification ---")
    
    # 1. Stats
    print("\n[1] Checking Stats...")
    stats = get_dataset_stats()
    print(stats)
    
    # 2. Cleaning
    print("\n[2] Checking Issue Detection...")
    # Check if 'val' has enough classes
    val_has_issue_potential = True
    val_counts = stats['val']['classes']
    non_zero_classes = [c for c, n in val_counts.items() if n > 0]
    if len(non_zero_classes) < 2:
        print(f"⚠️ Warning: Validation set has only {len(non_zero_classes)} class with data ({non_zero_classes}). Issue detection requires >= 2 classes.")
        val_has_issue_potential = False

    issues = detect_issues()
    print(f"✅ Found {len(issues)} issues total.")
    
    train_issues = [i for i in issues if i['split'] == 'train']
    val_issues = [i for i in issues if i['split'] == 'valid'] # Note: 'valid' is hardcoded in detect_issues return?
    # Actually In data_manager.py, I call detect_issues_in_split("valid", VAL_DIR). So split is "valid".
    
    print(f"  - Train Issues: {len(train_issues)}")
    print(f"  - Val Issues: {len(val_issues)}")
    
    if val_has_issue_potential and len(val_issues) == 0:
        print("  (Note: No issues found in val, but it was checked.)")
    elif not val_has_issue_potential:
        print("  (Skipped Val issue detection due to single class)")

    # 3. Training
    print("\n[3] Checking Training (Smoke Test)...")
    try:
        # Create subsets
        train_ds = CustomImageDataset(TRAIN_DIR)
        val_ds = CustomImageDataset(VAL_DIR)
        
        # Subset
        subset = 20
        train_ds.files = train_ds.files[:subset]
        train_ds.labels = train_ds.labels[:subset]
        val_ds.files = val_ds.files[:subset]
        val_ds.labels = val_ds.labels[:subset]
        
        # If val is empty/single class, training might error on metric calculation if not careful?
        # CrossEntropyLoss requires valid labels.
        # If val_ds is empty/single class it should be fine as long as labels are within 0..num_classes-1.
        
        res = run_automated_training(full_epochs=1, dataset_train=train_ds, dataset_val=val_ds)
        print("✅ Training success.")
        print(f"  - Best Params: {res['best_params']}")
        print(f"  - Val Acc: {res['val_accuracy']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_system()

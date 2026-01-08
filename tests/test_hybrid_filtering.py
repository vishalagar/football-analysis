
import os
import torch
import numpy as np
from PIL import Image
from backend.app.services.data import detect_issues_in_split
from backend.app.core.config import TRAIN_DIR

def create_mock_dataset(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "OK"), exist_ok=True)
    
    # 1. Create 10 normal images (gray noise)
    for i in range(10):
        img = Image.fromarray(np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8))
        img.save(os.path.join(temp_dir, "OK", f"normal_{i}.png"))
        
    # 2. Create 2 Outliers (very different - white vs black)
    img_white = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    img_white.save(os.path.join(temp_dir, "OK", "outlier_white.png"))
    
    img_black = Image.fromarray(np.full((64, 64, 3), 0, dtype=np.uint8))
    img_black.save(os.path.join(temp_dir, "OK", "outlier_black.png"))
    
    # 3. Create 2 near-duplicates
    img_dup = Image.fromarray(np.random.randint(200, 210, (64, 64, 3), dtype=np.uint8))
    img_dup.save(os.path.join(temp_dir, "OK", "dup_1.png"))
    img_dup.save(os.path.join(temp_dir, "OK", "dup_2.png"))

def test_filtering():
    print("\n--- Starting Hybrid Filtering Logic Test ---\n")
    
    temp_test_dir = "temp_test_dataset"
    create_mock_dataset(temp_test_dir)
    
    try:
        results = detect_issues_in_split("train", temp_test_dir)
        
        issue_types = [r['issue_type'] for r in results]
        print(f"Detected Issues: {issue_types}")
        
        # Verify Outliers
        has_outliers = "outlier" in issue_types
        print(f"[RESULT] Outlier Detection: {'PASS' if has_outliers else 'FAIL'}")
        
        # Verify Duplicates
        has_duplicates = "duplicate" in issue_types
        print(f"[RESULT] Duplicate Detection: {'PASS' if has_duplicates else 'FAIL'}")
        
        # Verify Categorization structure
        all_have_keys = all(all(k in r for k in ['file_path', 'issue_type', 'suggested_label']) for r in results)
        print(f"[RESULT] Metadata Integrity: {'PASS' if all_have_keys else 'FAIL'}")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)

if __name__ == "__main__":
    test_filtering()

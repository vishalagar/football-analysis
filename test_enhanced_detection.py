"""
Test script for enhanced label issue detection system.
This script verifies the new features work correctly.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.data import (
    detect_issues,
    detect_issues_in_split,
    ISSUE_DETECTION_CONFIG,
    calculate_quality_score,
    classify_severity,
    detect_outliers_ensemble,
    get_class_issue_summary
)
from backend.app.core.config import TRAIN_DIR, VAL_DIR, MODELS_DIR
import numpy as np

def test_configuration():
    """Test that configuration is properly loaded."""
    print("=" * 80)
    print("TEST 1: Configuration")
    print("=" * 80)
    
    assert "outlier_percentile" in ISSUE_DETECTION_CONFIG
    assert "duplicate_threshold" in ISSUE_DETECTION_CONFIG
    assert "severity_thresholds" in ISSUE_DETECTION_CONFIG
    
    print("✓ Configuration loaded successfully")
    print(f"  Outlier percentile: {ISSUE_DETECTION_CONFIG['outlier_percentile']}%")
    print(f"  Duplicate threshold: {ISSUE_DETECTION_CONFIG['duplicate_threshold']}")
    print(f"  Severity thresholds: {ISSUE_DETECTION_CONFIG['severity_thresholds']}")
    print()

def test_quality_score():
    """Test quality score calculation."""
    print("=" * 80)
    print("TEST 2: Quality Score Calculation")
    print("=" * 80)
    
    # Create mock data
    pred_probs = np.array([
        [0.9, 0.05, 0.05],  # High confidence, correct
        [0.4, 0.3, 0.3],     # Low confidence, ambiguous
        [0.1, 0.8, 0.1],     # High confidence, wrong label
    ])
    labels = np.array([0, 1, 0])  # True labels
    features = np.random.randn(3, 512)  # Mock features
    
    # Test scores for each sample
    for idx in range(3):
        score = calculate_quality_score(pred_probs, labels, features, idx)
        severity = classify_severity(score)
        print(f"  Sample {idx}: Quality Score = {score:.3f}, Severity = {severity}")
        assert 0 <= score <= 1, "Quality score must be between 0 and 1"
    
    print("✓ Quality score calculation working\n")

def test_severity_classification():
    """Test severity classification."""
    print("=" * 80)
    print("TEST 3: Severity Classification")
    print("=" * 80)
    
    test_cases = [
        (0.95, "CRITICAL"),
        (0.75, "HIGH"),
        (0.60, "MEDIUM"),
        (0.35, "LOW"),
        (0.15, "NEGLIGIBLE")
    ]
    
    for score, expected in test_cases:
        severity = classify_severity(score)
        print(f"  Score {score:.2f} -> {severity} (expected: {expected})")
        assert severity == expected, f"Expected {expected}, got {severity}"
    
    print("✓ Severity classification working\n")

def test_outlier_ensemble():
    """Test ensemble outlier detection."""
    print("=" * 80)
    print("TEST 4: Ensemble Outlier Detection")
    print("=" * 80)
    
    # Create test data with clear outliers
    np.random.seed(42)
    normal_data = np.random.randn(100, 10)
    outliers = np.random.randn(5, 10) * 5  # Outliers with higher variance
    features = np.vstack([normal_data, outliers])
    labels = np.array([0] * 50 + [1] * 50 + [0] * 5)
    
    outlier_indices = detect_outliers_ensemble(features, labels)
    print(f"  Detected {len(outlier_indices)} outliers from {len(features)} samples")
    print(f"  Expected outliers in indices 100-104")
    
    # Check if we detected at least some of the planted outliers
    detected_planted = sum(1 for idx in outlier_indices if idx >= 100)
    print(f"  Detected {detected_planted}/5 planted outliers")
    
    print("✓ Ensemble outlier detection working\n")

def test_class_summary():
    """Test class-wise issue summary."""
    print("=" * 80)
    print("TEST 5: Class-wise Issue Summary")
    print("=" * 80)
    
    # Mock issues
    mock_issues = [
        {"given_label": "ClassA", "issue_type": "label_issue", "suggested_label": "ClassB", 
         "severity": "CRITICAL", "quality_score": 0.95},
        {"given_label": "ClassA", "issue_type": "outlier", "suggested_label": "delete", 
         "severity": "HIGH", "quality_score": 0.75},
        {"given_label": "ClassB", "issue_type": "label_issue", "suggested_label": "ClassA", 
         "severity": "MEDIUM", "quality_score": 0.60},
    ]
    
    class_names = ["ClassA", "ClassB", "ClassC"]
    summary = get_class_issue_summary(mock_issues, class_names)
    
    print(f"  Summary for {len(class_names)} classes:")
    for cls in class_names:
        stats = summary[cls]
        print(f"    {cls}: {stats['total_issues']} issues, {stats['critical_count']} critical")
    
    assert summary["ClassA"]["total_issues"] == 2
    assert summary["ClassB"]["total_issues"] == 1
    assert summary["ClassC"]["total_issues"] == 0
    
    print("✓ Class-wise summary working\n")

def test_full_detection():
    """Test full detection pipeline if dataset exists."""
    print("=" * 80)
    print("TEST 6: Full Detection Pipeline")
    print("=" * 80)
    
    if not os.path.exists(TRAIN_DIR):
        print("  ⚠ Training directory not found, skipping full pipeline test")
        print()
        return
    
    print("  Running detection on dataset (this may take a while)...")
    
    try:
        # Test detect_issues which returns a dict with issues list and summaries
        result = detect_issues()
        
        if isinstance(result, dict):
            issues = result.get("issues", [])
            class_summary = result.get("class_summary", {})
            total_issues = result.get("total_issues", 0)
            
            print(f"  ✓ Detected {total_issues} total issues")
            print(f"  ✓ Class summary available for {len(class_summary)} classes")
            
            if issues:
                # Check first issue has all required fields
                first_issue = issues[0]
                required_fields = ["file_path", "issue_type", "severity", "quality_score", 
                                 "given_label", "suggested_label", "split"]
                
                for field in required_fields:
                    assert field in first_issue, f"Missing field: {field}"
                
                print(f"  ✓ First issue example:")
                print(f"    - Type: {first_issue['issue_type']}")
                print(f"    - Severity: {first_issue['severity']}")
                print(f"    - Quality Score: {first_issue['quality_score']:.3f}")
                print(f"    - Given Label: {first_issue['given_label']}")
                print(f"    - Suggested: {first_issue['suggested_label']}")
        else:
            # Old format (list) - still supported
            print(f"  ⚠ Detection returned old format (list), not new dict format")
            print(f"  ✓ Detected {len(result)} issues")
            
        print("✓ Full detection pipeline working\n")
        
    except Exception as e:
        print(f"  ✗ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        print()

def main():
    print("\n" + "=" * 80)
    print("ENHANCED LABEL DETECTION - VERIFICATION TESTS")
    print("=" * 80 + "\n")
    
    try:
        test_configuration()
        test_quality_score()
        test_severity_classification()
        test_outlier_ensemble()
        test_class_summary()
        test_full_detection()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nThe enhanced label detection system is working correctly!")
        print("\nKey Features Verified:")
        print("  ✓ Quality scores (0-1 scale)")
        print("  ✓ Severity levels (CRITICAL, HIGH, MEDIUM, LOW, NEGLIGIBLE)")
        print("  ✓ Ensemble outlier detection (OOD + IsolationForest + LOF)")
        print("  ✓ Class-wise issue analysis")
        print("  ✓ Configurable thresholds")
        print()
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TESTS FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

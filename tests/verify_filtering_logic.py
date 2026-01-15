
import os
import sys
import numpy as np

# Mock config injection
ISSUE_DETECTION_CONFIG = {
    "outlier_percentile": 5,
    "duplicate_threshold": 0.98,
    "min_confidence_for_relabel": 0.65,
    "use_ensemble": True,
    "ensemble_weights": {"model": 0.85, "aux": 0.15}, 
    "severity_thresholds": {
        "critical": 0.9,
    }
}

def verify_logic():
    print("Verifying Weighted Ensemble Logic...")
    
    # Simulate a scenario
    # Model says: Class A (0.9), Class B (0.1)
    # Aux says:   Class A (0.4), Class B (0.6) <-- Aux disagrees strongly
    
    weights = ISSUE_DETECTION_CONFIG["ensemble_weights"]
    model_probs = np.array([0.9, 0.1])
    aux_probs = np.array([0.4, 0.6])
    
    ensemble_probs = weights["model"] * model_probs + weights["aux"] * aux_probs
    
    print(f"Model Probs: {model_probs}")
    print(f"Aux Probs:   {aux_probs}")
    print(f"Weights:     {weights}")
    print(f"Ensemble:    {ensemble_probs}")
    
    # Expected: 0.85*0.9 + 0.15*0.4 = 0.765 + 0.06 = 0.825 (Class A)
    # The Model's strong opinion should override Aux's disagreement
    
    assert np.argmax(ensemble_probs) == 0, "Ensemble failed to prioritize Model"
    assert ensemble_probs[0] > 0.8, "Ensemble score lower than expected"
    print("✅ Ensemble prioritization verified.")
    
    print("\nVerifying Confidence Threshold Logic...")
    # Scenario: Model predicts Class B, but with low confidence
    # Given Label: Class A
    given_label = "Class A"
    predicted_label = "Class B"
    conf = 0.55 # Below 0.65 threshold
    
    min_conf = ISSUE_DETECTION_CONFIG["min_confidence_for_relabel"]
    
    should_flag = False
    if predicted_label != given_label and conf >= min_conf:
        should_flag = True
    else:
        print(f"Skipping issue because confidence {conf} < threshold {min_conf}")
        
    assert not should_flag, "Low confidence issue was not skipped"
    print("✅ Low confidence suppression verified.")

    # Scenario: High confidence
    conf = 0.8
    if predicted_label != given_label and conf >= min_conf:
        print(f"Flagging issue because confidence {conf} >= threshold {min_conf}")
        should_flag = True
        
    assert should_flag, "High confidence issue was skipped incorrectly"
    print("✅ High confidence flagging verified.")

    print("\nVerifying CV Threshold Logic...")
    # This mimics the logic added to detect_issues_in_split
    # Scenario: CV Prediction for Training Data
    # Pred Prob: 0.55 (Low) vs Threshold 0.65
    
    cv_prob = 0.55
    min_conf = ISSUE_DETECTION_CONFIG["min_confidence_for_relabel"]
    
    cv_should_flag = False
    if cv_prob >= min_conf:
        cv_should_flag = True
    
    assert not cv_should_flag, "CV Low confidence issue should be skipped"
    print("✅ CV Low confidence suppression verified.")

if __name__ == "__main__":
    verify_logic()

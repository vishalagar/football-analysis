
import ollama
from .data_manager import detect_issues, get_dataset_stats

def query_llama3(prompt):
    try:
        response = ollama.chat(model='llama3', messages=[
          {
            'role': 'user',
            'content': prompt,
          },
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def analyze_situation_and_decide():
    """
    Analyzes the current state and returns a decision.
    """
    print("Agent: Analyzing dataset health...")
    
    # 1. Get Stats
    stats = get_dataset_stats()
    
    # 2. Check for Issues (Simplified check first to save time, or full check)
    # For this agent, we'll do a full check.
    issues = detect_issues()
    
    if isinstance(issues, dict) and "error" in issues:
        return {
            "decision": "ERROR",
            "reason": issues["error"],
            "action": "none"
        }

    num_issues = len(issues)
    total_train = stats.get('train', {}).get('count', 0)
    
    # Construct Prompt for Llama3
    prompt = f"""
    You are an expert ML Engineer Agent.
    System Status:
    - Dataset Split: {stats}
    - Total Training Images: {total_train}
    - Detected Potential Label Issues/Outliers: {num_issues}
    
    Goal: Build a high-performance model.
    
    Rules:
    1. If there are significant label issues (>5), priority is 'data_cleaning'.
    2. If issues are minimal (<=5), you can recommend 'start_training' but mention the issues in analysis.
    3. If everything looks perfect, 'start_training'.
    
    Respond in JSON format:
    {{
        "analysis": "Your thought process...",
        "recommended_action": "data_cleaning" | "hyperparameter_tuning" | "start_training"
    }}
    """
    
    print("Agent: Asking Llama3...")
    response_text = query_llama3(prompt)
    
    # Basic parsing if Llama returns Markdown
    import json
    import re
    
    try:
        # Extract JSON using regex
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            decision = json.loads(json_str)
            decision['raw_issues_count'] = num_issues
            decision['issues_list'] = issues # Pass full list to frontend
            return decision
        else:
             return {"decision": "ERROR", "reason": "Failed to parse Agent response", "raw_response": response_text}
    except Exception as e:
        return {"decision": "ERROR", "reason": f"Parsing error: {str(e)}", "raw_response": response_text}

# ============== Phase 4: Post-Training Diagnosis ==============

def diagnose_after_exploration(exploration_results):
    """
    Diagnoses root cause after exhausting automated exploration.
    
    Args:
        exploration_results: dict with status, best_result, all_results
    
    Returns:
        dict with diagnosis, reasoning, and recommended_action
    """
    import numpy as np
    
    best = exploration_results["best_result"]
    all_results = exploration_results["all_results"]
    
    # Compute per-class accuracy from confusion matrix
    cm = np.array(best["confusion_matrix"])
    per_class_acc = {}
    class_names = ["Class_" + str(i) for i in range(len(cm))]  # Will be replaced with actual names
    
    for i in range(len(cm)):
        total = cm[i].sum()
        if total > 0:
            per_class_acc[class_names[i]] = cm[i, i] / total
        else:
            per_class_acc[class_names[i]] = 0.0
    
    # Find problematic classes (< 60% accuracy)
    problematic_classes = [cls for cls, acc in per_class_acc.items() if acc < 0.6]
    
    # Check for overfitting
    train_val_gap = best["train_acc"] - best["val_acc"]
    is_overfitting = train_val_gap > 0.15
    
    # Build prompt for Llama3
    prompt = f"""
    You are an ML expert analyzing training results after exhaustive hyperparameter exploration.
    
    Exploration Summary:
    - Configurations tried: {len(all_results)}
    - Best validation accuracy: {best['val_acc']:.2%}
    - Best training accuracy: {best['train_acc']:.2%}
    - Train-Val gap: {train_val_gap:.2%}
    - Target accuracy: 90%
    
    Per-class Performance:
    {', '.join([f'{cls}: {acc:.1%}' for cls, acc in per_class_acc.items()])}
    
    Confusion Matrix (rows=true, cols=pred):
    {cm}
    
    Question: What is the root cause of underperformance?
    
    Analysis Guidelines:
    1. If specific classes have very low accuracy (< 60%) AND there's class confusion → Likely DATA QUALITY issue (mislabeling)
    2. If overfitting (train >> val by 15%+) → Need MORE REGULARIZATION (but we already tried weight_decay variations)
    3. If all classes perform poorly uniformly → May need DIFFERENT MODEL ARCHITECTURE
    4. If performance is close to target (85%+) → Just need MINOR TUNING
    
    Respond in JSON:
    {{
        "diagnosis": "data_quality" | "need_different_model" | "nearly_there" | "need_more_exploration",
        "problematic_classes": [...],
        "reasoning": "Detailed explanation...",
        "recommended_action": "recheck_labels" | "try_resnet50" | "fine_tune" | "continue_exploration"
    }}
    """
    
    print("🤖 Agent: Diagnosing root cause...")
    response = query_llama3(prompt)
    
    # Parse response
    import json, re
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            diagnosis = json.loads(match.group(0))
            diagnosis["best_result"] = best
            diagnosis["all_results_summary"] = [
                {"config": r["config_name"], "val_acc": r["val_acc"]} 
                for r in all_results
            ]
            return diagnosis
        else:
            return {
                "diagnosis": "error",
                "reasoning": "Failed to parse agent response",
                "raw_response": response
            }
    except Exception as e:
        return {
            "diagnosis": "error",
            "reasoning": f"Parsing error: {str(e)}",
            "raw_response": response
        }

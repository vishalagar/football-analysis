
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from backend.core.agent_brain import analyze_situation_and_decide
from backend.core.data_manager import apply_fix, get_dataset_stats, CustomImageDataset
from backend.core.config import TRAIN_DIR
from backend.core.trainer import run_automated_training
import threading
import os
import json
from backend.core.config import MODELS_DIR

router = APIRouter()

# Simple In-Memory State for Training
training_state = {
    "status": "idle",
    "progress": [],
    "result": None
}

# Phase 4: Auto-Training State
auto_training_state = {
    "status": "idle",
    "current_config": 0,
    "total_configs": 0,
    "current_trial": 0,
    "total_trials": 0,
    "best_acc": 0.0,
    "exploration_results": None,
    "diagnosis": None,
    "iteration": 0,
    "max_iterations": 3,
    "current_epoch": 0,
    "total_epochs": 0
}

def restore_state():
    global auto_training_state
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                # Load all results for the leaderboard
                if "all_results" in data:
                    auto_training_state["results"] = data["all_results"]
                
                # If we have a valid summary, mark as completed or partial
                if "status" in data:
                    if data["status"] == "success":
                        auto_training_state["status"] = "completed"
                    elif data["status"] == "partial":
                        auto_training_state["status"] = "exploring" # Or keep exploring if it was in progress
                    
                    auto_training_state["exploration_results"] = {
                        "status": data["status"],
                        "best_result": data.get("best_result"),
                        "all_results": data.get("all_results", [])
                    }
                    if data.get("best_result"):
                        auto_training_state["best_acc"] = data["best_result"].get("val_acc", 0.0)
        except Exception as e:
            print(f"Failed to restore state: {e}")

restore_state()

class FixRequest(BaseModel):
    file_path: str
    action: str # 'delete', 'move', 'ignore'
    new_label: Optional[str] = None

@router.get("/status")
def get_system_status():
    return {
        "dataset_stats": get_dataset_stats(),
        "training_state": training_state,
        "auto_training_state": auto_training_state
    }

@router.get("/analyze")
def analyze_dataset():
    """
    Triggers the Agent to analyze the dataset.
    """
    decision = analyze_situation_and_decide()
    return decision

@router.get("/get_classes")
def get_available_classes():
    """
    Returns all available classes for the dropdown.
    """
    if not os.path.exists(TRAIN_DIR):
        return {"classes": []}
    
    dataset = CustomImageDataset(TRAIN_DIR)
    return {"classes": dataset.classes}

@router.post("/fix_issue")
def fix_data_issue(req: FixRequest):
    success, msg = apply_fix(req.file_path, req.action, req.new_label)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

class BatchFixRequest(BaseModel):
    file_paths: list[str]
    action: str
    new_label: Optional[str] = None

class BatchItem(BaseModel):
    file_path: str
    new_label: str

class BatchSuggestionRequest(BaseModel):
    items: List[BatchItem]

@router.post("/batch_fix_suggestions")
def batch_fix_suggestions(req: BatchSuggestionRequest):
    """
    Apply varying fixes (move or delete) for multiple files.
    """
    results = []
    
    for item in req.items:
        # Action is 'delete' if suggestion is 'delete', else 'move'
        action = 'delete' if item.new_label.lower() == 'delete' else 'move'
        success, msg = apply_fix(item.file_path, action, item.new_label)
        results.append({"path": item.file_path, "success": success, "message": msg})
        
    success_count = sum(1 for r in results if r["success"])
    return {
        "status": "completed",
        "total": len(req.items),
        "success": success_count,
        "results": results
    }

@router.post("/batch_fix")
def batch_fix_issues(req: BatchFixRequest):
    """
    Apply the same fix to multiple files.
    """
    results = []
    for path in req.file_paths:
        success, msg = apply_fix(path, req.action, req.new_label)
        results.append({"path": path, "success": success, "message": msg})
    
    success_count = sum(1 for r in results if r["success"])
    return {
        "status": "completed",
        "total": len(req.file_paths),
        "success": success_count,
        "failed": len(req.file_paths) - success_count,
        "results": results
    }

def run_training_background():
    global training_state
    training_state["status"] = "running"
    training_state["progress"] = []
    
    try:
        result = run_automated_training()
        training_state["result"] = result
        training_state["status"] = "completed"
    except Exception as e:
        training_state["status"] = "failed"
        training_state["error"] = str(e)

@router.post("/start_training")
def start_training_endpoint():
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    t = threading.Thread(target=run_training_background)
    t.start()
    
    return {"status": "started", "message": "Training started in background"}

# ============== Phase 4: Auto-Training Endpoints ==============

def run_auto_exploration_background():
    """Background thread for auto-exploration."""
    global auto_training_state
    from backend.core.trainer import auto_explore
    from backend.core.agent_brain import diagnose_after_exploration
    
    try:
        auto_training_state["status"] = "exploring"
        auto_training_state["iteration"] += 1
        
        print(f"\n[AUTO] Starting auto-exploration (Iteration {auto_training_state['iteration']})...")
        
        def progress_cb(status_dict):
            global auto_training_state
            auto_training_state.update(status_dict)
            
        # Run exploration with callback
        results = auto_explore(target_accuracy=0.90, max_time_hours=2, progress_callback=progress_cb)
        auto_training_state["exploration_results"] = results
        
        if results["status"] == "failed":
            error_msg = results.get('error', 'Unknown trainer error')
            print(f"[ERROR] Trainer returned failure: {error_msg}")
            raise Exception(error_msg)

        if results["status"] == "success":
            # Success! Training achieved target
            auto_training_state["status"] = "completed"
            auto_training_state["best_acc"] = results["best_result"]["val_acc"]
        else:
            # Need diagnosis
            auto_training_state["status"] = "diagnosing"
            diagnosis = diagnose_after_exploration(results)
            auto_training_state["diagnosis"] = diagnosis
            
            # Check if we should ask user or continue
            if diagnosis["diagnosis"] in ["data_quality"]:
                # Ask user to clean data
                auto_training_state["status"] = "waiting_user"
            elif auto_training_state["iteration"] >= auto_training_state["max_iterations"]:
                # Max iterations reached
                auto_training_state["status"] = "completed"
                auto_training_state["best_acc"] = results["best_result"]["val_acc"]
            else:
                # Continue with another iteration (shouldn't happen often)
                auto_training_state["status"] = "completed"
                
    except Exception as e:
        auto_training_state["status"] = "failed"
        auto_training_state["error"] = str(e)
        print(f"[ERROR] Auto-exploration failed: {e}")
        import traceback
        traceback.print_exc()

@router.post("/start_auto_training")
def start_auto_training():
    """Starts the smart auto-exploration process."""
    global auto_training_state
    
    if auto_training_state["status"] in ["exploring", "diagnosing"]:
        raise HTTPException(status_code=400, detail="Auto-training already in progress")
    
    # Reset state
    auto_training_state = {
        "status": "exploring",
        "current_config": 0,
        "total_configs": 0,
        "current_trial": 0,
        "total_trials": 0,
        "results": [],
        "best_acc": 0.0,
        "current_epoch": 0,
        "total_epochs": 0,
        "exploration_results": None,
        "diagnosis": None,
        "iteration": 0,
        "max_iterations": 3
    }
    
    t = threading.Thread(target=run_auto_exploration_background)
    t.start()
    
    return {"status": "started", "message": "Auto-training started"}

@router.get("/auto_training_status")
def get_auto_training_status():
    """Returns current auto-training state for frontend polling."""
    # Include stats for dashboard sync
    state = auto_training_state.copy()
    state["dataset_stats"] = get_dataset_stats()
    return state

@router.post("/user_feedback")
def handle_user_feedback(action: str):
    """
    Handles user feedback after diagnosis.
    action: 'recleaned' | 'satisfied' | 'continue'
    """
    global auto_training_state
    
    if action == "recleaned":
        # User cleaned data, restart exploration
        auto_training_state["status"] = "exploring"
        t = threading.Thread(target=run_auto_exploration_background)
        t.start()
        return {"status": "restarted", "message": "Restarting exploration with cleaned data"}
    elif action == "satisfied":
        # User is satisfied, mark as complete
        auto_training_state["status"] = "completed"
        return {"status": "completed", "message": "Marked as complete"}
    else:
        return {"status": "unknown_action"}

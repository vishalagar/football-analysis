
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from .models import create_model
from .data_manager import CustomImageDataset, TRAIN_DIR, VAL_DIR, TEST_DIR, transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss = running_loss / total
    acc = correct / total
    return loss, acc

def train_model(params, dataset_train, dataset_val, num_epochs=10):
    num_classes = len(dataset_train.classes)
    model = create_model(num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    train_loader = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=params['batch_size'], shuffle=False)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    return model, best_acc, history

def tune_hyperparameters(n_trials=5):
    # Dataset should be loaded once usually, but here for simplicity
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("Datasets missing")
        return None

    dataset_train = CustomImageDataset(TRAIN_DIR)
    dataset_val = CustomImageDataset(VAL_DIR)
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        params = {"lr": lr, "batch_size": batch_size}
        # Short training for tuning
        _, best_acc, _ = train_model(params, dataset_train, dataset_val, num_epochs=5)
        return best_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def run_automated_training(full_epochs=20, dataset_train=None, dataset_val=None):
    print("Starting Automated Training...")
    
    # Load defaults if not provided
    if dataset_train is None:
        dataset_train = CustomImageDataset(TRAIN_DIR)
    if dataset_val is None:
        dataset_val = CustomImageDataset(VAL_DIR)
    
    print("Phase 1: Hyperparameter Tuning")
    # We need to pass datasets to tune_hyperparameters too, or refactor it.
    # For now, let's just make tune_hyperparameters use the passed datasets if possible
    # or just use a simpler tuning call here.
    
    # Refactoring tune_hyperparameters to accept datasets is better.
    # But for minimal changes:
    def tune_wrapper(train_ds, val_ds, n_trials=5):
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            params = {"lr": lr, "batch_size": batch_size}
            _, best_acc, _ = train_model(params, train_ds, val_ds, num_epochs=3) # Reduced to 3 for speed
            return best_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    best_params = tune_wrapper(dataset_train, dataset_val, n_trials=3) # Reduced to 3 trials
    print(f"Best Params: {best_params}")
    
    print("Phase 2: Full Training")
    model, best_val_acc, history = train_model(best_params, dataset_train, dataset_val, num_epochs=full_epochs)
    
    # Save Model
    os.makedirs("../../models", exist_ok=True)
    save_path = os.path.abspath("../../models/best_model.pth")
    torch.save(model.state_dict(), save_path)
    
    # Test if exists
    test_acc = None
    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        # Check for actual images in subfolder... simplified check:
        dataset_test = CustomImageDataset(TEST_DIR)
        length = len(dataset_test)
        if length > 0:
             print("Test dataset found. Evaluating...")
             test_loader = DataLoader(dataset_test, batch_size=best_params['batch_size'], shuffle=False)
             criterion = nn.CrossEntropyLoss()
             _, test_acc = validate(model, test_loader, criterion)
             print(f"Test Accuracy: {test_acc:.4f}")
        else:
             print("Test dataset folder exists but is empty.")
    else:
        print("Test dataset not found. Skipping.")
        
    return {
        "best_params": best_params,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "history": history,
        "model_path": save_path
    }

# ============== Phase 4: Auto-Exploration Engine ==============

def compute_confusion_matrix(model, loader, num_classes):
    """Compute confusion matrix for model evaluation."""
    import numpy as np
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = sk_confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm

def auto_explore(target_accuracy=0.90, max_time_hours=2):
    """
    Automatically explores multiple configurations until success or exhaustion.
    
    Args:
        target_accuracy: Target validation accuracy to achieve
        max_time_hours: Maximum time budget in hours
    
    Returns:
        dict with status, best_result, and all_results
    """
    print("\n🚀 Starting Auto-Exploration...")
    
    # Load datasets
    dataset_train = CustomImageDataset(TRAIN_DIR)
    dataset_val = CustomImageDataset(VAL_DIR)
    dataset_size = len(dataset_train)
    
    # Adaptive budget based on dataset size
    if dataset_size < 1000:
        max_trials_per_config = 3
        max_configs = 3
        time_budget = 3600  # 1 hour
        print(f"  Dataset size: {dataset_size} (SMALL) - Budget: 3 configs x 3 trials")
    elif dataset_size < 10000:
        max_trials_per_config = 5
        max_configs = 5
        time_budget = 7200  # 2 hours
        print(f"  Dataset size: {dataset_size} (MEDIUM) - Budget: 5 configs x 5 trials")
    else:
        max_trials_per_config = 8
        max_configs = 6
        time_budget = min(max_time_hours * 3600, 21600)  # Max 6 hours
        print(f"  Dataset size: {dataset_size} (LARGE) - Budget: 6 configs x 8 trials")
    
    # Configuration queue to explore
    exploration_configs = [
        {
            "name": "ResNet18 + Standard HP",
            "lr_range": [1e-5, 1e-2],
            "batch_size_options": [16, 32, 64],
            "weight_decay": [1e-5, 1e-3]
        },
        {
            "name": "ResNet18 + High Regularization",
            "lr_range": [1e-6, 1e-3],
            "batch_size_options": [16, 32],
            "weight_decay": [1e-4, 1e-2]
        },
        {
            "name": "ResNet18 + Low LR",
            "lr_range": [1e-6, 1e-4],
            "batch_size_options": [32, 64],
            "weight_decay": [1e-5, 1e-4]
        },
        {
            "name": "ResNet18 + High LR",
            "lr_range": [1e-4, 1e-2],
            "batch_size_options": [16, 32],
            "weight_decay": [1e-5, 1e-3]
        },
        {
            "name": "ResNet18 + Conservative",
            "lr_range": [5e-5, 5e-4],
            "batch_size_options": [32],
            "weight_decay": [1e-4, 1e-3]
        },
    ]
    
    start_time = time.time()
    all_results = []
    best_overall = None
    
    for config_idx, config in enumerate(exploration_configs[:max_configs]):
        if time.time() - start_time > time_budget:
            print(f"\n⏰ Time budget exhausted ({time_budget / 3600:.1f}h)")
            break
        
        print(f"\n📊 Config {config_idx + 1}/{min(len(exploration_configs), max_configs)}: {config['name']}")
        
        # Run Optuna tuning for this config
        def objective(trial):
            lr = trial.suggest_float("lr", *config["lr_range"], log=True)
            batch_size = trial.suggest_categorical("batch_size", config["batch_size_options"])
            weight_decay = trial.suggest_float("weight_decay", *config["weight_decay"], log=True)
            
            params = {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay}
            
            # Short training for exploration (3 epochs)
            model, best_acc, history = train_model(params, dataset_train, dataset_val, num_epochs=3)
            return best_acc
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=max_trials_per_config, show_progress_bar=False)
        
        best_params = study.best_params
        print(f"  ✅ Best params: {best_params}, Val Acc: {study.best_value:.4f}")
        
        # Train full model with best params (10 epochs)
        print(f"  🏋️ Training full model...")
        model, val_acc, history = train_model(best_params, dataset_train, dataset_val, num_epochs=10)
        
        # Compute confusion matrix
        val_loader = DataLoader(dataset_val, batch_size=best_params['batch_size'], shuffle=False)
        cm = compute_confusion_matrix(model, val_loader, len(dataset_train.classes))
        
        # Get final metrics
        train_loader = DataLoader(dataset_train, batch_size=best_params['batch_size'], shuffle=False)
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = validate(model, train_loader, criterion)
        val_loss, val_acc_final = validate(model, val_loader, criterion)
        
        result = {
            "config_name": config["name"],
            "config_idx": config_idx,
            "best_params": best_params,
            "val_acc": val_acc_final,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "confusion_matrix": cm.tolist(),
            "history": history
        }
        
        all_results.append(result)
        
        # Track best
        if best_overall is None or val_acc_final > best_overall["val_acc"]:
            best_overall = result
            # Save best model
            os.makedirs("../../models", exist_ok=True)
            save_path = os.path.abspath("../../models/best_model.pth")
            torch.save(model.state_dict(), save_path)
            best_overall["model_path"] = save_path
        
        print(f"  🎯 Final: Train={train_acc:.4f}, Val={val_acc_final:.4f}")
        
        # Success criteria
        if val_acc_final >= target_accuracy:
            print(f"\n🎉 SUCCESS! Achieved {val_acc_final:.4f} >= {target_accuracy:.4f}")
            return {
                "status": "success",
                "best_result": best_overall,
                "all_results": all_results,
                "total_trials": config_idx + 1
            }
        
        # Early stopping: No improvement in last 3 configs
        if len(all_results) >= 3:
            recent = [r["val_acc"] for r in all_results[-3:]]
            if max(recent) - min(recent) < 0.02:  # Less than 2% improvement
                print(f"\n📉 No significant improvement in last 3 configs. Stopping.")
                break
    
    # Exhausted all options
    print(f"\n⚠️ Exploration exhausted. Best: {best_overall['val_acc']:.4f} (Target: {target_accuracy:.4f})")
    return {
        "status": "need_user_review",
        "best_result": best_overall,
        "all_results": all_results,
        "total_trials": len(all_results)
    }

def train_model_with_weight_decay(params, dataset_train, dataset_val, num_epochs=10):
    """Modified train_model to support weight_decay."""
    num_classes = len(dataset_train.classes)
    model = create_model(num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params.get('weight_decay', 0.0)
    )
    
    train_loader = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=params['batch_size'], shuffle=False)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    return model, best_acc, history

# Override train_model to use weight_decay version
train_model = train_model_with_weight_decay


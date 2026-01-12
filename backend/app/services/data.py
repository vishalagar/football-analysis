
import os
import shutil
import glob
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
try:
    from cleanlab.classification import CleanLearning
    from cleanlab.outlier import OutOfDistribution
    from cleanlab.filter import find_label_issues
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from backend.app.ml.networks import create_model

from backend.app.core.config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, LOGS_DIR

if HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

import numpy as np
from PIL import Image

# Image Transformations
if HAS_TORCH:
    # Standard validation/test transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Augmented training transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Added for industrial parts
        transforms.RandomRotation(30),   # Increased from 15 for better orientation robustness
        transforms.RandomGrayscale(p=0.1), # Handle lighting/color variations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    val_transform = None
    train_transform = None

# Backward compatibility or default
transform = val_transform

class CustomImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        if not os.path.exists(root_dir):
            self.classes = []
            self.class_to_idx = {}
            return
            
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_path in glob.glob(os.path.join(cls_dir, "*.*")): # Catch all extensions roughly
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.files.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if HAS_TORCH:
                if self.transform:
                    image = self.transform(image)
                elif transform: # Global fallback
                    image = transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if HAS_TORCH:
                image = torch.zeros((3, 224, 224))
            else:
                image = None
        
        label = self.labels[idx]
        return image, label, img_path

def get_dataset_stats():
    stats = {}
    for split, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not os.path.exists(path):
             stats[split] = {"count": 0, "classes": {}}
             continue
        
        classes = {}
        total = 0
        for cls_name in os.listdir(path):
            cls_path = os.path.join(path, cls_name)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                classes[cls_name] = count
                total += count
        stats[split] = {"count": total, "classes": classes}
    return stats

def extract_features(dataset: CustomImageDataset):
    """Extracts features using a pre-trained ResNet18."""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity() # Remove classification layer
    model.eval()
    model = model.to(DEVICE)  # Move model to same device as images
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(DEVICE) if HAS_TORCH else images
            output = model(images)
            features.append(output.cpu().numpy())
            
    return np.vstack(features)

def detect_issues_in_split(split_name, split_dir):
    """Helper to detect issues in a specific split."""
    if not HAS_CLEANLAB or not HAS_TORCH:
        print("Cleanlab or Torch not available for issue detection.")
        return []
        
    if not os.path.exists(split_dir):
        return []

    try:
        dataset = CustomImageDataset(split_dir, transform=val_transform)
        if len(dataset) == 0:
            return []
    except Exception as e:
        print(f"Error loading {split_name} dataset for issues: {e}")
        return []

    print(f"Extracting features for {split_name}...")
    features = extract_features(dataset)
    labels = np.array(dataset.labels)

    # Train a quick LogReg on features to get probs for Cleanlab
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    # If dataset is too small, cross_val_predict might fail.
    # We fallback to simple predict if len < 6 (just a heuristic)
    if len(dataset) < 10:
        print(f"Dataset {split_name} too small for cross-val issues detection.")
        return []

    clf = LogisticRegression(max_iter=100)
    try:
        pred_probs = cross_val_predict(clf, features, labels, cv=min(3, len(dataset)//2), method="predict_proba")
    except Exception as e:
        print(f"Error in cross_val_predict for {split_name}: {e}")
        return []
    
    print(f"Finding label issues in {split_name}...")
    issues_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )
    
    # 2. Outlier Detection (Unsupervised)
    print(f"Finding outliers in {split_name}...")
    ood = OutOfDistribution()
    ood_scores = ood.fit_score(features=features)
    # Heuristic: top 2% or score < 0.05
    outlier_indices = np.where(ood_scores < 0.05)[0]
    
    # 3. Near-Duplicate Detection
    print(f"Finding duplicates in {split_name}...")
    sim_matrix = cosine_similarity(features)
    # Mask diagonal
    np.fill_diagonal(sim_matrix, 0)
    duplicate_indices = []
    # Heuristic: similarity > 0.98
    for i in range(len(sim_matrix)):
        if np.max(sim_matrix[i]) > 0.98:
            duplicate_indices.append(i)
    
    results = []
    
    # Process Label Issues
    for idx in issues_indices:
        img_path = dataset.files[idx]
        given_label_idx = dataset.labels[idx]
        given_label = dataset.classes[given_label_idx]
        predicted_label_idx = np.argmax(pred_probs[idx])
        predicted_label = dataset.classes[predicted_label_idx]

        results.append({
            "file_path": img_path,
            "issue_type": "label_issue",
            "given_label": given_label,
            "suggested_label": predicted_label,
            "confidence": float(np.max(pred_probs[idx])),
            "split": split_name
        })

    # Process Outliers
    for idx in outlier_indices:
        # Avoid duplicate entries if already a label issue
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
            
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "outlier",
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete", # Common suggestion for outliers
            "confidence": float(ood_scores[idx]),
            "split": split_name
        })
        
    # Process Duplicates
    for idx in duplicate_indices:
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
            
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "duplicate",
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete",
            "confidence": 0.99, # Placeholder for duplicate confidence
            "split": split_name
        })

    return results

def detect_issues():
    """Detects label issues and outliers using cleanlab in both Train and Valid."""
    all_issues = []
    
    # Check Train
    all_issues.extend(detect_issues_in_split("train", TRAIN_DIR))
    
    # Check Valid
    all_issues.extend(detect_issues_in_split("val", VAL_DIR))
        
    return all_issues

def detect_issues_with_model(model_path):
    """
    Uses the trained model to find label issues with high precision (Hybrid Approach).
    """
    if not HAS_CLEANLAB or not HAS_TORCH:
        return {"error": "Dependencies missing"}
    
    if not os.path.exists(model_path):
        return {"error": "Model file not found"}

    print(f"Loading model from {model_path} for Hybrid Analysis...")
    
    # Load Data
    dataset = CustomImageDataset(TRAIN_DIR, transform=val_transform) # Use val_transform for deterministic eval
    if len(dataset) == 0:
        return {"error": "Training dataset empty"}
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load Model
    num_classes = len(dataset.classes)
    # We need to know the architecture. For now assuming ResNet18 as it's the default.
    # In a perfect world, we'd save metadata with the model.
    try:
        model = create_model(num_classes, "resnet18") 
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    # get probabilities
    all_probs = []
    all_labels = []
    
    import torch.nn.functional as F
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    # Cleanlab
    print("Running Cleanlab with model probabilities...")
    issues_indices = find_label_issues(
        labels=all_labels,
        pred_probs=all_probs,
        return_indices_ranked_by="self_confidence"
    )
    
    results = []
    for idx in issues_indices:
        img_path = dataset.files[idx]
        given_label_idx = dataset.labels[idx]
        given_label = dataset.classes[given_label_idx]
        predicted_label_idx = np.argmax(all_probs[idx])
        predicted_label = dataset.classes[predicted_label_idx]
        conf = float(np.max(all_probs[idx]))
        
        results.append({
            "file_path": img_path,
            "issue_type": "hybrid_label_issue", # Distinct type
            "given_label": given_label,
            "suggested_label": predicted_label,
            "confidence": conf,
            "split": "train"
        })
        
    return results

def apply_fix(file_path, action, new_label=None):
    """
    Actions:
    - 'delete': Delete the file.
    - 'move': Move to new label folder (keeping same split).
    - 'ignore': Do nothing (mark as resolved in UI state).
    """
    if not os.path.exists(file_path):
        return False, "File not found"
        
    try:
        if action == 'delete':
            os.remove(file_path)
            return True, "Deleted"
        elif action == 'move' and new_label:
            file_name = os.path.basename(file_path)
            # Find the split directory (parent of parent of file)
            # file is at .../split/class/image.jpg
            # dir(file) = .../split/class
            # dir(dir(file)) = .../split
            
            current_class_dir = os.path.dirname(file_path)
            split_dir = os.path.dirname(current_class_dir)
            
            new_dir = os.path.join(split_dir, new_label)
            os.makedirs(new_dir, exist_ok=True)
            
            new_path = os.path.join(new_dir, file_name)
            shutil.move(file_path, new_path)
            return True, f"Moved to {new_label}"
    except Exception as e:
        return False, str(e)
    
    return True, "Ignored"

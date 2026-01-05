
import os
import shutil
import glob
from typing import Dict, List, Tuple
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from cleanlab.classification import CleanLearning
from cleanlab.outlier import OutOfDistribution
from cleanlab.filter import find_label_issues
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Dataset Path
# Dataset Path
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mlcc")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
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
            image = transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle appropriately
            image = torch.zeros((3, 224, 224))
        
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
                count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                classes[cls_name] = count
                total += count
        stats[split] = {"count": total, "classes": classes}
    return stats

def extract_features(dataset: CustomImageDataset):
    """Extracts features using a pre-trained ResNet18."""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity() # Remove classification layer
    model.eval()
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    
    with torch.no_grad():
        for images, _, _ in loader:
            output = model(images)
            features.append(output.numpy())
            
    return np.vstack(features)

def detect_issues_in_split(split_name, split_dir):
    """Helper to detect issues in a specific split."""
    if not os.path.exists(split_dir):
        return []

    dataset = CustomImageDataset(split_dir)
    if len(dataset) == 0:
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
    
    results = []
    for idx in issues_indices:
        img_path = dataset.files[idx]
        given_label_idx = dataset.labels[idx]
        given_label = dataset.classes[given_label_idx]
        
        # Predicted label
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
    return results

def detect_issues():
    """Detects label issues and outliers using cleanlab in both Train and Valid."""
    all_issues = []
    
    # Check Train
    all_issues.extend(detect_issues_in_split("train", TRAIN_DIR))
    
    # Check Valid
    all_issues.extend(detect_issues_in_split("valid", VAL_DIR))
        
    return all_issues

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

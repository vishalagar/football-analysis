# Data Issue Detection & Filtering Architecture

This document provides a comprehensive deep-dive into the "Data Filter" logic implemented in `backend/app/services/data.py`. It explains the hybrid architecture, the mathematical algorithms used, and the decision-making strategies for cleaning machine learning datasets.

---

## 1. High-Level Architecture

The system is designed to identify three types of data issues:
1.  **Label Issues**: Images that are mislabeled (e.g., a "Good" chip labeled as "Defect").
2.  **Outliers**: Images that do not look like anything else in the dataset (e.g., corrupted files, OOD data).
3.  **Duplicates**: Identical or near-identical images that bias validation results.

### System Diagram

```mermaid
flowchart TD
    %% Global Styles
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:black;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black,shape:diamond;
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black,shape:cylinder;
    classDef result fill:#ffebee,stroke:#c62828,stroke-width:2px,color:black;

    Start([Start: detect_issues]) --> InitCheck{Check Dependencies}
    InitCheck -- "Torch & Cleanlab OK" --> ModelCheck{Model Exists?}
    
    subgraph FeaturePipeline ["Feature Extraction Core (Shared)"]
        direction TB
        FP_Input[Input Images] --> FP_Resize[Resize: 224x224]
        FP_Resize --> FP_ResNet[ResNet18 Backbone]
        FP_ResNet -- "Strip FC Layer" --> FP_Vector[512-dim Feature Vector]
        FP_Vector --> FP_Norm[L2 Normalization]
    end

    ModelCheck -- "No Model Found" --> StrategyCV
    ModelCheck -- "Model Found" --> StrategyHybrid

    %% STRATEGY 1: Training Data (Cross-Validation)
    subgraph StrategyCV ["Strategy A: Cross-Validation (Training Data)"]
        direction TB
        SCV_Start[Input: Training Split] --> SCV_KFold[Split: K-Fold Cross-Val]
        SCV_KFold --> SCV_Train[Train: Logistic Regression on Fold K-1]
        SCV_Train --> SCV_Predict[Predict: Probabilities on Fold K]
        SCV_Predict -- "Aggregate" --> SCV_Probs[Out-of-Sample Probabilities]
        
        %% Also computes feature steps
        SCV_Start -.-> FeaturePipeline
        FeaturePipeline -.-> SCV_Feats[Training Features]
    end

    %% STRATEGY 2: Validation/Test Data (Hybrid)
    subgraph StrategyHybrid ["Strategy B: Hybrid Ensemble (Val/Test Data)"]
        direction TB
        SH_Start[Input: Val/Test Split] --> SH_ModelPred[Model Inference]
        SH_Start -.-> FeaturePipeline
        FeaturePipeline -.-> SH_Feats[Val/Test Features]
        
        SH_Feats --> SH_AuxTrain[Train Aux LogReg]
        SH_ModelPred & SH_AuxTrain --> SH_Ensemble{Weighted Ensemble}
        SH_Ensemble -- "0.85 * Model + 0.15 * Aux" --> SH_Probs[Final Probabilities]
    end

    %% ALGORITHM CORE
    subgraph DetectionCore ["The Detection Engine"]
        direction TB
        
        %% Inputs
        DC_Probs(Probabilities)
        DC_Feats(Features 512-d)
        DC_Labels(Original Labels)
        
        StrategyCV --> DC_Probs & DC_Feats & DC_Labels
        StrategyHybrid --> DC_Probs & DC_Feats & DC_Labels

        %% 1. Label Issues (Cleanlab)
        subgraph Algo_CL ["1. Label Issue Detection"]
            CL_Input[Input: Probs & Labels] --> CL_Conf[Compute Self-Confidence]
            CL_Conf --> CL_Matrix[Compute Joint Noise Matrix]
            CL_Matrix --> CL_Result[List of Label Errors]
        end

        %% 2. Outlier Detection (Ensemble)
        subgraph Algo_OD ["2. Outlier Ensemble"]
            OD_Input[Input: Features] 
            OD_Method1[Method 1: Cleanlab OOD]
            OD_Method2[Method 2: Isolation Forest]
            OD_Method3[Method 3: Local Outlier Factor]
            OD_Logic{Vote >= 2?}
            
            OD_Method1 & OD_Method2 & OD_Method3 --> OD_Logic
            OD_Logic -- Yes --> OD_Result[Flag as Outlier]
        end

        %% 3. Duplicate Detection
        subgraph Algo_DD ["3. Duplicate Detection"]
            DD_Input[Input: Features] --> DD_Cos[Cosine Similarity Matrix]
            DD_Cos --> DD_Thresh{Sim > 0.98?}
            DD_Thresh -- Yes --> DD_Result[Flag as Duplicate]
        end

        DC_Probs --> Algo_CL
        DC_Feats --> Algo_OD & Algo_DD
        DC_Labels --> Algo_CL
    end

    %% SCORING SYSTEM
    subgraph ScoringSystem ["Quality Scoring System"]
        direction TB
        SS_Input(Detected Issue) --> SS_Metrics
        
        subgraph SS_Metrics ["Metric Calculation"]
            M_Conf["1-Max(Prob)"]
            M_Ent["Entropy(Prob)"]
            M_Iso["Distance(Centroid)"]
        end
        
        M_Conf -- "0.4" --> SS_Calc
        M_Ent -- "0.2" --> SS_Calc
        M_Iso -- "0.1" --> SS_Calc
        
        SS_Calc[Weighted Sum 0.0-1.0] --> SS_Classify{Grade Severity}
    end

    CL_Result & OD_Result & DD_Result --> SS_Input
    SS_Classify --> FinalOutput[Final JSON Report]
```

---

## 2. Detection Strategies

The system uses two different strategies depending on *which* dataset is being analyzed. This is critical to avoid bias.

### Strategy A: Cross-Validation (For Training Data)
**Problem:** If you ask a model to find errors in the data it was trained on, it will fail. The model has "memorized" the training images, even the mislabeled ones. It will likely say "Confidence 99% this is a Cat" even if it's a dog, because you told it so during training.

**Solution (CV):**
1.  We split the training data into $K$ parts (folds).
2.  We train a temporary, lightweight model (Logistic Regression) on Parts 1 & 2.
3.  We use that model to predict probabilities for Part 3.
4.  **Result:** The predictions for Part 3 are "Out-of-Sample." The model never saw these images during training, so its predictions are unbiased. If the model says "This looks like a Dog" but the label is "Cat," we likely have a real label error.

### Strategy B: Hybrid Ensemble (For Validation/Test Data)
**Problem:** Validation data is unseen. We want to use our *best* and most powerful model (ResNet18) to judge it, not a weak temporary one.

**Solution (Hybrid):**
1.  **Expert Model:** We load the fully trained `best_model.pth`.
2.  **Auxiliary Model:** We also train a lightweight Logistic Regression on the validation features.
3.  **Ensemble:** We combine predictions: `Final_Prob = 0.85 * Best_Model + 0.15 * Aux_Model`.
4.  **Why?** Deep Learning models can sometimes be "overconfident" on wrong answers. The simpler auxiliary model acts as a "sanity check" (regularizer) to smooth out the probability distributions, making the error detection more reliable.

---

## 3. Algorithm Deep-Dive

### A. Feature Extraction (ResNet18)
Before any analysis, images are converted into math.
*   **Backbone:** ResNet18 (Pre-trained on ImageNet).
*   **Process:** We cut off the final "Head" (the part that says "Cat" or "Dog").
*   **Output:** We take the output of the penultimate layer.
*   **Result:** Every image becomes a vector of **512 numbers**. This vector represents the "DNA" of the image—its shapes, textures, and patterns.

### B. Cleanlab (Confident Learning)
Used for: **Label Issues**
*   **Core Idea:** Instead of just looking for wrong predictions, Cleanlab estimates the "Joint Distribution of Label Noise."
*   **Mechanism:**
    1.  It calculates a matrix $C$ where $C_{ij}$ is the probability that an image labeled as Class $i$ actually belongs to Class $j$.
    2.  It uses "pruning" to remove images that fall into the "off-diagonal" of this matrix with high confidence.
    3.  This method is robust against **Class Imbalance**. If you have 1000 cats and 10 dogs, a simple threshold would ignore the dogs. Cleanlab normalizes by class frequency.

### C. Isolation Forest
Used for: **Outlier Detection**
*   **Intuition:** "Anomalies are few and different."
*   **Mechanism:**
    1.  It randomly cuts the data points with lines (decision trees).
    2.  If a point is "normal" (inside a cluster of other points), you need many cuts to isolate it.
    3.  If a point is an "outlier" (floating alone in space), it gets isolated very quickly (few cuts).
    4.  **metric:** Short path length in the tree = High Anomaly Score.

### D. Local Outlier Factor (LOF)
Used for: **Outlier Detection**
*   **Intuition:** "Density-based Isoloation."
*   **Mechanism:**
    1.  It compares the local density of point A (how close its neighbors are) to the local density of its neighbors.
    2.  If point A is in a sparse region, but its neighbors are in dense regions, point A is an outlier.
    3.  This is better than Isolation Forest for finding outliers that are near a cluster but not *part* of it.

### E. Cosine Similarity
Used for: **Duplicates** & **Quality Scoring**
*   **Formula:** $\text{Similarity} = \frac{A \cdot B}{||A|| \times ||B||}$
*   **Application:**
    *   **Duplicates:** If Similarity > 0.98, the images are effectively identical.
    *   **Centroids:** We calculate the "Center" (Average Vector) of the "Good" class. If a "Good" image has a low cosine similarity to this center, it is a low-quality or atypical example.

---

## 4. Quality Scoring System

Every detected issue is assigned a **Quality Score (0.0 - 1.0)** to help you prioritize what to fix. Higher score = Worse quality (more urgent).

The score is a weighted sum of four metrics:

1.  **Confidence Score (Weight: 0.4)**
    *   How confident is the model that the label is wrong?
    *   `1.0 - Max_Prob`
2.  **Mismatch Score (Weight: 0.3)**
    *   Does the prediction match the label?
    *   `1.0` if mismatch, `0.0` if match.
3.  **Entropy Score (Weight: 0.2)**
    *   Is the model confused? (e.g., 50% Cat, 50% Dog).
    *   High Entropy = High Uncertainty.
4.  **Isolation Score (Weight: 0.1)**
    *   How far is this image from the center of its class in feature space?
    *   Measured using Euclidean distance to Class Centroid.

**Severity Grading:**
*   **CRITICAL (> 0.9)**: Model is extremely confident this is wrong. Fix immediately.
*   **HIGH (> 0.7)**: Strong evidence of error.
*   **MEDIUM (> 0.5)**: Likely an error, or a very hard/ambiguous example.
*   **LOW (< 0.5)**: Potential duplicate or edge case.

---

## 5. Code Reference
All logic is contained in `backend/app/services/data.py`.

*   `detect_issues()`: Main entry point. Decides Strategy A or B.
*   `detect_issues_in_split()`: Implements **Strategy A (CV)**.
*   `detect_issues_with_model()`: Implements **Strategy B (Hybrid)**.
*   `extract_features()`: Runs ResNet18.
*   `detect_outliers_ensemble()`: Runs IsolationForest + LOF + Cleanlab OOD.
*   `calculate_quality_score()`: Implements the scoring math.

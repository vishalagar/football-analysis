# Understanding Data Filtering: A Simple Guide

If you've ever felt confused about how the system identifies "bad" data, you're not alone. This guide breaks down the complex logic into simple steps and analogies.

---

## 1. The Big Picture
Think of the Data Filter as a **Quality Control Inspector**. Its job is to look at every image in your dataset and flag three things:
1.  **Label Issues**: "You said this is a Defect, but it looks like a Good chip."
2.  **Outliers**: "This image looks like noise/garbage, it doesn't belong here."
3.  **Duplicates**: "You've uploaded this exact same image twice; that's cheating!"

---

## 2. When There is NO "Best Model" (Initial Analysis)
When you first upload a dataset, the system hasn't trained a model on your specific parts yet. 

**The Problem:** We can't use a model that doesn't exist.
**The Solution: Cross-Validation (The "Substitute Teacher" Approach)**

Imagine a classroom where the main teacher (the Best Model) is absent. We use "Substitute Teachers" to grade the students:
1.  We split the data into 3 groups (Folds).
2.  We take a generic "Brain" (a pre-trained ResNet18) and a simple math model.
3.  **First Round**: We show Group A and B to the simple model. Then we ask it to guess the labels for Group C.
4.  **Second Round**: We show B and C, then ask it to guess Group A.
5.  **Result**: Because the model *never saw the images it was guessing*, its opinion is unbiased. If the model says "This looks like a Defect" but you labeled it "OK", we flag a **Label Issue**.

---

## 3. When There IS a "Best Model" (After Training)
Once you've trained a model, we now have an **Expert**. We want to use this expert to find errors in new data (Validation/Test sets).

**The Strategy: The "Expert + Colleague" Ensemble**
1.  **The Expert**: We use your `best_model.pth`. It knows exactly what your chips look like.
2.  **The Colleague**: We also train a simple "Sanity Check" model on-the-fly.
3.  **The Vote**: We combine their opinions (85% Weight for the Expert, 15% for the Colleague).
4.  **Why?** Sometimes even experts get overconfident. The colleague acts as a "sanity check" to make sure the expert isn't just hallucinating.

---

## 4. How "Finding Label Issue" Actually Works (The Cleanlab Deep-Dive)
We use a technology called **Cleanlab (Confident Learning)**. It's much smarter than a simple "if prediction != label" check. 

### The 3-Step Cleanlab Process:
1.  **Calculate Self-Confidence**: For every image, it looks at the probability the model gave to the *current label*. (e.g., "The user says this is OK, and the model is 60% sure it's OK").
2.  **Learn the "Noise Pattern"**: It calculates an average confidence for every class. If "Defect" chips usually have a 90% confidence, but "OK" chips usually only have 70% confidence, it adjusts its expectations.
3.  **Cross-Reference**: It identifies an image as a "Label Issue" only if:
    *   The model is more confident in a *different* label (e.g., "Defect") than it is in its own average confidence for that class.

**Analogy**: Imagine a teacher who knows they are usually bad at grading handwriting. If they see a word that looks like "Dog" but the student says "Cat", and the teacher knows they usually struggle with "C" and "D", they won't flag it as an error immediately. They only flag it if they are *exceptionally* sure this specific time.

---

## 5. Does K-Fold work for the "Current Best Model"?
This is a point of confusion! Let's clarify: **K-Fold and the Best Model are two different ways to get the same thing: Probabilities.**

*   **Cleanlab** is the "Brain" that finds errors.
*   **Cleanlab Needs Numbers (Probabilities)** to work.

### The Problem of Circular Reasoning:
If you use the **Best Model** to judge the **Training Data**, it's like a student grading their own test after seeing the answer key. They will just say "Everything is 100% correct!" even if the answer key itself had a typo. This is called **Overfitting**.

### The Logic Flow in our Code:
1.  **For Training Data**: We **ignore** the Best Model and use **K-Fold**.
    *   We split training data into 3 parts.
    *   Model A trains on 1&2, predicts 3.
    *   Model B trains on 2&3, predicts 1.
    *   **Result**: We get probabilities for everything without the model ever "cheating".
2.  **For Val/Test Data**: We **use** the **Best Model**.
    *   Since the model has *never seen* these images before, it can't "cheat". It's an honest test.

**So, to answer your question**: We don't use K-Fold *on* the Best Model. We use K-Fold **instead of** the Best Model when we are looking at the Training Split.

---

## 5. Summary Table (Who analyzes what?)

| Scenario | Dataset Split | What is used? | Why? |
| :--- | :--- | :--- | :--- |
| **No Best Model** | Train, Val, Test | Cross-Validation (CV) | No expert exists yet; everyone gets the "Substitute Teacher" treatment. |
| **With Best Model** | **Training** | **Cross-Validation (CV)** | **CRITICAL**: We *never* use the Best Model to judge its own training data. It has "memorized" these images and would be biased. CV gives an honest, unbiased second opinion. |
| **With Best Model** | **Val & Test** | **Expert Model + Aux** | These images are new to the model, so we use its expert knowledge to find errors. |

## 6. Technical Deep Dive: The Math of Cleanlab
For those who want to see under the hood, here is how the "Confident Learning" algorithm actually works mathematically.

### Step 1: The Probability Matrix ($P$)
We start with a matrix $P$, where each row is an image and each column is a class. 
$P_{i,j}$ is the probability the model gives to Image $i$ belonging to Class $j$.

### Step 2: Class Thresholds ($T$)
Cleanlab calculates a threshold for every class. This is the key to its "intelligence".
For each class $c$, the threshold $t_c$ is the **average probability** the model gave to that class for all images that are *actually labeled* as class $c$.

$$t_c = \frac{1}{|X_c|} \sum_{i \in X_c} P_{i,c}$$

*Why this matters:* If a class is "hard" (e.g., subtle defects), the model might only be 60% confident on average. Cleanlab learns this and won't flag a 60% prediction as an error for that class.

### Step 3: The "Confident Joint" Matrix ($C$)
This is where we count the "confusion". We build a matrix $C$ (size: Classes x Classes).
We look at हर Image $i$ that is labeled as Class $A$, but the model is more confident in Class $B$ than Class $B$'s own threshold ($t_B$).

If $P_{i,B} \ge t_B$ and $B$ is the model's top guess, we add 1 to the cell $C_{A,B}$.

### Step 4: Normalization and Pruning
Finally, Cleanlab converts this count matrix $C$ into a "Joint Distribution of Noise". It uses this to calculate exactly how many images it should remove from each class to clean the dataset without losing too much important data.

**In summary**: Cleanlab doesn't just look for "wrong" guesses; it looks for guesses that are **statistically louder** than the usual noise for that specific category.

---

## 7. What do the scores mean?
Every issue gets a **Quality Score (0 to 1)**:
*   **CRITICAL (> 0.9)**: The system is ALMOST CERTAIN this is a mistake.
*   **HIGH (> 0.7)**: Very likely a mistake.
*   **LOW (< 0.5)**: Might be a mistake, or just a very confusing image.

**Pro-Tip**: Start by fixing the **CRITICAL** issues first. These are the ones where you and the AI disagree the most!

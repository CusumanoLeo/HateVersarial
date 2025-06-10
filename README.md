# HateVersarial

**Robust Hate Speech Detection via Adversarial Training**  
This project aims to improve the robustness of hate speech classifiers by augmenting data with adversarial perturbations. We explore character-level (e.g. homoglyphs, leetspeak) and embedding-based attacks  to simulate real-world evasion strategies, and train a model capable of detecting toxic content even under disguise.

---

## Project Structure

- `Data/`: Contains the raw HateXplain dataset used for training and evaluation.
- `augmented_data/`: Stores adversarially augmented examples for training robustness.
- `notebooks/`: Jupyter notebooks for data exploration, model evaluation, and visualization.

---

## 📁 Core Code Files

- `main_perturb.py`: Training of the Hardkuma rationale-extraction model training, as well as training of the 'black-box' model.
- `functions.py`: Utility functions for loading data, computing metrics, and processing inputs.
- `preprocessing.py`: Utilitiy functions for tokenization, filtering, and data preparation.
- `perturbations.py`: Utility functions used for perturbation logic including homoglyphs and leetspeak substitution.
- `rationale.py`: Utility functions used for rationale masking using HardKuma-based methods.
- `predictions.py`: Utility functions used for generating predictions on clean and adversarial inputs.
- `utils.py`: Helper functions
- `final.py`: Runs inference of both models on clean and perturbed data, and compares results to baseline

---

## 📦 Dependencies

This project requires the following libraries:

- **Core libraries**:  
  `numpy`, `pandas`, `matplotlib`, `requests`, `os`, `math`, `random`, `json`, `collections`, `typing`

- **Deep Learning & Optimization**:  
  `torch`, `torch.nn`, `torch.optim`, `torch.nn.functional`, `torch.utils.data`,  
  `torch.optim.lr_scheduler` (e.g. `LambdaLR`, `ReduceLROnPlateau`)

- **NLP & Transformers**:  
  `transformers` — includes `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`,  
  `get_linear_schedule_with_warmup`, `EarlyStoppingCallback`, `pipeline`, `PreTrainedTokenizerBase`

- **Adversarial Text Augmentation**:  
  `pyleetspeak` — used for generating leetspeak-based adversarial examples

- **Evaluation & ML Utilities**:  
  `scikit-learn` — includes `f1_score`, `accuracy_score`, `precision_recall_curve`, `train_test_split`

---

### 🔧 Setup

To install the dependencies, run:

```bash
pip install -r requirements.txt
```
---


## 🧪 Approach

1. **Dataset**: We use the [HateXplain](https://github.com/hate-alert/HateXplain) dataset, which includes multi-annotator rationales and labels for hate, offensive, and normal classes.

2. **Model**: We fine-tuned a BERT-based classifier using clean and adversarially perturbed inputs. Our perturbations target:
   - **Character-level**: homoglyphs, leetspeak, repeated characters
   - **Embedding-level**: word substitutions with similar contextual embeddings
   - **Rationale masking**: drop or mask model-identified rationales during training for robustness

3. **Evaluation**: Performance was measured on both clean and adversarially modified test sets using accuracy, F1-score, and robustness metrics.

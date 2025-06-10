# HateVersarial

**Robust Hate Speech Detection via Adversarial Training**  
This project investigates model robustness against disguised hate speech through adversarial training. We simulate real-world evasion strategies—like homoglyph substitution, leetspeak, and semantically similar word replacements—and train a miniBERT model on adversarially augmented data. Our goal is to equip hate speech classifiers with stronger generalization capabilities while maintaining high accuracy on clean data.

---

## 📁 Project Structure
```
HateVersarial/
├── Data/                               
├── Models/                             
│ ├── bert-hateXplain-aug-finetuned/
│ ├── hardkuma_ckpt/ 
│ └── White_box_ADV_model/ 
├── augmented_data/ 
├── notebooks/ 
│ ├── preprocessing.ipynb         # Data cleaning & label processing
│ ├── training_blackbox.ipynb     # Black-box training pipeline
│ ├── get_attended_words.ipynb    # White-box attention analysis
│ └── eval.ipynb                  # Model evaluation
├── main_perturb.py               # Hardkuma and white box training pipeline
├── preprocessing.py             
├── perturbations.py 
├── rationale.py
├── utils.py 
└── final.py                      # Inference and Comparative evaluation
```
---

## 🛠️ Core Code Files

- `main_perturb.py`: Trains the HardKuma rationale-extraction model and the black-box adversarially fine-tuned model.
- `functions.py`: Utility functions for dataset handling, metric computation, and training.
- `preprocessing.py`: Text cleaning, tokenization, filtering, and label binarization.
- `perturbations.py`: Applies leetspeak and homoglyph substitutions.
- `rationale.py`: Implements HardKuma-based rationale extraction (token selection using L0 relaxation).
- `predictions.py`: Inference and evaluation functions on clean and perturbed test sets.
- `utils.py`: Miscellaneous helpers (e.g. attention aggregation).
- `final.py`: Runs and compares inference across the baseline, black-box, and white-box models.

---

## 📦 Dependencies

This project requires the following Python libraries:

- **Core**: `numpy`, `pandas`, `matplotlib`, `requests`
- **Deep Learning**: `torch`, `scikit-learn`
- **Transformers & NLP**: `transformers` (incl. `AutoTokenizer`, `Trainer`, `EarlyStoppingCallback`, etc.)
- **Adversarial Text Augmentation**: `pyleetspeak` (for leetspeak generation)

## 🧪 Approach

### Dataset
We use the HateXplain dataset, which provides multi-annotator labels (hate, offensive, normal) and token-level rationales. Labels are binarized (toxic / non-toxic) using majority voting, with full disagreement cases excluded.

### Data Augmentation
- **GPT-4 generation**: Toxic and non-toxic examples generated via prompting and filtered manually.
- **T5 paraphrasing**: Style-preserving rewording to create hard positive/negative cases.

### Black-box perturbation
Homoglyph and leetspeak perturbations are applied to rationale tokens (from HateXplain or predicted by a HardKuma rationale extractor).

### White-box perturbation
High-attention tokens (identified via self-attention maps) are replaced with their nearest embedding neighbors to simulate targeted evasion.

### Training and Evaluation
A BERT-based classifier is fine-tuned on clean + adversarially perturbed data. Performance is evaluated on both clean and perturbed test splits using macro-averaged F1, recall, and accuracy.


## 🧠 Models
The `Models/` directory contains all saved model checkpoints:  

| Model                        | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `bert-hateXplain-aug-finetuned/` | BERT fine-tuned on clean + perturbed data. Robust against homoglyphs/leetspeak. |
| `hardkuma_ckpt/`             | HardKuma rationale extractor. Outputs token-level masks for perturbations.  |
| `White_box_ADV_model/`       | White-box adversarially trained BERT (self-attention + embedding similarity). |



## 🏁 Conclusion  
This project implements and compares two adversarial training approaches to improve hate speech detection robustness:  

**Key Contributions**  
- **Black-box augmentation**: Leetspeak/homoglyph perturbations on HardKuma-extracted rationales  
- **White-box method**: Attention-guided embedding substitutions  
- **Synthetic data pipeline**: GPT-4 generation + T5 paraphrasing for hard cases  

**Outcomes**  
✔ Black-box model achieves **94% F1** on perturbed text (+19pp over baseline)  
✔ Maintains **84% F1** on clean data, demonstrating generalization  
✔ Modular pipeline enables extension to new perturbation types  

**Limitations & Future Work**  
- Current evaluation limited to HateXplain-derived perturbations  
- White-box method showed weaker robustness gains  
- Adaptive adversaries may require continuous retraining  

Suggested extensions:  
- Contrastive learning for perturbation-invariant representations  
- Ensemble methods combining both approaches  
- Real-world deployment testing against evolving evasion tactics

## 👥 Authors
Group 26 – EE-559: Deep Learning, EPFL

Melina Cherchali, Léo Cusumano, Romane Vorwald


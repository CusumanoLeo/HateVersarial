####################################################################################################
######################################## PROJECT (GROUP 26) ########################################
#################################### MAIN FOR METHOD'S PART 1-3 ####################################
##################### Authors: Melina Cherchali, Romane Vorwald, Léo Cusumano ######################
####################################################################################################


################################ IMPORT IMPORTANT LIBRARIES & FILES ################################

import numpy as np
import pandas as pd
import json
import random
from collections import Counter

from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, pipeline

from utils import *


############################################ CONSTANTS ############################################

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

CKPT_DIR = "Models/hardkuma_ckpt"

MERGE_MODE = "union"

# For rational extractor model training
EPOCHS = 3
BATCH_SZ = 16
MAX_LEN = 128


########################################## LOAD DATASET ###########################################

with open('Data/dataset.json', 'r') as f:
    dataset = json.load(f)

with open('Data/post_id_divisions.json', 'r') as f:
    split_ids = json.load(f)

rationales = [
    (dataset[k]['rationales'])
    for k in dataset.keys()
]

posts = [
    (dataset[k]['post_tokens'])
    for k in dataset.keys()
]

print("[INFO] HateXplain dataset loaded")

# load generated data

toxic_path = "augmented_data/combined-toxic.csv"
generated_toxic_posts = pd.read_csv(toxic_path)["text"].tolist()

non_toxic_path = "augmented_data/combined-non-toxic.csv" 
generated_non_toxic_posts = pd.read_csv(non_toxic_path)["text"].tolist()

print("[INFO] Generated dataset loaded")


########################################## PREPROCESSING ##########################################

print("[INFO] Resolving disagreement in label between annotators...")
disagreement_stats, top_disagreements = compute_annotator_disagreement(dataset, verbose=True, return_top_n=5, plot=False)
resolved_examples_custom = resolve_disagreements_custom(dataset, disagreement_stats)

print("[INFO] Merging rationale labels from all annotators for toxic posts & preprocessing...")
records = {pid: preprocess_record(post) for pid, post in dataset.items()}
data_binary = [
    (text, "toxic") if label == "hatespeech" else (text, "non-toxic")
    for text, label in resolved_examples_custom
    if label in {"hatespeech", "normal"}
]

print(f"length of data_binary: {len(data_binary)}")


############################### RATIONALE EXTRACTOR MODEL TRAINING ################################

print("[INFO] ---- RATIONAL EXTRACTOR MODEL TRAINING ----")

# Split data

print("[INFO] Splitting data into train, validation and test sets...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
records = {pid: out for pid, post in dataset.items()
           if (out := preprocess_record(post)) is not None} # we skip posts that were removed due to 'offensive'

train_pairs, train_rats = make_subset(split_ids["train"], records)
valid_pairs, valid_rats = make_subset(split_ids["val"], records)
test_pairs, test_rats = make_subset(split_ids["test"], records)

print("Train label distribution:", Counter([label for _, label in train_pairs]))
print("Valid label distribution:", Counter([label for _, label in valid_pairs]))
print("Test  label distribution:", Counter([label for _, label in test_pairs]))

train_ds = HateXplainDataset(train_pairs, train_rats, tokenizer, MAX_LEN)
val_ds   = HateXplainDataset(valid_pairs,  valid_rats,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SZ,
                          shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SZ,
                          shuffle=False, collate_fn=collate)

model = HardKumaRationaleModel(max_len=MAX_LEN).to(DEVICE)

pos_weight = torch.tensor([len([l for _,l in train_pairs if l=="non-toxic"]) /
                      len([l for _,l in train_pairs if l=="toxic"])],
                     device=DEVICE) # to balance classes -> class weight
clf_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
lambdas = {"len": 3.0, # stronger sparsity
           "cont": 2.0,
           "sup": 1.0}


# linear warm-up to keep BERT stable
warmup_steps = int(0.1 * len(train_loader) * EPOCHS)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0  # after warm-up, use base LR until ROP kicks in

warmup_scheduler = LambdaLR(optim, lr_lambda)

# plateau scheduler
rop_scheduler = ReduceLROnPlateau(
        optim,
        mode="max",          # we monitor F1 (higher is better)
        factor=0.5,          # halve LR
        patience=1,          # wait 1 epoch w/out improvement
        threshold=1e-3,      # minimum change considered "improvement"
        verbose=True
)

train_losses, val_losses = [], []

best_f1 = 0.0
patience_cnt = 0
best_model_path = "best_hardkuma.bin"

# TRAINING

for epoch in range(EPOCHS):
    print(f"---------- EPOCH NUMBER {epoch} ----------")
    tr_loss, tr_f1 = run_epoch(train_loader, model, clf_loss, lambdas, optim, warmup_scheduler, train=True)
    print("[INFO] train epoch completed")
    vl_loss, vl_f1 = run_epoch(val_loader, model, clf_loss, lambdas, optim, warmup_scheduler, train=False)
    print("[INFO] validation epoch completed")

    rop_scheduler.step(vl_f1) # adjust LR if plateau detected

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)

    print(f"Epoch {epoch}: train loss {tr_loss:.3f} F1 {tr_f1:.2%} | "
          f"val loss {vl_loss:.3f} F1 {vl_f1:.2%}")
    
    if vl_f1 > best_f1 + 1e-3: # significant improvement
        best_f1 = vl_f1
        torch.save(model.state_dict(), best_model_path)
        patience_cnt = 0
    else:
        patience_cnt += 1

    if patience_cnt == 3:
        print(f"[EARLY-STOP] no val F1 gain for 3 epochs "
              f"(best = {best_f1:.2%}).")
        break

print("[INFO] Rational extractor model trained.")

model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"\nBest checkpoint (F1 {best_f1:.2%} loaded.")

# TESTING

test_ds = HateXplainDataset(test_pairs, test_rats,
                            tokenizer=tokenizer, max_len=MAX_LEN)

test_loader = DataLoader(test_ds, batch_size=BATCH_SZ,
                         shuffle=False, collate_fn=collate)

test_loss, test_f1 = evaluate(test_loader, model, clf_loss, lambdas)
print(f"[TEST] loss {test_loss:.3f} | F1 {test_f1:.2%}")


############################ RATIONALE EXTRACTION FROM GENERATED DATA #############################

rationale_outputs = find_rationales(generated_toxic_posts, model=model, tokenizer=tokenizer, max_len=128 , batch_size=16)
print("[INFO] Rationales found for generated data")

################ LEETSPEAK TRANSFORMATIONS & HOMOGLYPH SUBSTITUTIONS ON RATIONALES #################

# Turn generated posts into the same "pair + mask" format

# toxic
gen_tox_pairs = [(text, "toxic") for text in generated_toxic_posts]
gen_tox_rats  = [out["rationale_mask"] for out in rationale_outputs]

# non-toxic  (mask = None -> will become zeros)
gen_nontox_pairs = [(t, "non-toxic") for t in generated_non_toxic_posts]
gen_nontox_rats  = [None] * len(gen_nontox_pairs)

# Split generated data 8:1:1 and merge with original splits
(t_tr, r_tr, t_va, r_va, t_te, r_te) = split(gen_tox_pairs, gen_tox_rats, 1.0)
(nt_tr, nr_tr, nt_va, nr_va, nt_te, nr_te) = split(gen_nontox_pairs, gen_nontox_rats, 1.0)

train_pairs += t_tr + nt_tr;    train_rats += r_tr + nr_tr
valid_pairs += t_va + nt_va;    valid_rats += r_va + nr_va
test_pairs  += t_te + nt_te;    test_rats  += r_te + nr_te

random.seed(13)
homoglyph_map = build_homoglyph_map()
leeter = SimpleLeeter()

print("[INFO] Dataset augmentation with perturbations...")
# 20% augmentation of each split
train_aug, train_lab = partial_augment(train_pairs, train_rats, tokenizer, homoglyph_map, leeter, frac=0.2)
valid_aug,   valid_lab   = partial_augment(valid_pairs, valid_rats, tokenizer, homoglyph_map, leeter, frac=0.2)

# No augmentation of test 
test_aug, test_lab = test_pairs,  test_rats

aug_train_pairs = list(zip(train_aug, train_lab))
aug_valid_pairs = list(zip(valid_aug, valid_lab))
aug_test_pairs  = list(zip(test_aug,  test_lab))

# rat_masks stay identical to originals for HardKuma supervision
aug_train_rats, aug_valid_rats, aug_test_rats = train_rats, valid_rats, test_rats


################### FINE-TUNING OF PRETRAINED BERT MODEL WITH AUGMENTED DATASET ####################

print("[INFO] ---- FINE-TUNING OF PRETRAINED BERT MODEL WITH AUGMENTED DATASET ----")

# tokenize the augmented dataset and turn them into torch dataset
tokenizer = AutoTokenizer.from_pretrained("tum-nlp/bert-hateXplain")

def encode_texts(texts, labels):
    enc = tokenizer(texts,
                    padding='longest', # dynamic pad per batch
                    truncation=True,
                    max_length=128)
    enc["labels"] = labels
    return enc

# Unpack text and label from augmented pairs
train_aug, train_lab = zip(*aug_train_pairs)
valid_aug, valid_lab = zip(*aug_valid_pairs)
test_aug,  test_lab  = zip(*test_pairs)

# Recreate binary label lists for each split
train_labels = [1 if l == "toxic" else 0 for l in train_lab]
val_labels   = [1 if l == "toxic" else 0 for l in valid_lab]
test_labels  = [1 if l == "toxic" else 0 for l in test_lab]

train_texts = train_aug
val_texts   = valid_aug
test_texts  = test_aug

print("[INFO] Augmented dataset splitted into train, validation and test sets")
print("Test label distribution:", Counter(test_labels))
print("Valid label distribution:", Counter(val_labels))
print("Train label distribution", Counter(train_labels))

train_tokenized = encode_texts(train_texts, train_labels)
val_tokenized   = encode_texts(val_texts,   val_labels)
test_tokenized  = encode_texts(test_texts,  test_labels)

train_ds = TorchDataset(train_tokenized)
val_ds   = TorchDataset(val_tokenized)
test_ds  = TorchDataset(test_tokenized)

# Trainer setup
model = AutoModelForSequenceClassification.from_pretrained(
            "tum-nlp/bert-hateXplain",
            num_labels=2,
            use_safetensors=True # <- tells HF to load the .safetensors
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    label_smoothing_factor=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    compute_metrics = compute_metrics,
    callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("[INFO] Training...")
trainer.train()
print("[INFO] BERT model fine-tuned.")
print("Best checkpoint:", trainer.state.best_model_checkpoint)

# evaluate
test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
print("[TEST] Results of testing:")
print(test_results)

# Save model & tokenizer
save_dir = "Models/bert-hateXplain-aug-finetuned_"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)


############## COMPARISON OF PERFORMANCE BETWEEN FINE-TUNED MODEL AND ORIGINAL MODEL ###############

clf_aug   = pipeline("text-classification", model=save_dir,  tokenizer=tokenizer,
                     return_all_scores=False)
clf_base  = pipeline("text-classification",
                     model="tum-nlp/bert-hateXplain",
                     tokenizer=tokenizer, return_all_scores=False)

pred_aug  = predict(clf_aug, test_texts)
pred_base = predict(clf_base, test_texts)

print("----- FINAL RESULTS AND COMPARISON WITH ORIGINAL MODEL'S PERFORMANCE -----")

print("Augmented fine-tuned:\n",
      classification_report(test_labels, pred_aug, target_names=["non-toxic", "toxic"]))

print("\nOriginal checkpoint:\n",
      classification_report(test_labels, pred_base, target_names=["non-toxic", "toxic"]))
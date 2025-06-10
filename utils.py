#################################### IMPORT IMPORTANT LIBRARIES ####################################

import numpy as np
import matplotlib.pyplot as plt
import requests
import random
from collections import Counter, defaultdict
from typing import Optional, Tuple

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModel


############################################ CONSTANTS ############################################

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

CKPT_DIR = "hardkuma_ckpt"

MERGE_MODE = "union"

# For rational extractor model training
EPOCHS = 3
BATCH_SZ = 16
MAX_LEN = 128


########################################## PREPROCESSING ##########################################

def compute_annotator_disagreement(dataset, verbose=False, return_top_n=0, plot=False):
    """
    Compute the number of unique labels assigned by annotators for each post,
    and optionally plot disagreement distribution.
    
    Args:
        dataset (dict): Loaded JSON dataset
        verbose (bool): Print summary stats
        return_top_n (int): If >0, return top N most disagreed examples
        plot (bool): If True, plot a histogram of disagreement levels

    Returns:
        disagreement_stats (list of tuples): (post_id, disagreement_count, label_counter)
        top_disagreements (optional): top N posts with highest disagreement
    """
    disagreement_stats = []

    for post_id, content in dataset.items():
        labels = [ann['label'] for ann in content.get('annotators', [])]
        label_counter = Counter(labels)
        disagreement_count = len(label_counter)
        disagreement_stats.append((post_id, disagreement_count, label_counter))

    if verbose:
        total = len(disagreement_stats)
        unanimous = sum(1 for _, c, _ in disagreement_stats if c == 1)
        mild_disagreement = sum(1 for _, c, _ in disagreement_stats if c == 2)
        full_disagreement = sum(1 for _, c, _ in disagreement_stats if c >= 3)

        print(f"Total examples: {total}")
        print(f"Unanimous (all annotators agree): {unanimous} ({unanimous/total:.2%})")
        print(f"Two-label disagreement: {mild_disagreement} ({mild_disagreement/total:.2%})")
        print(f"Three-label disagreement: {full_disagreement} ({full_disagreement/total:.2%})")

    if plot:
        disagreement_counts = [c for _, c, _ in disagreement_stats]
        count_dist = Counter(disagreement_counts)
        plt.bar(count_dist.keys(), count_dist.values(), color='gray')
        plt.xlabel("Number of unique labels (Disagreement level)")
        plt.ylabel("Number of posts")
        plt.title("Annotator Disagreement Distribution")
        plt.xticks([1, 2, 3])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    if return_top_n > 0:
        top_disagreements = sorted(disagreement_stats, key=lambda x: -x[1])[:return_top_n]
        return disagreement_stats, top_disagreements

    return disagreement_stats


def resolve_disagreements_custom(dataset, disagreement_stats):
    resolved = []
    counter_unanimous = 0
    counter_hatespeech = 0
    counter_off_normal = 0
    counter_skipped = 0

    for post_id, disagreement, label_counts in disagreement_stats:
        if disagreement == 3:
            counter_skipped += 1
            continue  # skip level 3 disagreements

        text = " ".join(dataset[post_id]['post_tokens'])
        labels = list(label_counts.elements())
        label_set = set(label_counts.keys())

        # Case 1: unanimous
        if disagreement == 1:
            counter_unanimous += 1
            resolved_label = labels[0]

        # Case 2: offensive vs hatespeech -> resolve as hatespeech
        elif disagreement == 2 and label_set == {"offensive", "hatespeech"}:
            counter_hatespeech += 1
            resolved_label = "hatespeech"

        # Case 3: normal vs offensive or normal vs hatespeech -> majority
        else:
            counter_off_normal += 1
            resolved_label = Counter(labels).most_common(1)[0][0]

        resolved.append((text, resolved_label))

    print(f"Unanimous: {counter_unanimous}, Offensive vs Hatespeech: {counter_hatespeech}, Normal vs Other: {counter_off_normal}, Skipped (3-label): {counter_skipped}")
    return resolved


def merge_rationales(masks, n_tokens, mode="union"):
    """
    masks    : List[List[int]] (length 0–3, possibly wrong sizes)
    n_tokens : int  = len(post['post_tokens'])
    Returns   : merged mask (list[int]) or None if no valid annotator
    """
    # Keep only masks whose length matches the post
    arrs = [np.array(m) for m in masks if len(m) == n_tokens]

    if len(arrs) == 0:
        return None  # no reliable annotator mask

    if mode == "union":
        merged = np.bitwise_or.reduce(arrs)
    elif mode == "majority":
        thr = (len(arrs) + 1) // 2 # majority among remaining annotators
        merged = (np.sum(arrs, axis=0) >= thr).astype(int)
    else: # intersection
        merged = np.bitwise_and.reduce(arrs)
    return merged.tolist()


def preprocess_record(post):
    text = " ".join(post["post_tokens"])
    labels = [ann["label"] for ann in post["annotators"]]
    majority_label = Counter(labels).most_common(1)[0][0]

    # Skip "offensive" posts entirely
    if majority_label == "offensive":
        return None

    n_tokens = len(post["post_tokens"])
    merged_rat = None

    if majority_label == "hatespeech" and post.get("rationales"):
        merged_rat = merge_rationales(post["rationales"], n_tokens, mode=MERGE_MODE)

    bin_label = "toxic" if majority_label == "hatespeech" else "non-toxic"
    return text, bin_label, merged_rat


############################### RATIONALE EXTRACTOR MODEL TRAINING #################################

class HardKumaSampler(nn.Module):
    """Samples a binary (0/1) mask via the HardKuma reparameterisation.

    Given shape (alpha) and rate (beta) > 0, the sampler draws a sample z in
    the open interval (0, 1) during the forward pass, then hard-rounds it to
    {0,1} while using the soft value for gradient flow (straight-through).
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor, hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_hard, z_soft).

        z_soft is the continuous sample; z_hard is detached to be exactly 0/1
        but uses z_soft in the backward pass (straight‑through).
        """
        # Sample u ~ Uniform(0, 1)
        u = torch.rand_like(alpha)
        v = (1 - u.pow(1.0 / beta)).clamp(self.eps, 1 - self.eps)
        z_soft = (1 - v.pow(1.0 / alpha)).clamp(self.eps, 1 - self.eps)

        if hard:
            z_hard = (z_soft > 0.5).float()
            # replace hard with soft in backward
            z_hard = z_hard.detach() - z_soft.detach() + z_soft
        else:
            z_hard = z_soft
        return z_hard, z_soft


# Rationale Generator + Predictor model

class HardKumaRationaleModel(nn.Module):
    """Generator-Predictor architecture for rationale extraction.

    * Generator: produces binary mask z over tokens via HardKuma.
    * Predictor: applies BERT to the masked input and predicts toxicity.
    """

    def __init__(self, bert_name: str = "bert-base-uncased", max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size

        # Generator head -> 2 positives (alpha, beta = HardKuma distribution parameters) per token
        self.gen_head = nn.Linear(hidden_size, 2)
        self.hardkuma = HardKumaSampler()

        # Classifier head on [CLS]
        self.cls_head = nn.Linear(hidden_size, 1)

    # Mask application helper
    
    @staticmethod
    def apply_mask(embeddings: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Element-wise multiply embeddings by z (shape: batch x seq_len x 1)."""
        return embeddings * z.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rationale_labels: Optional[torch.Tensor] = None,
        lambda_len: float = 1.0,
        lambda_cont: float = 1.0,
        lambda_sup: float = 1.0,
    ) -> dict:
        # BERT contextual embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, L, H)

        # Generator: predict alpha, beta > 0 via softplus
        alpha_beta = F.softplus(self.gen_head(hidden)) + 1e-4
        alpha, beta = alpha_beta[..., 0], alpha_beta[..., 1]

        z, z_soft = self.hardkuma(alpha, beta)
        masked_hidden = self.apply_mask(hidden, z)

        # Predictor on masked sequence (take [CLS])
        cls_repr = masked_hidden[:, 0, :]
        logits = self.cls_head(cls_repr).squeeze(-1)

        # Loss components
        out = {"logits": logits, "z_hard": z, "z_soft": z_soft}

        if rationale_labels is not None:
            # Binary cross‑entropy for token supervision
            sup_loss = F.binary_cross_entropy(z_soft, rationale_labels.float(), reduction="none")
            sup_loss = (sup_loss * attention_mask).sum() / attention_mask.sum()
            out["sup_loss"] = sup_loss * lambda_sup
        else:
            out["sup_loss"] = 0.0 * logits.sum()

        # Length regulariser: encourage sparse mask
        avg_len = z_soft.mean()
        len_loss = avg_len * lambda_len
        out["len_loss"] = len_loss

        # Continuity regulariser: encourage contiguous spans
        diff = torch.abs(z_soft[:, 1:] - z_soft[:, :-1]) * attention_mask[:, 1:]
        cont_loss = diff.mean() * lambda_cont
        out["cont_loss"] = cont_loss

        return out


# Training step utility
def training_step(batch, model, criterion, lambdas):
    # unpack & move to device
    input_ids, attn_mask, labels, rat_labels = (t.to(DEVICE) for t in batch)

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        rationale_labels=rat_labels,
        lambda_len=lambdas["len"],
        lambda_cont=lambdas["cont"],
        lambda_sup=lambdas["sup"],
    )

    sup_loss_raw = out["sup_loss"]
    has_mask     = (rat_labels.sum(dim=1) > 0).float()
    sup_loss     = (sup_loss_raw * has_mask).sum() / has_mask.sum().clamp(min=1)

    clf_loss   = criterion(out["logits"], labels)
    total_loss = clf_loss + sup_loss + out["len_loss"] + out["cont_loss"]
    out.update({"clf_loss": clf_loss, "total_loss": total_loss})

    # Cast labels to int ONCE for metric calculation later
    labels_long = labels.long()
    return out, labels_long


class HateXplainDataset(Dataset):
    """
    Each item from `data_binary` is a (text, label_str) tuple where
        label_str ∈ {"toxic", "non-toxic"}.
    Optionally supply `rat_masks` - a list (same length) where each element is
    either:
        * list[int] of 0/1 at **word** level  (toxic post with annotation)
        * None                                (toxic post w/out annot OR non-toxic)
    The class:
        • tokenises with HF tokenizer
        • aligns word-level rationale to WordPiece level
        • outputs (input_ids, attention_mask, label_float, rat_mask, sup_weight)
    """

    LABEL_MAP = {"non-toxic": 0.0, "toxic": 1.0}

    def __init__(self, pairs, ratm, tokenizer, max_len):
        self.pairs, self.ratm = pairs, ratm
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        text, lbl = self.pairs[idx]
        label = torch.tensor(self.LABEL_MAP[lbl])
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=self.max_len, return_offsets_mapping=True,
                       return_tensors="pt")
        ids  = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        offs = enc["offset_mapping"].squeeze(0)

        # build rationale mask
        if self.ratm[idx] is not None: # toxic with annotation
            word_mask = torch.tensor(self.ratm[idx], dtype=torch.float)
            wp2word = (offs[:,0]==0).cumsum(0)-1
            wp2word[wp2word < 0] = -1
            wp2word[wp2word >= len(word_mask)] = -1
            sel   = wp2word >= 0
            rmask = torch.zeros_like(ids, dtype=torch.float)
            rmask[sel] = word_mask[wp2word[sel]]
        else:
            rmask = torch.zeros_like(ids, dtype=torch.float)

        return ids, attn, label, rmask


def make_subset(id_list, records):
    texts, labels, ratm = [], [], []
    for pid in id_list:
        if pid not in records:
            continue  # skip posts that were removed ("offensive")
        text, lab, rat = records[pid]
        texts.append(text)
        labels.append(lab)
        ratm.append(rat)
    return list(zip(texts, labels)), ratm


def collate(batch):
    return tuple(torch.stack(items) for items in zip(*batch))


def run_epoch(loader, model, criterion, lambdas, optim, warmup_scheduler, train=True):
    model.train() if train else model.eval()
    running, y_true, y_pred, all_logits = 0.0, [], [], []
    for batch in loader:
        out, lbls = training_step(batch, model, criterion, lambdas)

        if train:
            out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); warmup_scheduler.step(); optim.zero_grad()

        probs = (out["logits"].detach() > 0).long().cpu()
        y_pred.extend(probs.tolist()); y_true.extend(lbls.cpu().tolist())
        running += out["total_loss"].item()
    return running/len(loader), f1_score(y_true, y_pred)


def evaluate(loader, model, criterion, lambdas):
    model.eval()
    running, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            out, lbls = training_step(batch, model, criterion, lambdas)
            preds     = (out["logits"] > 0).long().cpu()
            y_pred.extend(preds.tolist()); y_true.extend(lbls.cpu().tolist())
            running  += out["total_loss"].item()
    return running / len(loader), f1_score(y_true, y_pred)


############################ RATIONALE EXTRACTION FROM GENERATED DATA #############################

def find_rationales(texts, model, tokenizer, max_len=128, batch_size=16):
    """
    Args:
        texts      : List[str] toxic posts (e.g. from CSV)
        model      : trained HardKumaRationaleModel (already loaded and .eval())
        tokenizer  : tokenizer matching the model
        max_len    : max token length (same as used during training)
        batch_size : batch size for inference

    Returns:
        List[Dict] — for each text:
            {
              'tokens':            List[str] (WordPiece tokens),
              'rationale_mask':    List[int] (0/1, aligned with tokens),
              'rationale_tokens':  List[str] (non-special tokens where mask == 1)
            }
    """
     # wrap each string with a dummy "toxic" label
    dataset = HateXplainDataset(
                [(t, "toxic") for t in texts], # (text, label)
                ratm=[None] * len(texts), # no human rationales
                tokenizer=tokenizer,
                max_len=max_len
             )
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate)

    results = []
    model.eval()
    with torch.no_grad():
        for ids, attn, _, _ in loader:
            ids, attn = ids.to(DEVICE), attn.to(DEVICE)
            z_hard    = model(ids, attn)["z_hard"].cpu()

            for wp_ids, wp_mask, z in zip(ids.cpu(),
                                          attn.cpu(),
                                          z_hard):
                tokens = tokenizer.convert_ids_to_tokens(wp_ids.tolist())
                rationale_tokens = [
                    tok for tok, m, zh in zip(tokens, wp_mask, z)
                    if m == 1 and zh == 1
                       and tok not in ("[CLS]", "[SEP]", "[PAD]")
                ]
                results.append(
                    dict(tokens=tokens,
                         rationale_mask=z.int().tolist(),
                         rationale_tokens=rationale_tokens)
                )
    return results


################ LEETSPEAK TRANSFORMATIONS & HOMOGLYPH SUBSTITUTIONS ON RATIONALES #################

def split(lst, rats, train_p=0.8, val_p=0.1):
    idx = np.random.permutation(len(lst))
    n_train = int(train_p*len(lst)); n_val = int(val_p*len(lst))
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    return ([lst[i] for i in tr], [rats[i] for i in tr],
            [lst[i] for i in va], [rats[i] for i in va],
            [lst[i] for i in te], [rats[i] for i in te])


def build_homoglyph_map():
    url = "https://www.unicode.org/Public/security/latest/confusables.txt" 
    response = requests.get(url) # Fetch the confusables data
    raw_text = response.text # Get text content

    homoglyph_map = defaultdict(list) 

    for line in raw_text.splitlines():
        if line.startswith('#') or not line.strip(): # Skip comments and empty lines
            continue
        try:
            src_hex, target_hex, *_ = line.split(';') # 
            src_char = chr(int(src_hex.strip(), 16))
            target_chars = ''.join([chr(int(h, 16)) for h in target_hex.strip().split()])

            # We only want visually similar substitutions that map to 1 character
            if len(src_char) == 1 and len(target_chars) == 1:
                ascii_base = target_chars.lower()
                if ascii_base.isascii() and ascii_base.isalnum():
                    homoglyph_map[ascii_base].append(src_char)
        except Exception as e:
            continue  # skip malformed lines

    # Convert defaultdict to normal dict and deduplicate entries
    homoglyph_map = {k: list(set(v)) for k, v in homoglyph_map.items()}

    return homoglyph_map


# simple leet converter
class SimpleLeeter:
    _map = str.maketrans("aeios", "43105")
    def text2leet(self, word): return word.translate(self._map)


# convert rationale_tokens list -> target_words set (lower-case)
def toks_to_wordset(tok_list):
    return set(t.lower() for t in tok_list)


# make HateXplain-style record
def make_record(text, label_str, rat_mask=None):
    return (text, label_str), rat_mask


def perturb_token(token, homoglyph_map, leeter, mode):
    # strip ## for sub-words, re-attach later
    prefix = "##" if token.startswith("##") else ""
    core   = token[2:] if prefix else token

    if mode in ("homoglyph", "both"):
        core = ''.join(
            random.choice(homoglyph_map[c]) if c in homoglyph_map and random.random()<0.5 else c
            for c in core
        )
    if mode in ("leet", "both"):
        core = leeter.text2leet(core)
    return prefix + core


def random_homoglyph_substitution(text, homoglyph_map, prob=0.4):
    new_text = ""
    for char in text:
        if char.isalpha() and char.lower() in homoglyph_map and random.random() < prob:
            replacement = random.choice(homoglyph_map[char.lower()])
            new_text += replacement
        else:
            new_text += char
    return new_text


def smart_homoglyph_substitution(text, homoglyph_map, target_words, prob=0.5):
    tokens = text.split()
    new_tokens = []

    for token in tokens:
        if any(word in token.lower() for word in target_words) and random.random() < prob:
            new_token = ''.join(
                random.choice(homoglyph_map[c.lower()]) if c.lower() in homoglyph_map and random.random() < 0.5 else c
                for c in token
            )
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)

    return ' '.join(new_tokens)


def leet_some_words(text, leeter, word_prob=0.6):
    """
    Randomly leet some words in the text based on a probability.
    """
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < word_prob:
            new_words.append(leeter.text2leet(word))
        else:
            new_words.append(word)
    return ' '.join(new_words)


def augment_text(text, leeter, homoglyph_map, target_words=None):
    mode = random.choice(["leet", "homoglyph", "both"])

    if mode == "leet":
        text = leet_some_words(text, leeter)
    elif mode == "homoglyph":
        text = smart_homoglyph_substitution(text, homoglyph_map, target_words or [])
    elif mode == "both":
        text = leet_some_words(text, leeter)
        text = smart_homoglyph_substitution(text, homoglyph_map, target_words or [])

    return text


def augment_tokens(tokens, tokenizer, mask, homoglyph_map, leeter, p_apply=0.8):
    """
    tokens: list[str] WordPiece
    mask  : list[int] 0/1 aligned (None → treat as all zeros)
    """
    if mask is None or random.random() > p_apply:
        return tokenizer.convert_tokens_to_string(tokens)

    mode = random.choice(["homoglyph", "leet", "both"])
    new_tokens = [
        perturb_token(tok, homoglyph_map, leeter, mode) if m==1 else tok
        for tok, m in zip(tokens, mask)
    ]
    return tokenizer.convert_tokens_to_string(new_tokens)


def build_augmented_texts(pairs, rats, tokenizer, homoglyph_map, leeter):
    aug_texts, aug_labels = [], []
    for (txt, lab), mask in zip(pairs, rats):
        # tokenize original because rats is word-level for original splits
        if mask is not None and isinstance(mask[0], int):
            ids = tokenizer(txt, add_special_tokens=True)["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            # For original data, we already aligned rat_mask to WordPiece earlier
            wp_mask = mask                                           
        else: # for generated tox we already have WordPiece tokens
            tokens  = tokenizer.tokenize(txt)
            wp_mask = mask if mask is not None else [0]*len(tokens)

        augmented = augment_tokens(tokens, tokenizer, wp_mask, homoglyph_map, leeter)
        aug_texts.append(augmented); aug_labels.append(lab)
        
    return aug_texts, aug_labels


def partial_augment(pairs, rats, tokenizer, homoglyph_map, leeter, frac=0.2):
    # Select % of the data to augment
    indices = random.sample(range(len(pairs)), int(len(pairs) * frac))
    selected_pairs = [pairs[i] for i in indices]
    selected_rats  = [rats[i]  for i in indices]

    aug_texts, aug_labels = build_augmented_texts(selected_pairs, selected_rats, tokenizer, homoglyph_map, leeter)

    # Combine original data + augmented samples
    orig_texts  = [txt for (txt, _), _ in zip(pairs, rats)]
    orig_labels = [lab for (_, lab), _ in zip(pairs, rats)]

    all_texts  = orig_texts + aug_texts
    all_labels = orig_labels + aug_labels
    return all_texts, all_labels


################### FINE-TUNING OF PRETRAINED BERT MODEL WITH AUGMENTED DATASET ####################

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self):
        return len(self.enc["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted}


############## COMPARISON OF PERFORMANCE BETWEEN FINE-TUNED MODEL AND ORIGINAL MODEL ###############

def predict(pipeline_model, texts):
    preds = []
    for text in texts:
        out = pipeline_model(text, truncation=True, padding=True)[0]
        label = out['label'].lower()
        pred = 1 if label in {"toxic", "hate"} else 0
        preds.append(pred)
    return preds
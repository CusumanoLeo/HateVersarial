import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from collections import Counter
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers import AutoModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MERGE_MODE = "union" # other possibilities: "majority" -> but may drop rare buzt valid toxic terms / else: intersection (POOR COVERAGE, TOO STRICT!!!!!!!!!!)

def merge_rationales(masks, n_tokens, mode="union"):
    """
    masks    : List[List[int]] (length 0–3, possibly wrong sizes)
    n_tokens : int  = len(post['post_tokens'])
    Returns   : merged mask (list[int]) or None if no valid annotator
    """
    # Keep only masks whose length matches the post
    arrs = [np.array(m) for m in masks if len(m) == n_tokens]

    if len(arrs) == 0:
        return None                      # no reliable annotator mask

    if mode == "union":
        merged = np.bitwise_or.reduce(arrs)
    elif mode == "majority":
        thr = (len(arrs) + 1) // 2       # majority among remaining annotators
        merged = (np.sum(arrs, axis=0) >= thr).astype(int)
    else:                                # "intersection"
        merged = np.bitwise_and.reduce(arrs)
    return merged.tolist()

def preprocess_record(post):
    """
    Preprocesses a single post record to extract text, binary label, and merged rationales.

    Args:
        post (dict): A dictionary containing the post data. Expected keys include:
            - "post_tokens" (list of str): Tokens of the post text.
            - "annotators" (list of dict): A list of annotator dictionaries, each containing a "label" key.
            - "rationales" (optional, list): A list of rationale annotations for the post.

    Returns:
        tuple: A tuple containing:
            - text (str): The reconstructed text from the post tokens.
            - bin_label (str): The binary label, either "toxic" or "non-toxic", based on the majority label.
            - merged_rat (optional): The merged rationales if applicable, otherwise None.
    """
    text   = " ".join(post["post_tokens"])
    labels = [ann["label"] for ann in post["annotators"]]
    majority_label = Counter(labels).most_common(1)[0][0]

    n_tokens   = len(post["post_tokens"])
    merged_rat = None
    if majority_label in {"offensive", "hatespeech"} and post.get("rationales"):
        merged_rat = merge_rationales(post["rationales"], n_tokens, mode=MERGE_MODE)

    bin_label = "toxic" if majority_label in {"offensive", "hatespeech"} else "non-toxic"
    return text, bin_label, merged_rat






# ----- HARDKODE RATIONALE EXTRACTION ----- #


# -------------------------------------------------------------
# HardKuma distribution utilities (Bastings et al., 2020)
# -------------------------------------------------------------

class HardKumaSampler(nn.Module):
    """Samples a binary (0/1) mask via the HardKuma re‑parameterisation.

    Given shape (alpha) and rate (beta) > 0, the sampler draws a sample z in
    the open interval (0, 1) during the forward pass, then hard‑rounds it to
    {0,1} while using the soft value for gradient flow (straight‑through).
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
            # Straight‑through gradient: replace hard with soft in backward
            z_hard = z_hard.detach() - z_soft.detach() + z_soft
        else:
            z_hard = z_soft
        return z_hard, z_soft

# -------------------------------------------------------------
# Rationale Generator + Predictor model
# -------------------------------------------------------------
class HardKumaRationaleModel(nn.Module):
    """Generator‑Predictor architecture for rationale extraction.

    * Generator: produces binary mask z over tokens via HardKuma.
    * Predictor: applies BERT to the masked input and predicts toxicity.
    """

    def __init__(self, bert_name: str = "bert-base-uncased", max_len: int = 128):
        super().__init__()
        self.max_len = max_len
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size

        # Generator head → 2 positives (alpha, beta) per token
        self.gen_head = nn.Linear(hidden_size, 2)
        self.hardkuma = HardKumaSampler()

        # Classifier head on [CLS]
        self.cls_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Mask application helper
    # ------------------------------------------------------------------
    @staticmethod
    def apply_mask(embeddings: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Element‑wise multiply embeddings by z (shape: batch × seq_len × 1)."""
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
        alpha_beta = F.softplus(self.gen_head(hidden)) + 1e-4  # (B, L, 2)
        alpha, beta = alpha_beta[..., 0], alpha_beta[..., 1]

        z, z_soft = self.hardkuma(alpha, beta)  # (B, L)
        masked_hidden = self.apply_mask(hidden, z)

        # Predictor on masked sequence (take [CLS])
        cls_repr = masked_hidden[:, 0, :]
        logits = self.cls_head(cls_repr).squeeze(-1)  # (B,)

        # ----------------------------------------------------------
        # Loss components
        # ----------------------------------------------------------
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

        # Classification loss must be computed outside (BCEWithLogitsLoss)
        return out

# -------------------------------------------------------------
# Training step utility
# -------------------------------------------------------------
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

    sup_loss_raw = out["sup_loss"]                     # (scalar per sample)
    has_mask     = (rat_labels.sum(dim=1) > 0).float()
    sup_loss     = (sup_loss_raw * has_mask).sum() / has_mask.sum().clamp(min=1)

    clf_loss   = criterion(out["logits"], labels)
    total_loss = clf_loss + sup_loss + out["len_loss"] + out["cont_loss"]
    out.update({"clf_loss": clf_loss, "total_loss": total_loss})

    # Cast labels to int **once** for metric calculation later
    labels_long = labels.long()
    return out, labels_long


# dataset wrapper

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

        # ----- build rationale mask -----
        if self.ratm[idx] is not None:            # toxic with annotation
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

def collate(batch):
    return tuple(torch.stack(items) for items in zip(*batch))

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
     # -- dataset: wrap each string with a dummy "toxic" label
    dataset = HateXplainDataset(
                [(t, "toxic") for t in texts],   # pairs (text, label)
                ratm=[None] * len(texts),        # no human rationales
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
        for ids, attn, _, _ in loader:           # 4-tuple per collate
            ids, attn = ids.to(DEVICE), attn.to(DEVICE)
            z_hard    = model(ids, attn)["z_hard"].cpu()   # (B, L)

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



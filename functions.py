############################ IMPORT IMPORTANT LIBRARIES ############################

import numpy as np
import torch
import random
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import preprocessing as prep
import perturbations as perturb
import rationale
import predictions as pred
from sklearn.metrics import classification_report

from datasets import Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "Models/hardkuma_ckpt"

#################################### FUNCTIONS ####################################


def set_seed(seed):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   


def load_test_data():
    """
    Load the test data from the specified files.
    
    Returns:
        tuple: A tuple containing:
            - data_classes_two (np.ndarray): The classes for the two-class classification.
            - data_classes (np.ndarray): The classes for the multi-class classification.
            - post_id_divisions (dict): A dictionary mapping post IDs to their divisions.
            - data_test (dict): The test dataset filtered by post IDs.
    """
    data_classes_two = np.load('Data/classes_two.npy', allow_pickle=True)
    data_classes = np.load('Data/classes.npy', allow_pickle=True)

    with open('Data/post_id_divisions.json', 'r') as f:
        post_id_divisions = json.load(f)

    with open('Data/dataset.json', 'r') as f:
        data_ = json.load(f)

    test_ids = post_id_divisions['test']
    data_test = {k: v for k, v in data_.items() if k in test_ids}

    return data_classes_two, data_classes, post_id_divisions, data_test


def clean_data(data_test):
    """
    Clean the test data by removing posts with level 3 disagreements and resolving disagreements.
    
    Args:
        data_test (dict): The test dataset containing post data.
        
    Returns:
        list: A list of tuples containing cleaned text and labels.
    """
    disagreement_stats = prep.compute_annotator_disagreement(data_test, verbose=True)
    resolved_examples_custom = prep.resolve_disagreements_custom(data_test, disagreement_stats)
    
    cleaned_data = [
        (text, "toxic") if label == "hatespeech" else (text, "non-toxic")
        for text, label in resolved_examples_custom
        if label in {"hatespeech", "normal"}
    ]
    
    return cleaned_data



def separate_data(data_test, test_data_clean):
    """
    Separate the test data into toxic and non-toxic posts.
    
    Args:
        data_test (dict): The test dataset containing post data.
        
    Returns:
        tuple: A tuple containing lists of toxic and non-toxic posts.
    """
    
    rationales = [
        (data_test[k]['rationales'])
        for k in data_test.keys()
    ]

    posts = [
        (data_test[k]['post_tokens'])
        for k in data_test.keys()
    ]

    toxic_data = [post for post, label in test_data_clean if label == "toxic"]
    non_toxic_data = [post for post, label in test_data_clean if label == "non-toxic"] 
    
    return toxic_data, non_toxic_data



### ---- RATIONALE EXTRACTION ---- #


def rationale_extraction(toxic_data):
    """
    Extract rationales from the toxic data using a customed trained model.
    
    Args:
        toxic_data (list): List of toxic posts.
        non_toxic_data (list): List of non-toxic posts.
        
    Returns:
        tuple: A tuple containing lists of generated toxic and non-toxic posts with rationales.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)

    # 3.2  Read metadata, recreate the architecture skeleton
    meta  = json.load(open(f"{CKPT_DIR}/training_meta.json"))
    model = rationale.HardKumaRationaleModel(
                bert_name = meta["bert_model_name"],
                max_len   = meta["max_len"]
            )
    model.load_state_dict(torch.load(f"{CKPT_DIR}/pytorch_model.bin",
                                    map_location=DEVICE))
    model.to(DEVICE).eval()

    rationale_outputs = rationale.find_rationales(toxic_data, model=model, tokenizer=tokenizer, max_len=128 , batch_size=16)
    
    return rationale_outputs

def get_toxic_data_with_rationales(rationale_outputs, toxic_data, non_toxic_data):

    # Turn generated posts into the same “pair + mask” format

    # 1. Toxic generated posts
    gen_tox_pairs = [(text, "toxic") for text in toxic_data]
    gen_tox_rats  = [out["rationale_mask"] for out in rationale_outputs]

    # 2. Non-toxic generated posts (no rationales)
    gen_nontox_pairs = [(text, "non-toxic") for text in non_toxic_data]
    gen_nontox_rats  = [None] * len(gen_nontox_pairs)

    test_pairs = gen_tox_pairs + gen_nontox_pairs
    test_rats = gen_tox_rats + gen_nontox_rats  
    
    return test_pairs, test_rats


def perturb_data(test_pairs, test_rats, tokenizer):
    """
    Perturb the test data using homoglyphs and leet speak.
    
    Args:
        test_pairs (list): List of (text, label) pairs.
        test_rats (list): List of rationale masks corresponding to the texts.
        tokenizer: The tokenizer used for text processing.
        
    Returns:
        tuple: A tuple containing augmented texts and their corresponding labels.
    """
    


    ## ---- PERTURB DATA ---- ##
    homoglyph_map = perturb.build_homoglyph_map()
    leeter = perturb.SimpleLeeter()

    test_aug,  test_lab  = perturb.build_augmented_texts(test_pairs,  test_rats,
                                             tokenizer, homoglyph_map, leeter) 

    aug_test_pairs  = list(zip(test_aug,  test_lab))
    aug_test_rats = test_rats # rat_masks stay identical to originals for HardKuma supervision


    return aug_test_pairs, aug_test_rats


def prepare_final_data(test_final_clean, test_final_perturb):
    """
    Prepare the final test data for evaluation.
    
    Args:
        test_final_clean (list): List of clean test data pairs.
        test_final_perturb (list): List of perturbed test data pairs.
        
    Returns:
        tuple: A tuple containing lists of texts and labels for clean and perturbed data.
    """
    
    texts_final_clean = []
    labels_final_clean = []
    for test in test_final_clean:
        texts_final_clean.append(test[0])
        labels_final_clean.append(test[1])

    texts_final_perturb = []
    labels_final_perturb = []
    for test in test_final_perturb:
        texts_final_perturb.append(test[0])
        labels_final_perturb.append(test[1])
        
    print("Data preparation complete.")

    return texts_final_clean, labels_final_clean, texts_final_perturb, labels_final_perturb

    

# --- FINAL EVALUATION --- #

def load_models(model_path):
    """
    Load the pretrained and fine-tuned models from the specified path.
    
    Args:
        model_path (str): The path to the model directory.
        
    Returns:
        tuple: A tuple containing the tokenizer and model.
    """
    print("Loading model from:", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)
    return tokenizer, model, clf
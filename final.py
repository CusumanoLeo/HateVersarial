####################################################################################################
######################################## PROJECT (GROUP 26) ########################################
#######################################  File for Inference  #######################################
##################### Authors: Melina Cherchali, Romane Vorwald, Léo Cusumano ######################
####################################################################################################


################################ IMPORT IMPORTANT LIBRARIES & FILES ################################


import numpy as np
import torch
import random
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import preprocessing as prep
import perturbations as perturb
import rationale
import predictions as pred
import functions as fn
from sklearn.metrics import classification_report
from datasets import Dataset

############################################ CONSTANTS ############################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CKPT_DIR = "Models/hardkuma_ckpt"
fn.set_seed(SEED)


if __name__ == "__main__":
    
    
    ###################### LOAD AND PREPROCESS DATA ######################
    
    
    # Load the test data
    data_classes_two, data_classes, post_id_divisions, data_test = fn.load_test_data()
    test_data_clean =  fn.clean_data(data_test)
    print(f"Number of test posts after cleaning: {len(test_data_clean)}")

    # Separate the test data into toxic and non-toxic posts
    toxic_data, non_toxic_data = fn.separate_data(data_test, test_data_clean)
    print(f"Number of toxic posts: {len(toxic_data)}")

    # Perform rationale extraction on the toxic data
    print("Extracting rationales from toxic data...")
    rationale_outputs = fn.rationale_extraction(toxic_data)
    test_pairs, test_rats = fn.get_toxic_data_with_rationales(rationale_outputs, toxic_data, non_toxic_data)
    print(f"Number of test posts with rationales: {len(test_pairs)}")
    
    # Perturb the test data
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
    aug_test_pairs, aug_test_rats = fn.perturb_data(test_pairs, test_rats, tokenizer)
    print(f"Number of augmented test posts: {len(aug_test_pairs)}")


    # Prepare the clean and perturbed test data
    test_final_clean = test_data_clean       # Original test data
    test_final_perturb = aug_test_pairs      # Augmented test data with leet/homoglyph perturbations
    
    
    
   ###################### PREPARE FINAL DATA FOR MODEL INFERENCE ######################
   
    texts_final_clean, labels_final_clean, texts_final_perturb, labels_final_perturb = fn.prepare_final_data(test_final_clean, test_final_perturb)
    
    pretrained_model_path = "tum-nlp/bert-hateXplain" 
    finetuned_aug_model_path = "Models/bert-hateXplain-aug-finetuned"
    attention_model_path = "Models/bert-hateXplain-Adv_Attention"
    
    _,_, clf_pretrained = fn.load_models(pretrained_model_path)
    _,_, clf_finetuned_aug = fn.load_models(finetuned_aug_model_path)
    _,_, clf_attention = fn.load_models(attention_model_path)
    
    # Run predictions on clean test data
    print("Running predictions on clean test data...")
    preds_pretrained_clean = pred.get_preds(clf_pretrained, texts_final_clean)
    preds_finetuned_aug_clean = pred.get_preds(clf_finetuned_aug, texts_final_clean)
    preds_attention_clean = pred.get_preds(clf_attention, texts_final_clean)

    # Run predictions on perturbed test data
    print("Running predictions on perturbed test data...")
    preds_pretrained_perturb = pred.get_preds(clf_pretrained, texts_final_perturb)
    preds_finetuned_clean_aug_perturb = pred.get_preds(clf_finetuned_aug, texts_final_perturb)
    preds_attention_perturb = pred.get_preds(clf_attention, texts_final_perturb)


    # Use label_maps to convert labels to integers
    label_map = {"non-toxic": 0, "toxic": 1}
    # Convert labels to integers
    labels_final_clean_int = [label_map[label] for label in labels_final_clean]
    labels_final_perturb_int = [label_map[label] for label in labels_final_perturb]
    
    
    ###################### GENERATION CLASSIFICATION REPORTS ###################### 
    
    
    # Evaluate the models on clean test data
    print("Pretrained model on clean test data:")
    print(classification_report(labels_final_clean_int, preds_pretrained_clean, target_names=["non-toxic", "toxic"]))

    print("Fine-tuned augmented model on clean test data:")
    print(classification_report(labels_final_clean_int, preds_finetuned_aug_clean, target_names=["non-toxic", "toxic"]))

    print("Attention model on clean test data:")
    print(classification_report(labels_final_clean_int, preds_attention_clean, target_names=["non-toxic", "toxic"]))

    # Evaluate the models on perturbed test data

    print("Pretrained model on perturbed test data:")
    print(classification_report(labels_final_perturb_int, preds_pretrained_perturb, target_names=["non-toxic", "toxic"]))

    print("Fine-tuned augmented model on perturbed test data:")
    print(classification_report(labels_final_perturb_int, preds_finetuned_clean_aug_perturb, target_names=["non-toxic", "toxic"]))

    print("Attention model on perturbed test data:")
    print(classification_report(labels_final_perturb_int, preds_attention_perturb, target_names=["non-toxic", "toxic"]))


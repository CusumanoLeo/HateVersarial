from sklearn.metrics import f1_score, accuracy_score
import torch



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for a classification task.

    Args:
        eval_pred (tuple): A tuple containing two elements:
            - logits (ndarray): The predicted logits from the model.
            - labels (ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
            - "accuracy" (float): The accuracy score of the predictions.
            - "f1_macro" (float): The F1 score with macro averaging.
            - "f1_weighted" (float): The F1 score with weighted averaging.
    """
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted}
    
def get_preds(pipeline_model, texts):
    """
    Generate predictions for a list of texts using a given pipeline model.

    Args:
        pipeline_model (Callable): A pre-trained model pipeline that takes a text input 
                                   and returns a list of dictionaries containing predictions.
        texts (List[str]): A list of text strings to classify.

    Returns:
        List[int]: A list of integer predictions where 1 represents "toxic" or "hate" 
                   labels, and 0 represents all other labels.
    """
    preds = []
    for text in texts:
        out = pipeline_model(text)[0]['label']
        pred = 1 if out.lower() in {"toxic", "hate"} else 0
        preds.append(pred)
    return preds


def tokenize(tokenizer, example):
    ''''
    Tokenizes the input text using the provided tokenizer.
    '''
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128) 

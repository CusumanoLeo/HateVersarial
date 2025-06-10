
import random
from collections import defaultdict
import requests

### --- PERTURBATION FUNCTIONS --- ###

def build_homoglyph_map():
    """
    Builds a mapping of ASCII alphanumeric characters to their homoglyphs 
    (visually similar Unicode characters) based on the Unicode Consortium's 
    confusables data.

    The function fetches the latest confusables data from the Unicode Consortium's 
    public repository, parses it, and constructs a dictionary where the keys are 
    ASCII alphanumeric characters (lowercase) and the values are lists of homoglyphs 
    (Unicode characters that look similar to the key character).

    Returns:
        dict: A dictionary mapping ASCII alphanumeric characters to lists of their 
              homoglyphs. Each homoglyph list contains unique characters.
    """
    url = "https://www.unicode.org/Public/security/latest/confusables.txt" 
    response = requests.get(url) # Fetch the confusables data
    raw_text = response.text     # Get the text content

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

# 0.2  simple leet converter (stub)
class SimpleLeeter:
    _map = str.maketrans("aeios", "43105")  # toy map
    def text2leet(self, word): return word.translate(self._map)

# 1.1  convert rationale_tokens list ➜ target_words set (lower-case)
def toks_to_wordset(tok_list):
    return set(t.lower() for t in tok_list)

# 1.2  make HateXplain-style record
def make_record(text, label_str, rat_mask=None):
    return (text, label_str), rat_mask

def perturb_token(token, homoglyph_map, leeter, mode):
    """
    Applies perturbations to a given token using homoglyph substitution and/or leetspeak transformation.

    Parameters:
        token (str): The input token to be perturbed. If the token starts with "##", it is treated as a sub-word.
        homoglyph_map (dict): A dictionary mapping characters to their homoglyph alternatives.
        leeter (object): An object with a `text2leet` method that converts text to leetspeak.
        mode (str): The perturbation mode. Can be one of the following:
            - "homoglyph": Apply homoglyph substitution.
            - "leet": Apply leetspeak transformation.
            - "both": Apply both homoglyph substitution and leetspeak transformation.

    Returns:
        str: The perturbed token with the same prefix (if any) as the input token.
    """
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
    """
    Applies random homoglyph substitutions to characters in the input text.

    Homoglyphs are visually similar characters that can be used to replace 
    letters in the text. This function substitutes characters in the input 
    text with their homoglyphs based on a given probability.

    Args:
        text (str): The input text to be modified.
        homoglyph_map (dict): A dictionary where keys are characters (str) 
            and values are lists of homoglyphs (str) for those characters.
        prob (float, optional): The probability of substituting a character 
            with one of its homoglyphs. Defaults to 0.4.

    Returns:
        str: The modified text with random homoglyph substitutions applied.
    """
    new_text = ""
    for char in text:
        if char.isalpha() and char.lower() in homoglyph_map and random.random() < prob:
            replacement = random.choice(homoglyph_map[char.lower()])
            new_text += replacement
        else:
            new_text += char
    return new_text

def smart_homoglyph_substitution(text, homoglyph_map, target_words, prob=0.5):
    """
    Applies smart homoglyph substitution to a given text by replacing characters in 
    target words with visually similar characters (homoglyphs) based on a provided map.

    Args:
        text (str): The input text to process.
        homoglyph_map (dict): A dictionary mapping characters to lists of homoglyphs.
                              For example, {'a': ['а', 'à'], 'e': ['е', 'è']}.
        target_words (list): A list of words (case-insensitive) to target for homoglyph substitution.
        prob (float, optional): The probability of applying homoglyph substitution to a target word.
                                Defaults to 0.5.

    Returns:
        str: The text with homoglyph substitutions applied to target words.

    Example:
        >>> homoglyph_map = {'a': ['а', 'à'], 'e': ['е', 'è']}
        >>> target_words = ['example', 'text']
        >>> text = "This is an example text."
        >>> smart_homoglyph_substitution(text, homoglyph_map, target_words, prob=1.0)
        "This is аn ехample tеxt."
    """
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
    """
    Augments the input text by applying one or more text transformation techniques.

    This function randomly selects a transformation mode ("leet", "homoglyph", or "both")
    and applies the corresponding text augmentation. If the mode is "leet", it applies
    leetspeak transformations to some words in the text. If the mode is "homoglyph", it
    substitutes characters in the text with visually similar homoglyphs. If the mode is
    "both", it applies both transformations sequentially.

    Parameters:
        text (str): The input text to be augmented.
        leeter (callable): A function or object responsible for performing leetspeak
            transformations on the text.
        homoglyph_map (dict): A mapping of characters to their homoglyph equivalents.
        target_words (list, optional): A list of specific words to target for homoglyph
            substitution. If None, homoglyph substitution is applied to the entire text.

    Returns:
        str: The augmented text after applying the selected transformation(s).
    """
    mode = random.choice(["leet", "homoglyph", "both"])

    if mode == "leet":
        text = leet_some_words(text, leeter)
    elif mode == "homoglyph":
        text = smart_homoglyph_substitution(text, homoglyph_map, target_words or [])
    elif mode == "both":
        text = leet_some_words(text, leeter)
        text = smart_homoglyph_substitution(text, homoglyph_map, target_words or [])
    # mode == "none": return as is
    return text

def augment_tokens(tokens, tokenizer, mask, homoglyph_map, leeter, p_apply=0.8):
    """
    tokens: list[str] WordPiece
    mask  : list[int] 0/1 aligned (None → treat as all zeros)
    """
    if mask is None or random.random() > p_apply:
        return tokenizer.convert_tokens_to_string(tokens)   # leave unchanged

    mode = random.choice(["homoglyph", "leet", "both"])
    new_tokens = [
        perturb_token(tok, homoglyph_map, leeter, mode) if m==1 else tok
        for tok, m in zip(tokens, mask)
    ]
    return tokenizer.convert_tokens_to_string(new_tokens)

def build_augmented_texts(pairs, rats, tokenizer, homoglyph_map, leeter):
    """
    Builds augmented texts and their corresponding labels by applying token-level 
    augmentations based on a given mask and augmentation strategies.

    Args:
        pairs (list of tuples): A list of (text, label) pairs where `text` is a string 
            and `label` is the corresponding label.
        rats (list of lists): A list of masks corresponding to the `pairs`. Each mask 
            is either a word-level or WordPiece-level mask indicating which tokens 
            should be augmented.
        tokenizer (Tokenizer): A tokenizer object used to tokenize the input text 
            and convert tokens to IDs.
        homoglyph_map (dict): A dictionary mapping characters to their homoglyph 
            equivalents for augmentation.
        leeter (callable): A function or callable object that applies leetspeak 
            transformations to tokens.

    Returns:
        tuple: A tuple containing:
            - aug_texts (list of str): A list of augmented texts.
            - aug_labels (list): A list of labels corresponding to the augmented texts.
    """
    aug_texts, aug_labels = [], []
    for (txt, lab), mask in zip(pairs, rats):
        # tokenize original because rats is word-level for original splits
        if mask is not None and isinstance(mask[0], int):           # word-level?
            ids = tokenizer(txt, add_special_tokens=True)["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            # For original data, we already aligned rat_mask to WordPiece earlier
            wp_mask = mask                                           
        else:   # for generated tox we already have WordPiece tokens
            tokens  = tokenizer.tokenize(txt)
            wp_mask = mask if mask is not None else [0]*len(tokens)

        augmented = augment_tokens(tokens, tokenizer, wp_mask,
                                   homoglyph_map, leeter)
        aug_texts.append(augmented); aug_labels.append(lab)
    return aug_texts, aug_labels




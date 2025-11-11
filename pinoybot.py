"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
import re
import pandas as pd

# 1. Load your trained model from disk (e.g., using pickle or joblib)
with open('phase3_output/pinoybot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('phase3_output/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

def extract_filipino_affix_features(word):
        word_lowered = word.lower()
        features = {}

        #Prefixes
        features['has_nag_prefix'] = bool(re.match(r"^nag", word_lowered))
        features['has_na_prefix'] = bool(re.match(r"^na", word_lowered))
        features['has_mag_prefix'] = bool(re.match(r"^mag", word_lowered))
        features['has_ma_prefix'] = bool(re.match(r"^ma", word_lowered))
        features['has_pag_prefix'] = bool(re.match(r"^pag", word_lowered))
        features['has_pinag_prefix'] = bool(re.match(r"^pinag", word_lowered))
        features['has_pina_prefix'] = bool(re.match(r"^pina", word_lowered))

        #Suffixes
        features['has_an_suffix'] = bool(re.search(r"an$", word_lowered))
        features['has_hin_suffix'] = bool(re.search(r"hin$", word_lowered))
        features['has_in_suffix'] = bool(re.search(r"in$", word_lowered))

        #Infixes
        #infix should be in between a consonant and a vowel making words like kUMain FIL and words like hUMan NOT FIL (ENG)
        features['has_in_infix'] = bool(re.match(r'^[bcdfghjklmnpqrstvwxyz]in[aeiou]', word_lowered))  
        features['has_um_infix'] = bool(re.match(r'^[bcdfghjklmnpqrstvwxyz]um[aeiou]', word_lowered))

        return features


def extract_english_affixes(word):
    word_lowered = word.lower()
    features = {}

    #Prefixes
    features['has_un_prefix'] = bool(re.match(r"^un", word_lowered))
    features['has_re_prefix'] = bool(re.match(r"^re", word_lowered))
    features['has_pre_prefix'] = bool(re.match(r"^pre", word_lowered))
    features['has_dis_prefix'] = bool(re.match(r"^dis", word_lowered))
    features['has_im_prefix'] = bool(re.match(r"^im", word_lowered))
    features['has_mis_prefix'] = bool(re.match(r"^mis", word_lowered))
    features['has_non_prefix'] = bool(re.match(r"^non", word_lowered))

    #Suffixes
    features['has_ed_suffix'] = bool(re.search(r"ed$", word_lowered))
    features['has_ing_suffix'] = bool(re.search(r"ing$", word_lowered))
    features['has_tion_suffix'] = bool(re.search(r"tion$", word_lowered))
    features['has_sion_suffix'] = bool(re.search(r"sion$", word_lowered))
    features['has_ly_suffix'] = bool(re.search(r"ly$", word_lowered))
    features['has_ness_suffix'] = bool(re.search(r"ness$", word_lowered))
    features['has_ment_suffix'] = bool(re.search(r"ment$", word_lowered))
    features['has_er_suffix'] = bool(re.search(r"er$", word_lowered))
    features['has_est_suffix'] = bool(re.search(r"est$", word_lowered))
    features['has_es_s_suffix'] = bool(re.search(r"e?s$", word_lowered))
    features['has_ful_suffix'] = bool(re.search(r"ful$", word_lowered))
    features['has_less_suffix'] = bool(re.search(r"less$", word_lowered))
    features['has_able_suffix'] = bool(re.search(r"able$", word_lowered))
    features['has_ible_suffix'] = bool(re.search(r"ible$", word_lowered))

    return features

def extract_character_features(word):
    word_lowered = word.lower()
    features = {}

    #Filipino specific patterns
    features['has_ng'] = bool(re.search(r"ng", word_lowered))
    features['ng_count'] = len(re.findall(r"ng", word_lowered)) #re.findall returns a list of non-overlapping mathes of ng, then we count the length of that list
    features['has_double_vowel'] = bool(re.search(r"(aa)|(ee)|(ii)|(oo)|(uu)", word_lowered))
    features['has_enye'] = bool(re.search(r"ñ", word_lowered))

    #vowel consonant ratios (FIL has strong emphasis on vowels) (ENG has consonant heavy words)
    if len(word_lowered) == 0:
        features['vowel_count'] = 0
        features['consonant_count'] = 0
        features['vowel_ratio'] = 0
        features['consonant_ratio'] = 0
    else:
        features['vowel_count'] = len(re.findall(r"[aeiou]", word_lowered))
        features['consonant_count'] = len(re.findall(r"[qwrtypsdfghjklzxcvbnm]", word_lowered))
        features['vowel_ratio'] = features['vowel_count']/len(word_lowered)
        features['consonant_ratio'] = features['consonant_count']/len(word_lowered)

    #bigram count (bigrams are 2 adjacent characters)
    features['fil_bigram_count'] = len(re.findall(r"(ng)|(ay)|(an)|(in)|(ka)|(sa)", word_lowered))
    features['eng_bigram_count'] = len(re.findall(r"(th)|(he)|(er)|(ed)|(es)|(ly)", word_lowered))

    return features
    
def extract_special_token_features(word):
    features = {}

    #numbers
    features['is_number'] = bool(re.match(r'^[0-9]+([.,][0-9]+)*$', word))
    features['has_digit'] = bool(re.search(r"[0-9]", word))

    #capitalization
    features['is_first_cap'] = bool(re.match(r"^[A-Z]", word))
    features['is_all_caps'] = bool(re.match(r"^[A-Z]{2,}$", word))
    features['is_all_lower'] = word.islower()

    #symbols
    features['is_punctuation'] = bool(re.match(r'^[^a-zA-Z0-9\s]+$', word))
    features['has_special_char'] = bool(re.search(r'[^a-zA-Z0-9\s]', word)) 

    #alphanumeric patterns
    features['is_alphanumeric'] = bool(re.search(r"[a-zA-Z]", word) and re.search(r"[0-9]", word))

    #length
    features['word_length'] = len(word)
    features['is_very_short'] = bool(len(word) <= 2)

    return features


def extract_all_features(word):
    #combines all features into one dictionary
    features = {}

    features.update(extract_filipino_affix_features(word))
    features.update(extract_english_affixes(word))
    features.update(extract_character_features(word))
    features.update(extract_special_token_features(word))

    return features


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """

    
    if not tokens: #if token list is empty
        return[]
    if not model or not feature_names: #checks if model and feature_names are empty, if so it returns others for all the tokens (prevents crash)
        print("Model is not loaded return all tokens as OTH")
        return ['OTH' for _ in tokens]
    
    word_features_list = []

    # 2. Extract features from the input tokens to create the feature matrix
    #create our mini feature matrix using the given tokens
    for word in tokens:
        features_of_word = extract_all_features(word)
        word_features_list.append(features_of_word)

    #New data frame for the model
    X_new = pd.DataFrame(word_features_list)

    #ensures all columns are the same as the training data
    X_new = X_new.reindex(columns=feature_names, fill_value=0)

    #convert to int for multinomialNB model
    X_new_nb = X_new.astype(int)

    # 3. Use the model to predict the tags for each token
    predictions = model.predict(X_new_nb)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    tags = [str(tag) for tag in predictions]


    # 5. Return the list of tags
    #    return tags
    return tags
    

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Check", "mo", "yung", "new", "update", "sa", "game", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)

    print("\n--- Results ---")
    output = ""
    for word, tag in zip(example_tokens, tags):
        output += f"{word}[{tag}] "
    print(output)
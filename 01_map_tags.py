import pandas as pd
import re

#Map the fine-grained tags (Fil, Eng, CS, etc.) into 3 classes (FIL, ENG, OTH)
#This will be used to train the model

#load validated csv file
df = pd.read_csv("Group1_ValidatedDataSet.xlsx - Data.csv")

#reads the tag and determines its class
def map_to_three_classes(tag):
    if not isinstance(tag, str) or not tag: #tag is empty/non-string
        return "OTH"
    elif "fil" in tag.lower():
        return "FIL"
    elif "eng" in tag.lower():
        return "ENG"
    elif "cs" in tag.lower(): #codeswitched words are still considered as filipino
        return "FIL"
    else:
        return "OTH"


#determines if "is_correct" == true to mark if we should check label or corrected_label
def label_is_true(row):
    is_correct_str = str(row["is_correct"]).lower()

    if is_correct_str == "true": #specifically check the column is_correct
        return row["label"]
    else:
        return row["corrected_label"]
    

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
    #Prefixes
    #Suffixes
    #Infixes
    
    pass

#use the function in apply to determine the tag and creates a panda series (kind of like a list)
tag = df.apply(lambda row: label_is_true(row), axis = 1)

#creates a new column that will contain either "ENG", "FIL", or "OTH"
df['three_class_label'] = tag.apply(map_to_three_classes)



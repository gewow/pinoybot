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
    
def extract_special_token_features(word): # I renamed the function
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
    

#use the function in apply to determine the tag and creates a panda series (kind of like a list)
tag = df.apply(lambda row: label_is_true(row), axis = 1)

#creates a new column that will contain either "ENG", "FIL", or "OTH"
df['three_class_label'] = tag.apply(map_to_three_classes)

# --- Test for Task 2.5 ---
print("\nTesting Master Feature Extractor:")
test_words = ['naglunch', 'kumain', 'corrupt', 'playing', 'Manila', '.', '2023']

for word in test_words:
    features = extract_all_features(word)
    print(f"\nWord: '{word}' ({len(features)} features extracted)")
    
    # Let's just show a few features to prove it works
    print(f"  ... has_nag_prefix: {features.get('has_nag_prefix')}")
    print(f"  ... has_ing_suffix: {features.get('has_ing_suffix')}")
    print(f"  ... vowel_ratio: {features.get('vowel_ratio')}")
    print(f"  ... is_number: {features.get('is_number')}")






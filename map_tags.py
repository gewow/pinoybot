import pandas as pd
import re
import pickle
import os

#Map the fine-grained tags (Fil, Eng, CS, etc.) into 3 classes (FIL, ENG, OTH)
#This will be used to train the model

#load validated csv file
df = pd.read_csv("Group2_ValidatedDataSet.xlsx - Data.csv")

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
    

def create_feature_matrix(data):
    feature_dicts = []

    for idx, row in data.iterrows():
        word = str(row['word'])
        features = extract_all_features(word)
        feature_dicts.append(features)

    
    X = pd.DataFrame(feature_dicts)
    y = data['three_class_label']

    feature_names  = list(X.columns)

    return X, y, feature_names

#FOR DEBUGGING:
# def analyze_features(X, y):
   
#     print("\n" + "="*50)
#     print("FEATURE ANALYSIS")
#     print("="*50)
    
#     # combine X and y into one DataFrame for easy filtering
#     analysis_df = X.copy()
#     analysis_df['label'] = y
    
#     # check FIL features
#     print("\n--- Top 5 Features for FIL ---")
#     # get all rows that are FIL, sum up their feature columns, and show the biggest
#     fil_features = analysis_df[analysis_df['label'] == 'FIL'].sum(numeric_only=True)
#     print(fil_features.sort_values(ascending=False).head(5))
    
#     # check ENG features
#     print("\n--- Top 5 Features for ENG ---")
#     eng_features = analysis_df[analysis_df['label'] == 'ENG'].sum(numeric_only=True)
#     print(eng_features.sort_values(ascending=False).head(5))
    
#     # check OTH features
#     print("\n--- Top 5 Features for OTH ---")
#     oth_features = analysis_df[analysis_df['label'] == 'OTH'].sum(numeric_only=True)
#     print(oth_features.sort_values(ascending=False).head(5))



def save_feature_data(X, y, feature_names, output_dir='phase2_output'):
    print("\n" + "="*50)
    print("SAVING DATA FOR PHASE 3")
    print("="*50)
    
    # create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # save as pickle
    with open(f'{output_dir}/feature_matrix.pkl', 'wb') as f:
        pickle.dump(X, f)
        
    with open(f'{output_dir}/labels.pkl', 'wb') as f:
        pickle.dump(y, f)
        
    with open(f'{output_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # CSV version 
    combined = X.copy()
    combined['label'] = y
    combined.to_csv(f'{output_dir}/feature_matrix.csv', index=False)
    
    print(f"All data saved to '{output_dir}/' folder.")



# use the function in apply to determine the tag and creates a panda series (kind of like a list)
tag = df.apply(lambda row: label_is_true(row), axis = 1)

# creates a new column that will contain either "ENG", "FIL", or "OTH"
df['three_class_label'] = tag.apply(map_to_three_classes)
print("Tag mapping complete.")
print(f"Total rows: {len(df)}")
print(f"Label distribution:\n{df['three_class_label'].value_counts()}\n")


X, y, feature_names = create_feature_matrix(df)

print("\n--- Feature Matrix Preview (X) ---")
print(X.head())
print("\n--- Labels Preview (y) ---")
print(y.head())



#analyze_features(X, y)
save_feature_data(X, y, feature_names)



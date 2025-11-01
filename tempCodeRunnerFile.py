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
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


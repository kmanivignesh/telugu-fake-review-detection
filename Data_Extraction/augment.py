import pandas as pd
import random

# Step 1: Define templates and patterns for fake reviews
# Based on sample: hyperbole, vague comments, unusual ratings, call-to-actions
fake_templates = [
    # Extreme emotional language
    "నవ్వి నవ్వి చచ్చిపోయాను {extra}!",
    "అదిరిపోయింది {rating}/10, సూపర్ డూపర్!",
    "వావ్, ఇది గజ్జిబిజ్జి {extra}!",
    "నవ్వి నవ్వి కడుపు పగిలింది, {extra}!",
    "అద్భుతం, నీవు నెంబర్ 1 {extra}!",
    "సచ్చేసాను, ఇంత బాగుంటుందని అనుకోలేదు!",
    # Poor quality of writing
    "వావ్!",
    "సూపర్!",
    "ఓకే!",
    "గుడ్ {extra}!",
    "అదిరింది!",
    # Unusual ratings
    "ఇది {rating}/10, నిజంగా అద్భుతం!",
    "సూపర్ {rating}/10, లైక్ చేయండి!",
    # Call-to-actions
    "సబ్‌స్క్రైబ్ చేయండి, {rating}/10!",
    "లైక్ చేయండి, సూపర్ వీడియో {extra}!",
    "షేర్ చేయండి, నవ్వి నవ్వి సచ్చాను!"
]

# Extras for variety
extras = ["బ్రో", "గుయ్స్", "మామా", "సిస్టర్", "అన్నా", "", "OMG", "సో ఫన్నీ"]
ratings = [23, 45, 58, 67, 81, 99, 100, 150]  # Unusual ratings like in sample

# Reasons distribution (based on sample: 84.5% extreme, 14.6% poor quality, 1% unusual)
reason_weights = {
    "Extreme emotional language": 0.845,
    "Poor quality of writing": 0.146,
    "Unusual ratings": 0.009
}

# Step 2: Function to preprocess comments (add spaces between characters)
def preprocess_comment(comment):
    # Add space between each character, preserve spaces and punctuation
    return " ".join(char for char in comment)

# Step 3: Function to generate a fake review
def generate_fake_review():
    template = random.choice(fake_templates)
    extra = random.choice(extras)
    rating = random.choice(ratings) if "{rating}" in template else ""
    
    # Format the comment
    comment = template.format(extra=extra, rating=rating)
    
    # Assign reason based on weighted probability
    reason = random.choices(
        list(reason_weights.keys()),
        weights=list(reason_weights.values()),
        k=1
    )[0]
    
    # If template is short (e.g., "వావ్"), force "Poor quality" reason
    if len(comment.split()) <= 2 and "వావ్" in comment or "సూపర్" in comment or "ఓకే" in comment:
        reason = "Poor quality of writing"
    elif "{rating}" in template and rating:
        reason = random.choices(
            ["Extreme emotional language", "Unusual ratings"],
            weights=[0.5, 0.5],
            k=1
        )[0]
    
    # Preprocess the comment
    preprocessed = preprocess_comment(comment)
    
    return {
        "comment": comment,
        "preprocessed_comment": preprocessed,
        "label": 1,
        "reason": reason
    }

# Step 4: Generate 700 fake reviews
fake_reviews = [generate_fake_review() for _ in range(700)]

# Step 5: Load the existing dataset
# Replace 'labeled_telugu_comments_extended.xlsx' with your actual file path
input_file = "labeled_telugu_comments_extended.xlsx"
try:
    df = pd.read_excel(input_file)
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Creating a new DataFrame.")
    df = pd.DataFrame(columns=["comment", "preprocessed_comment", "label", "reason"])

# Verify original dataset size
print(f"Original dataset size: {len(df)} comments")

# Step 6: Create DataFrame for new fake reviews
new_reviews_df = pd.DataFrame(fake_reviews)

# Step 7: Append new reviews to original dataset
updated_df = pd.concat([df, new_reviews_df], ignore_index=True)

# Step 8: Verify new dataset size
print(f"New dataset size: {len(updated_df)} comments")
print(f"Real comments: {len(updated_df[updated_df['label'] == 0])}")
print(f"Fake comments: {len(updated_df[updated_df['label'] == 1])}")

# Step 9: Save the updated dataset to a new Excel file
output_file = "labeled_telugu_comments_extended_with_fake.xlsx"
updated_df.to_excel(output_file, index=False)
print(f"Updated dataset saved to '{output_file}'")
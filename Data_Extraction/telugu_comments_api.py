from googleapiclient.discovery import build
import re
import pandas as pd
import emoji
from indic_transliteration.sanscript import transliterate, ITRANS, TELUGU

API_KEY = ""

# 🔹 List of Telugu YouTube Video IDs
VIDEO_IDS = [
    # "cT2SlpsaGvg",  
    # "9cBp4yVHhYs",
    # "ZOlamEAhwYM",
    # "cT2SlpsaGvg",
    # "eQ3OUnYj5dE",
    # "4_JkZ3MhO0M",
    # "fF-Fp18qpyk",
    # "IG2UrMxLkDs", #before 
    "-tgQJekJ8vg",
    "Zg0xNeJbpDI",
    "4fgPvS6srmw",
    "cS2GlDd10ao"

    # Add 20+ more video IDs
]

# Remove emojis from text
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# Attempt transliteration from English (if detected as Telugu phonetically)
def transliterate_if_telugu_phonetic(text):
    if re.search(r'[a-zA-Z]', text) and not re.search(r'[\u0C00-\u0C7F]', text):
        return transliterate(text, ITRANS, TELUGU)
    return text

def get_telugu_comments(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if re.search(r'[\u0C00-\u0C7F]', comment) or re.search(r'[a-zA-Z]', comment):
                clean = remove_emojis(comment)
                clean = transliterate_if_telugu_phonetic(clean)
                if re.search(r'[\u0C00-\u0C7F]', clean):
                    comments.append(clean)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# 🔹 Combine Comments from All Videos
all_comments = []

for vid in VIDEO_IDS:
    print(f"📥 Fetching from Video ID: {vid}")
    all_comments.extend(get_telugu_comments(vid))

# 🔹 Save to Excel
if all_comments:
    df = pd.DataFrame({"Telugu Comments": all_comments})
    df.to_excel("telugu_reviews_all_new.xlsx", index=False)
    print(f"✅ Saved {len(all_comments)} Telugu comments to 'telugu_reviews_all.xlsx'")
else:
    print("❌ No comments found.")

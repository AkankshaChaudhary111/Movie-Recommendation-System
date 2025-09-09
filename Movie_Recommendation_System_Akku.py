import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Movie Recommendation System - EDA + Content-based recommender
# Copy/paste into a Jupyter/Colab cell and run.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# Replace with path if different
FILE_PATH = "C:/Users/akank/Downloads/Movie_Reccomendation__System_Dataset.csv"
assert os.path.exists(FILE_PATH), "Dataset file not found at path."

# 1. Load
df = pd.read_csv(FILE_PATH)
df.columns = [c.strip() for c in df.columns]

# 2. Basic cleaning helpers
def clean_multivalue(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = re.sub(r'[\|;/]+', ',', x)
    x = re.sub(r'\s*,\s*', ',', x)
    tokens = [t.strip() for t in x.split(',') if t.strip()!='']
    tokens = [t.lower().replace(' ', '_') for t in tokens]
    return " ".join(tokens)

# Identify multi-value columns and clean them
for col in df.columns:
    if any(k in col.lower() for k in ["genre", "tag", "keywords", "cast", "preferred", "platform"]):
        df[col] = df[col].apply(clean_multivalue)

# Numeric conversions for candidate columns
def to_numeric_col(df, col):
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_candidates = ["IMDb Rating", "Rotten Tomatoes (%)", "Box Office Collection (in millions)", "Duration (minutes)", "Liked By (Number of People)"]
for col in numeric_candidates:
    to_numeric_col(df, col)

# Fix Liked By (commas)
if "Liked By (Number of People)" in df.columns:
    df["Liked By (Number of People)"] = df["Liked By (Number of People)"].astype(str).str.replace(',', '')
    df["Liked By (Number of People)"] = pd.to_numeric(df["Liked By (Number of People)"], errors='coerce')

# Fill numeric missing values with median where sensible
for col in ["IMDb Rating", "Rotten Tomatoes (%)", "Box Office Collection (in millions)"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 3. Create combined text for TF-IDF
parts = []
for c in ["Genre", "Keywords / Tags", "Cast", "Director", "Preferred by Generation", "Streaming Platform"]:
    parts.append(df[c].astype(str) if c in df.columns else pd.Series([""]*len(df)))
df["combined_text"] = parts[0].fillna("") + " " + parts[1].fillna("") + " " + parts[2].fillna("") + \
                      " " + parts[3].fillna("") + " " + parts[4].fillna("") + " " + parts[5].fillna("")
df["combined_text"] = df["combined_text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# 4. EDA plots (matplotlib only)
# 4a: Genre distribution
if "Genre" in df.columns:
    genres = df["Genre"].str.replace('_', ' ').str.split().explode()
    genre_counts = genres.value_counts().head(30)
    plt.figure(figsize=(10,6))
    genre_counts.plot(kind='bar')
    plt.xlabel("Genre"); plt.ylabel("Count"); plt.xticks(rotation=45, ha='right')
    plt.title("Top Genres Distribution")
    plt.tight_layout(); plt.show()

# 4b: Language pie
if "Language" in df.columns:
    lang_counts = df["Language"].replace('', 'Unknown').value_counts().head(10)
    plt.figure(figsize=(7,7))
    lang_counts.plot(kind='pie', autopct='%1.1f%%', pctdistance=0.85)
    plt.ylabel("")
    plt.title("Top Languages Distribution")
    plt.tight_layout(); plt.show()

# 4c: IMDb vs Rotten Tomatoes scatter
if "IMDb Rating" in df.columns and "Rotten Tomatoes (%)" in df.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df["IMDb Rating"], df["Rotten Tomatoes (%)"])
    plt.xlabel("IMDb Rating (out of 10)"); plt.ylabel("Rotten Tomatoes (%)")
    plt.title("IMDb Rating vs Rotten Tomatoes (%)")
    plt.tight_layout(); plt.show()

# 4d: Median Box Office by Year
if "Release Year" in df.columns and "Box Office Collection (in millions)" in df.columns:
    box_by_year = df.groupby("Release Year")["Box Office Collection (in millions)"].median().dropna().sort_index()
    if len(box_by_year) > 0:
        plt.figure(figsize=(10,5))
        plt.plot(box_by_year.index, box_by_year.values, marker='o')
        plt.xlabel("Release Year"); plt.ylabel("Median Box Office (millions)")
        plt.xticks(rotation=45)
        plt.title("Median Box Office by Release Year")
        plt.tight_layout(); plt.show()

# 4e: Top Directors
if "Director" in df.columns:
    top_dirs = df["Director"].value_counts().head(15)
    plt.figure(figsize=(10,6))
    top_dirs.plot(kind='bar')
    plt.xlabel("Director"); plt.ylabel("Number of Movies")
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Directors by Movie Count")
    plt.tight_layout(); plt.show()

# 5. Build TF-IDF and cosine similarity (content-based)
tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(df["combined_text"].fillna(""))
cosine_sim = cosine_similarity(X, X)

# map titles to indices
if "Movie Title" in df.columns:
    titles = df["Movie Title"].astype(str).tolist()
else:
    titles = df.iloc[:,0].astype(str).tolist()
indices = pd.Series(range(len(df)), index=titles)

def get_recommendations(title, k=10):
    title = str(title)
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]  # skip itself
    movie_indices = [i[0] for i in sim_scores]
    results = df.iloc[movie_indices][["Movie Title", "Genre"]].copy()
    results["similarity_score"] = [round(s[1],3) for s in sim_scores]
    return results.reset_index(drop=True)

# Show example: first 3 titles' recommendations (print)
example_titles = titles[:3]
for t in example_titles:
    print("Recommendations for:", t)
    recs = get_recommendations(t, k=8)
    print(recs, end="\n\n")

# 6. Model checking: optional regression on Recommendation Score (if exists)
if "Recommendation Score" in df.columns:
    y = pd.to_numeric(df["Recommendation Score"], errors='coerce').fillna(df["Recommendation Score"].median())
    n_comp = min(50, X.shape[1]-1) if X.shape[1]>1 else 5
    n_comp = max(5, n_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=42, n_iter=7)
    X_reduced = svd.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print("Regression RMSE on Recommendation Score:", rmse)

# 7. Save sample recommendations to CSV for report
out_recs = []
for t in example_titles:
    recs = get_recommendations(t, k=8)
    if recs is None:
        continue
    for _, row in recs.iterrows():
        out_recs.append({"query_title": t, "recommended_title": row["Movie Title"],
                         "genre": row.get("Genre", ""), "similarity_score": row["similarity_score"]})
out_df = pd.DataFrame(out_recs)
out_path = "sample_recommendations.csv"
out_df.to_csv(out_path, index=False)
print("Saved sample recommendations to", os.path.abspath(out_path))


# --- User Interaction for Recommendations ---
print("\n🎬 Welcome to the Movie Recommendation System 🎬")

# Ask user preferences
genre = input("Enter a genre you like (e.g., Action, Comedy, Drama): ").strip().title()
language = input("Enter preferred language (e.g., English, Hindi, Korean): ").strip().title()
min_rating = float(input("Enter minimum IMDb rating (e.g., 7.5): "))

# Filter dataset based on input
recommended = df[
    (df['Genre'].str.contains(genre, case=False, na=False)) &
    (df['Language'].str.contains(language, case=False, na=False)) &
    (df['IMDb Rating'] >= min_rating)
]

# Show recommendations
if not recommended.empty:
    print("\n✨ Recommended Movies for You ✨")
    print(recommended[['Movie Title', 'Genre', 'Language', 'IMDb Rating', 'Director']].head(10))

    # Ask if user wants a link
    choice = input("\nDo you want a link for any movie? Enter the exact movie title (or 'no' to skip): ").strip()
    if choice.lower() != "no":
        if choice in df["Movie Title"].values:
            # Generate YouTube search link
            yt_link = "https://www.youtube.com/results?search_query=" + choice.replace(" ", "+") + "+trailer"
            print(f"\n🔗 Here’s a YouTube link for {choice}: {yt_link}")
        else:
            print("\n⚠️ Movie not found in dataset. Please type the exact title shown in recommendations.")
else:
    print("\n😢 Sorry, no movies found matching your preferences.")

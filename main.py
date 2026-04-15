import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. Load dataset
df = pd.read_csv("resume_data.csv", encoding='latin1')

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns)

# 2. Graph (Category Distribution)
plt.figure()
df['ï»¿job_position_name'].value_counts().head(10).plot(kind='bar')

plt.title("Top Job Positions")
plt.xlabel("Job Role")
plt.ylabel("Count")

plt.xticks(rotation=90)
plt.show()
# 3. Cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['text_features'] = (
    df['skills_required'].fillna('') + ' ' +        # JD skills
    df['responsibilities.1'].fillna('') + ' ' +     # JD responsibilities
    df['skills'].fillna('') + ' ' +                 # Resume skills
    df['career_objective'].fillna('') + ' ' +
    df['responsibilities'].fillna('')
)

df['cleaned_resume'] = df['text_features'].apply(clean_text)

print("\nSample cleaned text:")
print(df['cleaned_resume'][0][:200])
def score_to_label(score):
    if score >= 0.80:
        return 'High'
    elif score >= 0.60:
        return 'Medium'
    else:
        return 'Low'

df['label'] = df['matched_score'].apply(score_to_label)

print("\nClass distribution:")
print(df['label'].value_counts())

# 4. TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
X = tfidf.fit_transform(df['cleaned_resume'])

# 5. Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['label'])

# 6. Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. ENSEMBLE MODEL (MAIN PART)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm = SVC(kernel='linear', probability=True, max_iter=1000)

lr = LogisticRegression(max_iter=1000)


ensemble_model = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm)],


    voting='soft'
)

print("\nTraining Ensemble Model...")

ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))

import joblib
joblib.dump(ensemble_model, "resume_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("Models saved!")

# 8. Prediction function
def predict_resume(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    return le.inverse_transform(ensemble_model.predict(vec))[0]

# Test prediction
sample_resume = "Python developer with machine learning and SQL experience"
print("\nSample Prediction:", predict_resume(sample_resume))

# 9. Resume Matching using Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

job_description = """
Looking for a Data Science candidate with Python, machine learning,
SQL, data analysis, pandas, numpy, and visualization skills.
"""

jd_vec = tfidf.transform([clean_text(job_description)])
resume_vecs = tfidf.transform(df['cleaned_resume'])

similarity_scores = cosine_similarity(jd_vec, resume_vecs).flatten()

df['similarity'] = similarity_scores

top_resumes = df.sort_values(by='similarity', ascending=False).head(5)

print("\nTop 5 Matching Resumes:")
print(top_resumes[['ï»¿job_position_name', 'similarity']])

# 10. Skill extraction (bonus feature)
SKILLS = ['python', 'java', 'sql', 'machine learning', 'aws', 'docker']

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILLS if skill in text]

print("\nExtracted skills:")
print(extract_skills(sample_resume))

# 11. Interactive mode (bonus)
print("\n--- Resume Screening System ---")

while True:
    text = input("\nPaste resume (or type exit): ")

    if text.lower() == "exit":
        break

    prediction = predict_resume(text)
    skills = extract_skills(text)

    print("Prediction:", prediction)
    print("Skills found:", skills)
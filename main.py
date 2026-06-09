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
    text = str(text).lower()#force to lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)#removes punctautions
    text = re.sub(r'\s+', ' ', text).strip()#collapses multiple spaces into one
    return text #returns a clean string suitable for TF-IDF.

# Apply cleaning
df['text_features'] = (  #combine columns
    df['skills_required'].fillna('') + ' ' +        # JD skills
    df['responsibilities.1'].fillna('') + ' ' +     # JD responsibilities
    df['skills'].fillna('') + ' ' +                 # Resume skills
    df['career_objective'].fillna('') + ' ' +
    df['responsibilities'].fillna('')
)

df['cleaned_resume'] = df['text_features'].apply(clean_text)

print("\nSample cleaned text:")
print(df['cleaned_resume'][0][:200])
def score_to_label(score): #Before training, converts the matched_score into categorical labels based on defined thresholds.
    if score >= 0.80:
        return 'High'
    elif score >= 0.60:
        return 'Medium'
    else:
        return 'Low'

df['label'] = df['matched_score'].apply(score_to_label)# create a new column 'label' by applying the score_to_label function to the 'matched_score' column.

print("\nClass distribution:")
print(df['label'].value_counts())#Counts how many rows belong to each class.This helps check whether the dataset is balanced.
#Example output:

# High      500
# Medium    300
# Low       200

# 4. TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1500, stop_words='english') #stop_words='english' tells TF-IDF to ignore common English words that don't carry much meaning.
X = tfidf.fit_transform(df['cleaned_resume'])

# 5. Encode labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #creates a LabelEncoder object and stores it in the variable le.
#A LabelEncoder object is simply an object whose job is to remember the mapping between labels and numbers.
y = le.fit_transform(df['label'])
import joblib

joblib.dump(le, "label_encoder.pkl")
#When fit() runs, LabelEncoder first finds all unique labels in the 'label' column (e.g., 'High', 'Medium', 'Low') and it assigns a unique integer to each label (e.g., 'High' → 0, 'Medium' → 1, 'Low' → 2). 
print("\nLabel Mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")
# 6. This code splits data into training data and testing data.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. ENSEMBLE MODEL
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm = SVC(kernel='linear', probability=True, max_iter=1000)
#create a SVM model. Support Vector Machine is a classification algorithm.Its job is to find a boundary that separates classes.
#kernel='linear' means we are using a linear SVM, a straight-line (or hyperplane) separator.
#Stop after 1000 iterations if convergence hasn't happened.convergence means the model has found a good solution and further training won't improve it.
#Keep optimizing until you find the best solution, but stop if you reach 1000 steps.
lr = LogisticRegression(max_iter=1000)#Create Logistic Regression model
#Another classifier. Logistic Regression naturally predicts probabilities.


ensemble_model = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm)],


    voting='soft'
    # Soft voting: Each model gives probabilities for each class, and the final prediction is based on the average of these probabilities. This often leads to better performance than hard voting, especially when the individual models are well-calibrated.


)

print("\nTraining Ensemble Model...")

ensemble_model.fit(X_train, y_train)
#X_train contains TF-IDF vectors.
#Example:
# Resume 1 -> [0.2, 0.5, 0.1, ...]
# Resume 2 -> [0.8, 0.1, 0.4, ...]
# Resume 3 -> [0.3, 0.7, 0.2, ...]
# y_train contains labels encoded as integers (e.g., 0 for High, 1 for Medium, 2 for Low).
# # The model sees:

# Vector A -> High
# Vector B -> Low
# Vector C -> Medium
# many thousands of times.
# It starts learning patterns:
# Python + SQL + Machine Learning
#         ↓
# often High
# Few relevant skills
#         ↓
# often Low

# This learning process is called:

# fit(X_train, y_train)
#ensemble.fit(X_train, y_train)-> both models (svm and lr) are trained on the same training data. and later predictions are combined.

y_pred = ensemble_model.predict(X_test)# generates predictions for unseen test data.test data was kept aside and never shown during training.Used to evaluate the model.

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))#compares those predictions with the true labels and calculates the percentage of correct predictions

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
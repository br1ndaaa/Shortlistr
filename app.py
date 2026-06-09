import gradio as gr
import joblib
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer

# Load NLP model
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Stopwords
stopwords = set([
    "project", "experience", "company", "team",
    "work", "role", "year", "india", "user",
    "users", "application", "data", "system"
])

# Known tech keywords
tech_keywords = [
    "python", "java", "sql", "aws", "docker", "linux",
    "machine learning", "deep learning", "html", "css",
    "javascript", "react", "node", "mongodb", "c++",
    "tensorflow", "pandas", "numpy", "kubernetes", "redis"
]

# -------- SKILL EXTRACTION --------

def extract_skills_auto(text):
    doc = nlp(text[:1000])#Pass the text to the NLP model for processing and store the processed result in doc
    skills = set() #Creates an empty set.it automatically removes duplicates.

    for token in doc:#Loop through every word.
        #token.pos_ gives the part of speech tag for the word, like NOUN, VERB etc.The code keeps only: NOUN, PROPN (proper noun) and ignores common stop words like "the", "and", etc.
        #  token.is_stop checks if the word is a common stop word like the, is a, an etc. If it is, we skip it.
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            word = token.text.lower()#convert word to lower case to ensure uniformity. This helps in matching with our tech keywords later.
            if len(word) > 2 and word not in stopwords:# Keep only words longer than 2 characters and Checks against your custom stopword list.
                skills.add(word)#Add the filtered word to the set.

    return list(skills)

def extract_skills_smart(text):
    auto_skills = extract_skills_auto(text)# First, we extract potential skills using the NLP-based method. This gives us a list of candidate skills based on the content of the text.
    final_skills = set()
#Keyword scan
    for word in auto_skills:
        if word in tech_keywords:
            final_skills.add(word)

    for skill in tech_keywords:
        if skill in text.lower():
            final_skills.add(skill)

    return list(final_skills)

# -------- LOAD MODELS --------

model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")
print("MODEL CLASSES:", model.classes_)

# -------- TEXT CLEANING --------

def normalize_text(text):
    text = text.lower()
    replacements = { #dictionary created
        "ml": "machine learning",
        "dl": "deep learning",
        "js": "javascript",
        "py": "python"
    }
    for k, v in replacements.items():#iterate through. 1st iteration gives(k=ml v=machine learning) 2nd(k=dl) and so on
        text = re.sub(r'\b' + k + r'\b', v, text)
        #re module re.sub(pattern, replacement, string) pattern in string replaced w replacement
        #r'\b' + "ml" + r'\b'=r'\bml\b'
        #r is raw string notation,Without r, Python may treat \ as an escape character. 
        # \b is word boundary, so it matches "ml" as a whole word only, not part of "email" or "html"
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text) #Keep only letters, numbers, spaces, and common tech symbols like +, #, . 
    text = re.sub(r'\s+', ' ', text).strip() #Collapses multiple spaces into one and removes leading/trailing spaces.
    return text

# -------- PDF EXTRACTION --------

def extract_pdf(file):
    reader = PdfReader(file.name)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# -------- MAIN FUNCTION --------
import os  # ← add this at the top of the file

def rank_resumes(files, job_desc):
    if not files:
        return "❌ Please upload at least one resume"
    if not job_desc.strip():
        return "❌ Please enter a job description"

    jd_vec = bert_model.encode(normalize_text(job_desc))
    jd_skills = extract_skills_smart(job_desc)

    results = []  # ← BEFORE the loop

    for file in files:  # ← indented inside function
        try:
            text = extract_pdf(file)
            cleaned = clean_text(text)

            resume_vec = bert_model.encode(normalize_text(text))
            bert_score = cosine_similarity([jd_vec], [resume_vec])[0][0]

            skills = extract_skills_smart(text)
            if jd_skills:
                skill_match = len(set(skills) & set(jd_skills)) / len(jd_skills)
            else:
                skill_match = 0

            cleaned_jd = clean_text(job_desc)
            combined_text = cleaned_jd + " " + cleaned
            vec = tfidf.transform([combined_text])
            proba = model.predict_proba(vec)[0]
            high_idx = list(le.classes_).index("High")
            ml_score = float(proba[high_idx])

            final_score = (0.6 * bert_score) + (0.2 * skill_match) + (0.2 * ml_score)

            if final_score >= 0.55:
                final_label = "High"
            elif final_score >= 0.38:
                final_label = "Medium"
            else:
                final_label = "Low"

            results.append((os.path.basename(file.name), final_score, final_label))

        except Exception as e:
            print("ERROR:", e)
            results.append((os.path.basename(file.name), 0, f"Error: {str(e)}"))

    # -------- SORT & OUTPUT --------
    results.sort(key=lambda x: x[1], reverse=True)

    output = "## 📊 Resume Ranking\n\n"
    if results:
        best = results[0]
        output += f"## 🏆 Best Candidate: {best[0]} (Score: {best[1]*100:.1f}%)\n\n"
    for i, (name, score, pred) in enumerate(results, 1):
        output += f"### {i}. {name}\n"
        output += f"Score: {score*100:.1f}%\n"
        output += f"Prediction: {pred}\n\n"
    return output


# -------- UI --------

with gr.Blocks() as app:
    gr.Markdown("# Shortlistr")
    gr.Markdown("### Enter Job Description and Upload Multiple Resumes")

    job_desc = gr.Textbox(
        lines=4,
        placeholder="e.g. Python developer with machine learning and SQL"
    )

    files = gr.File(
        file_count="multiple",
        label="📂 Upload Multiple Resume PDFs"
    )

    output = gr.Markdown(label="Ranking Result")

    btn = gr.Button("Rank Resumes")

    btn.click(
        fn=rank_resumes,
        inputs=[files, job_desc],
        outputs=output
    )

app.launch()
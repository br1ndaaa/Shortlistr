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
    doc = nlp(text[:1000])
    skills = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            word = token.text.lower()
            if len(word) > 2 and word not in stopwords:
                skills.add(word)

    return list(skills)

def extract_skills_smart(text):
    auto_skills = extract_skills_auto(text)
    final_skills = set()

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

# -------- TEXT CLEANING --------

def normalize_text(text):
    text = text.lower()
    replacements = {
        "ml": "machine learning",
        "dl": "deep learning",
        "js": "javascript",
        "py": "python"
    }
    for k, v in replacements.items():
        text = re.sub(r'\b' + k + r'\b', v, text)
    return text

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

def rank_resumes(files, job_desc):

    if not files:
        return "❌ Please upload at least one resume"

    if not job_desc.strip():
        return "❌ Please enter a job description"

    jd_vec = bert_model.encode(normalize_text(job_desc))
    jd_skills = extract_skills_smart(job_desc)

    results = []

    for file in files:
        try:
            text = extract_pdf(file)
            cleaned = clean_text(text)

            # -------- BERT SIMILARITY --------
            resume_vec = bert_model.encode(normalize_text(text))
            bert_score = cosine_similarity([jd_vec], [resume_vec])[0][0]

            # -------- SKILL MATCH --------
            skills = extract_skills_smart(text)
            if jd_skills:
                skill_match = len(set(skills) & set(jd_skills)) / len(jd_skills)
            else:
                skill_match = 0

            # -------- ML MODEL --------
            cleaned_jd = clean_text(job_desc)
            combined_text = cleaned_jd + " " + cleaned   # JD + Resume together
            vec = tfidf.transform([combined_text])
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            classes = model.classes_
            high_idx = list(classes).index("High")
            ml_score = float(proba[high_idx])
            # -------- FINAL SCORE --------
            final_score = (0.6 * bert_score) + (0.2 * skill_match) + (0.2 * ml_score)

            # -------- FINAL LABEL (BASED ON SCORE) --------
            if final_score >= 0.55:
                final_label = "High"
            elif final_score >= 0.38:
                final_label = "Medium"
            else:
                final_label = "Low"
            results.append((
                 file.name.split("\\")[-1],
                 final_score,
                 final_label
             ))
            # Keep prediction from model.predict(), use it as the label
            
        
            


        except Exception as e:
            results.append((
                file.name.split("\\")[-1],
                0,
                "Error"
            ))

    # -------- SORT --------
    results.sort(key=lambda x: x[1], reverse=True)

    # -------- OUTPUT --------
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
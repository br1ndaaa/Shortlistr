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

# Known tech keywords (used as boost, NOT restriction)
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
        # Keep if:
        # 1. Known tech keyword
        # 2. OR looks like technical (alphabetic, not junk)
        if word in tech_keywords or word.isalpha():
            final_skills.add(word)

    # Ensure multi-word keywords are also captured
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
        text = text.replace(k, v)

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

    jd_clean = clean_text(job_desc)
    jd_vec = bert_model.encode(normalize_text(job_desc))

    results = []
    jd_skills = extract_skills_smart(job_desc)

    for file in files:
        try:
            text = extract_pdf(file)
            cleaned = clean_text(text)
            skills = extract_skills_smart(text)

            resume_vec = bert_model.encode(normalize_text(text))
            prediction = "N/A"
            score = cosine_similarity([jd_vec], [resume_vec])[0][0]

            # Skill match
            if jd_skills:
                skill_match = len(set(skills) & set(jd_skills)) / len(jd_skills)
            else:
                skill_match = 0

            final_score = (score + skill_match) / 2

            results.append((file.name.split("\\")[-1], final_score, skills, prediction))

        except Exception as e:
            results.append((file.name.split("\\")[-1], 0, [], "Error"))

    # Sort results
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 🔥 NORMALIZE SCORES (ADD THIS BLOCK HERE)
    # scores = [r[1] for r in results]
    # min_s = min(scores)
    # max_s = max(scores)

    # if max_s - min_s == 0:
    #    max_s += 1e-6


    # new_results = []
    # for name, score, skills, pred in results:
    #    norm_score = (score - min_s) / (max_s - min_s + 1e-6)
    #    norm_score = 0.1 + 0.8 * norm_score
    #    new_results.append((name, norm_score, skills, pred))

    # results = new_results

    # Output
    output = "## 📊 Resume Ranking\n\n"

    if results:
        best = results[0]
        output += f"## 🏆 Best Candidate: {best[0]} (Score: {best[1]*100:.1f}%)\n\n"

    for i, (name, score, skills, _) in enumerate(results, 1):
        output += f"### {i}. {name}\n"
        output += f"Score: {score*100:.1f}%\n\n"

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
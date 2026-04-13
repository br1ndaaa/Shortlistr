import gradio as gr
import joblib
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")
skills_list = ["python", "machine learning", "sql", "aws", "data analysis"]

matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(skill) for skill in skills_list]
matcher.add("SKILLS", patterns)
def extract_skills_spacy(text):
    doc = nlp(text)
    matches = matcher(doc)
    
    skills = set()
    for _, start, end in matches:
        skills.add(doc[start:end].text.lower())
    
    return list(skills)
#  Load model ONCE (important for speed)
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract PDF text
def extract_pdf(file):
    reader = PdfReader(file.name)   #  important fix
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text[:3000]   #  limit size for speed

# Main ranking function
def rank_resumes(files, job_desc):

    if not files:
        return "❌ Please upload at least one resume"

    if not job_desc.strip():
        return "❌ Please enter a job description"

    jd_clean = clean_text(job_desc)
    jd_vec = tfidf.transform([jd_clean])

    results = []

    # limit files to avoid lag
    for file in files:
        try:
            text = extract_pdf(file)
            cleaned = clean_text(text)
            skills = extract_skills_spacy(text)
            resume_vec = tfidf.transform([cleaned])
            score = cosine_similarity(jd_vec, resume_vec)[0][0]

            results.append((file.name.split("\\")[-1], score))

        except Exception as e:
            results.append((file.name, 0))

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    # Format output nicely
    output = " Resume Ranking:\n\n"
    for i, (name, score) in enumerate(results, 1):
        output += f"{i}. {name} → {round(score,2)}\n"

    return output

# UI
with gr.Blocks() as app:
    gr.Markdown("#  Resume Ranking System")

    gr.Markdown("### Enter Job Description and Upload Multiple Resumes")

    job_desc = gr.Textbox(
        lines=4,
        placeholder="e.g. Python developer with machine learning and SQL"
    )

    files = gr.File(
        file_count="multiple",
        label="📂 Upload Multiple Resume PDFs"
    )

    output = gr.Textbox(label=" Ranking Result")

    btn = gr.Button(" Rank Resumes")

    btn.click(
        fn=rank_resumes,
        inputs=[files, job_desc],
        outputs=output
    )

app.launch()
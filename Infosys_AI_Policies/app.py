# -------------------- Libraries --------------------
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# -------------------- Load CSV and Prepare Text --------------------
df = pd.read_csv(r"C:\Users\Aadhya\Downloads\Infosys_AI_Policies\updated_data.csv")

# Combine multiple columns for NLP
df['text_for_nlp'] = (
    df['scheme_name'].astype(str) + " " +
    df['details'].astype(str) + " " +
    df['benefits'].astype(str) + " " +
    df['eligibility'].astype(str) + " " +
    df['application'].astype(str) + " " +
    df['documents'].astype(str)
).str.lower()

# -------------------- TF-IDF Vectorization --------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df['text_for_nlp'])

# Save vectorizer and matrix
joblib.dump(vectorizer, "scheme_vectorizer.pkl")
joblib.dump({"matrix": tfidf_matrix, "df": df}, "scheme_tfidf_matrix.pkl")

# -------------------- FastAPI Setup --------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------- Helper Function --------------------
def search_policies(query: str, top_k: int = 3):
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "scheme_name": row["scheme_name"],
            "benefits": row["benefits"],
            "eligibility": row["eligibility"],
            "summary": textwrap.shorten(row.get("details", ""), width=250, placeholder="..."),
            "similarity": round(float(sims[idx]), 3)
        })
    return results

# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = search_policies(query)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

# -------------------- Example Query (Optional) --------------------
if __name__ == "__main__":
    question = "scholarship for students"
    results = search_policies(question, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['scheme_name']} | Similarity: {res['similarity']}")
        print(f" Benefits: {res['benefits']}")
        print(f" Eligibility: {res['eligibility']}\n")

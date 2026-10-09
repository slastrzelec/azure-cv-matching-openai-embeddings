"""CV Job Matcher: match a CV to job offers with OpenAI embeddings (Streamlit UI)."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.embeddings import (
    EMBEDDING_MODELS,
    EmbeddingError,
    cosine_similarities,
    embed_texts,
    extract_skills_with_ai,
    get_openai_client,
    get_similarity_rating,
    rank_jobs,
)
from utils.jobs import export_results_to_csv, fetch_fresh_jobs, last_update, load_jobs, safe_url, save_jobs
from utils.pdf_handler import MAX_PAGES, MAX_UPLOAD_BYTES, METHODS, extract_text_from_pdf

load_dotenv()

st.set_page_config(page_title="CV Job Matcher", page_icon="🎯", layout="wide")

for key in ("cv_text", "cv_skills", "results", "jobs", "jobs_source"):
    st.session_state.setdefault(key, None)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_job_embeddings(model: str, jobs_key: str, texts: tuple[str, ...]):
    """Embeddings of the offer texts, cached for an hour per model and offer set."""
    return embed_texts(list(texts), model=model, client=get_openai_client())


def job_text(job: dict) -> str:
    return f"{job['title']}. {job['description']} Requirements: {job['requirements']}"


def jobs_fingerprint(jobs: list[dict]) -> str:
    return hashlib.sha256("|".join(str(j.get("id")) + job_text(j) for j in jobs).encode()).hexdigest()


# ---------------------------------------------------------------- header
st.title("🎯 CV Job Matcher")
st.markdown(
    "Upload a CV (PDF), let an LLM condense it into skills, and rank job offers by "
    "**cosine similarity of OpenAI embeddings**."
)
st.info(
    "**Privacy:** the text of your CV is sent to OpenAI (skills extraction and embeddings). "
    "The PDF is processed in memory and is **not stored**."
)

# ---------------------------------------------------------------- job database
if st.session_state.jobs is None:
    st.session_state.jobs, st.session_state.jobs_source = load_jobs()

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    jobs = st.session_state.jobs
    if st.session_state.jobs_source == "sample":
        st.warning(f"Showing {len(jobs)} **fictional sample offers**. Click *Refresh DB* for live offers from The Muse API.")
    elif jobs:
        st.success(f"{len(jobs)} offers from The Muse API")
    else:
        st.warning("No job offers available. Click *Refresh DB*.")
with col2:
    stamp = last_update()
    st.caption(f"Last update: {stamp}" if stamp else "No live update yet")
with col3:
    if st.button("🔄 Refresh DB", use_container_width=True, help="Fetch current offers from The Muse API"):
        with st.spinner("Fetching offers..."):
            fresh = fetch_fresh_jobs()
        if fresh:
            save_jobs(fresh)
            st.session_state.jobs, st.session_state.jobs_source = fresh, "cache"
            st.session_state.results = None
            st.rerun()
        else:
            st.error("Could not fetch offers. Keeping the current list.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    embedding_model = st.selectbox(
        "Embedding model",
        EMBEDDING_MODELS,
        help="small: cheaper, 1536 dimensions; large: more expensive, 3072 dimensions",
    )
    pdf_method = st.selectbox("PDF extraction", METHODS)
    top_n = st.slider("Offers to show", 3, 20, 10)

    st.markdown("---")
    st.markdown(
        "**Score guide** (heuristic, not calibrated):\n\n"
        "- above 60 % 🟢 excellent\n- 50-60 % 🟠 good\n- 40-50 % 🟡 average\n- below 40 % 🔴 poor"
    )

# ---------------------------------------------------------------- step 1: CV
st.header("Step 1: Upload CV")
uploaded = st.file_uploader(f"PDF, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB (first {MAX_PAGES} pages are read)", type=["pdf"])

if uploaded is not None:
    data = uploaded.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        st.error("File is too large.")
    elif st.button("🔍 Analyze CV", type="primary", use_container_width=True):
        text = extract_text_from_pdf(io.BytesIO(data), method=pdf_method)
        if text:
            st.session_state.cv_text = text
            st.session_state.cv_skills = None
            st.session_state.results = None
            st.success(f"Text extracted: {len(text)} characters")
        else:
            st.error("Could not extract text (is the PDF a scan without a text layer?).")

# ---------------------------------------------------------------- step 2: skills
if st.session_state.cv_text:
    st.header("Step 2: Skills extraction")
    if st.button("🔧 Extract key skills (OpenAI)", use_container_width=True):
        client = get_openai_client()
        if client is None:
            st.error("OPENAI_API_KEY is not configured.")
        else:
            with st.spinner("Extracting skills..."):
                st.session_state.cv_skills = extract_skills_with_ai(st.session_state.cv_text, client=client)
            if not st.session_state.cv_skills:
                st.error("Skills extraction failed.")
    if st.session_state.cv_skills:
        st.info(st.session_state.cv_skills)

# ---------------------------------------------------------------- step 3: matching
if st.session_state.cv_skills and jobs:
    st.header("Step 3: Match")
    if st.button("🚀 Find matching offers", type="primary", use_container_width=True):
        client = get_openai_client()
        if client is None:
            st.error("OPENAI_API_KEY is not configured.")
        else:
            try:
                with st.spinner("Generating embeddings..."):
                    cv_vec = embed_texts([st.session_state.cv_skills], model=embedding_model, client=client)[0]
                    job_vecs = cached_job_embeddings(
                        embedding_model, jobs_fingerprint(jobs), tuple(job_text(j) for j in jobs)
                    )
                sims = cosine_similarities(cv_vec, job_vecs)
                st.session_state.results = rank_jobs(jobs, sims, top_n=top_n)
            except EmbeddingError as exc:
                st.session_state.results = None
                st.error(f"{exc}. No ranking was produced.")

# ---------------------------------------------------------------- results
if st.session_state.results:
    st.header("Results")
    if st.session_state.jobs_source == "sample":
        st.caption("These are fictional sample offers, shown for demonstration.")
    stamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "📥 Download CSV",
        data=export_results_to_csv(st.session_state.results, get_similarity_rating),
        file_name=f"cv_matches_{stamp_now}.csv",
        mime="text/csv",
    )
    for rank, job in enumerate(st.session_state.results, 1):
        emoji, label, _ = get_similarity_rating(job["similarity"])
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"### #{rank} {emoji} {job['title']}")
            st.markdown(f"**{job['company']}** | {job['location']}")
        with right:
            st.metric("Match", f"{job['similarity'] * 100:.1f}%", delta=label)
        with st.expander("Offer details"):
            st.markdown(f"**Description:** {job['description'][:500]}...")
            st.markdown(f"**Requirements:** {job['requirements']}")
            link = safe_url(job.get("link"))
            if link:
                st.markdown(f"[Open offer]({link})")
        st.markdown("---")

st.caption("OpenAI embeddings and GPT-4o-mini · The Muse API · Streamlit")

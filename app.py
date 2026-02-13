"""
CV Job Matcher - Streamlit Application
Match your CV to job offers using OpenAI embeddings
"""

import streamlit as st
import json
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Azure imports
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure SDK not installed. Install with: pip install azure-storage-blob")

# Import our modules
from utils.pdf_handler import extract_text_from_pdf, compare_pdf_methods
from utils.embeddings import (
    get_openai_client,
    calculate_embedding,
    extract_skills_with_ai,
    calculate_job_similarity,
    rank_jobs,
    get_similarity_rating
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎯 CV Job Matcher",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'cv_text' not in st.session_state:
    st.session_state.cv_text = None
if 'cv_skills' not in st.session_state:
    st.session_state.cv_skills = None
if 'cv_embedding' not in st.session_state:
    st.session_state.cv_embedding = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'cv_blob_url' not in st.session_state:
    st.session_state.cv_blob_url = None


# ========================================
# AZURE BLOB STORAGE FUNCTIONS
# ========================================

def upload_cv_to_azure(file_path, original_filename):
    """
    Uploads CV to Azure Blob Storage
    
    Args:
        file_path (str): Path to temporary CV file
        original_filename (str): Original filename
    
    Returns:
        str: URL to uploaded blob or None on error
    """
    
    if not AZURE_AVAILABLE:
        print("Azure SDK not available")
        return None
    
    try:
        # Get connection string from environment
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        if not connection_string:
            print("AZURE_STORAGE_CONNECTION_STRING not found in .env")
            return None
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Container name
        container_name = "cv-uploads"
        
        # Create container if doesn't exist
        try:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.get_container_properties()
        except:
            container_client = blob_service_client.create_container(container_name)
        
        # Generate unique blob name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{timestamp}_{original_filename}"
        
        # Upload file
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Return URL
        return blob_client.url
        
    except Exception as e:
        print(f"Error uploading to Azure: {e}")
        return None


def list_azure_cvs(limit=10):
    """
    Lists recent CVs from Azure Blob Storage
    
    Args:
        limit (int): Maximum number of CVs to return
    
    Returns:
        list: List of blob names
    """
    
    if not AZURE_AVAILABLE:
        return []
    
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            return []
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client("cv-uploads")
        
        # List blobs
        blobs = list(container_client.list_blobs())
        
        # Sort by creation time (newest first)
        blobs.sort(key=lambda x: x.creation_time, reverse=True)
        
        return [blob.name for blob in blobs[:limit]]
        
    except:
        return []


# ========================================
# FUNCTIONS FOR FETCHING JOB OFFERS
# ========================================

def fetch_fresh_jobs():
    """Fetches fresh job offers from The Muse API"""
    
    BASE_URL = "https://www.themuse.com/api/public/jobs"
    
    # Search keywords
    search_terms = [
        "machine learning",
        "data scientist", 
        "python developer",
        "AI engineer",
        "deep learning"
    ]
    
    all_jobs = []
    
    for term in search_terms:
        try:
            params = {"search": term, "page": 0}
            response = requests.get(BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                jobs = data.get("results", [])
                
                # Add without duplicates
                existing_ids = {j.get("id") for j in all_jobs}
                for job in jobs:
                    if job.get("id") not in existing_ids:
                        all_jobs.append(job)
        except Exception as e:
            print(f"Error fetching '{term}': {e}")
            continue
    
    # Format offers to our structure
    formatted_jobs = []
    
    for job in all_jobs:
        # Locations
        locations = job.get("locations", [])
        location_str = ", ".join([loc.get("name", "") for loc in locations]) if locations else "Remote"
        
        # Tags as requirements
        tags = job.get("tags", [])
        requirements = ", ".join([tag.get("name", "") for tag in tags]) if tags else "Not specified"
        
        formatted_jobs.append({
            "id": job.get("id"),
            "title": job.get("name", "Unknown"),
            "company": job.get("company", {}).get("name", "Unknown"),
            "description": job.get("contents", "")[:1000],  # Limit to 1000 chars
            "requirements": requirements,
            "location": location_str,
            "salary": "Not specified",
            "link": job.get("refs", {}).get("landing_page", "")
        })
    
    # Save to file
    if formatted_jobs:
        os.makedirs('data', exist_ok=True)  # Ensure folder exists
        
        with open('data/muse_jobs.json', 'w', encoding='utf-8') as f:
            json.dump(formatted_jobs, f, indent=2, ensure_ascii=False)
        
        # Save timestamp
        with open('data/last_update.txt', 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return formatted_jobs


@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
def load_jobs(refresh=False):
    """
    Loads job offers.
    
    Args:
        refresh (bool): If True, fetches fresh offers from API
    
    Returns:
        list: List of job offers
    """
    
    if refresh:
        st.cache_data.clear()  # Clear cache
        return fetch_fresh_jobs()
    
    try:
        # Try to load from file
        with open('data/muse_jobs.json', 'r', encoding='utf-8') as f:
            jobs = json.load(f)
        
        return jobs
        
    except FileNotFoundError:
        # If file doesn't exist, fetch fresh offers
        st.info("📥 First initialization - fetching offers from API...")
        return fetch_fresh_jobs()
    except Exception as e:
        st.error(f"❌ Error loading offers: {e}")
        return []


# ========================================
# EXPORT FUNCTIONS
# ========================================

def export_results_to_csv(results):
    """
    Exports match results to CSV format
    
    Args:
        results (list): List of matched jobs with similarity scores
    
    Returns:
        str: CSV formatted string
    """
    
    export_data = []
    
    for rank, job in enumerate(results, 1):
        emoji, rating, color = get_similarity_rating(job['similarity'])
        
        export_data.append({
            'Rank': rank,
            'Match Score': f"{job['similarity']*100:.1f}%",
            'Rating': rating,
            'Title': job['title'],
            'Company': job['company'],
            'Location': job['location'],
            'Salary': job.get('salary', 'Not specified'),
            'Requirements': job['requirements'],
            'Link': job['link']
        })
    
    # Create DataFrame
    df = pd.DataFrame(export_data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    return csv


# ========================================
# MAIN APPLICATION
# ========================================

# Application header
st.title("🎯 CV Job Matcher")
st.markdown("""
Find the best job offers matched to your CV using **AI and OpenAI embeddings**.

### 📋 How it works?
1. 📄 Upload your CV (PDF)
2. 🤖 AI extracts key skills
3. 🧠 Generates embeddings (semantic vectors)
4. 🎯 Compares with job offers and shows TOP matches
""")

# ========================================
# JOB DATABASE MANAGEMENT SECTION (top)
# ========================================
st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    jobs = load_jobs()
    if jobs:
        st.success(f"✅ Job database: {len(jobs)} positions from The Muse API")
    else:
        st.warning("⚠️ No jobs in database")

with col2:
    try:
        with open('data/last_update.txt', 'r') as f:
            last_update = f.read().strip()
        st.caption(f"🕐 Last update:\n{last_update}")
    except:
        st.caption("🕐 No update data")

with col3:
    if st.button("🔄 Refresh DB", help="Fetch latest offers from API", use_container_width=True):
        with st.spinner("Fetching fresh offers from The Muse API..."):
            jobs = load_jobs(refresh=True)
            st.success(f"✅ Fetched {len(jobs)} new offers!")
            st.rerun()

st.markdown("---")

# Sidebar - settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Embedding model:",
        ["text-embedding-3-small", "text-embedding-3-large"],
        help="Small: cheaper, 1536 dimensions | Large: more expensive, 3072 dimensions"
    )
    
    # PDF extraction method selection
    pdf_method = st.selectbox(
        "PDF extraction method:",
        ["pdfplumber", "pypdf2"],
        help="pdfplumber usually gives better formatting results"
    )
    
    # Number of results
    top_n = st.slider(
        "How many offers to show:",
        min_value=3,
        max_value=20,
        value=10,
        help="Top N best matched offers"
    )
    
    # Azure Blob Storage toggle
    st.markdown("---")
    st.markdown("### ☁️ Azure Storage")
    
    azure_enabled = st.checkbox(
        "Save CV to Azure Blob Storage",
        value=AZURE_AVAILABLE and bool(os.getenv("AZURE_STORAGE_CONNECTION_STRING")),
        help="Requires AZURE_STORAGE_CONNECTION_STRING in .env",
        disabled=not AZURE_AVAILABLE
    )
    
    if azure_enabled and st.session_state.cv_blob_url:
        st.success("✅ CV uploaded to Azure")
        st.caption(f"[View CV]({st.session_state.cv_blob_url})")
    
    # Show recent CVs from Azure
    if azure_enabled:
        recent_cvs = list_azure_cvs(limit=5)
        if recent_cvs:
            st.markdown("**Recent CVs:**")
            for cv in recent_cvs:
                st.caption(f"📄 {cv}")
    
    st.markdown("---")
    st.markdown("### 📊 Results interpretation")
    st.markdown("""
    - **>60%** 🟢 Excellent match
    - **50-60%** 🟠 Good match  
    - **40-50%** 🟡 Average match
    - **<40%** 🔴 Poor match
    """)

# Main section - CV Upload
st.header("📄 Step 1: Upload CV")

uploaded_file = st.file_uploader(
    "Select PDF file with CV:",
    type=['pdf'],
    help="We only support PDF files"
)

if uploaded_file is not None:
    # Save file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Upload to Azure if enabled
    if azure_enabled:
        if st.session_state.cv_blob_url is None:
            with st.spinner("Uploading CV to Azure Blob Storage..."):
                blob_url = upload_cv_to_azure(temp_path, uploaded_file.name)
                
                if blob_url:
                    st.session_state.cv_blob_url = blob_url
                    st.success(f"☁️ CV uploaded to Azure!")
                    st.caption(f"[View in Azure]({blob_url})")
                else:
                    st.warning("⚠️ Failed to upload to Azure (check connection string)")
    
    # Extract button
    if st.button("🔍 Analyze CV", type="primary", use_container_width=True):
        with st.spinner("Extracting text from PDF..."):
            # Text extraction
            cv_text = extract_text_from_pdf(temp_path, method=pdf_method, clean=True)
            
            if cv_text:
                st.session_state.cv_text = cv_text
                st.success(f"✅ Text extracted: {len(cv_text)} characters")
                
                # Show preview
                with st.expander("📝 CV Preview (first 500 characters)"):
                    st.text(cv_text[:500] + "...")
            else:
                st.error("❌ Failed to extract text from PDF")
    
    # Remove temporary file
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except:
            pass

# If we have CV text, show next steps
if st.session_state.cv_text:
    st.markdown("---")
    st.header("🤖 Step 2: Skills Extraction")
    
    if st.button("🔧 Extract key skills (AI)", use_container_width=True):
        with st.spinner("AI analyzing CV and extracting skills..."):
            client = get_openai_client()
            
            if client:
                skills = extract_skills_with_ai(
                    st.session_state.cv_text,
                    model="gpt-4o-mini",
                    client=client
                )
                
                if skills:
                    st.session_state.cv_skills = skills
                    st.success("✅ Skills extracted!")
                    
                    st.markdown("### 🎯 Your key skills:")
                    st.info(skills)
                    
                    st.caption(f"Length: {len(skills)} characters (vs full CV: {len(st.session_state.cv_text)} characters)")
                else:
                    st.error("❌ Failed to extract skills")
            else:
                st.error("❌ Missing OPENAI_API_KEY in .env file")

# If we have skills, do matching
if st.session_state.cv_skills:
    st.markdown("---")
    st.header("🎯 Step 3: Matching to Job Offers")
    
    # Load offers
    jobs = load_jobs()
    
    if jobs:
        st.info(f"📋 Ready to compare with {len(jobs)} job offers")
        
        if st.button("🚀 Find matching offers", type="primary", use_container_width=True):
            with st.spinner("Generating embeddings and calculating similarity..."):
                client = get_openai_client()
                
                if client:
                    # 1. Generate embedding for CV skills
                    cv_embedding = calculate_embedding(
                        st.session_state.cv_skills,
                        model=embedding_model,
                        client=client
                    )
                    
                    if cv_embedding is not None:
                        st.session_state.cv_embedding = cv_embedding
                        
                        # 2. Generate embeddings for offers
                        job_embeddings = []
                        progress_bar = st.progress(0)
                        
                        for i, job in enumerate(jobs):
                            job_text = f"{job['title']}. {job['description']} Requirements: {job['requirements']}"
                            job_emb = calculate_embedding(job_text, model=embedding_model, client=client)
                            
                            if job_emb is not None:
                                job_embeddings.append(job_emb)
                            else:
                                st.warning(f"⚠️ Failed to generate embedding for: {job['title']}")
                            
                            progress_bar.progress((i + 1) / len(jobs))
                        
                        progress_bar.empty()
                        
                        # 3. Calculate similarities
                        similarities = calculate_job_similarity(cv_embedding, job_embeddings)
                        
                        # 4. Rank offers
                        ranked_jobs = rank_jobs(jobs, similarities, top_n=top_n)
                        
                        st.session_state.results = ranked_jobs
                        
                        st.success(f"✅ Analysis complete! Found {len(ranked_jobs)} best offers")
                    else:
                        st.error("❌ Failed to generate embedding for CV")
                else:
                    st.error("❌ Missing OPENAI_API_KEY")
    else:
        st.warning("⚠️ No job offers in database. Click 'Refresh DB' at the top of the page.")

# Display results
if st.session_state.results:
    st.markdown("---")
    st.header("🏆 Match Results")
    
    # ========================================
    # EXPORT BUTTON (CSV)
    # ========================================
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Found {len(st.session_state.results)} best matching offers**")
    
    with col2:
        # Generate CSV
        csv_data = export_results_to_csv(st.session_state.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"cv_matches_{timestamp}.csv",
            mime="text/csv",
            help="Download match results as CSV file",
            use_container_width=True
        )
    
    with col3:
        st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    
    # Display job cards
    for rank, job in enumerate(st.session_state.results, 1):
        similarity = job['similarity']
        emoji, rating, color = get_similarity_rating(similarity)
        
        # Medal for TOP 3
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        
        # Job card
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {medal} {emoji} {job['title']}")
                st.markdown(f"**{job['company']}** | 📍 {job['location']}")
                
                if 'salary' in job and job['salary'] != "Not specified":
                    st.caption(f"💰 {job['salary']}")
            
            with col2:
                st.metric(
                    "Match",
                    f"{similarity * 100:.1f}%",
                    delta=rating
                )
            
            # Details
            with st.expander("📝 Offer details"):
                st.markdown(f"**Description:**\n{job['description'][:500]}...")
                st.markdown(f"**Requirements:**\n{job['requirements']}")
                st.markdown(f"**Link:** [{job['link']}]({job['link']})")
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎯 CV Job Matcher | Powered by OpenAI Embeddings & The Muse API | Made with Streamlit</p>
    <p>☁️ Azure Blob Storage enabled | 📊 CSV Export available</p>
</div>
""", unsafe_allow_html=True)
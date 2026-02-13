# 🎯 CV Job Matcher

AI-powered CV to job matching using OpenAI embeddings and Azure Cloud.

## Features
- 📄 PDF CV extraction
- 🤖 AI skills extraction (GPT-4o-mini)
- 🧠 Semantic matching (OpenAI embeddings)
- ☁️ Azure Blob Storage for CV uploads
- 📊 CSV export of results
- 🔄 Live job data from The Muse API

## Tech Stack
- **Frontend:** Streamlit
- **AI:** OpenAI API (embeddings, GPT-4o-mini)
- **Cloud:** Azure Web App, Azure Blob Storage
- **ML:** scikit-learn (cosine similarity)
- **PDF:** pdfplumber, PyPDF2

## Setup

1. Clone repo:
```bash
git clone https://github.com/your-username/cv-job-matcher.git
cd cv-job-matcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env`:
```
OPENAI_API_KEY=your-key
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
```

4. Run:
```bash
streamlit run app.py
```

## Deployment

Deployed on Azure Web App: [Live Demo](https://your-app.azurewebsites.net)

## Author
Sławomir Strzelec - Portfolio project demonstrating Azure + AI skills
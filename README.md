# CV Job Matcher

[![tests](https://github.com/slastrzelec/azure-cv-matching-openai-embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/slastrzelec/azure-cv-matching-openai-embeddings/actions/workflows/ci.yml)

Streamlit app that ranks job offers against a CV. The CV is condensed into a short skills list by an LLM
(`gpt-4o-mini`), skills and offers are embedded (`text-embedding-3-small/large`), and offers are ranked by
cosine similarity. Offers come from The Muse public API.

**Live demo:** https://cv-matching-openai-embeddings.streamlit.app/ (Streamlit Community Cloud, may need a
click to wake up). **Portfolio page:** https://slastrzelec.github.io/portfolio/18_cv-matching-openai-embeddings/

> This is a demonstration of an embedding pipeline, **not a validated recommender**. There is no labelled data,
> so match quality has not been measured. The 60/50/40 % score bands are heuristic.

## What happens to your CV

| Step | What happens |
|---|---|
| Upload | the PDF is read in memory (5 MB, first 10 pages); it is never written to disk |
| Skills extraction | the CV text (max 20,000 characters) is **sent to OpenAI** |
| Matching | only the short skills summary is embedded and sent to OpenAI |
| Storage | **nothing is stored by default.** Optional Azure Blob storage exists for deployments that set `CV_STORAGE_ENABLED=true`; even then the visitor must tick a consent box, the blob gets a random name, and there is no listing of stored CVs |

## Features

- PDF text extraction (`pdfplumber` or `pypdf`)
- Skills condensation with `gpt-4o-mini`
- Batched embeddings (up to 100 texts per request), cached for one hour per offer set
- Ranking by cosine similarity; if any embedding request fails, no ranking is shown
- Live offers from The Muse API (*Refresh DB*); the first start uses 10 bundled **fictional** sample offers
- CSV export (cells that could be interpreted as spreadsheet formulas are neutralised)

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # put your OPENAI_API_KEY there; .env is git-ignored
streamlit run app.py
```

## Testing

**40 automated tests** (pytest) plus 2 repository checks that run only in a git checkout. They run without an
API key, without network and without an Azure account: OpenAI is replaced by a deterministic fake client.

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

What is verified:

- **Ranking:** ordering, a length mismatch raises (the old code silently attached scores to the wrong offers),
  input is not mutated, rating thresholds.
- **Embeddings:** batching (calls = ceil(n/100)), row *i* belongs to text *i* (also when the API returns items out
  of order), a failed request raises instead of returning partial results, long and empty texts are handled.
- **Similarity:** cosine on known vectors.
- **PDF:** text extracted from generated PDFs with both libraries, garbage input returns `None`, page limit.
- **Job data:** HTML stripped, Muse payload formatted, duplicates removed, network errors tolerated, `javascript:`
  links dropped, cache/sample fallback without network.
- **CSV export:** formula-injection cells neutralised.
- **Privacy rules:** storage is off unless explicitly enabled, blob names never contain the user's file name,
  there is no function that lists stored CVs, no "Recent CVs" in the UI.
- **App:** Streamlit `AppTest` runs the whole script (sample offers, privacy notice, matching end to end with the fake
  client, and "no ranking on embedding failure").
- **Repository hygiene:** no `.env`, PDF or generated job files tracked; no e-mail or phone number in tracked
  notebook outputs.

Not covered: the real OpenAI and The Muse APIs, match quality (no ground truth), the Azure deployment itself,
and uploading a file through the browser widget. The Azure deploy workflow also runs the tests first and deploys only if
they pass.

## Deployment

The public demo runs on **Streamlit Community Cloud** (`OPENAI_API_KEY` as a secret).

`.github/workflows/main_azure-cv-matching-openai-embeddings.yml` builds and deploys to an Azure App Service
(`startup.sh` listens on `$PORT`, default 8000). It is **manual-only (`workflow_dispatch`) and currently not
active**: the last run failed at the Azure login step (*No subscriptions found*), so the app is not deployed
on Azure at the moment. The workflow, `startup.sh` and the optional Blob Storage module are kept as a reference
for that setup.

## Repository layout

```
app.py                 Streamlit UI
utils/embeddings.py    batching, cosine similarity, ranking, skills extraction
utils/jobs.py          The Muse client, cache, CSV export
utils/pdf_handler.py   PDF text extraction
utils/azure_storage.py optional, opt-in CV storage
data/sample_jobs.json  fictional sample offers
notebooks/             exploration (outputs stripped; they may have contained personal data)
SPEC.md                requirements and data-security rules for this release
tests/                 pytest suite
```

## Limitations

- Match quality is unmeasured; the notebooks compare strategies on 5 fictional offers only, which is an
  illustration, not evidence.
- The Muse API returns mostly US-centric offers and only the first page (20) per search term.
- Scores from `text-embedding-3` models are compressed (rarely above 0.7), so percentages are relative.

## Author

Sławomir Strzelec · [Portfolio](https://slastrzelec.github.io/portfolio/) ·
[GitHub](https://github.com/slastrzelec) · [LinkedIn](https://linkedin.com/in/sławomir-strzelec)

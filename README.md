# CV Job Matcher

[![tests](https://github.com/slastrzelec/azure-cv-matching-openai-embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/slastrzelec/azure-cv-matching-openai-embeddings/actions/workflows/ci.yml)

Streamlit app that ranks job offers against a CV. The CV is condensed into a short skills list by an LLM
(`gpt-4o-mini`), skills and offers are embedded (`text-embedding-3-small/large`), and offers are ranked by
cosine similarity. Offers come from The Muse public API.

**Live demo:** https://cv-matching-openai-embeddings.streamlit.app/ (Streamlit Community Cloud, may need a
click to wake up). **Portfolio page:** https://slastrzelec.github.io/portfolio/18_cv-matching-openai-embeddings/

![Ranked offers with match scores](assets/app_results.png)
*Ranking of the bundled fictional sample offers for one CV. Scores are cosine similarities of OpenAI embeddings; the bands (60/50/40 %) are heuristic.*

> This is a demonstration of an embedding pipeline, **not a validated recommender**. There is no labelled data,
> so match quality has not been measured. The 60/50/40 % score bands are heuristic.

## What happens to your CV

| Step | What happens |
|---|---|
| Upload | the PDF is read in memory (5 MB, first 10 pages); it is never written to disk |
| Skills extraction | the CV text (max 20,000 characters) is **sent to OpenAI** |
| Matching | only the short skills summary is embedded and sent to OpenAI |
| Storage | **nothing is stored.** The app has no storage code and no cloud-storage dependency |

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

**38 automated tests** (pytest; 2 of them need a git checkout and are skipped otherwise). They run without an
API key and without network: OpenAI is replaced by a deterministic fake client.

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
- **Privacy rules:** no "Recent CVs" in the UI, no storage option, and no storage module or cloud-storage
  dependency in code, requirements, workflows or settings.
- **App:** Streamlit `AppTest` runs the whole script (sample offers, privacy notice, matching end to end with the fake
  client, and "no ranking on embedding failure").
- **Repository hygiene:** no `.env`, PDF or generated job files tracked; no e-mail or phone number in tracked
  notebook outputs.

Not covered: the real OpenAI and The Muse APIs, match quality (no ground truth), and uploading a file through
the browser widget. GitHub Actions runs the tests and `ruff` on every push and pull request.

## Repository layout

```
app.py                 Streamlit UI
assets/                README screenshot
utils/embeddings.py    batching, cosine similarity, ranking, skills extraction
utils/jobs.py          The Muse client, cache, CSV export
utils/pdf_handler.py   PDF text extraction
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

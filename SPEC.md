# SPEC: hardening release (privacy, correctness, tests, CI)

Scope: the existing CV Job Matcher (Streamlit, OpenAI embeddings, The Muse API, optional Azure Blob).
No new product features. The release removes a privacy problem, fixes a ranking bug, makes cost
predictable, and makes the repository verifiable without any API key or network.

## 1. Goals

1. Visitors' CVs are not stored and not shown to other visitors, by default and in the UI.
2. The ranking can never silently attach a similarity score to the wrong job offer.
3. `pytest` on a clean clone runs with **no API key, no network, no Azure account** (fake OpenAI client).
4. Every claim in README and on the portfolio page is backed by code or a test, or is removed.

## 2. Out of scope

New models, a labelled relevance data set, user accounts, a database, Azure AI Search.
Match quality is **not** measured (no ground truth exists); the documentation says so.

## 3. Data flow and data security (binding)

| Data | Where it goes | Rule |
|---|---|---|
| Uploaded CV (PDF) | read from the upload in memory | never written to local disk (no `temp_*` files) |
| CV text | sent to OpenAI (`gpt-4o-mini`) for skills extraction | truncated to 20,000 characters; UI states that it is sent to OpenAI |
| CV skills summary | sent to OpenAI (embeddings) | short text only |
| CV file in Azure Blob | **off by default**; enabled only when the deployment sets `CV_STORAGE_ENABLED=true` **and** the visitor ticks an explicit consent box | blob name is random (`uuid4`), never the original file name; no listing of stored CVs anywhere in the UI |
| Job offers | public data (The Muse API) | cached in memory and in `data/muse_jobs.json` (git-ignored) |
| Secrets | `.env` / platform settings only | never logged, never in the repo; `.env` is git-ignored (a test asserts it) |

- No file named `*.pdf`, `.env`, `muse_jobs.json` or `last_update.txt` may be tracked (test on `git ls-files` when
  a git checkout is present).
- Notebook outputs may contain personal data (CV text). Outputs of notebooks that read a real CV are stripped
  before commit; a test fails if a tracked notebook contains an e-mail address or a phone number in its outputs.
- Uploads: PDF only, at most 5 MB, at most 10 pages processed.
- External content (job titles, links) is untrusted: links are rendered only when they start with `http://` or
  `https://`; CSV export neutralises cells that start with `=`, `+`, `-`, `@` (spreadsheet formula injection).

## 4. Correctness

- The current code skips a job whose embedding failed but keeps the job in the list, so `zip(jobs, similarities)`
  attaches scores to the wrong offers. New rule: embedding is all-or-nothing per request; on failure the app shows
  an error and produces **no** ranking. `rank_jobs` raises `ValueError` if lengths differ and does not mutate its input.
- Job embeddings are requested in batches (up to 100 texts per call) instead of one call per offer, and cached for
  one hour per (model, offer ids).
- First start never calls the network: cached file, else the bundled `data/sample_jobs.json`. The UI says when
  it is showing sample (fictional) offers. Network only on "Refresh".

## 5. Tests (what is verified, what is not)

Run without API key, network or Azure; OpenAI is replaced by a deterministic fake client.

| Area | Verified |
|---|---|
| Ranking | order by similarity; length mismatch raises; input not mutated; rating thresholds |
| Embeddings | batching (calls = ceil(n/100)), output order equals input order, failure raises, text truncation |
| Similarity | cosine on known vectors (identical = 1, orthogonal = 0) |
| PDF | text extracted from a generated PDF (both libraries); garbage input returns `None`; size/page limits |
| Jobs | HTML stripped, Muse payload formatted, duplicates removed by id, network error handled, unsafe links dropped |
| CSV export | formula-injection cells neutralised |
| Storage | disabled unless env flag set; blob name never contains the original file name; no listing function exists |
| App | Streamlit `AppTest`: renders with sample offers, shows the OpenAI/privacy notice, storage checkbox absent by default, no "Recent CVs" section |
| Hygiene | `.env`, PDFs and generated job files are not tracked; no PII in tracked notebook outputs |

Not covered (stated in README): match quality (no labelled data), the real OpenAI and Muse APIs, the Azure
deployment itself, upload through the browser widget.

## 6. CI

GitHub Actions on Python 3.11 (the version used for the Azure deployment): `ruff check`, `pytest`.
The deploy workflow runs the same tests first and deploys only if they pass. Dependabot weekly (pip) and
monthly (actions). `pip-audit` as an informational job.

## 7. Documentation honesty

- README: where the public demo runs, what is stored (nothing by default), what is sent to OpenAI, how to run
  tests, what is not covered. No placeholder URLs.
- Portfolio page: results table is labelled as a demonstration on 5 fictional sample offers, not a quality
  measurement; deployment statement matches reality; Testing section added.

## 8. Acceptance criteria

1. Clean clone: `pip install -r requirements-dev.txt && pytest` passes with no environment variables.
2. `grep -R "Recent CVs\|temp_" app.py utils` finds nothing.
3. `git ls-files` contains no `.pdf`, `.env`, `muse_jobs.json`.
4. Deploy workflow cannot deploy when tests fail.
5. README and portfolio page contain no statement the repository does not support.

## 9. Risks

- Dependency upgrades (the old pins are from 2023–24) could break the Azure build: pins are tested on Python 3.11
  in a clean venv; `startup.sh` keeps working with the default port 8000 and also honours `$PORT`.
- Already-published notebook outputs remain in git history until the history is rewritten (owner action,
  documented separately); the repository itself stops adding new personal data.

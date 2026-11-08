# API Reference

Auto-generated from the FastAPI OpenAPI schema (`python -c ...` in the build). Interactive docs: `http://127.0.0.1:8000/docs` · schema: `/openapi.json`.

All endpoints are prefixed with `/api`. Errors: `400` invalid input, `404` unknown id, `413` upload too large, `415` non-PDF upload, `422` invalid/unreadable PDF or bad body, `500` generic (details only in server logs; secrets are redacted).

| Method | Path | Summary |
|---|---|---|
| GET | `/api/config` | Config |
| GET | `/api/documents` | List Docs |
| POST | `/api/documents` | Upload |
| DELETE | `/api/documents/{paper_id}` | Delete |
| GET | `/api/documents/{paper_id}/text` | Text |
| GET | `/api/health` | Health |
| GET | `/api/projects` | List Projects |
| POST | `/api/projects` | Create |
| GET | `/api/projects/{pid}` | Get |
| GET | `/api/projects/{pid}/comparison` | Comparison |
| POST | `/api/projects/{pid}/discover` | Discover |
| GET | `/api/projects/{pid}/findings` | Findings |
| GET | `/api/projects/{pid}/gaps` | Gaps |
| GET | `/api/projects/{pid}/graph` | Graph |
| GET | `/api/projects/{pid}/report` | Report |
| GET | `/api/projects/{pid}/report/download` | Download |
| POST | `/api/projects/{pid}/run` | Run |
| GET | `/api/projects/{pid}/runs` | Runs |
| POST | `/api/projects/{pid}/select` | Select |
| GET | `/api/projects/{pid}/verifications` | Verifications |
| GET | `/api/stats` | Stats |

## Typical flow

```bash
curl -X POST :8000/api/projects -H 'content-type: application/json' -d '{"question":"How can explainable AI improve medical diagnosis reliability?"}'   # -> {project_id}
curl -X POST :8000/api/projects/1/discover -H 'content-type: application/json' -d '{}'          # planner + discovery (human review step)
curl -X POST :8000/api/documents -F file=@paper.pdf -F project_id=1                             # optional: your own PDFs
curl -X POST :8000/api/projects/1/run -H 'content-type: application/json' -d '{"paper_ids":["arxiv_2301_00001"]}'  # background run
curl :8000/api/projects/1/runs                                                                   # real agent status (poll)
curl -OJ ':8000/api/projects/1/report/download?format=pdf'                                       # pdf | docx | md
```

`POST /projects/{id}/run` body: `{paper_ids?: [str], uploaded_ids?: [str], config?: {enable_comparison, enable_gap, enable_verification, validate_dois, max_papers}, background: bool=true}`.
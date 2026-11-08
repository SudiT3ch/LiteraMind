# Database Schema

Generated from `database/models.py` (SQLAlchemy 2.x). SQLite by default; set `DATABASE_URL` (e.g. `postgresql+psycopg://user:pw@host/db`, add `psycopg` to requirements) for PostgreSQL — no code changes needed.

Relationships: `projects` 1—* `research_queries`, `agent_runs`, `findings`, `comparisons`, `gaps`, `verifications`, `reports`; `projects` *—* `papers` through `project_papers` (carries relevance + selection); `papers` 1—* `documents`, `chunks`, `findings`.

## `papers`

| column | type | notes |
|---|---|---|
| paper_id | VARCHAR(120) | PK |
| title | TEXT | not null |
| authors | JSON | not null |
| year | INTEGER |  |
| abstract | TEXT | not null |
| doi | VARCHAR(200) |  |
| arxiv_id | VARCHAR(50) |  |
| s2_id | VARCHAR(60) |  |
| openalex_id | VARCHAR(60) |  |
| venue | VARCHAR(300) |  |
| url | TEXT |  |
| pdf_url | TEXT |  |
| sources | JSON | not null |
| citation_count | INTEGER |  |
| is_uploaded | BOOLEAN | not null |
| fulltext_status | VARCHAR(40) | not null |
| extra | JSON | not null |
| created_at | DATETIME | not null |

## `projects`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| title | VARCHAR(300) | not null |
| status | VARCHAR(30) | not null |
| created_at | DATETIME | not null |

## `agent_runs`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| agent | VARCHAR(60) | not null |
| status | VARCHAR(20) | not null |
| attempts | INTEGER | not null |
| started_at | DATETIME | not null |
| finished_at | DATETIME |  |
| duration_s | FLOAT |  |
| summary | TEXT | not null |
| output | JSON |  |
| error | TEXT |  |
| warnings | JSON | not null |

## `chunks`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| chunk_uid | VARCHAR(160) | unique, not null |
| paper_id | VARCHAR(120) | FK → papers.paper_id, not null |
| idx | INTEGER | not null |
| page | INTEGER |  |
| section | VARCHAR(200) | not null |
| text | TEXT | not null |

## `comparisons`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| payload | JSON | not null |

## `documents`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| paper_id | VARCHAR(120) | FK → papers.paper_id, not null |
| filename | VARCHAR(300) | not null |
| path | TEXT | not null |
| sha256 | VARCHAR(64) | not null |
| size_bytes | INTEGER | not null |
| n_pages | INTEGER | not null |
| status | VARCHAR(30) | not null |
| error | TEXT |  |
| created_at | DATETIME | not null |

## `findings`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| paper_id | VARCHAR(120) | FK → papers.paper_id, not null |
| field | VARCHAR(40) | not null |
| text | TEXT | not null |
| page | INTEGER |  |
| section | VARCHAR(200) |  |
| passage | TEXT | not null |
| method | VARCHAR(20) | not null |

## `gaps`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| title | TEXT | not null |
| payload | JSON | not null |
| confidence | VARCHAR(20) | not null |

## `project_papers`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| paper_id | VARCHAR(120) | FK → papers.paper_id, not null |
| relevance | FLOAT | not null |
| selected | BOOLEAN | not null |

## `reports`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| title | TEXT | not null |
| markdown | TEXT | not null |
| payload | JSON | not null |
| created_at | DATETIME | not null |

## `research_queries`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| text | TEXT | not null |
| plan | JSON |  |
| created_at | DATETIME | not null |

## `verifications`

| column | type | notes |
|---|---|---|
| id | INTEGER | PK |
| project_id | INTEGER | FK → projects.id, not null |
| claim | TEXT | not null |
| claim_source | VARCHAR(40) | not null |
| paper_id | VARCHAR(120) |  |
| passage | TEXT | not null |
| page | INTEGER |  |
| section | VARCHAR(200) |  |
| status | VARCHAR(30) | not null |
| score | FLOAT | not null |
| contradicting | JSON | not null |
| method | VARCHAR(20) | not null |

## Not implemented
`users` — authentication was optional in the spec and is not implemented (single-user local app). Add a `users` table and a `user_id` FK on `projects` to extend.
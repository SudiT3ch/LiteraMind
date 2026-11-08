"""Research knowledge graph (NetworkX): papers, authors, topics, algorithms, datasets, limitation themes."""
from __future__ import annotations
import networkx as nx
from agents.base import short_title
from agents.lexicon import themes_of, THEMES
from database import repository as repo
from retrieval.embeddings import tokenize



def build_graph(project_id: int) -> nx.Graph:
    g = nx.Graph()
    papers = repo.project_papers(project_id, selected_only=True)
    findings = repo.list_findings(project_id)
    plan = (repo.get_query(project_id).plan or {}) if repo.get_query(project_id) else {}
    texts = {p["paper_id"]: set(tokenize(p["title"] + " " + (p["abstract"] or ""))) for p in papers}
    analysed = {f["paper_id"] for f in findings}
    for p in papers:
        if p["paper_id"] not in analysed: continue
        pid = p["paper_id"]
        g.add_node(pid, kind="paper", label=short_title(p["title"], 40), year=p["year"])
        for a in (p["authors"] or [])[:4]:
            g.add_node("author:" + a, kind="author", label=a); g.add_edge(pid, "author:" + a, rel="authored_by")
        for kw in (plan.get("keywords") or [])[:8]:
            t = set(tokenize(kw))
            if t and t <= texts[pid]:
                g.add_node("topic:" + kw.lower(), kind="topic", label=kw); g.add_edge(pid, "topic:" + kw.lower(), rel="addresses")
    for f in findings:
        pid = f["paper_id"]
        if pid not in g: continue
        if f["field"] == "algorithm":
            g.add_node("algo:" + f["text"], kind="algorithm", label=f["text"]); g.add_edge(pid, "algo:" + f["text"], rel="uses")
        elif f["field"] == "dataset":
            g.add_node("data:" + f["text"], kind="dataset", label=f["text"]); g.add_edge(pid, "data:" + f["text"], rel="evaluated_on")
        elif f["field"] == "limitations":
            for t in themes_of(f["text"]):
                g.add_node("lim:" + t, kind="limitation", label=THEMES[t][0]); g.add_edge(pid, "lim:" + t, rel="has_limitation")
    return g


from visualization.plotting import to_json, to_plotly, COLORS  # noqa: F401  (re-exported)

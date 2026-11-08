"""Pure graph->JSON/Plotly helpers (no DB imports; safe for the frontend)."""
from __future__ import annotations
import networkx as nx

COLORS = {"paper": "#1f3a5f", "author": "#7a8ba3", "topic": "#2a9d8f", "algorithm": "#e76f51", "dataset": "#e9c46a", "limitation": "#9b2226"}


def to_json(g: nx.Graph) -> dict:
    return {"nodes": [{"id": n, **d} for n, d in g.nodes(data=True)], "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)]}


def to_plotly(g: nx.Graph):
    import plotly.graph_objects as go
    if g.number_of_nodes() == 0: return go.Figure()
    pos = nx.spring_layout(g, seed=7, k=0.9)
    ex, ey = [], []
    for u, v in g.edges():
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#b8c1cc"), hoverinfo="none", showlegend=False))
    for kind, color in COLORS.items():
        ns = [n for n, d in g.nodes(data=True) if d["kind"] == kind]
        if ns:
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=kind.title(),
                                     text=[g.nodes[n]["label"] for n in ns], textposition="top center", textfont=dict(size=9),
                                     marker=dict(size=18 if kind == "paper" else 11, color=color, line=dict(width=1, color="white")),
                                     hovertext=[f"{kind}: {g.nodes[n]['label']}" for n in ns], hoverinfo="text"))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", legend=dict(orientation="h", y=-0.02))
    return fig


def json_to_graph(data: dict) -> nx.Graph:
    g = nx.Graph()
    for n in data.get("nodes", []):
        n = dict(n); g.add_node(n.pop("id"), **n)
    for e in data.get("edges", []):
        e = dict(e); g.add_edge(e.pop("source"), e.pop("target"), **e)
    return g

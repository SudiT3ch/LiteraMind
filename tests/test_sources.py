import httpx, pytest
from academic_sources import arxiv_client, semantic_scholar, openalex, crossref
from academic_sources.base import PaperRecord, deduplicate, http_get, SourceError, normalize_doi
from academic_sources import aggregator

# Payloads below follow the documented response formats of each API (hand-written format fixtures, not real papers).
ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2301.00001v2</id><published>2023-01-01T00:00:00Z</published>
<title>Fixture  Title
 One</title><summary> An abstract. </summary><author><name>A One</name></author><author><name>B Two</name></author>
<link title="pdf" href="http://arxiv.org/pdf/2301.00001v2" type="application/pdf"/><arxiv:doi>10.1234/Fix.1</arxiv:doi></entry>
<entry><id>http://arxiv.org/abs/2301.00002v1</id><title></title></entry></feed>"""


def test_arxiv_parse():
    r = arxiv_client.parse_feed(ARXIV)
    assert len(r) == 1 and r[0].arxiv_id == "2301.00001" and r[0].title == "Fixture Title One"
    assert r[0].year == 2023 and r[0].authors == ["A One", "B Two"] and r[0].paper_id == "arxiv_2301.00001"
    with pytest.raises(SourceError):
        arxiv_client.parse_feed("<not xml")


def test_s2_parse_missing_fields():
    r = semantic_scholar.parse({"data": [{"paperId": "x", "title": "T", "abstract": None, "year": None, "authors": None,
                                         "externalIds": {"DOI": "10.1/ABC"}, "openAccessPdf": None}, {"title": None}]})
    assert len(r) == 1 and r[0].doi == "10.1/abc" and r[0].abstract == "" and r[0].authors == []


def test_openalex_parse_inverted_abstract():
    r = openalex.parse({"results": [{"id": "https://openalex.org/W1", "title": "T", "publication_year": 2020,
                                     "doi": "https://doi.org/10.5/X", "abstract_inverted_index": {"Hello": [0], "world": [1]},
                                     "authorships": [{"author": {"display_name": "Z"}}], "locations": []}]})
    assert r[0].abstract == "Hello world" and r[0].doi == "10.5/x" and r[0].openalex_id == "W1"


def test_crossref_parse():
    p = crossref.parse_item({"title": ["T"], "DOI": "10.1/Z", "issued": {"date-parts": [[2019, 5]]},
                             "author": [{"given": "A", "family": "B"}]})
    assert p.year == 2019 and p.authors == ["A B"] and crossref.parse_item({}) is None


def test_dedup_by_doi_arxiv_title_and_merge():
    a = PaperRecord("Deep Learning for X", doi="10.1/A", sources=["arxiv"], abstract="short")
    b = PaperRecord("deep learning for x!", doi="https://doi.org/10.1/a", sources=["openalex"], abstract="a longer abstract", year=2020)
    c = PaperRecord("Other", arxiv_id="2301.1v1"); d = PaperRecord("Different title", arxiv_id="2301.1")
    e = PaperRecord("  "); f = PaperRecord("Same Title"); g = PaperRecord("same   title")
    out = deduplicate([a, b, c, d, e, f, g])
    assert len(out) == 3
    m = next(x for x in out if x.doi)
    assert m.abstract == "a longer abstract" and m.year == 2020 and m.sources == ["arxiv", "openalex"]


def test_http_get_handles_429_and_network(monkeypatch):
    calls = {"n": 0}
    def fake(url, **kw):
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake)
    with pytest.raises(SourceError) as e:
        http_get("https://x.org/a", retries=2)
    assert calls["n"] == 3 and "429" in str(e.value)
    def boom(url, **kw): raise httpx.ConnectTimeout("t")
    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(SourceError):
        http_get("https://x.org/a", retries=0)


def test_aggregator_degrades_gracefully(monkeypatch):
    ok = lambda q, n: [PaperRecord("Good paper", sources=["arxiv"])]
    def bad(q, n): raise SourceError("HTTP 429 from api")
    monkeypatch.setitem(aggregator.REGISTRY, "arxiv", ok)
    monkeypatch.setitem(aggregator.REGISTRY, "openalex", bad)
    recs, warns = aggregator.search_all(["q"], 5, ["arxiv", "openalex"])
    assert len(recs) == 1 and any("openalex" in w for w in warns)
    recs, warns = aggregator.search_all([""], 5, ["arxiv"])
    assert recs == [] and warns


def test_doi_normalisation_and_crossref_validate(monkeypatch):
    assert normalize_doi("DOI:10.1/AbC") == "10.1/abc" and normalize_doi("") is None
    monkeypatch.setattr(crossref, "lookup_doi", lambda d: (_ for _ in ()).throw(SourceError("net")))
    assert crossref.validate_doi("10.1/x") is None

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def make_pdf(path, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=10)
    doc.save(str(path)); doc.close()
    return str(path)

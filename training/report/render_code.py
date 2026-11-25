"""Render the report's short code listings as syntax-highlighted PNG images (pygments + Pillow).
Run: python render_code.py  -> ../figures/code_collect.png, ../figures/code_query.png"""
import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTINGS = {
 "code_collect.png": '''# collect_gtr.py - fetch_page(): polite, retrying API client
for attempt in range(max_retries):
    try:
        r = session.get(url, params={"p": page, "s": 100}, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt); continue          # exponential back-off
        raise RuntimeError(f"HTTP {r.status_code}")
    except (requests.ConnectionError, requests.Timeout, ValueError):
        time.sleep(2 ** attempt)
# collect(): skip pages already cached as page_XXXXX.json.gz; sleep 1 s between requests''',
 "code_query.png": '''# src/fundmatch/engine.py - one query (simplified)
masked = mask_funders(title + ". " + abstract)         # same masking as training
x = self.vectorize([masked])                           # TF-IDF uni/bi-grams, stop words removed
p = softmax((x @ self.coef.T + self.intercept) / T)    # temperature-calibrated, NumPy only
contrib = x.data * self.coef[c, x.indices]             # exact word contributions
rrf, svd, lex, _ = self._search(masked)                # int8 SVD + lexical, rank fusion
band = self._award(top30, cpih=cpih, ...)              # P25-P75; £0 and 2026+ starts left out''',
}
for name, code in LISTINGS.items():
    png = highlight(code, PythonLexer(), ImageFormatter(style="friendly", font_name="DejaVu Sans Mono", font_size=26,
                                                       line_numbers=False, image_pad=18, line_pad=6))
    open(f"{T}/figures/{name}", "wb").write(png)
    print("saved", name)

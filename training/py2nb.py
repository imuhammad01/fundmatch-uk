"""Convert a '# %%' percent-format script into a .ipynb (small helper, our own code).
Usage: python py2nb.py notebook_src.py out.ipynb
'# %% [markdown]' cells: each line starts with '# ' which is stripped.
A line '# %% INCLUDE path' inserts the source of a .py file (minus its __main__ block)."""
import sys, re, nbformat

src = open(sys.argv[1]).read()
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    header = header.strip()
    if header.startswith("[markdown]"):
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    elif header.startswith("INCLUDE"):
        path = header.split(None, 1)[1]
        code = open(path).read().split('\nif __name__ == "__main__":')[0]
        cells.append(nbformat.v4.new_code_cell(f"# ---- source of {path} (also runnable from the command line) ----\n" + code.strip() + "\n" + body.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, sys.argv[2])
print(len(cells), "cells ->", sys.argv[2])

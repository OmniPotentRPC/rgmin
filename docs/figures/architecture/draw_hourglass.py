"""Draw the rgmin hourglass from the headers and the Rust source.

Run from anywhere in the repository; needs Graphviz ``dot`` on PATH:

    python3 docs/figures/architecture/draw_hourglass.py

Writes ``hourglass.png`` beside this script and prints what it counted.
"""

import pathlib
import posixpath
import re
import subprocess
import sys

here = pathlib.Path(__file__).resolve().parent
root = here.parents[2]
include, src = root / "include", root / "src"

# One caption per public header; a new header must be added here.
roles = {
    "rgmin/optimize.hpp": "C++ API, namespace rgmin",
    "xts/optimize.hpp": "old xts::optimize names",
    "xts/xtensor.hpp": "xt::xarray adapter",
    "xts.h": "old xts_* C names",
}


def words(text):
    return [w for w in re.split(r"[\s,]+", text) if w]


header = re.sub(r"/\*.*?\*/", "", (include / "rgmin.h").read_text(), flags=re.S)
declared = set(re.findall(r"^(?!typedef)[A-Za-z_][\w *]*?\b(rgmin_\w+)\s*\(", header, re.M))
abi = dict(re.findall(r"#define RGMIN_ABI_(\w+) (\d+)", header))
exported = set()
for path in src.rglob("*.rs"):
    exported |= set(re.findall(r'extern "C" fn (rgmin_\w+)', path.read_text()))
if declared != exported:
    sys.exit(f"rgmin.h and src/ disagree on {sorted(declared ^ exported)}")

home = {}
for module, names in re.findall(r"^pub use (\w+)::\{?([^;]*?)\}?;", (src / "lib.rs").read_text(), re.M | re.S):
    for name in words(names):
        home[name] = module
ffi = (src / "ffi.rs").read_text()
reached = {}
for names in re.findall(r"^use crate::\{([^}]*)\}", ffi, re.M):
    for name in words(names):
        reached.setdefault(home.get(name, name), set()).add(name)
for module, names in re.findall(r"^use crate::(\w+)::\{([^}]*)\}", ffi, re.M):
    reached.setdefault(module, set()).update(words(names))
for name in re.findall(r"\bcrate::([A-Z]\w*)", ffi):
    reached.setdefault(home[name], set()).add(name)

headers = sorted(str(p.relative_to(include)) for p in include.rglob("*.h*") if p.parent.name != "dlpack")
edges = []
for name in headers:
    for target in re.findall(r'#include "([^"]+)"', (include / name).read_text()):
        edges.append((name, posixpath.normpath(posixpath.join(posixpath.dirname(name), target))))
missing = set(headers) - set(roles) - {"rgmin.h"}
if missing:
    sys.exit(f"no caption for {sorted(missing)}")

waist = f"{len(declared)} functions\\nversion {abi['VERSION_MAJOR']}.{abi['VERSION_MINOR']}, layout revision {abi['LAYOUT_REVISION']}"
cols = 4
cells = []
for i, module in enumerate(sorted(reached)):
    items = "<br/>".join(sorted(reached[module] - {module}))
    cells.append(f"<td><b>{module}</b><br/>{items}</td>")
rows = "".join("<tr>" + "".join(cells[i : i + cols]) + "</tr>" for i in range(0, len(cells), cols))
dot = [
    "digraph hourglass {",
    '  node [shape=box, style=rounded, fontname="Helvetica", fontsize=11];',
    '  edge [fontname="Helvetica", fontsize=10];',
]
for name in headers:
    label = roles.get(name, "C ABI: " + waist)
    dot.append(f'  "{name}" [label="include/{name}\\n{label}"];')
dot.append(f'  ffi [label="src/ffi.rs\\n{len(exported)} extern \\"C\\" functions"];')
dot.append(
    '  core [shape=plain, label=<<table border="1" cellborder="0" cellspacing="4" style="rounded">'
    f'<tr><td colspan="{cols}"><b>Rust core: modules src/ffi.rs calls</b></td></tr>{rows}</table>>];'
)
dot += [f'  "{a}" -> "{b}" [label="#include"];' for a, b in edges if b in headers]
dot += ['  "rgmin.h" -> ffi [label="same names"];', "  ffi -> core;", "}"]
subprocess.run(["dot", "-Tpng", "-Gdpi=120", "-o", str(here / "hourglass.png")], input="\n".join(dot), text=True, check=True)

print(f"rgmin.h declares {len(declared)} functions; src/ defines {len(exported)} extern \"C\" functions")
print(f"ABI {abi['VERSION_MAJOR']}.{abi['VERSION_MINOR']}, layout revision {abi['LAYOUT_REVISION']}")
print(f"src/ffi.rs reaches {len(reached)} modules: {', '.join(sorted(reached))}")
for a, b in edges:
    print(f"{a} includes {b}")

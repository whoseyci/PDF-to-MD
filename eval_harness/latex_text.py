"""LaTeX → plain text conversion.

Two-stage strategy:
  1. Try ``pandoc`` (highest fidelity)
  2. Fall back to a custom regex-based stripper (always succeeds,
     lower fidelity but enough for char/word-level WER computation)

The custom stripper is intentionally conservative: it removes math,
TeX commands, citation keys, and comment lines, then collapses
whitespace. It does NOT try to inline included sub-files, expand
macros, or resolve cross-references -- it gives us a "reasonable
plain-text version of the body" that's stable enough to compare
against pipeline output.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


# ----------------------------------------------------------------------
# Pandoc path
# ----------------------------------------------------------------------

def pandoc_to_text(main_tex: Path, out: Path,
                    *, timeout: int = 60) -> bool:
    """Convert LaTeX → plain text via pandoc, return True on success."""
    if shutil.which("pandoc") is None:
        return False
    try:
        proc = subprocess.run(
            ["pandoc", "--from=latex", "--to=plain",
             "--wrap=none", str(main_tex), "-o", str(out)],
            cwd=str(main_tex.parent),
            capture_output=True, text=True, timeout=timeout,
        )
        if out.exists() and out.stat().st_size > 500:
            return True
        return False
    except (subprocess.TimeoutExpired, Exception):
        return False


# ----------------------------------------------------------------------
# Custom regex stripper
# ----------------------------------------------------------------------

# A LaTeX command: \name, \name*, \name{arg}, \name[opt]{arg}{arg2}, ...
# We don't try to fully parse braces here -- we strip iteratively.

_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)

# Strip whole environments that contain only "noise" content
_NOISY_ENVS = (
    "equation", "equation*", "align", "align*", "eqnarray", "eqnarray*",
    "gather", "gather*", "split", "multline", "multline*",
    "math", "displaymath",
    "figure", "figure*", "table", "table*",
    "thebibliography", "verbatim", "lstlisting", "minted",
    "tikzpicture", "pgfpicture",
)


def _drop_env(text: str, env: str) -> str:
    """Remove \\begin{env}...\\end{env} blocks (non-greedy)."""
    pat = re.compile(
        r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{"
        + re.escape(env) + r"\}",
        re.DOTALL,
    )
    return pat.sub(" ", text)


# Inline math: $...$ or \( ... \) or \[ ... \]
_INLINE_MATH_RES = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"(?<!\\)\$(?:[^$\\]|\\.)*\$"),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
]


_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")


def inline_inputs(main_tex: Path, max_depth: int = 4) -> str:
    """Read main_tex, recursively inline \\input{} / \\include{} children."""
    base = main_tex.parent

    def _read(p: Path, depth: int) -> str:
        if depth > max_depth or not p.exists():
            return ""
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        def _sub(m):
            name = m.group(1)
            for ext in ("", ".tex", ".TEX"):
                cand = base / (name + ext)
                if cand.exists() and cand.is_file():
                    return _read(cand, depth + 1)
            return ""
        return _INPUT_RE.sub(_sub, text)

    return _read(main_tex, 0)


# Patterns we strip
_BRACE_CMD = re.compile(r"\\([a-zA-Z@]+)\s*\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
_NO_ARG_CMD = re.compile(r"\\([a-zA-Z@]+)\*?")
_BRACES = re.compile(r"[{}]")

# Citation/ref commands we want to KEEP the bare text of (for citation
# preservation). But for WER we'd actually just strip them — they're
# rendered as numbers in the PDF anyway. So strip them out.
_KEEP_TEXT_CMDS = {
    "textbf", "textit", "texttt", "emph", "underline",
    "section", "subsection", "subsubsection", "paragraph", "chapter",
    "title", "author", "abstract",
}


def regex_strip_latex(main_tex: Path) -> str:
    """Pure-python LaTeX → text. Always returns SOMETHING."""
    text = inline_inputs(main_tex)
    if not text:
        return ""

    # 1. Strip block comments
    text = _COMMENT_RE.sub("", text)

    # 2. Drop preamble (everything before \begin{document})
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    # And anything after \end{document}
    m = re.search(r"\\end\{document\}", text)
    if m:
        text = text[:m.start()]

    # 3. Drop noisy environments
    for env in _NOISY_ENVS:
        text = _drop_env(text, env)

    # 4. Drop inline math
    for pat in _INLINE_MATH_RES:
        text = pat.sub(" ", text)

    # 5. Iteratively expand brace commands: \cmd{X} -> X for KEEP cmds,
    #    -> "" for others (citations, refs, labels, etc.)
    for _ in range(8):  # iterate to handle nesting
        def _sub(m):
            cmd, arg = m.group(1), m.group(2)
            if cmd in _KEEP_TEXT_CMDS:
                return arg
            # Strip citation / reference / label / footnote commands entirely
            if cmd in ("cite", "citep", "citet", "ref", "eqref", "label",
                          "footnote", "url", "href", "includegraphics",
                          "bibliography", "bibliographystyle"):
                return " "
            # Default: replace with the argument so we preserve any text
            # (e.g. \textsc{Foo} -> Foo)
            return arg
        new = _BRACE_CMD.sub(_sub, text)
        if new == text:
            break
        text = new

    # 6. Remove any leftover bare \commands
    text = _NO_ARG_CMD.sub(" ", text)

    # 7. Remove leftover braces
    text = _BRACES.sub(" ", text)

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def latex_to_text(main_tex: Path, out: Path,
                   *, pandoc_timeout: int = 60) -> str:
    """Try pandoc (with a short budget), fall back to regex stripper.
    Always writes ``out``."""
    if pandoc_to_text(main_tex, out, timeout=pandoc_timeout):
        return "pandoc"
    text = regex_strip_latex(main_tex)
    out.write_text(text, encoding="utf-8")
    return "regex-stripper"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: latex_text.py main.tex out.txt")
        sys.exit(1)
    main = Path(sys.argv[1])
    out = Path(sys.argv[2])
    method = latex_to_text(main, out)
    print(f"converted via {method}: {out} ({out.stat().st_size} bytes)")

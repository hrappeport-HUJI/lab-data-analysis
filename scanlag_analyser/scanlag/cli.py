"""Command-line helpers."""

from __future__ import annotations

import re


def split_prompt_args(argv: list[str]) -> list[str]:
    """Split arguments that arrived glued together, e.g. from PyCharm's $Prompt$ box, which passes
    'D:\\scans\\exp --redo --detector classical' as one argument, or from a quoted path ending in a
    backslash ("D:\\scans\\exp\\" --redo), where Windows reads \\" as a literal quote.

    An argument is split only before ' --', so folder names with spaces stay whole.
    """
    out = []
    for arg in argv:
        parts = re.split(r"\s+(?=--)", arg.strip())
        if len(parts) == 1:
            out.append(arg)
            continue
        if parts[0] and not parts[0].startswith("--"):
            out.append(parts[0].strip().strip('"').strip())
            parts = parts[1:]
        for option in parts:
            name, _, value = option.partition(" ")
            out.append(name)
            if value.strip():
                out.append(value.strip().strip('"').strip())
    return [a.strip('"') if a.count('"') == 1 else a for a in out]

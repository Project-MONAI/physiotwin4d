---
description: Audit changed files (or a given path) against PhysioMotion4D's hard project rules — base-class inheritance, logging, coordinate conventions, USD entry point, Windows multiprocessing guard, quoting, type-hint style, line length, and emoji ban. Reports violations without auto-fixing.
---

Audit PhysioMotion4D source for hard-rule violations.

$ARGUMENTS

By default, audit every Python file modified since `HEAD`. If $ARGUMENTS names
specific files or directories, audit those instead.

## Determine the file set

```powershell
# Default: changed, non-deleted .py files since HEAD
git diff --diff-filter=d --name-only HEAD -- '*.py'
```

If a path was passed in $ARGUMENTS, expand it to all `.py` files under that
path (recursively). Skip files that no longer exist on disk.

## Rules to check

For each file, read the **entire file** (rules below depend on surrounding
context such as class inheritance), then flag every occurrence of:

### Base class and logging
- [ ] A class that orchestrates workflow / segmentation / registration / USD
      conversion but does **not** inherit from `PhysioMotion4DBase`.
- [ ] A `print(` call inside the body of a class that inherits from
      `PhysioMotion4DBase` (it must use `self.log_info()` / `self.log_debug()`).
      Standalone scripts and helper / data-container classes may use `print()`.

### USD / coordinate conventions
- [ ] An `import` of `physiomotion4d.vtk_to_usd` (or `from ... vtk_to_usd ...`)
      from a file that is **not** `src/physiomotion4d/convert_vtk_to_usd.py`
      and is **not** itself inside `src/physiomotion4d/vtk_to_usd/`.
      Experiments, CLIs, tests, and tutorials must use `ConvertVTKToUSD`.
- [ ] A docstring or comment claiming PyVista surfaces are in **RAS** — they
      are in **LPS** internally; convert to USD Y-up only at export.

### Windows multiprocessing
- [ ] A module-level instantiation of `SegmentChestTotalSegmentator` (or a
      module-level call into it) that is not guarded by
      `if __name__ == "__main__":`. Required on Windows because
      `torch.multiprocessing` re-imports the module in child workers.

### Code style
- [ ] `X | None` in a type hint (use `Optional[X]`; ruff `UP007` is suppressed).
- [ ] `Any` in a public signature without a comment explaining why.
- [ ] A docstring delimited with `'''` (use `"""`).
- [ ] A string literal using `"..."` for ordinary inline strings (use `'...'`;
      docstrings stay on `"""`).
- [ ] A line longer than 88 characters.
- [ ] An emoji or other non-ASCII glyph inside a `.py` file (Windows cp1252
      encoding has broken builds; keep emojis out of source).

### Public API hygiene
- [ ] A public method (no leading underscore) without a NumPy-style docstring.
- [ ] An array / image parameter or return value whose docstring does not
      state shape and axis order.

## Output

Group findings by file. For each finding, print:

```text
<path>:<line>  <rule short name>  <one-line excerpt>
```

End with a one-line summary: total findings per rule category.

Do **not** auto-fix. The point is to surface violations the user can decide
how to address. If `$ARGUMENTS` includes `--fix`, ask before mutating anything
and limit fixes to the trivially mechanical rules (line length is not one of
them — `ruff format` covers that).

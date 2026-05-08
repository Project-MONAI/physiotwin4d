---
description: Inspect changed PhysioMotion4D code and existing docstrings, update docstrings and inline comments with accurate shape/axis information, and regenerate docs/API_MAP.md if public APIs changed.
---

Update documentation for the following in PhysioMotion4D:

$ARGUMENTS

Instructions:
1. Read the changed source file(s) in full.
2. Read existing docstrings for every public method or class that changed.
3. Update docstrings to reflect current behavior using NumPy docstring style.
   State image/tensor shape and axis order wherever arrays are involved.
4. Add inline comments only for non-obvious logic (coordinate transforms, shape permutations).
5. Do not create new `.md` files unless explicitly asked.
6. If any public class, method, or function signature changed, regenerate the API map:
   `python utils/generate_api_map.py`
7. Do not paraphrase the method name as the docstring — explain what it does and why.

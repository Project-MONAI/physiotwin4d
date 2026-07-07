---
description: Regenerate docs/API_MAP.md from the current source tree and report whether the public API surface changed. Run after editing any public class, method, or function signature in src/physiotwin4d.
---

Regenerate the PhysioTwin4D API map and report what changed.

$ARGUMENTS

Instructions:

1. Run the generator from the active `.\venv` (activate it first if needed,
   or call `.\venv\Scripts\python.exe` directly):

   ```powershell
   python utils/generate_api_map.py
   ```

2. Inspect the diff:

   ```powershell
   git diff -- docs/API_MAP.md
   ```

3. Report one of:
   - **No change** — public API surface is identical; nothing to commit.
   - **Changed** — list each added, removed, or signature-modified entry
     (class.method, file, one-line summary). Group by file.

4. If `docs/API_MAP.md` changed, remind the user to stage it together with
   the source changes that produced it so the API map never lags the code.

5. Do not edit `docs/API_MAP.md` by hand. If the generator output looks wrong,
   investigate `utils/generate_api_map.py` and the source it scans, not the
   generated file.

6. Do not commit. The user controls when to stage and commit.

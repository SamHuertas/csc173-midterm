# Setup for the scoring step (SummaC, AlignScore, ROUGE-L)

Use a **separate virtual environment**, `.venv-scoring`. Your main `.venv` is never touched.

**Why:** SummaC and AlignScore declare version pins that clash with the main environment (`huggingface-hub<=0.17.0` for SummaC, `torch<2` and `numpy<2` for AlignScore). Installing them normally either fails or downgrades packages and breaks `transformers`. Installing them with `--no-deps` into their own environment avoids both.

## Steps (Windows; macOS / Linux in comments)
Python 3.13 is needed (see step 1 of `SETUP.md`). Skip the GPU step there. Then, from the repo folder:

```
deactivate                                  # leave .venv if its prompt is showing
py -3.13 -m venv .venv-scoring              # macOS / Linux: python3.13 -m venv .venv-scoring
.venv-scoring\Scripts\activate              # macOS / Linux: source .venv-scoring/bin/activate
pip install -r requirements-scoring.txt
pip install summac --no-deps
pip install https://github.com/yuh-zha/AlignScore/archive/refs/heads/main.zip --no-deps
python scripts/patch_summac.py
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
The prompt must start with `(.venv-scoring)`. `--no-deps` is the part that prevents the conflicts: the two packages go in without their pins, and their real dependencies are already in `requirements-scoring.txt`.

## SummaC compatibility patch

`scripts/patch_summac.py` fixes two tokenizer calls in the installed SummaC 0.0.4 package for compatibility with Transformers 4.x. It does not change model weights or evaluation data.

### What to do

1. After installing SummaC, run from the project root:

   ```powershell
   .venv-scoring\Scripts\python.exe scripts\patch_summac.py
   ```

2. Restart the notebook kernel if SummaC was already imported.
3. Run the Quick test below to verify scoring works.

Reapply the patch after reinstalling SummaC or recreating the scoring environment. It is not needed before every run.

### Expected messages and safeguards

| Message | Action |
|---|---|
| `Patched both SummaC tokenizer calls for Transformers 4.x.` | Patch applied successfully. |
| `SummaC compatibility patch is already applied.` | Nothing else to patch; repeating the command is safe. |
| `Run this script with .venv-scoring/Scripts/python.exe` | Use the explicit interpreter command above. |
| `This patch supports SummaC 0.0.4 only; review newer versions first.` | Check the installed version; do not force the patch. |
| `Unexpected SummaC source; no changes made.` | Inspect the installation before continuing. |

The patch only supports SummaC 0.0.4 and the expected source layout. It edits the installed package without creating a backup. A successful patch message does not verify model inference or CUDA; use the Quick test.

**AlignScore checkpoint** (about 1.83 GiB, keep it out of git): download
https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt
into a `checkpoints/` folder in the repo.

**Notebooks:** pick the `.venv-scoring` kernel for the scoring notebooks, and `.venv` for notebooks 01-04.

## Quick test
```python
import torch, transformers
transformers.AdamW = torch.optim.AdamW        # AlignScore imports it; recent transformers removed it
from alignscore import AlignScore
from summac.model_summac import SummaCZS

abstract = "Paste an abstract here."
summary = "Paste a summary here."

align = AlignScore(model="roberta-base", batch_size=16, device="cpu",
                   ckpt_path="checkpoints/AlignScore-base.ckpt", evaluation_mode="nli_sp")
print("AlignScore:", align.score(contexts=[abstract], claims=[summary]))

summac = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")   # "cuda" if you have a GPU
print("SummaC:", summac.score([abstract], [summary])["scores"])
```

## Rules
- Never run `pip install summac` or `pip install alignscore` without `--no-deps`, and never in `.venv`.
- Close VS Code and stop notebook kernels before any `pip install`, then restart the kernel afterwards.
- `requirements-scoring.txt` pins `pytorch-lightning<2` because AlignScore calls `load_from_checkpoint` on an instance, which Lightning 2.x rejects. It also pins `setuptools<81` because Lightning 1.x needs `pkg_resources`. Keep these pins in `.venv-scoring` only.

## Status
Validated on Windows with Python 3.13.17, torch 2.15.0 (CPU), Transformers 4.57.6, Lightning 1.9.5, setuptools 80.10.2, and the SummaC patch above. Imports, ROUGE-L, spaCy, NLTK, and one-pair AlignScore/SummaC inference passed, including a repeat with Hugging Face offline mode after caching the models. This is a runtime smoke test, not a validation of metric quality. `pip check` still reports the legacy dependency pins intentionally bypassed by the two `--no-deps` installs.


## RTX 5050 local GPU setup (9 November 2026)

The scoring environment uses PyTorch 2.11.0+cu128 through a directory junction:
`.venv-scoring/Lib/site-packages/torch` points to `.venv/Lib/site-packages/torch`.
This reuses the existing CUDA installation because unpacking another wheel failed
with insufficient disk space. Other scoring dependencies remain in `.venv-scoring`;
SummaC and AlignScore were not installed into the main environment.

The previous CPU package and metadata remain as `torch_cpu_backup` and
`torch_cpu_metadata_backup` inside the scoring environment's site-packages directory.
The CUDA distribution metadata was copied into the scoring environment. This is a
local workaround, not a portable standalone environment: keep the main `.venv` at
its current location. Do not upgrade/uninstall torch in either environment while
using this shared link; a package manager can modify the shared files.
When disk space permits, replace this arrangement with an independent CUDA installation.

The script now requires CUDA for scoring, prints the GPU name, and uses
`outputs/faithfulness/run_01/`. The previous CPU outputs were deleted at the user?s request.
Restart the notebook kernel before running 06 so it imports the new PyTorch and script.

GPU validation status: CUDA detection and matrix multiplication passed. Full metric
validation has not passed: AlignScore checkpoint loading exhausted system memory;
a memory-mapped diagnostic also failed. Only about 0.32 GiB disk space remained.
Shut down old notebook kernels and free disk space before retesting. No full GPU
scoring run was started, and no CPU results were removed.

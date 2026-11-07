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
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
The prompt must start with `(.venv-scoring)`. `--no-deps` is the part that prevents the conflicts: the two packages go in without their pins, and their real dependencies are already in `requirements-scoring.txt`.

**AlignScore checkpoint** (about 1 GB, keep it out of git): download
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
- If AlignScore fails to load its checkpoint with `pytorch-lightning` 2.x, try `pip install "pytorch-lightning<2"` in `.venv-scoring` only.

## Status
The two `--no-deps` installs were run in a clean test environment and completed. The imports and the quick test above were not run (no GPU, torch or model downloads were available), so run it once before relying on either metric.

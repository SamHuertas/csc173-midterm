# Setup

## 1. Install Python 3.13
This project uses **Python 3.13**. Check what you have:
```
py -3.13 --version
```
If it prints `Python 3.13.x`, go to step 2. If not, install it with winget (Windows), or download it from https://www.python.org/downloads/ and tick "Add to PATH":
```
winget install -e --id Python.Python.3.13
```
Close and reopen the terminal afterwards so the new Python is found.

## 2. Create the virtual environment
In a terminal in the repo folder:
```
py -3.13 -m venv .venv             # Windows   (macOS / Linux: python3.13 -m venv .venv)
.venv\Scripts\activate             # Windows   (macOS / Linux: source .venv/bin/activate)
python --version                   # must print Python 3.13.x
```
The prompt should now start with `(.venv)`. Run every later command in this terminal.

## 3. NVIDIA GPU and torch: **NOT NEEDED, skip to step 4**

**Do not install anything for this step.** The model is already trained and uploaded (see "The trained model" below), so nobody trains it again. Notebooks 01, 03 and 04 run fine on a CPU, and a GPU only makes generation faster.

<details>
<summary>Only if you still want your GPU used</summary>

1. Run `nvidia-smi`. If it prints a table, note the GPU name.
2. Install the matching torch build **before step 4**:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/<tag>
   ```
   | Your GPU | `<tag>` |
   |---|---|
   | GTX 10-series, RTX 20 / 30 / 40-series | `cu126` |
   | RTX 50-series | `cu128` |
3. Check: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
</details>

## 4. Install the packages
```
pip install -r requirements.txt
```
Do not add `--upgrade`: it can pull in `transformers` 5.x, which this project does not support.

## 5. Check it works
```
python -c "import torch, transformers; from transformers import AutoModelForSeq2SeqLM; print(torch.cuda.is_available(), transformers.__version__)"
```
You want no error and a `transformers` version starting with `4.`. `False` just means CPU only, which is fine.

## 6. Open the notebooks
In VS Code, open a notebook in `notebooks/`, click the kernel picker (top right) and choose the `.venv` interpreter.

---

## The trained model

**https://huggingface.co/SamHuertas/SciTLDR-trained-BART**

The BART model was fine-tuned on the SciTLDR training abstracts and uploaded there, so **there is no need to retrain it**. Notebook 02 is only for retraining. Notebooks 03 and 04 use `outputs/model/` if that folder exists, and otherwise download this model automatically (about 560 MB on first use, then cached).

To use it in your own code, for example to produce several candidate summaries to re-rank:
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "SamHuertas/SciTLDR-trained-BART"        # or "outputs/model" if you have the folder
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()

abstract = "Paste an abstract here."
batch = tok(abstract, max_length=512, truncation=True, return_tensors="pt")
with torch.inference_mode():
    ids = model.generate(**batch, num_beams=5, num_return_sequences=5, max_new_tokens=64)
candidates = tok.batch_decode(ids, skip_special_tokens=True)   # 5 candidate TLDRs
```
Beam search is only an example. Choose how the candidates are generated (beams or sampling) before running, and keep it fixed.
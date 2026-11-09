# Running faithfulness evaluation on CPU

CUDA is not required by the research method. The current notebooks 06 and 08 explicitly require it as a local performance choice. This guide explains how to change those notebooks to CPU execution without overwriting the GPU experiment.

These are instructions only: adding this document does not change notebook code, installed packages, or saved results.

## 1. Decide what you need to reproduce

| Goal | Work needed |
|---|---|
| Inspect results or recompute gains, bootstrap intervals, and ROUGE-L | Use the completed run files with notebook 09; no CUDA or metric inference is needed |
| Recompute baseline and candidate faithfulness scores | Follow the CPU changes for 06 and 08 below; reuse the same abstract/summary inputs |
| Regenerate candidates as well | Also configure notebook 07 for CPU; this is a separate, potentially slow generation step |

To reproduce scoring of the same summaries, reuse `evaluation_pairs.csv` and `generated_candidates.csv`. Regeneration is not necessary and can introduce different candidate text across software/hardware configurations.

## 2. Set up the scoring environment

On a teammate's machine, create a fresh `.venv-scoring` using the installation steps in [SETUP_SCORING.md](SETUP_SCORING.md). Those steps install the scoring requirements, SummaC/AlignScore with `--no-deps`, the SummaC tokenizer patch, and language resources.

An existing CUDA-enabled PyTorch installation can also execute on CPU: choosing `device='cpu'` is sufficient. You do not have to uninstall CUDA just to run these notebooks on CPU. On a CPU-only machine, a compatible CPU build of PyTorch is sufficient.

Select `.venv-scoring` as the kernel for notebooks 05, 06, 08, and 09. A terminal environment activation does not change an already running notebook kernel.

The current author's scoring environment uses a machine-local directory link to another environment's torch package. Do not copy that virtual environment to a teammate's machine or modify torch through that link. Create their environment from the setup instructions instead.

Verify the selected interpreter with:

```python
import sys
import torch

print('Python:', sys.executable)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
```

`CUDA available: False` is expected and acceptable on a CPU-only machine. GPU availability does not determine the device once the function explicitly requests CPU.

## 3. Use a separate output folder

In the setup cell of each notebook you will use, replace:

```python
RUN = ROOT / 'outputs' / 'faithfulness' / 'run_01'
```

with:

```python
RUN = ROOT / 'outputs' / 'faithfulness' / 'run_cpu_01'
```

Use the same path in all stages of this CPU experiment. Keep `run_01` unchanged as the existing GPU experiment.

For scoring the existing summaries, create `outputs/faithfulness/run_cpu_01/` and copy only these input files from the completed run:

- `evaluation_pairs.csv`
- `generated_candidates.csv`

Do not copy score tables, metric/score manifests, or analysis outputs into the new folder. The CPU run must calculate its own scores and settings records. Downloaded model files and the AlignScore checkpoint can be reused; they are not tied to the output folder.

Alternatively, notebook 05 can prepare evaluation pairs directly from the original baseline, using the new `RUN`. Candidate generation is still unnecessary when reusing `generated_candidates.csv`.

## 4. Change `run_scoring()` in BOTH 06 and 08

Each notebook contains its own function definition. Apply the same edits in both notebooks.

### Remove the CUDA requirement and GPU-name query

Find this block inside `run_scoring()`:

```python
    require(torch.cuda.is_available(),
            'CUDA is unavailable in this kernel. Install CUDA-enabled PyTorch in .venv-scoring and restart.')
    print('Scoring GPU:', torch.cuda.get_device_name(0), flush=True)
```

Replace it with:

```python
    print('Scoring device: CPU', flush=True)
```

Removing only the requirement is insufficient: `torch.cuda.get_device_name(0)` would still attempt a GPU query on a CPU-only machine.

### Set the actual scoring device

Change the first line of `PROTOCOL` from:

```python
        'device': 'cuda', 'outer_batch': 8, 'internal_batch': 8,
```

to:

```python
        'device': 'cpu', 'outer_batch': 8, 'internal_batch': 8,
```

Both metric constructors already receive `device=PROTOCOL['device']`. This change therefore places SummaC and AlignScore on CPU. No change to the constructor device arguments is required.

Keep the existing scientific settings: SummaC entailment-only configuration, AlignScore model/checkpoint and evaluation mode, input limits, and summary inputs. Do not change these just to switch devices.

Start with the existing batch sizes of eight. If RAM is insufficient, reduce `outer_batch` and `internal_batch` consistently in both notebooks and record the change. Smaller batches may reduce temporary inference memory, but cannot eliminate the RAM needed to load model weights. CPU execution can be substantially slower.

## 5. Run the CPU scoring stages

After editing, restart the kernel and run each relevant notebook's setup, shared helper definitions, main function definition, and execution cells.

1. Run notebook 06 against `run_cpu_01` to produce `baseline_scores.csv`.
2. Run notebook 08 against the same folder to produce `candidate_scores.csv`.
3. Run notebook 09 against that folder to calculate selected summaries, paired changes, ROUGE-L, intervals, and plots.

Expected complete counts remain 618 baseline rows and 3,090 candidate rows. The notebook should print `Scoring device: CPU`. No CUDA error or GPU-name query should occur.

The first scoring call can resolve/download metric files before switching subsequent model loading to offline mode. CPU execution does not remove download requirements: an offline run requires all needed model/tokenizer files and language resources to be cached first. Keep `checkpoints/AlignScore-base.ckpt` available.

Do not edit a GPU run's manifest to force CPU scores into it. `Protocol changed` is the expected protection when device/settings differ. Use the new folder. Interrupted CPU scoring can resume from its own saved batches with unchanged inputs/settings.

## 6. Optional: regenerate candidates on CPU

This is only needed to repeat generation rather than score the existing summaries.

Notebook 07 uses the separate main `.venv` kernel. Its current device choice is:

```python
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

On a machine without CUDA this already selects CPU. To force CPU even if a GPU is present, replace that line with:

```python
    device = 'cpu'
```

Set this notebook's `RUN` to the same CPU run folder, and use the intended trained model/tokenizer and revision. Keep the existing seed and decoding settings: ten beams, ten returned outputs, and the first five distinct normalized summaries retained. No backup-generation policy change is needed for CPU.

The `CUBLAS_WORKSPACE_CONFIG` environment setting is for CUDA operations; leaving it in the execution cell is harmless for CPU and it does not enable CUDA. The existing cuDNN configuration assignments do not move computation to GPU either; the chosen device controls that.

Start from evaluation pairs without incompatible previous generation files/manifests. Do not overwrite an existing candidate input after scoring it. Generation can be particularly slow on CPU; sharing the completed candidate CSV is usually preferable for reproducing the scoring and analysis.

## 7. What reproducibility does and does not mean here

Use the same model checkpoints/revisions, input CSV bytes, metric settings, tokenizer resources, and software versions where possible. Record the Python version and installed packages for the CPU experiment, for example in a terminal using the scoring environment:

```powershell
.venv-scoring\Scripts\python.exe --version
.venv-scoring\Scripts\python.exe -m pip freeze > outputs\faithfulness\run_cpu_01\scoring_packages.txt
```

The folder must exist before redirecting that inventory. The requirements file contains ranges and the setup downloads a moving AlignScore branch, so installing the requirements alone does not recreate an exact historical environment. For strict repeatability, also retain resolved package versions, metric/model revisions, and checkpoint identity.

CPU and GPU floating-point computations can produce slightly different scores; matching results need not mean bit-for-bit equality. Small differences can also change a winner when candidate scores are very close. Regenerating candidates can introduce further differences. The seed alone does not guarantee cross-device identity.

CPU inference still needs substantial RAM and adequate Windows pagefile capacity. Error 1455 or an allocation failure is a system-memory problem that CPU execution does not automatically resolve. Keep enough disk space for models, temporary downloads, and virtual memory.

If a teammate only needs Member 3's analysis on the already computed scores, they can use notebook 09 with the completed run and its manifests. That calculation does not require recomputing neural scores or changing notebooks 06 and 08.

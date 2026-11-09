from importlib.metadata import distribution
from pathlib import Path
import sys


def main():
    expected = Path(__file__).resolve().parents[1] / ".venv-scoring"
    if Path(sys.prefix).resolve() != expected:
        raise SystemExit("Run this script with .venv-scoring/Scripts/python.exe")
    package = distribution("summac")
    if package.version != "0.0.4":
        raise SystemExit("This patch supports SummaC 0.0.4 only; review newer versions first.")
    target = Path(package.locate_file("summac/model_summac.py"))
    original = target.read_text(encoding="utf-8")
    old = 'padding=True, truncation=True, max_length=self.max_input_length, return_tensors="pt", truncation_strategy="only_first"'
    new = 'padding=True, truncation="only_first", max_length=self.max_input_length, return_tensors="pt"'
    if original.count(old) == 2:
        target.write_text(original.replace(old, new), encoding="utf-8")
        print("Patched both SummaC tokenizer calls for Transformers 4.x.")
    elif old not in original and original.count(new) == 2:
        print("SummaC compatibility patch is already applied.")
    else:
        raise SystemExit("Unexpected SummaC source; no changes made.")


if __name__ == "__main__":
    main()

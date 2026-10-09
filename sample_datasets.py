"""
sample_datasets.py

Étape 1 : récupérer un petit échantillon de chaque dataset candidat (streaming,
rien n'est téléchargé en entier) et produire un rapport de qualité.

Usage :
    python sample_datasets.py                     # tous les datasets, 200 exemples
    python sample_datasets.py fineweb2_fr 500     # un seul dataset, 500 exemples

Sorties (dossier samples/) :
    <nom>.jsonl   exemples nettoyés qui passent le filtre
    report.md     taux de rejet par raison, longueur moyenne, 3 extraits à lire
"""

import json
import os
import sys
from collections import Counter

from datasets import load_dataset

from dataset_utils import CATALOG, process

OUT_DIR = "samples"


def sample_one(name: str, spec: dict, n: int) -> dict:
    kwargs = {"split": spec["split"], "streaming": True}
    if spec["config"]:
        ds = load_dataset(spec["path"], spec["config"], **kwargs)
    else:
        ds = load_dataset(spec["path"], **kwargs)

    kept, reasons, lengths = [], Counter(), []
    seen = 0
    for example in ds:
        seen += 1
        text, ok, reason = process(example, spec)
        reasons[reason] += 1
        if ok:
            kept.append(text)
            lengths.append(len(text.split()))
        if seen >= n:
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for t in kept:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    return {
        "seen": seen,
        "kept": len(kept),
        "reasons": reasons,
        "avg_words": sum(lengths) / len(lengths) if lengths else 0,
        "extracts": kept[:3],
    }


def main():
    names = list(CATALOG)
    n = 200
    if len(sys.argv) >= 2:
        names = [sys.argv[1]]
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])

    lines = ["# Rapport d'échantillonnage\n"]
    for name in names:
        spec = CATALOG[name]
        print(f"[{name}] échantillonnage de {n} exemples...")
        try:
            r = sample_one(name, spec, n)
        except Exception as e:  # dataset inaccessible, config incorrecte, etc.
            print(f"  ERREUR : {e}")
            lines.append(f"## {name}\n\nERREUR : {e}\n")
            continue

        rate = 100 * r["kept"] / max(r["seen"], 1)
        print(f"  gardés : {r['kept']}/{r['seen']} ({rate:.0f}%)")
        lines.append(f"## {name}\n")
        lines.append(f"- Source : `{spec['path']}` ({spec['note']})")
        lines.append(f"- Licence : {spec['license']}")
        lines.append(f"- Gardés : {r['kept']}/{r['seen']} ({rate:.0f}%)")
        lines.append(f"- Longueur moyenne : {r['avg_words']:.0f} mots")
        lines.append(f"- Raisons : {dict(r['reasons'])}\n")
        for i, ex in enumerate(r["extracts"], 1):
            lines.append(f"**Extrait {i}**\n\n> " + ex[:500].replace("\n", "\n> ") + "\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nRapport : {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()

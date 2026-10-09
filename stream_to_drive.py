"""
stream_to_drive.py

Étape 2 : une fois un dataset validé par l'échantillonnage, le récupérer en
streaming, le nettoyer et l'écrire par fichiers (shards) directement dans un
dossier de sortie, sans jamais stocker le dataset brut sur le PC.

Conçu pour Google Colab, où Drive se monte comme un dossier :

    from google.colab import drive
    drive.mount("/content/drive")
    !python stream_to_drive.py fineweb2_fr /content/drive/MyDrive/llm_data 200000

Usage :
    python stream_to_drive.py <dataset> <dossier_sortie> <nb_documents_cibles>

Chaque shard contient au plus SHARD_SIZE documents et environ SHARD_MAX_BYTES
octets (dépassement d'un seul document au maximum) : ainsi aucun fichier
n'approche la limite de 100 Mo de GitHub.

Reprise : si le script est relancé, il repart du dernier shard complet
(état dans state.json). Les shards sont écrits en fichier temporaire puis
renommés, donc jamais à moitié écrits.
"""

import json
import os
import sys

from datasets import load_dataset

from dataset_utils import CATALOG, process

SHARD_SIZE = 10_000  # documents maximum par fichier
SHARD_MAX_BYTES = 50 * 1024 * 1024  # 50 Mo maximum par fichier


def load_state(out_dir: str) -> dict:
    path = os.path.join(out_dir, "state.json")
    if not os.path.exists(path):
        return {"shards_done": 0, "examples_seen": 0, "kept_total": 0}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(out_dir: str, state: dict):
    tmp = os.path.join(out_dir, "state.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join(out_dir, "state.json"))


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    name, base_dir, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
    spec = CATALOG[name]
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    state = load_state(out_dir)
    print(
        f"[{name}] reprise : {state['shards_done']} shards, "
        f"{state['kept_total']} documents, {state['examples_seen']} exemples lus"
    )
    if state["kept_total"] >= target:
        print("Cible déjà atteinte.")
        return

    kwargs = {"split": spec["split"], "streaming": True}
    ds = (
        load_dataset(spec["path"], spec["config"], **kwargs)
        if spec["config"]
        else load_dataset(spec["path"], **kwargs)
    )
    if state["examples_seen"]:
        ds = ds.skip(state["examples_seen"])

    seen = state["examples_seen"]
    buffer: list[str] = []
    buffer_bytes = 0

    def flush():
        nonlocal buffer, buffer_bytes
        idx = state["shards_done"]
        path = os.path.join(out_dir, f"{name}_{idx:05d}.jsonl")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for t in buffer:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        state["shards_done"] += 1
        state["kept_total"] += len(buffer)
        state["examples_seen"] = seen
        save_state(out_dir, state)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(
            f"  shard {state['shards_done']} écrit ({size_mb:.1f} Mo) "
            f"- {state['kept_total']} documents au total"
        )
        buffer, buffer_bytes = [], 0

    for example in ds:
        seen += 1
        text, ok, _ = process(example, spec)
        if not ok:
            continue
        buffer.append(text)
        buffer_bytes += len(text.encode("utf-8")) + 16  # +16 : syntaxe JSON
        if len(buffer) >= SHARD_SIZE or buffer_bytes >= SHARD_MAX_BYTES:
            flush()
            if state["kept_total"] >= target:
                break

    if buffer:
        flush()

    print(f"Terminé : {state['shards_done']} shards dans {out_dir}")


if __name__ == "__main__":
    main()

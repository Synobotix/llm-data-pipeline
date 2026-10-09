"""
dataset_utils.py

Catalogue des datasets candidats + fonctions communes (extraction du texte,
nettoyage léger, filtre qualité). Utilisé par sample_datasets.py et
stream_to_drive.py.
"""

import re

# ---------------------------------------------------------------------------
# Catalogue des datasets candidats (Hugging Face)
# kind = "text"         -> un champ texte par exemple
# kind = "conversation" -> une liste de tours de parole
# ---------------------------------------------------------------------------
CATALOG = {
    "fineweb2_fr": {
        "path": "HuggingFaceFW/fineweb-2",
        "config": "fra_Latn",
        "split": "train",
        "kind": "text",
        "field": "text",
        "license": "ODC-By 1.0",
        "note": "Web français filtré et dédoublonné (CommonCrawl)",
    },
    "wikipedia_fr": {
        "path": "wikimedia/wikipedia",
        "config": "20231101.fr",
        "split": "train",
        "kind": "text",
        "field": "text",
        "license": "CC-BY-SA 3.0 / GFDL",
        "note": "Articles Wikipédia français (dump 2023-11-01)",
    },
    "fineweb_edu_en": {
        "path": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "kind": "text",
        "field": "text",
        "license": "ODC-By 1.0",
        "note": "Web anglais à contenu éducatif (échantillon 10 milliards de tokens)",
    },
    "french_instruct": {
        "path": "angeluriot/french_instruct",
        "config": None,
        "split": "train",
        "kind": "conversation",
        "field": "conversation",
        "license": "MIT",
        "note": "Conversations utilisateur/assistant en français (~276K)",
    },
}

# ---------------------------------------------------------------------------
# Extraction du texte
# ---------------------------------------------------------------------------


def extract_text(example: dict, spec: dict) -> str:
    """Retourne le texte brut d'un exemple, selon le type du dataset."""
    value = example.get(spec["field"])
    if value is None:
        return ""

    if spec["kind"] == "text":
        return str(value)

    # Conversation : liste de tours. Le nom exact des clés n'est pas garanti,
    # on lit "text" et, si présent, "role"/"speaker"; sinon on alterne.
    lines = []
    for i, turn in enumerate(value):
        if isinstance(turn, dict):
            text = turn.get("text") or turn.get("content") or ""
            role = turn.get("role") or turn.get("speaker")
        else:
            text, role = str(turn), None
        if not role:
            role = "Utilisateur" if i % 2 == 0 else "Assistant"
        if text.strip():
            lines.append(f"{role} : {text.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nettoyage léger + filtre qualité
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def quality_check(
    text: str,
    min_words: int = 50,
    max_words: int = 20_000,
    min_alpha_ratio: float = 0.6,
    max_dup_4gram_ratio: float = 0.3,
) -> tuple[bool, str]:
    """
    Filtre qualité simple. Retourne (ok, raison).

    La répétition est mesurée par la part de 4-grammes de mots dupliqués
    (et non par le ratio de mots uniques, qui pénalise les longs documents).
    """
    if not text:
        return False, "vide"

    words = text.split()
    if len(words) < min_words:
        return False, "trop_court"
    if len(words) > max_words:
        return False, "trop_long"

    alpha = sum(1 for c in text if c.isalpha())
    if alpha / len(text) < min_alpha_ratio:
        return False, "ratio_alpha"

    n4 = len(words) - 3
    if n4 > 0:
        grams = {tuple(words[i : i + 4]) for i in range(n4)}
        if 1 - len(grams) / n4 > max_dup_4gram_ratio:
            return False, "repetition"

    return True, "ok"

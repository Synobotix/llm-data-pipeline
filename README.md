# llm-data-pipeline

Récupération de datasets de qualité (textes complets, conversations) depuis Hugging Face pour entraîner le LLM du projet. Deux étapes : **valider sur un petit échantillon**, puis **télécharger en masse en streaming**, sans stocker les données brutes sur le PC.

## Contenu du dépôt

```
llm-data-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_utils.py       catalogue des datasets, nettoyage, filtre qualité
├── sample_datasets.py     étape 1 : échantillon + rapport de qualité
├── stream_to_drive.py     étape 2 : streaming en masse, par fichiers
└── samples/               sorties de l'étape 1 (rapport suivi par Git)
```

## Datasets du catalogue

| Nom | Source Hugging Face | Contenu | Licence |
|---|---|---|---|
| `wikipedia_fr` | `wikimedia/wikipedia`, config `20231101.fr` | Articles Wikipédia, phrases complètes | CC-BY-SA 3.0 / GFDL |
| `fineweb2_fr` | `HuggingFaceFW/fineweb-2`, config `fra_Latn` | Web français filtré et dédoublonné | ODC-By 1.0 |
| `french_instruct` | `angeluriot/french_instruct` | ~276 000 conversations utilisateur/assistant | MIT |
| `fineweb_edu_en` | `HuggingFaceFW/fineweb-edu`, config `sample-10BT` | Web anglais à contenu éducatif | ODC-By 1.0 |

Ajouter un dataset : une entrée de plus dans `CATALOG` (`dataset_utils.py`) avec `path`, `config`, `split`, `kind` (`text` ou `conversation`) et `field`.

## Utilisation

### 0. Installation

```
pip install -r requirements.txt
```

### 1. Valider un dataset sur un échantillon

```
python sample_datasets.py                    # tous les datasets, 200 exemples chacun
python sample_datasets.py wikipedia_fr 500   # un seul dataset, 500 exemples
```

Produit `samples/report.md` (taux de rejet par raison, longueur moyenne, 3 extraits) et `samples/<nom>.jsonl`. À lire avant de continuer : les extraits sont-ils des phrases complètes, dans la bonne langue, sans bruit de navigation ou de publicité ? Le taux de rejet est-il raisonnable ?

### 2. Télécharger en masse (Colab + Google Drive)

Dans un notebook Colab :

```python
from google.colab import drive
drive.mount("/content/drive")

!pip install -q datasets
!python stream_to_drive.py wikipedia_fr /content/drive/MyDrive/llm_data 200000
```

Arguments : `<dataset> <dossier_sortie> <nombre de documents à garder>`.

- Les données arrivent par flux (streaming) : rien n'est téléchargé en entier.
- Sortie : `<dossier>/<dataset>/<dataset>_00000.jsonl`, `_00001.jsonl`, etc.
- Chaque fichier contient au plus **10 000 documents** et environ **50 Mo** (dépassement d'un document au maximum), pour rester loin de la limite de 100 Mo de GitHub.
- **Reprise automatique** : relancer la même commande repart du dernier fichier complet (état dans `state.json`).

## Format de sortie

Une ligne JSON par document : `{"text": "..."}`. C'est le même format que `train.jsonl` du pipeline principal, donc ces fichiers sont utilisables tels quels par `train_tokenizer.py` et `tokenize_dataset.py`. Les conversations sont aplaties en `Utilisateur : ...` / `Assistant : ...`.

## Nettoyage et filtre qualité

Nettoyage : suppression du HTML, des URLs, des emails, normalisation des espaces.

Un document est rejeté si :

| Raison | Règle |
|---|---|
| `trop_court` | moins de 50 mots |
| `trop_long` | plus de 20 000 mots |
| `ratio_alpha` | moins de 60 % de caractères alphabétiques |
| `repetition` | plus de 30 % de séquences de 4 mots dupliquées |

Filtres propres à certains datasets (options du `CATALOG`) :

| Dataset | Option | Effet |
|---|---|---|
| `fineweb2_fr` | `web_cleanup` | supprime les lignes de moins de 4 mots (menus, titres isolés) et les lignes répétées dans un document |
| `french_instruct` | `exclude_regex` | écarte les puzzles de logique traduits (variables `x_8`, `x_12`…) ; les rôles `user`/`assistant` deviennent `Utilisateur`/`Assistant` |

Les seuils sont des arguments de `quality_check` dans `dataset_utils.py`.

## Où stocker quoi

| Emplacement | À y mettre | Pourquoi |
|---|---|---|
| **Google Drive** | Les datasets en masse | 15 Go gratuits, se monte directement dans Colab |
| **GitHub** | Scripts, documentation, `samples/report.md`, éventuellement de petits fichiers nettoyés | Voir les limites ci-dessous |

GitHub n'est pas illimité :

- un fichier de plus de **100 Mo** est refusé au push (d'où le plafond de 50 Mo par fichier) ;
- GitHub recommande de garder un dépôt sous **1 Go**, et fortement sous **5 Go** ;
- avec Git LFS, la limite par fichier est de 2 Go sur un compte gratuit.

`data/` et `output/` sont dans `.gitignore`. Pour versionner volontairement un fichier nettoyé : `git add -f chemin/du/fichier.jsonl`.

## Licences

- **Wikipédia (CC-BY-SA)** : attribution obligatoire et partage dans les mêmes conditions.
- **FineWeb / FineWeb-2 (ODC-By)** : attribution obligatoire. FineWeb-2 est issu de CommonCrawl et soumis à ses conditions d'utilisation.
- **French Instruct (MIT)** : mélange de sources, dont des traductions faites avec l'API ChatGPT.

Avant de publier des données nettoyées dans un dépôt public, vérifier ces conditions. Un dépôt **privé** évite la question pour un usage interne.

## État des vérifications

Vérifié hors-ligne, avec un faux dataset de streaming :

- reprise après interruption sans doublon ni document perdu ;
- plafond de taille par fichier (respecté à un document près) ;
- filtre qualité sur 579 documents C4 réels (526 conservés) ;
- extraction des conversations.

**Non vérifié** : l'accès réel à Hugging Face (non joignable depuis l'environnement de développement). Le premier lancement de `sample_datasets.py` le confirmera, notamment pour la structure de `french_instruct` (champ `conversation`, tours avec une clé `text`).

## Suite prévue

1. ~~Valider les datasets avec `samples/report.md`~~ : fait, filtres ajoutés pour FineWeb-2 et French Instruct (relancer `sample_datasets.py` pour les vérifier sur de vrais exemples).
2. Télécharger les datasets retenus vers Drive.
3. Entraîner le tokenizer BPE sur ces données, puis relancer l'entraînement du Student.

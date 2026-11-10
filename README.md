# Anticipez les besoins en consommation de bâtiments

Prédiction de la **consommation énergétique** et des **émissions de CO₂** des bâtiments non
destinés à l'habitation de la ville de Seattle, à partir des relevés 2016.

> Projet 3 du parcours *Data Scientist / AI Engineer* — Faïza Khelladi

---

## Contexte

Seattle s'est fixé un objectif de neutralité carbone à l'horizon 2050. Pour le piloter, la
ville doit estimer la consommation et les émissions de son parc tertiaire **sans attendre un
relevé annuel coûteux sur chaque bâtiment**. L'enjeu est donc de prédire ces deux grandeurs à
partir des seules **caractéristiques structurelles déclaratives** — surface, nombre d'étages,
année de construction, quartier, types d'usage — avec un modèle par cible.

## Résultats

Modèle retenu pour les deux cibles : **GradientBoostingRegressor**, après recherche
d'hyperparamètres en deux temps (`RandomizedSearchCV` 50 tirages puis `GridSearchCV`
resserrée), validation croisée à 5 plis.

| Cible | R² test | MAE |
|---|---|---|
| `SiteEnergyUseWN(kBtu)` — consommation | **0,735** | 2,47 M kBtu |
| `TotalGHGEmissions` — émissions CO₂ | **0,840** | 55,9 t CO₂e |

> **Ces R² sont optimistes.** Les variables de mix énergétique (`ElectricityPercentage`,
> `NaturalGasPercentage`, `OtherEnergyPercentage`) divisent des consommations mesurées par une
> grandeur quasi identique à la cible : c'est une fuite de données, ces parts n'étant pas
> connues au moment de la prédiction. Les remplacer par de simples indicateurs de présence
> (`HasElectricity`, `HasNaturalGas`, `HasSteam`) abaisserait les performances, surtout sur
> les émissions.

Cinq algorithmes ont été comparés par cible (régression linéaire, SVR, RandomForest,
GradientBoosting, XGBoost). Les modèles linéaires sont inadaptés — jusqu'à un R² test de
**−2,74** sur les émissions : le lien entre caractéristiques du bâtiment et consommation est
multiplicatif et riche en interactions.

**Enseignement principal :** le **profil d'usage** est le premier déterminant de la
consommation (importance 0,368), devant la surface (0,287) ; pour les émissions, la hiérarchie
s'inverse au profit de la surface (0,339). Les usages à processus techniques intensifs —
hôpitaux, laboratoires — dominent les deux modèles.

→ **Analyse détaillée :** section *Conclusions*, en fin de chaque notebook.

## Données

| | |
|---|---|
| **Source** | City of Seattle Open Data (`data.seattle.gov`), jeu *2016 Building Energy Benchmarking* |
| **Fichier** | [data/raw/2016_Building_Energy_Benchmarking.csv](data/raw/2016_Building_Energy_Benchmarking.csv) — versionné (1,2 Mo) pour la reproductibilité |
| **Volume brut** | 3 376 bâtiments × 46 variables |
| **Après filtrage métier** | **1 479 bâtiments × 76 variables** (1 480 pour la cible CO₂) |

- Restriction aux bâtiments **non résidentiels**, via le type de bâtiment puis les trois
  niveaux d'usage déclaré.
- **`ENERGYSTARScore` écarté** : 24,97 % de valeurs manquantes, et surtout fuite de données —
  ce score est calculé à partir des consommations à prédire.
- Variables construites : mix énergétique, âge du bâtiment, nombre de types d'usage, et
  **part de surface par type d'usage** plutôt qu'une étiquette d'usage dominant, afin de
  représenter correctement les bâtiments mixtes.

## Structure du dépôt

```
.
├── data/raw/              Jeu de données source (versionné)
├── notebooks/
│   ├── 01_prediction_consommation_energie.ipynb
│   └── 02_prediction_emission_co2.ipynb
├── src/                   Fonctions partagées par les deux notebooks
│   └── Khelladi_Faïza_3_Fonctions_Projet3_1912.py
├── reports/
│   ├── figures/           40 graphiques exportés (PNG, 300 dpi)
│   └── presentation_projet3.pdf
├── references/            Carte des quartiers de Seattle
├── pyproject.toml
└── uv.lock                Versions verrouillées
```

| Fonction de `src/` | Rôle |
|---|---|
| `prepare_data` | Split train/test et scaling optionnel de `X` et `y` (standard ou min-max) |
| `train_evaluate_model` | Entraînement, inférence, inversion du scaling, métriques R²/RMSE/MAE, graphique prédit vs réel |
| `grid_around_best` | Grille resserrée autour du point optimal d'une `RandomizedSearchCV` |
| `group_feature_name` | Regroupe les colonnes one-hot par famille, pour une lecture non biaisée des *feature importances* |

## Installation et exécution

```bash
uv sync              # environnement à partir du lock (Python 3.14)
uv run jupyter lab
```

Les deux notebooks sont **indépendants** : chacun reprend l'intégralité du pipeline et ne
diffère que par la cible prédite. L'ordre d'ouverture est donc libre.

Les chemins sont résolus dynamiquement par une cellule en tête de chaque notebook
(`PROJECT_ROOT`, `DATA_RAW`, `FIG_DIR`), ce qui les rend exécutables depuis n'importe quel
répertoire de travail. Les figures sont écrites dans `reports/figures/`.

## Limites connues

- **Fuite de données dans le mix énergétique** (voir *Résultats*) : principale réserve, à
  corriger avant toute exploitation des chiffres.
- **Une seule année (2016)** : aucune dynamique temporelle n'est modélisable.
- **Cibles très asymétriques** : les métriques globales sont dominées par la masse des petits
  bâtiments, alors que l'enjeu porte sur les gros consommateurs et émetteurs.
- **Unique découpage train/test** (`random_state=42`) : les écarts entre modèles proches
  mériteraient une validation croisée répétée.
- **XGBoost non restitué** : la cellule d'évaluation affiche `results_gb` au lieu de
  `results_xgb` dans les deux notebooks. Le modèle est bien entraîné et optimisé, mais ses
  métriques n'ont jamais été affichées et les valeurs XGBoost des graphiques de comparaison
  dupliquent celles du GradientBoosting.

## Livrables

Les fichiers portent ici un nom décrivant leur contenu ; la convention de soumission
OpenClassrooms en attend un autre. Le renommage est sans effet sur le code : les chemins sont
résolus dynamiquement et le module `src/` est importé par son propre nom.

| # | Livrable | Fichier dans le dépôt | Nom de soumission |
|---|---|---|---|
| 1 | Notebook — consommation | [notebooks/01_prediction_consommation_energie.ipynb](notebooks/01_prediction_consommation_energie.ipynb) | `Khelladi_Faiza_1_notebook_consommation_energie_<date>.ipynb` |
| 2 | Notebook — émissions CO₂ | [notebooks/02_prediction_emission_co2.ipynb](notebooks/02_prediction_emission_co2.ipynb) | `Khelladi_Faiza_2_notebook_emission_co2_<date>.ipynb` |
| 3 | Module de fonctions | [src/](src/) | `Khelladi_Faïza_3_Fonctions_Projet3_<date>.py` |
| 4 | Présentation de soutenance | [reports/presentation_projet3.pdf](reports/presentation_projet3.pdf) | `Khelladi_Faïza_4_Présentation_Projet3_<date>.pdf` |

## Couverture des étapes de la mission

Chaque notebook couvre **l'intégralité** des étapes 1 à 5, sous ces titres de section :

| Étape | Section du notebook |
|---|---|
| 1 — Analyse exploratoire | *Partie 1 : Analyse Exploratoire* |
| 2 — Feature engineering | *Feature Engineering* |
| 3 — Préparation des features | *Préparation des features pour la modélisation* |
| 4 — Comparaison de modèles supervisés | *Comparaison de différents modèles supervisés* |
| 5 — Optimisation et interprétation | *Optimisation et interprétation du modèle* |
| 6 — Formalisation des résultats | *Conclusions* + [reports/presentation_projet3.pdf](reports/presentation_projet3.pdf) |

# Interface graphique — Dungeon RL

Lancer avec `.venv\Scripts\python.exe src\main.py`.

---

## Layout général

```
┌──────────────────────────────────────────────────────────────────────────┐  ← HUD (108px)
│  Déplacements: 0   Note: 0   Info:                   ← ↑ → ↓ | R restart │  Ligne 1
│  [Génération terrain]  Seed:[____]   Trail X/N | Victoires Y/N | Note ZZ  │  Ligne 2
│  [IA simple model] [Start SM det] [Start SM stoch]                        │  Ligne 3
│  [IA multi model]  [Start MM det] [Start MM stoch]                        │  Ligne 4
│  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░                            │  Ligne 5 — barre
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                      Grille 10×10  (800×800 px)                          │  ← Terrain
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

Résolution totale : 800×908 px (grille + HUD). Cases : 80×80 px, pas de séparateur.

---

## Terrain de jeu

| Élément | Sprite | Description |
|---------|--------|-------------|
| **Herbe** | `assets/tiles/grass.png` | Case franchissable, coût 1 déplacement |
| **Roche** | `assets/tiles/rock.png` | Obstacle infranchissable — tenter d'y entrer coûte 1 déplacement sans bouger |
| **Eau** | `assets/tiles/water.png` | Case franchissable, coût 2 déplacements |
| **Personnage** | `assets/tiles/player.png` (guerrier pixel art 80×80) | Position courante du joueur |
| **Sortie** | `assets/tiles/castle.png` (porte) | Objectif à atteindre |

Fallback : si un fichier PNG est absent, une couleur de remplacement est utilisée.

---

## Trails (tracés)

| Trail | Couleur | Décalage | Quand affiché |
|-------|---------|---------|---------------|
| **Joueur** (clavier) | Bleu foncé `(0, 70, 180)` | Centré | En continu pendant la partie |
| **Chemin optimal** (Dijkstra) | Jaune `(255, 255, 0)` | −5px à gauche | Uniquement en fin de partie (victoire) |
| **IA simple model** | Orange `(255, 140, 0)` | +5px à droite | Après clic sur [Start SM ...] |
| **IA multi model** | Dégradé orange→rouge | +5px à droite | Animation checkpoint par checkpoint |

Le dégradé IA multi : alpha 50 (premier checkpoint) → 220 (dernier), calculé linéairement.
Les trails se remettent à zéro à chaque nouveau terrain ou clic sur [IA restart].

---

## HUD — Ligne 1 : statistiques de jeu

| Élément | Description |
|---------|-------------|
| **Déplacements** | Compteur cumulé. Herbe=1, eau=2, choc mur=1 (sans bouger) |
| **Note** | Score final 0–100. `round(100 × coût_optimal / coût_joueur)`. 100 = chemin optimal |
| **Info** | Vide pendant la partie, affiche **GAGNE** à l'arrivée |
| Touches (gris) | Rappel `← ↑ → ↓ | R restart` |

---

## HUD — Ligne 2 : terrain et statistiques IA

| Élément | Interaction | Description |
|---------|-------------|-------------|
| **[Génération terrain]** | Clic | Génère un nouveau terrain aléatoire solvable |
| **Seed : [champ]** | Clic + saisie + Entrée | Génère le terrain du seed saisi (reproductible) |
| **Trail X/N** | Lecture seule | Progression de l'animation multi (checkpoint courant / total) |
| **Victoires Y/N** | Lecture seule | Épisodes gagnés / total des épisodes joués |
| **Note moy ZZ** | Lecture seule | `scores_sum / total_episodes` — les échecs (score 0) sont inclus |

Les stats se remettent à zéro à chaque nouveau terrain et à chaque clic sur [IA restart].

---

## HUD — Lignes 3–4 : modèles IA

### Modèle simple (SM)

| Élément | États | Description |
|---------|-------|-------------|
| **[IA simple model]** | Blanc = non chargé / **Cyan** = chargé | Ouvre sélecteur `.pt` (DQN) ou `.zip` (PPO). SM et MM sont **mutuellement exclusifs**. |
| **[Start SM déterministe]** | Grisé si non chargé | Lance 1 épisode déterministe (argmax). Trail orange. |
| **[Start SM stochastique]** | Grisé si non chargé | Lance 1 épisode stochastique (PPO uniquement). Résultat différent à chaque clic. |

### Modèle multi (MM)

| Élément | États | Description |
|---------|-------|-------------|
| **[IA multi model]** | Blanc = non chargé / **Cyan** = chargé | Ouvre sélecteur de dossier `*_run/`. Charge tous les modèles en fond (barre de progression visible). |
| **[Start MM déterministe]** | Grisé tant que chargement en cours | Lance tous les épisodes déterministes. Animation 200ms/trail, dégradé orange→rouge. |
| **[Start MM stochastique]** | Grisé tant que chargement en cours | Lance tous les épisodes stochastiques. Recalcul systématique. |
| **[IA restart]** | Toujours actif si MM chargé | Recalcule tous les épisodes sur le terrain courant. Bouton grisé + texte cyclique `Calcul.` / `Calcul..` / `Calcul...` pendant le recalcul. |

> **Exclusivité SM / MM :** charger un modèle simple efface le mode multi et vice versa.

---

## HUD — Ligne 5 : barre de chargement

Barre de 8px de hauteur, toujours réservée (même vide). Visible pendant :
- Chargement des modèles multi (`[IA multi model]`)
- Calcul des épisodes (`[Start MM ...]` et `[IA restart]`)

---

## Contrôles clavier

| Touche | Action |
|--------|--------|
| **← ↑ → ↓** | Déplace le personnage d'une case |
| **R** | Nouveau terrain aléatoire (= bouton [Génération terrain]) |
| **Entrée** (dans champ Seed) | Génère le terrain du seed saisi |

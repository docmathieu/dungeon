"""analyze_runs.py — Comparaison des chaînes de runs pool3→pool6→pool10.

Pour chaque chaîne curriculum (pool3 → pool6 → pool10), calcule et affiche :
    - win rate MAX  : meilleur taux de victoire sur une fenêtre glissante de 100 épisodes
    - win rate FINAL: taux de victoire sur les 100 derniers épisodes

Permet de comparer les architectures (MLP, Task-cond input, FiLM) sur les mêmes
métriques, et d'identifier quel run a produit les meilleurs résultats.

Résultats de référence (2026-05-28, vérifiés via poids des checkpoints) :
    Architecture  Run      p6_max  p6_fin  p10_max  p10_fin
    MLP           27/1534   23%     13%     24%       21%
    MLP           28/0835   71%     55%     33%        5%
    Task-cond     28/1039   69%     68%     59%        7%   ← meilleur pic pool10
    Task-cond     28/1411   44%     19%     51%        1%
    FiLM          28/1620   55%     51%     34%       16%   ← meilleur final pool10

Utilisation :
    cd dungeon/claude
    .venv\\Scripts\\python.exe analyze/analyze_runs.py
"""

import json, pathlib


def analyze(path, window=100):
    """Calcule win rate max et final sur un fichier de log JSONL.

    Filtre la ligne meta (type='meta') avant l'analyse.
    Retourne (nb_episodes, win_rate_max%, win_rate_final%).
    """
    lines = pathlib.Path(path).read_text().splitlines()
    # Ignorer la ligne meta si présente
    data = [json.loads(l) for l in lines if l.strip() and json.loads(l).get("type") != "meta"]
    scores = [d["score"] for d in data]
    n = len(scores)
    wins = [1 if s > 0 else 0 for s in scores]
    if n >= window:
        max_wr = max(sum(wins[i:i+window])/window for i in range(n-window+1))
    else:
        max_wr = sum(wins)/max(n,1)
    final_wr = sum(wins[-window:])/window if n >= window else sum(wins)/max(n,1)
    return n, round(max_wr*100,1), round(final_wr*100,1)


# Chaînes de runs à comparer : (date, tag_pool3, tag_pool6, tag_pool10)
# Chaque tag correspond au début du nom de fichier log (date_tag*.jsonl)
chains = [
    ("20260527", "1534_pool3", "1542_pool6", "1551_pool10"),  # MLP, 2000 ep/stage
    ("20260528", "0835_pool3", "0853_pool6", "0912_pool10"),  # MLP, 5000 ep/stage
    ("20260528", "1039_pool3", "1059_pool6", "1116_pool10"),  # Task-cond input
    ("20260528", "1411_pool3", "1431_pool6", "1451_pool10"),  # Task-cond input
    ("20260528", "1620_pool3", "1653_pool6", "1723_pool10"),  # FiLM
]

logs_dir = pathlib.Path("logs")
print("Chain          p3_max  p3_fin  p6_max  p6_fin  p10_max p10_fin")
print("-"*67)
for date, p3, p6, p10 in chains:
    def find(tag):
        matches = list(logs_dir.glob(date + "_" + tag + "*.jsonl"))
        return matches[0] if matches else None
    f3, f6, f10 = find(p3), find(p6), find(p10)
    r3  = analyze(f3)  if f3  else (0, 0, 0)
    r6  = analyze(f6)  if f6  else (0, 0, 0)
    r10 = analyze(f10) if f10 else (0, 0, 0)
    label = date[4:6] + "/" + p3[:4]
    print(f"{label}         {r3[1]:>5}%  {r3[2]:>5}%  {r6[1]:>5}%  {r6[2]:>5}%  {r10[1]:>6}%  {r10[2]:>6}%")

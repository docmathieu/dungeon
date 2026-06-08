# Historique des runs — Dungeon RL

---

## Ère DQN (algorithme maison, mai 2026)

| # | Date | Config | Seeds testés | Algo | Architecture | Win rate peak | Win rate final | Conclusion |
|---|------|--------|-------------|------|--------------|--------------|----------------|------------|
| D1 | 2026-05-21 | seed=42 fixe, 3 000 ep | 1 seed | DQN | MLP 304→128→64→4 | **99.6%** (ep 1000–1500) | 3.6% | Catastrophic forgetting confirmé |
| D2 | 2026-05-22 | pool20, 3 000 ep | 20 seeds | DQN | MLP 304→128→64→4 | 20% | 20% stable | Gradients conflictuels — politiques contradictoires |
| D3 | 2026-05-22 | pool10, 10 000 ep | 10 seeds | DQN | MLP 304→128→64→4 | 24% | 0% | LR trop fort, déclin inévitable |
| D4 | 2026-05-27 | Curriculum 1→3→6→10, 2 000 ep/stage | 10 seeds | DQN | MLP 304→128→64→4 | 35% (pool3) | 3% (pool10) | Transfer aide mais CF frappe à mi-parcours |
| D5 | 2026-05-27 | Curriculum + archi ×2 | 10 seeds | DQN | **MLP 304→256→128→64→4** | 32% (pool10) | 9% | +15 pts max — architecture augmentée prouve son utilité |
| D6 | 2026-05-28 | Curriculum + Task-cond | 10 seeds | DQN | MLP 314→256→128→64→4 (seed one-hot) | 59% pool10 | 7% | Pic élevé mais toujours instable |
| D7 | 2026-05-28 | Curriculum + FiLM | 10 seeds | DQN | FiLMDQNetwork 304+10→256→128→64→4 | 34% pool10 | **16%** | Plus stable en fin de run, pic inférieur |

**Blocage DQN :** catastrophic forgetting systématique au-delà de 3 seeds — deux seeds exigeant des actions
opposées dans des états visuellement similaires créent des gradients conflictuels qui écrasent les poids appris.

---

## Ère PPO — Stable-Baselines3 (juin 2026)

| # | Date | Dossier | ts cumulés | Algo | Architecture | Seeds entraînement | Det inconnus | Stoch ×3 inconnus | Score moy wins (det) | Conclusion |
|---|------|---------|-----------|------|--------------|-------------------|-------------|------------------|---------------------|------------|
| P1 | 2026-06-02 | 20260602_1028 | 2M | **PPO** | MLP 304→256→128→64→4 | pool100 | 35% training / **3%** inconnus | ~11% | 94.1 | PPO résout CF ; MLP ne généralise pas |
| P2 | 2026-06-02 | 20260602_1448 | 2M | PPO | **CNN** 10×10×5→16→32→128→64→4 | pool100 | 76% training / **3%** inconnus | ~11% | 94.5 | CNN = mémorisation parfaite, même plafond généralisation |
| P3 | 2026-06-02 | 20260602_1556 | **2M** | PPO | CNN | **full-random** (seed=None) | **10.5%** | **32.2%** | 94.5 | **Percée** : full-random force vraie généralisation |
| P4 | 2026-06-03 | 20260603_0811 | 10M | PPO | CNN | full-random | **56.0%** | **73.2%** | 93.7 | Progression régulière ~+6 pts/run |
| P5 | 2026-06-03 | 20260603_1448 | 15M | PPO | CNN | full-random | **66.5%** | **81.5%** | 95.0 | Tendance linéaire confirmée |
| P6 | 2026-06-03 | 20260603_1608 | 20M | PPO | CNN | full-random | **73.0%** | **85.8%** | 95.1 | — |
| P7 | 2026-06-05 | 20260605_1417 | 25M | PPO | CNN | full-random | **78.5%** | **89.0%** | 95.3 | — |
| **P8** | **2026-06-05** | **20260605_1600** | **30M** | PPO | CNN | full-random | **85.0%** | **90.0%** | 94.8 | ✅ **Objectif 80% atteint** |
| P9 | 2026-06-08 | 20260608_0922 | 35M | PPO | CNN | full-random | 81.0% | 89.5% | 95.9 | Plateau structurel confirmé (~81–85%) |
| P10 | 2026-06-08 | 20260608_1406 | **40M** | PPO | CNN + **Dijkstra shaping** | full-random | 76.8% | 89.2% | 94.8 | Shaping stabilise l'entraînement (92% stable) mais ne casse pas le plateau |

**Percée P3 :** seeds 0–99 (8%) ≈ seeds 100–299 (10.5%) — le modèle ne distingue plus "vu" vs "non vu".
C'est de la vraie généralisation. Pour CNN pool100 (P2), l'écart était 76% vs 3%.

**Plateau P9–P10 :** blocage résiduel sur les terrains à chemin long (coût Dijkstra > 16).

### Analyse des échecs (Run P10, 400 seeds inconnus)

| Groupe | Coût optimal | Win rate |
|--------|-------------|---------|
| Facile | 1–8 | 95.9% |
| Rochers | 9–12 | 77.3% |
| Mixte | 13–16 | 62.3% |
| Complexe | 17–20 | 19.4% |
| Difficile | 21+ | 11.8% |

Le modèle résout naturellement les terrains courts (coût ≤ 12) mais échoue sur les terrains nécessitant
15+ mouvements avec contournements importants — limite de planification longue distance des CNN sans mémoire.

---

## Synthèse des décisions architecturales

| Décision | Résultat | Statut |
|----------|---------|--------|
| DQN seed unique | Convergence rapide (99.6% en <2000 ep) | ✅ fonctionne, inutilisable en généralisation |
| DQN pool fixe + curriculum | 59% max, forgetting au-delà de pool3 | ✅ testé, limité |
| PPO MLP pool fixe | 76% training, 3% inconnus | ✅ testé |
| PPO CNN pool fixe | 76% training, 3% inconnus | ✅ testé |
| **PPO CNN full-random** | **85% inconnus déterministe (30M ts)** | ✅ **retenu** |
| Dijkstra reward shaping | Entraînement plus stable, généralisation identique | ✅ testé, non concluant |

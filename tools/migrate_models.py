"""
migrate_models.py — Regroupe les stages de curriculum dans des dossiers *_run/.

Usage:
    python tools/migrate_models.py [--models-dir models] [--logs-dir logs] [--dry-run]

Algorithme:
    - Un nom de stage a le format: {yyyymmdd_hhmm}_{label}[_from_{yyyymmdd_hhmm}]
    - Les stages sans _from_ (ou dont le _from_ pointe vers un timestamp absent) sont des racines.
    - Chaque racine entraine une chaine de stages relies par _from_.
    - Tous les stages d'une chaine sont deplaces dans {root_ts}_run/.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

# Regex pour extraire le timestamp initial et le _from_ timestamp d'un nom de stage
STAGE_RE = re.compile(r'^(\d{8}_\d{4})_.+?(?:_from_(\d{8}_\d{4}).*)?$')


def parse_stage_name(name: str) -> tuple[str | None, str | None]:
    """Retourne (own_ts, from_ts) ou (None, None) si le nom ne correspond pas."""
    m = STAGE_RE.match(name)
    if not m:
        return None, None
    own_ts = m.group(1)
    # Cherche _from_ dans le nom complet (le groupe 2 peut etre None)
    from_match = re.search(r'_from_(\d{8}_\d{4})', name)
    from_ts = from_match.group(1) if from_match else None
    return own_ts, from_ts


def collect_stages(base_dir: Path, is_file: bool = False) -> dict[str, dict]:
    """
    Liste les entrees dans base_dir et construit un dict:
        name -> {own_ts, from_ts, path}
    Ignore les entrees se terminant par _run (deja migrees).
    """
    stages = {}
    if not base_dir.exists():
        return stages

    for entry in base_dir.iterdir():
        # Pour les modeles: on veut les dossiers; pour les logs: les fichiers .jsonl
        if is_file:
            if not entry.is_file() or entry.suffix != '.jsonl':
                continue
            name = entry.stem  # sans .jsonl
        else:
            if not entry.is_dir():
                continue
            name = entry.name

        if name.endswith('_run'):
            continue

        own_ts, from_ts = parse_stage_name(name)
        if own_ts is None:
            continue

        stages[name] = {
            'own_ts': own_ts,
            'from_ts': from_ts,
            'path': entry,
        }

    return stages


def build_chains(stages: dict) -> list[list[str]]:
    """
    Construit les chaines de stages.
    Retourne une liste de chaines (chaque chaine est une liste ordonnee de noms).
    """
    # Index des timestamps presents
    ts_to_names: dict[str, list[str]] = {}
    for name, info in stages.items():
        ts = info['own_ts']
        ts_to_names.setdefault(ts, []).append(name)

    # Determiner les racines: stages sans _from_, ou dont le _from_ n'est pas dans ts_to_names
    roots = []
    for name, info in stages.items():
        from_ts = info['from_ts']
        if from_ts is None or from_ts not in ts_to_names:
            roots.append(name)

    # Pour chaque racine, suivre la chaine
    chains = []
    visited = set()

    for root in roots:
        chain = []
        current_names = [root]
        while current_names:
            # S'il y a plusieurs stages avec le meme timestamp (edge case), on les traite tous
            next_names = []
            for current in current_names:
                if current in visited:
                    continue
                visited.add(current)
                chain.append(current)
                own_ts = stages[current]['own_ts']
                # Chercher les successeurs: stages dont _from_ == own_ts de current
                for candidate_name, candidate_info in stages.items():
                    if candidate_info['from_ts'] == own_ts and candidate_name not in visited:
                        next_names.append(candidate_name)
            current_names = next_names
        if chain:
            chains.append(chain)

    # Stages non visites (orphelins non detectes comme racine — ne devrait pas arriver)
    for name in stages:
        if name not in visited:
            chains.append([name])

    return chains


def migrate(base_dir: Path, chains: list[list[str]], stages: dict,
            is_file: bool, dry_run: bool, label: str) -> int:
    """Deplace les entrees dans les dossiers *_run/. Retourne le nombre d'operations."""
    count = 0
    for chain in chains:
        root_name = chain[0]
        root_ts = stages[root_name]['own_ts']
        run_dir_name = f"{root_ts}_run"
        run_dir = base_dir / run_dir_name

        for name in chain:
            info = stages[name]
            src = info['path']
            dst = run_dir / src.name

            print(f"MOVE {label}/{src.name} -> {label}/{run_dir_name}/{src.name}")
            if not dry_run:
                run_dir.mkdir(exist_ok=True)
                shutil.move(str(src), str(dst))
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Regroupe les stages de curriculum dans des dossiers *_run/."
    )
    parser.add_argument('--models-dir', default='models',
                        help="Chemin du dossier models (relatif au CWD, defaut: models)")
    parser.add_argument('--logs-dir', default='logs',
                        help="Chemin du dossier logs (relatif au CWD, defaut: logs)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Affiche les operations sans les effectuer")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)

    if args.dry_run:
        print("[DRY RUN] Aucun fichier ne sera deplace.\n")

    total = 0

    # --- Models (dossiers) ---
    model_stages = collect_stages(models_dir, is_file=False)
    if model_stages:
        model_chains = build_chains(model_stages)
        n = migrate(models_dir, model_chains, model_stages,
                    is_file=False, dry_run=args.dry_run, label=args.models_dir)
        total += n
    else:
        print(f"Aucun stage trouve dans {models_dir}/")

    # --- Logs (fichiers .jsonl) ---
    log_stages = collect_stages(logs_dir, is_file=True)
    if log_stages:
        log_chains = build_chains(log_stages)
        n = migrate(logs_dir, log_chains, log_stages,
                    is_file=True, dry_run=args.dry_run, label=args.logs_dir)
        total += n
    else:
        print(f"Aucun log trouve dans {logs_dir}/")

    print(f"\n{total} operation(s) {'simulee(s)' if args.dry_run else 'effectuee(s)'}.")


if __name__ == '__main__':
    main()

"""Tests unitaires pour curriculum.py (_win_rate, _train_stage, run_curriculum)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from curriculum import (
    WIN_RATE_WINDOW,
    _pad_lr,
    _train_stage,
    _win_rate,
    run_curriculum,
)


# ===========================================================================
# _win_rate
# ===========================================================================

class TestWinRate:
    def test_empty_returns_zero(self):
        assert _win_rate([]) == 0.0

    def test_all_losses_returns_zero(self):
        assert _win_rate([0] * 100) == 0.0

    def test_all_wins_returns_one(self):
        assert _win_rate([50] * 100) == pytest.approx(1.0)

    def test_half_wins(self):
        scores = [100, 0] * 50
        assert _win_rate(scores) == pytest.approx(0.5)

    def test_uses_last_window_only(self):
        scores = [0] * 200 + [100] * 100
        assert _win_rate(scores, window=100) == pytest.approx(1.0)

    def test_partial_window_uses_available(self):
        scores = [100] * 50   # moins que la fenêtre
        assert _win_rate(scores, window=100) == pytest.approx(1.0)

    def test_score_zero_counts_as_loss(self):
        assert _win_rate([0]) == 0.0

    def test_positive_score_counts_as_win(self):
        assert _win_rate([1]) == pytest.approx(1.0)

    def test_custom_window(self):
        scores = [0] * 10 + [100] * 5
        assert _win_rate(scores, window=5) == pytest.approx(1.0)


# ===========================================================================
# _pad_lr
# ===========================================================================

class TestPadLr:
    def test_single_element_padded_to_n(self):
        assert _pad_lr([1e-3], 3) == pytest.approx([1e-3, 1e-3, 1e-3])

    def test_exact_length_unchanged(self):
        assert _pad_lr([3e-4, 1e-4, 5e-5], 3) == pytest.approx([3e-4, 1e-4, 5e-5])

    def test_shorter_pads_with_last_value(self):
        assert _pad_lr([3e-4, 1e-4], 4) == pytest.approx([3e-4, 1e-4, 1e-4, 1e-4])

    def test_longer_than_n_truncated(self):
        assert _pad_lr([3e-4, 1e-4, 5e-5, 1e-5], 2) == pytest.approx([3e-4, 1e-4])


# ===========================================================================
# _train_stage
# ===========================================================================

class TestTrainStage:
    def test_returns_final_pt_path(self, tmp_path):
        result = _train_stage(
            stage_pool         = [42],
            max_episodes       = 3,
            win_rate_threshold = 0.8,
            lr                 = 1e-3,
            pretrained         = None,
            log_path           = tmp_path / "test.jsonl",
            model_dir          = tmp_path / "models",
            verbose            = False,
        )
        assert result == tmp_path / "models" / "final.pt"

    def test_creates_log_file(self, tmp_path):
        _train_stage(
            stage_pool         = [42],
            max_episodes       = 3,
            win_rate_threshold = 0.8,
            lr                 = 1e-3,
            pretrained         = None,
            log_path           = tmp_path / "test.jsonl",
            model_dir          = tmp_path / "models",
            verbose            = False,
        )
        assert (tmp_path / "test.jsonl").exists()

    def test_creates_final_checkpoint(self, tmp_path):
        _train_stage(
            stage_pool         = [42],
            max_episodes       = 3,
            win_rate_threshold = 0.8,
            lr                 = 1e-3,
            pretrained         = None,
            log_path           = tmp_path / "test.jsonl",
            model_dir          = tmp_path / "models",
            verbose            = False,
        )
        assert (tmp_path / "models" / "final.pt").exists()

    def test_log_line_count_equals_episodes_run(self, tmp_path):
        _train_stage(
            stage_pool         = [42],
            max_episodes       = 5,
            win_rate_threshold = 0.8,
            lr                 = 1e-3,
            pretrained         = None,
            log_path           = tmp_path / "test.jsonl",
            model_dir          = tmp_path / "models",
            verbose            = False,
        )
        lines = (tmp_path / "test.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5

    def test_stops_early_when_mastery_reached(self, tmp_path):
        """L'entraînement s'arrête dès que le win rate est atteint."""
        with patch("curriculum._win_rate", return_value=1.0):
            _train_stage(
                stage_pool         = [42],
                max_episodes       = 2000,
                win_rate_threshold = 0.8,
                lr                 = 1e-3,
                pretrained         = None,
                log_path           = tmp_path / "test.jsonl",
                model_dir          = tmp_path / "models",
                verbose            = False,
            )
        lines = (tmp_path / "test.jsonl").read_text().strip().splitlines()
        assert len(lines) == WIN_RATE_WINDOW

    def test_pretrained_weights_loaded(self, tmp_path):
        """Un checkpoint pretrained doit être chargé avant l'entraînement."""
        import torch
        from model import FiLMDQNetwork
        pre_dir = tmp_path / "pre"
        pre_dir.mkdir()
        net = FiLMDQNetwork()
        for p in net.parameters():
            p.data.fill_(0.5)
        torch.save(net.state_dict(), pre_dir / "final.pt")

        result = _train_stage(
            stage_pool         = [42],
            max_episodes       = 2,
            win_rate_threshold = 0.8,
            lr                 = 1e-3,
            pretrained         = pre_dir / "final.pt",
            log_path           = tmp_path / "test.jsonl",
            model_dir          = tmp_path / "models",
            verbose            = False,
        )
        assert result.exists()


# ===========================================================================
# run_curriculum
# ===========================================================================

class TestRunCurriculum:
    def test_returns_path_to_final_pt(self, tmp_path):
        result = run_curriculum(
            pool                   = [42, 0, 1],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        assert str(result).endswith("final.pt")

    def test_creates_one_log_per_stage(self, tmp_path):
        run_curriculum(
            pool                   = [42, 0, 1],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        logs = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(logs) == 2

    def test_creates_one_model_dir_per_stage(self, tmp_path):
        run_curriculum(
            pool                   = [42, 0, 1],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        dirs = [d for d in (tmp_path / "models").iterdir() if d.is_dir()]
        assert len(dirs) == 2

    def test_each_stage_has_final_pt(self, tmp_path):
        run_curriculum(
            pool                   = [42, 0, 1],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        for d in (tmp_path / "models").iterdir():
            if d.is_dir():
                assert (d / "final.pt").exists()

    def test_second_stage_name_contains_from(self, tmp_path):
        """Le run name de l'étape 2+ doit contenir _from_ (transfer learning)."""
        run_curriculum(
            pool                   = [42, 0],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        logs = sorted((tmp_path / "logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        assert "_from_" in logs[1].name

    def test_first_stage_uses_seed_label(self, tmp_path):
        """La première étape avec 1 seed doit avoir le label seed{N}."""
        run_curriculum(
            pool                   = [42, 0],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        logs = sorted((tmp_path / "logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        assert "seed42" in logs[0].name

    def test_second_stage_uses_pool_label(self, tmp_path):
        """La deuxième étape avec 2 seeds doit avoir le label pool2."""
        run_curriculum(
            pool                   = [42, 0],
            stages                 = [1, 2],
            max_episodes_per_stage = 3,
            win_rate_threshold     = 0.8,
            lr                     = [1e-3],
            log_dir                = tmp_path / "logs",
            model_dir              = tmp_path / "models",
            verbose                = False,
        )
        logs = sorted((tmp_path / "logs").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        assert "pool2" in logs[1].name

    def test_per_stage_lr_routes_correctly(self, tmp_path):
        """Chaque étape reçoit le lr correspondant à sa position dans la liste."""
        lrs_seen = []

        def spy(*args, **kwargs):
            lrs_seen.append(kwargs["lr"])
            return _train_stage(*args, **kwargs)

        with patch("curriculum._train_stage", side_effect=spy):
            run_curriculum(
                pool                   = [42, 0],
                stages                 = [1, 2],
                max_episodes_per_stage = 3,
                win_rate_threshold     = 0.8,
                lr                     = [3e-4, 1e-4],
                log_dir                = tmp_path / "logs",
                model_dir              = tmp_path / "models",
                verbose                = False,
            )
        assert lrs_seen == pytest.approx([3e-4, 1e-4])

    def test_single_lr_applied_to_all_stages(self, tmp_path):
        """Un seul lr fourni doit être utilisé pour toutes les étapes."""
        lrs_seen = []

        def spy(*args, **kwargs):
            lrs_seen.append(kwargs["lr"])
            return _train_stage(*args, **kwargs)

        with patch("curriculum._train_stage", side_effect=spy):
            run_curriculum(
                pool                   = [42, 0],
                stages                 = [1, 2],
                max_episodes_per_stage = 3,
                win_rate_threshold     = 0.8,
                lr                     = [5e-4],
                log_dir                = tmp_path / "logs",
                model_dir              = tmp_path / "models",
                verbose                = False,
            )
        assert lrs_seen == pytest.approx([5e-4, 5e-4])

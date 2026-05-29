"""Tests unitaires pour curriculum.py (_win_rate, _train_stage, run_curriculum).

Removed : TestPadLr (utilitaire interne de padding), TestTrainStage.test_creates_log_file et
          test_creates_final_checkpoint (couverts par TestRunCurriculum), TestRunCurriculum labels
          de fichiers (test_first/second_stage_uses_*_label), TestArchitecture.test_train_stage_default
          et test_architectures_dict_exported (couverts par test_train.py).
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from curriculum import (
    ARCHITECTURES,
    WIN_RATE_WINDOW,
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
        scores = [100] * 50
        assert _win_rate(scores, window=100) == pytest.approx(1.0)

    def test_score_zero_counts_as_loss(self):
        assert _win_rate([0]) == 0.0

    def test_positive_score_counts_as_win(self):
        assert _win_rate([1]) == pytest.approx(1.0)

    def test_custom_window(self):
        scores = [0] * 10 + [100] * 5
        assert _win_rate(scores, window=5) == pytest.approx(1.0)


# ===========================================================================
# _train_stage
# ===========================================================================

class TestTrainStage:
    def test_returns_final_pt_path(self, tmp_path):
        result = _train_stage(
            stage_pool=         [42],
            max_episodes=       3,
            win_rate_threshold= 0.8,
            lr=                 1e-3,
            pretrained=         None,
            log_path=           tmp_path / "test.jsonl",
            model_dir=          tmp_path / "models",
            verbose=            False,
        )
        assert result == tmp_path / "models" / "final.pt"

    def test_log_line_count_equals_episodes_run(self, tmp_path):
        _train_stage(
            stage_pool=         [42],
            max_episodes=       5,
            win_rate_threshold= 0.8,
            lr=                 1e-3,
            pretrained=         None,
            log_path=           tmp_path / "test.jsonl",
            model_dir=          tmp_path / "models",
            verbose=            False,
        )
        lines = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()
                 if json.loads(l).get("type") != "meta"]
        assert len(lines) == 5

    def test_stops_early_when_mastery_reached(self, tmp_path):
        with patch("curriculum._win_rate", return_value=1.0):
            _train_stage(
                stage_pool=         [42],
                max_episodes=       2000,
                win_rate_threshold= 0.8,
                lr=                 1e-3,
                pretrained=         None,
                log_path=           tmp_path / "test.jsonl",
                model_dir=          tmp_path / "models",
                verbose=            False,
            )
        lines = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()
                 if json.loads(l).get("type") != "meta"]
        assert len(lines) == WIN_RATE_WINDOW

    def test_pretrained_weights_loaded(self, tmp_path):
        import torch
        from model import FiLMDQNetwork
        pre_dir = tmp_path / "pre"
        pre_dir.mkdir()
        net = FiLMDQNetwork()
        for p in net.parameters():
            p.data.fill_(0.5)
        torch.save(net.state_dict(), pre_dir / "final.pt")
        result = _train_stage(
            stage_pool=         [42],
            max_episodes=       2,
            win_rate_threshold= 0.8,
            lr=                 1e-3,
            pretrained=         pre_dir / "final.pt",
            log_path=           tmp_path / "test.jsonl",
            model_dir=          tmp_path / "models",
            verbose=            False,
        )
        assert result.exists()


# ===========================================================================
# run_curriculum
# ===========================================================================

class TestRunCurriculum:
    def _base_args(self, tmp_path):
        return dict(
            pool=                   [42, 0, 1],
            stages=                 [1, 2],
            max_episodes_per_stage= 3,
            win_rate_threshold=     0.8,
            lr=                     [1e-3],
            log_dir=                tmp_path / "logs",
            model_dir=              tmp_path / "models",
            verbose=                False,
        )

    def test_returns_path_to_final_pt(self, tmp_path):
        result = run_curriculum(**self._base_args(tmp_path))
        assert str(result).endswith("final.pt")

    def test_creates_one_log_per_stage(self, tmp_path):
        run_curriculum(**self._base_args(tmp_path))
        # Les logs sont maintenant dans un sous-dossier *_run/
        assert len(list((tmp_path / "logs").rglob("*.jsonl"))) == 2

    def test_each_stage_has_final_pt(self, tmp_path):
        run_curriculum(**self._base_args(tmp_path))
        # Les model dirs sont dans un sous-dossier *_run/ — chercher récursivement
        for d in (tmp_path / "models").rglob("final.pt"):
            assert d.exists()
        # S'assurer qu'il y a bien 2 final.pt (un par stage)
        assert len(list((tmp_path / "models").rglob("final.pt"))) == 2

    def test_second_stage_name_contains_from(self, tmp_path):
        run_curriculum(**self._base_args(tmp_path))
        logs = sorted((tmp_path / "logs").rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        assert "_from_" in logs[1].name

    def test_per_stage_lr_routes_correctly(self, tmp_path):
        lrs_seen = []

        def spy(*args, **kwargs):
            lrs_seen.append(kwargs["lr"])
            return _train_stage(*args, **kwargs)

        with patch("curriculum._train_stage", side_effect=spy):
            run_curriculum(
                pool=[42, 0], stages=[1, 2], max_episodes_per_stage=3,
                win_rate_threshold=0.8, lr=[3e-4, 1e-4],
                log_dir=tmp_path / "logs", model_dir=tmp_path / "models", verbose=False,
            )
        assert lrs_seen == pytest.approx([3e-4, 1e-4])

    def test_single_lr_applied_to_all_stages(self, tmp_path):
        lrs_seen = []

        def spy(*args, **kwargs):
            lrs_seen.append(kwargs["lr"])
            return _train_stage(*args, **kwargs)

        with patch("curriculum._train_stage", side_effect=spy):
            run_curriculum(
                pool=[42, 0], stages=[1, 2], max_episodes_per_stage=3,
                win_rate_threshold=0.8, lr=[5e-4],
                log_dir=tmp_path / "logs", model_dir=tmp_path / "models", verbose=False,
            )
        assert lrs_seen == pytest.approx([5e-4, 5e-4])


# ===========================================================================
# Architecture — propagation et logs
# ===========================================================================

class TestArchitecture:
    def test_train_stage_taskcond_arch(self, tmp_path):
        import torch
        from model import DQNetwork
        _train_stage(
            stage_pool=[42], max_episodes=2, win_rate_threshold=0.8, lr=1e-3,
            pretrained=None, log_path=tmp_path / "test.jsonl",
            model_dir=tmp_path / "models", verbose=False, arch="taskcond",
        )
        DQNetwork().load_state_dict(
            torch.load(tmp_path / "models" / "final.pt", weights_only=True))

    def test_train_stage_meta_logs_architecture(self, tmp_path):
        _train_stage(
            stage_pool=[42], max_episodes=2, win_rate_threshold=0.8, lr=1e-3,
            pretrained=None, log_path=tmp_path / "test.jsonl",
            model_dir=tmp_path / "models", verbose=False, arch="taskcond",
        )
        meta = json.loads((tmp_path / "test.jsonl").read_text().splitlines()[0])
        assert meta["architecture"] == "DQNetwork"

    def test_run_curriculum_arch_propagated(self, tmp_path):
        arch_seen = []

        def spy(*args, **kwargs):
            arch_seen.append(kwargs.get("arch", "film"))
            return _train_stage(*args, **kwargs)

        with patch("curriculum._train_stage", side_effect=spy):
            run_curriculum(
                pool=[42, 0], stages=[1, 2], max_episodes_per_stage=3,
                win_rate_threshold=0.8, lr=[1e-3], arch="taskcond",
                log_dir=tmp_path / "logs", model_dir=tmp_path / "models", verbose=False,
            )
        assert all(a == "taskcond" for a in arch_seen)

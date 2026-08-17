from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def load_main_module():
    """Load the project entry point as an importable test module."""

    spec = importlib.util.spec_from_file_location(
        "airaola_main",
        MAIN_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load Project Airaola entry point: {MAIN_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def main_module():
    """Provide a freshly loaded main module for each test."""

    return load_main_module()


def build_arguments(
    *,
    autonomous: bool = False,
    auto_apply: bool = False,
    dry_run: bool = False,
    advance_gameweek: bool = False,
    send_email: bool = False,
) -> argparse.Namespace:
    """Build the command-line namespace used by helper tests."""

    return argparse.Namespace(
        autonomous=autonomous,
        auto_apply=auto_apply,
        dry_run=dry_run,
        advance_gameweek=advance_gameweek,
        send_email=send_email,
    )


def build_intelligence(
    *,
    state_status: str,
    hours_until_deadline: float | None,
) -> SimpleNamespace:
    """Build the deadline fields consumed by the safety gate."""

    return SimpleNamespace(
        state_status=state_status,
        hours_until_deadline=hours_until_deadline,
    )


def test_non_autonomous_run_bypasses_gate(main_module):
    arguments = build_arguments(
        autonomous=False,
    )
    intelligence = build_intelligence(
        state_status=main_module.SEASON_NOT_STARTED,
        hours_until_deadline=400.0,
    )

    assert main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )


def test_autonomous_run_is_blocked_during_preseason(
    main_module,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    intelligence = build_intelligence(
        state_status=main_module.SEASON_NOT_STARTED,
        hours_until_deadline=400.0,
    )

    allowed = main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )

    output = capsys.readouterr().out

    assert not allowed
    assert "Status: BLOCKED" in output
    assert "disabled during preseason" in output


def test_autonomous_run_is_blocked_without_deadline(
    main_module,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=None,
    )

    allowed = main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )

    output = capsys.readouterr().out

    assert not allowed
    assert "no valid upcoming deadline" in output


def test_autonomous_run_is_blocked_after_deadline(
    main_module,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=0.0,
    )

    allowed = main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )

    output = capsys.readouterr().out

    assert not allowed
    assert "deadline has already passed" in output


def test_autonomous_run_is_blocked_before_final_window(
    main_module,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=24.01,
    )

    allowed = main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )

    output = capsys.readouterr().out

    assert not allowed
    assert "inside the final 24 hours" in output
    assert "24.01" in output


@pytest.mark.parametrize(
    "hours_remaining",
    [
        24.0,
        12.0,
        0.01,
    ],
)
def test_autonomous_run_is_allowed_inside_final_window(
    main_module,
    capsys,
    hours_remaining,
):
    arguments = build_arguments(
        autonomous=True,
    )
    intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=hours_remaining,
    )

    allowed = main_module.autonomous_run_is_allowed(
        arguments,
        intelligence,
    )

    output = capsys.readouterr().out

    assert allowed
    assert "Status: APPROVED" in output
    assert "autonomous decision window" in output


def test_autonomous_confirmation_approves_without_input(
    main_module,
    monkeypatch,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )

    def fail_if_input_is_requested(*args, **kwargs):
        raise AssertionError(
            "Autonomous mode requested interactive input."
        )

    monkeypatch.setattr(
        "builtins.input",
        fail_if_input_is_requested,
    )

    approved = main_module.confirm_state_change(
        arguments=arguments,
        prompt="Approve this test decision?",
    )

    output = capsys.readouterr().out

    assert approved
    assert "approved its own state change" in output


def test_dry_run_confirmation_never_applies_state(
    main_module,
    capsys,
):
    arguments = build_arguments(
        dry_run=True,
    )

    approved = main_module.confirm_state_change(
        arguments=arguments,
        prompt="Approve this test decision?",
    )

    output = capsys.readouterr().out

    assert not approved
    assert "state change not applied" in output


def test_autonomous_and_dry_run_are_mutually_exclusive(
    main_module,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--autonomous",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main_module.parse_arguments()

    output = capsys.readouterr().err

    assert error.value.code == 2
    assert "not allowed with argument" in output


def test_autonomous_and_advance_are_rejected(
    main_module,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--autonomous",
            "--advance-gameweek",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main_module.parse_arguments()

    output = capsys.readouterr().err

    assert error.value.code == 2
    assert "--autonomous cannot be combined" in output


def test_autonomous_sync_is_noop_for_non_autonomous_run(
    main_module,
):
    arguments = build_arguments(
        autonomous=False,
    )
    state = SimpleNamespace(
        current_gameweek=1,
    )
    intelligence = build_intelligence(
        state_status=main_module.ADVANCEMENT_REQUIRED,
        hours_until_deadline=6.0,
    )

    returned_state, returned_intelligence, synced = (
        main_module.autonomous_sync_gameweek_state(
            arguments=arguments,
            intelligence=intelligence,
            manager_state=state,
            bootstrap_data={},
        )
    )

    assert returned_state is state
    assert returned_intelligence is intelligence
    assert not synced


def test_autonomous_sync_does_not_advance_aligned_state(
    main_module,
):
    arguments = build_arguments(
        autonomous=True,
    )
    state = SimpleNamespace(
        current_gameweek=1,
    )
    intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=6.0,
    )

    returned_state, returned_intelligence, synced = (
        main_module.autonomous_sync_gameweek_state(
            arguments=arguments,
            intelligence=intelligence,
            manager_state=state,
            bootstrap_data={},
        )
    )

    assert returned_state is state
    assert returned_intelligence is intelligence
    assert not synced


def test_autonomous_sync_refuses_incomplete_gameweek(
    main_module,
    monkeypatch,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    state = SimpleNamespace(
        current_gameweek=1,
    )
    intelligence = build_intelligence(
        state_status=main_module.ADVANCEMENT_REQUIRED,
        hours_until_deadline=6.0,
    )

    monkeypatch.setattr(
        main_module,
        "gameweek_is_processed",
        lambda *_args, **_kwargs: False,
    )

    def fail_if_advanced(*args, **kwargs):
        raise AssertionError(
            "Incomplete Gameweek was advanced."
        )

    monkeypatch.setattr(
        main_module,
        "advance_gameweek",
        fail_if_advanced,
    )
    monkeypatch.setattr(
        main_module,
        "save_manager_state",
        fail_if_advanced,
    )

    returned_state, returned_intelligence, synced = (
        main_module.autonomous_sync_gameweek_state(
            arguments=arguments,
            intelligence=intelligence,
            manager_state=state,
            bootstrap_data={},
        )
    )

    output = capsys.readouterr().out

    assert returned_state is state
    assert returned_intelligence is intelligence
    assert not synced
    assert "Status: BLOCKED" in output
    assert "not fully processed" in output


def test_autonomous_sync_advances_one_completed_gameweek(
    main_module,
    monkeypatch,
    capsys,
):
    arguments = build_arguments(
        autonomous=True,
    )
    state = SimpleNamespace(
        current_gameweek=1,
    )
    intelligence = build_intelligence(
        state_status=main_module.ADVANCEMENT_REQUIRED,
        hours_until_deadline=6.0,
    )
    advanced_state = SimpleNamespace(
        current_gameweek=2,
    )
    refreshed_intelligence = build_intelligence(
        state_status="ALIGNED",
        hours_until_deadline=72.0,
    )

    monkeypatch.setattr(
        main_module,
        "gameweek_is_processed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        main_module,
        "advance_gameweek",
        lambda state: advanced_state,
    )

    saved = []

    def record_save(*, state, state_path):
        saved.append(
            (state, state_path)
        )

    monkeypatch.setattr(
        main_module,
        "save_manager_state",
        record_save,
    )
    monkeypatch.setattr(
        main_module,
        "analyse_deadline_intelligence",
        lambda *, bootstrap_data, saved_gameweek: refreshed_intelligence,
    )

    returned_state, returned_intelligence, synced = (
        main_module.autonomous_sync_gameweek_state(
            arguments=arguments,
            intelligence=intelligence,
            manager_state=state,
            bootstrap_data={"events": []},
        )
    )

    output = capsys.readouterr().out

    assert returned_state is advanced_state
    assert returned_intelligence is refreshed_intelligence
    assert synced
    assert saved == [
        (
            advanced_state,
            main_module.STATE_PATH,
        )
    ]
    assert "Status: ADVANCED" in output
    assert "Gameweek 1 to Gameweek 2" in output
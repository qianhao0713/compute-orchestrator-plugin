from pathlib import Path

import pytest

from handoff_manager import HandoffRecord, inspect_artifacts, write_handoff


def test_handoff_written_atomically_under_stable_root(tmp_path: Path):
    workdir = tmp_path / "project"
    result = write_handoff(
        working_directory=str(workdir),
        stable_root=tmp_path,
        record=HandoffRecord(
            projectId=1,
            clientMessageId="msg-1",
            objective="reproduce paper",
        ),
    )
    assert Path(result["handoffPath"]).exists()


def test_path_outside_stable_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        write_handoff(
            working_directory="/tmp/outside",
            stable_root=tmp_path,
            record=HandoffRecord(
                projectId=1,
                clientMessageId="msg-1",
                objective="reproduce paper",
            ),
        )


def test_partial_artifact_not_ready(tmp_path: Path):
    partial = tmp_path / "model.part"
    partial.write_text("partial", encoding="utf-8")
    result = inspect_artifacts([str(partial)], tmp_path)
    assert result["allReady"] is False

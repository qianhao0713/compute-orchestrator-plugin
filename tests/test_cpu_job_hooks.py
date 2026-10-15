from hooks.block_portal_bash import is_direct_portal_request
from hooks.inject_session_id import build_hook_output


def test_submit_cpu_job_receives_trusted_hook_session():
    output = build_hook_output(
        {
            "tool_name": "mcp__res__submit_cpu_job",
            "session_id": "trusted-session",
            "tool_input": {"request_id": "request-1", "session_id": "invented"},
        }
    )
    assert output["hookSpecificOutput"]["updatedInput"]["session_id"] == "trusted-session"


def test_unrelated_question_or_tool_cannot_create_submission_parameters():
    try:
        build_hook_output(
            {
                "tool_name": "AskUserQuestion",
                "session_id": "trusted-session",
                "tool_input": {},
            }
        )
    except ValueError as exc:
        assert "unexpected tool" in str(exc)
    else:
        raise AssertionError("non-submission tool should be rejected")


def test_direct_k8s_portal_request_is_blocked():
    assert is_direct_portal_request(
        "curl https://portal.example/scientist/project/1001/jobs/resources"
    )

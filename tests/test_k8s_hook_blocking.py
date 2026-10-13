"""Tests for K8s API path blocking in the Bash PreToolUse hook."""

from hooks.block_portal_bash import (
    K8S_JOB_PATH_PATTERN,
    build_hook_output,
    is_direct_portal_request,
)


class TestK8sPathBlocking:
    def test_k8s_job_path_pattern_matches_submit(self):
        assert K8S_JOB_PATH_PATTERN.search(
            "/scientist/project/1001/jobs/submit"
        )

    def test_k8s_job_path_pattern_matches_resources(self):
        assert K8S_JOB_PATH_PATTERN.search(
            "/scientist/project/1001/jobs/resources"
        )

    def test_k8s_job_path_pattern_matches_job_detail(self):
        assert K8S_JOB_PATH_PATTERN.search(
            "/scientist/project/1001/jobs/cj-123"
        )

    def test_k8s_job_path_pattern_matches_cancel(self):
        assert K8S_JOB_PATH_PATTERN.search(
            "/scientist/project/1001/jobs/cj-123/cancel"
        )

    def test_k8s_job_path_pattern_does_not_match_unrelated_paths(self):
        assert not K8S_JOB_PATH_PATTERN.search("/scientist/other/path")
        assert not K8S_JOB_PATH_PATTERN.search("/other/project/1001/jobs")

    def test_curl_to_k8s_submit_is_blocked(self):
        output = build_hook_output(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        'curl -X POST '
                        'https://portal.example.com/scientist/project/1001/jobs/submit '
                        '-H "Content-Type: application/json" '
                        '-d \'{"requestId":"r1"}\''
                    )
                },
            }
        )
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "K8s job requests" in output["hookSpecificOutput"]["permissionDecisionReason"]

    def test_wget_to_k8s_job_detail_is_blocked(self):
        output = build_hook_output(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        "wget -qO- "
                        "https://portal.example.com/scientist/project/1001/jobs/cj-123"
                    )
                },
            }
        )
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_python_requests_to_k8s_resources_is_blocked(self):
        output = build_hook_output(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        'python3 -c "import requests; '
                        "requests.get('https://portal.example.com"
                        "/scientist/project/1001/jobs/resources')\""
                    )
                },
            }
        )
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_httpx_post_to_k8s_cancel_is_blocked(self):
        output = build_hook_output(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": (
                        'python3 -c "import httpx; '
                        "httpx.post('https://portal.example.com"
                        "/scientist/project/1001/jobs/cj-123/cancel')\""
                    )
                },
            }
        )
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_normal_curl_without_k8s_path_is_not_blocked(self):
        output = build_hook_output(
            {
                "tool_name": "Bash",
                "tool_input": {
                    "command": "curl https://example.com/api/data"
                },
            }
        )
        assert output == {}

    def test_is_direct_portal_request_with_k8s_path(self):
        assert is_direct_portal_request(
            "curl https://portal.example.com/scientist/project/42/jobs/submit"
        )

    def test_is_direct_portal_request_without_k8s_or_deployment_path(self):
        assert not is_direct_portal_request(
            "curl https://example.com/some/other/endpoint"
        )

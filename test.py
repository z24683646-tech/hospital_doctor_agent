"""Local test-mode entrypoint for the baseline agent.

By default this script mirrors the payload shape used by ``web_server`` when it
calls a deployed ModelScope Studio agent:

    {"contestServiceToken": "..."}

For local debugging you can opt in to extra SDK override fields with
``--include-local-overrides``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
LOCAL_SDK_DIR = REPO_DIR / "hospital_agent_sdk"
if LOCAL_SDK_DIR.exists() and str(LOCAL_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_SDK_DIR))

DEFAULT_AGENT_TEST_URL = "http://127.0.0.1:7860/test"
DEFAULT_SERVICE_BASE_URL = "http://127.0.0.1:8001"


def main() -> None:
    args = parse_args()
    payload = build_web_server_like_payload(args)

    if args.print_payload:
        print(json.dumps({"payload": redact_payload(payload)}, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    if args.http:
        result = invoke_agent_test_endpoint(args.agent_url, payload)
    else:
        from agent import MyDoctorAgent, build_memory
        from hospital_agent_sdk import load_config

        config = load_config(BASE_DIR / args.config)
        memory = build_memory(config)
        agent = MyDoctorAgent(config=config, memory=memory)
        result = agent.run_test(payload=payload)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline test mode with the same payload shape used by web_server."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Baseline config file relative to baseline_example/.",
    )
    parser.add_argument(
        "--service-token",
        default=env_text("SERVICE_TEST_TOKEN", "CONTEST_SERVICE_TOKEN", "SERVICE_TOKEN"),
        help=(
            "Temporary contest service bearer token. Defaults to SERVICE_TEST_TOKEN, "
            "CONTEST_SERVICE_TOKEN, or SERVICE_TOKEN."
        ),
    )
    parser.add_argument(
        "--service-base-url",
        default=env_text("SERVICE_BASE_URL", default=DEFAULT_SERVICE_BASE_URL),
        help="Contest service base URL passed as contestServiceBaseUrl.",
    )
    parser.add_argument(
        "--team-id",
        default=env_text("TEAM_ID", "ACCOUNT", default="team_demo"),
        help="Team/account id passed as both account and teamId.",
    )
    parser.add_argument(
        "--dataset-key",
        default=env_text("DATASET_KEY", default="primary_test"),
        help="Dataset key included only with --include-local-overrides.",
    )
    parser.add_argument(
        "--include-local-overrides",
        action="store_true",
        help=(
            "Also include contestServiceBaseUrl/account/teamId/datasetKey. "
            "Leave this off to match web_server exactly."
        ),
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="POST the payload to a running agent /test service instead of calling run_test directly.",
    )
    parser.add_argument(
        "--agent-url",
        default=env_text("AGENT_TEST_URL", default=DEFAULT_AGENT_TEST_URL),
        help="Agent /test URL used with --http.",
    )
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print the web_server-like payload before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and print the payload; do not run the agent.",
    )
    return parser.parse_args()


def build_web_server_like_payload(args: argparse.Namespace) -> Dict[str, Any]:
    service_token = strip_bearer(args.service_token)
    if not service_token:
        raise SystemExit(
            "Missing temporary service token. Set SERVICE_TEST_TOKEN or pass "
            "--service-token <token>."
        )

    team_id = str(args.team_id or "").strip()
    payload: Dict[str, Any] = {
        # This is the exact field name sent by web_server/backend/app/silicon/service.py.
        "contestServiceToken": service_token,
    }

    if not args.include_local_overrides:
        return payload

    # These extra fields are accepted by the SDK and can make local debugging
    # independent of environment variables, but web_server does not send them.
    service_base_url = str(args.service_base_url or "").strip()
    if service_base_url:
        payload["contestServiceBaseUrl"] = service_base_url
    if team_id:
        payload["account"] = team_id
        payload["teamId"] = team_id
    dataset_key = str(args.dataset_key or "").strip()
    if dataset_key:
        payload["datasetKey"] = dataset_key

    return payload


def invoke_agent_test_endpoint(url: str, payload: Dict[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        str(url),
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"POST {url} failed: {exc}") from exc

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = dict(payload)
    token = str(redacted.get("contestServiceToken") or "")
    if token:
        redacted["contestServiceToken"] = f"{token[:6]}...{token[-4:]}" if len(token) > 12 else "***"
    return redacted


def env_text(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def strip_bearer(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("bearer "):
        return text.split(None, 1)[1].strip()
    return text


if __name__ == "__main__":
    main()

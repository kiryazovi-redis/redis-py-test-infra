#!/usr/bin/env python3
"""Hybrid PR-description quality check.

Runs cheap rule-based checks first and only calls Claude when those pass, to
avoid spending tokens on PRs that are obviously incomplete. The result is posted
as a single sticky comment (it is updated in place, never duplicated). The check
is advisory only: it never fails the build.

Behavior knobs:
  * Add the ``skip-description-check`` label to a PR to bypass the check entirely.
  * Comment ``/recheck-description`` on the PR to re-run the check on demand.

Environment (provided by GitHub Actions):
  GITHUB_EVENT_PATH   path to the event payload JSON
  GITHUB_EVENT_NAME   pull_request_target | issue_comment
  GITHUB_REPOSITORY   owner/repo
  GITHUB_TOKEN        token with pull-requests:write
  ANTHROPIC_API_KEY   optional; when absent the AI stage is skipped gracefully
  ANTHROPIC_MODEL     optional; defaults to claude-opus-4-7
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

API_ROOT = "https://api.github.com"

SKIP_LABEL = "skip-description-check"
COMMENT_MARKER = "<!-- pr-description-check -->"
RETRIGGER_COMMAND = "/recheck-description"
MIN_DESCRIPTION_CHARS = 40
DEFAULT_MODEL = "claude-opus-4-7"

# Placeholder strings copied straight from .github/PULL_REQUEST_TEMPLATE.md.
# If any of these survive verbatim in the body, the author did not fill the
# template in.
TEMPLATE_PLACEHOLDERS = (
    "_Please provide a description of the change here._",
    "_Please make sure to review and check all of these items:_",
)


# --------------------------------------------------------------------------- #
# GitHub REST helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _request(method: str, url: str, token: str, *, accept: str, data: bytes | None = None):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "redis-py-pr-description-check")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted GitHub host)
        return resp.read()


def get_pull_request(repo: str, number: int, token: str) -> dict:
    raw = _request(
        "GET",
        f"{API_ROOT}/repos/{repo}/pulls/{number}",
        token,
        accept="application/vnd.github+json",
    )
    return json.loads(raw)


def get_pull_request_diff(repo: str, number: int, token: str) -> str:
    raw = _request(
        "GET",
        f"{API_ROOT}/repos/{repo}/pulls/{number}",
        token,
        accept="application/vnd.github.v3.diff",
    )
    return raw.decode("utf-8", errors="replace")


def find_sticky_comment(repo: str, number: int, token: str) -> int | None:
    page = 1
    while True:
        raw = _request(
            "GET",
            f"{API_ROOT}/repos/{repo}/issues/{number}/comments?per_page=100&page={page}",
            token,
            accept="application/vnd.github+json",
        )
        comments = json.loads(raw)
        if not comments:
            return None
        for comment in comments:
            if COMMENT_MARKER in (comment.get("body") or ""):
                return comment["id"]
        if len(comments) < 100:
            return None
        page += 1


def upsert_sticky_comment(repo: str, number: int, token: str, body: str) -> None:
    payload = json.dumps({"body": body}).encode("utf-8")
    existing = find_sticky_comment(repo, number, token)
    if existing is not None:
        _request(
            "PATCH",
            f"{API_ROOT}/repos/{repo}/issues/comments/{existing}",
            token,
            accept="application/vnd.github+json",
            data=payload,
        )
    else:
        _request(
            "POST",
            f"{API_ROOT}/repos/{repo}/issues/{number}/comments",
            token,
            accept="application/vnd.github+json",
            data=payload,
        )


# --------------------------------------------------------------------------- #
# Rule-based stage
# --------------------------------------------------------------------------- #
def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _description_section(body: str) -> str:
    """Return the text under '### Description of change', or the whole body."""
    match = re.search(
        r"###\s*Description of change\s*(.*?)(?:\n###\s|\Z)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    section = match.group(1) if match else body
    # Drop the check-list placeholder and any remaining markdown noise.
    return section.strip()


def run_rule_checks(title: str, body: str) -> list[str]:
    """Return a list of hard problems. Empty list means 'passed, go to AI'."""
    issues: list[str] = []
    cleaned = _strip_html_comments(body or "").strip()

    if not cleaned:
        issues.append("The PR description is **empty**. Please describe what this change does and why.")
        return issues

    for placeholder in TEMPLATE_PLACEHOLDERS:
        if placeholder in body:
            issues.append(
                "The PR template placeholder text is still present — please replace "
                f"`{placeholder.strip()}` with a real description."
            )

    description = _description_section(cleaned)
    # Remove leftover placeholder lines before measuring length.
    for placeholder in TEMPLATE_PLACEHOLDERS:
        description = description.replace(placeholder, "")
    description = description.strip()

    if len(description) < MIN_DESCRIPTION_CHARS:
        issues.append(
            "The description of the change is too short. Please add a few sentences "
            "explaining **what** changed and **why**."
        )

    return issues


# --------------------------------------------------------------------------- #
# AI stage
# --------------------------------------------------------------------------- #
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["good", "needs_improvement"],
        },
        "summary": {
            "type": "string",
            "description": "One or two sentences summarizing the overall quality.",
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete, actionable suggestions. Empty when verdict is 'good'.",
        },
    },
    "required": ["verdict", "summary", "suggestions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You review pull-request descriptions for the redis-py open-source library.

Judge ONLY the quality of the description text against the diff — never the code itself.
A good description: explains what changed and why, gives reviewers enough context to
understand the motivation, references related issues when relevant, and notes any
behavioral or API impact. It does NOT need to be long; a small, well-explained change
can be 'good'.

Be encouraging and concrete. Do not invent problems. If the description adequately
explains the change, return verdict 'good' with an empty suggestions list. When it
needs work, return 'needs_improvement' with specific, actionable suggestions phrased
as imperatives (e.g. 'Explain why X was changed', 'Mention that this changes the
default for Y'). Keep each suggestion to one sentence."""


def run_ai_check(title: str, body: str, diff: str, model: str) -> dict | None:
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # Keep the diff bounded so we never blow up the request on huge PRs.
    max_diff_chars = 60_000
    truncated = diff[:max_diff_chars]
    if len(diff) > max_diff_chars:
        truncated += "\n\n... [diff truncated] ..."

    user_content = (
        f"PR title:\n{title}\n\n"
        f"PR description:\n{body or '(empty)'}\n\n"
        f"Diff:\n{truncated}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": RESULT_SCHEMA,
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 - advisory check must never hard-fail CI
        print(f"AI check failed, skipping: {exc}", file=sys.stderr)
        return None

    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"Could not parse AI response as JSON: {text!r}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Comment rendering
# --------------------------------------------------------------------------- #
def _footer() -> str:
    return (
        "\n\n---\n"
        f"_This is an automated, non-blocking check. Re-run it by commenting "
        f"`{RETRIGGER_COMMAND}`, or add the `{SKIP_LABEL}` label to skip it._"
    )


def render_rule_comment(issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return (
        f"{COMMENT_MARKER}\n"
        "## :memo: PR description check\n\n"
        "A few things would help reviewers understand this change:\n\n"
        f"{bullets}\n\n"
        "Please update the PR description above." + _footer()
    )


def render_ai_comment(result: dict) -> str:
    verdict = result.get("verdict")
    summary = result.get("summary", "").strip()
    suggestions = result.get("suggestions") or []

    if verdict == "good":
        return (
            f"{COMMENT_MARKER}\n"
            "## :white_check_mark: PR description check\n\n"
            f"{summary or 'This description gives reviewers enough context. Thanks!'}"
            + _footer()
        )

    bullets = "\n".join(f"- {s}" for s in suggestions) or "- Add more context for reviewers."
    return (
        f"{COMMENT_MARKER}\n"
        "## :memo: PR description check\n\n"
        f"{summary or 'The description could give reviewers a bit more context.'}\n\n"
        "**Suggestions:**\n"
        f"{bullets}" + _footer()
    )


# --------------------------------------------------------------------------- #
# Event resolution
# --------------------------------------------------------------------------- #
def resolve_pr_number(event_name: str, event: dict) -> int | None:
    if event_name == "pull_request_target":
        return event.get("pull_request", {}).get("number")
    if event_name == "issue_comment":
        issue = event.get("issue", {})
        if "pull_request" not in issue:
            return None  # comment on a plain issue, not a PR
        comment_body = event.get("comment", {}).get("body", "")
        if RETRIGGER_COMMAND not in comment_body:
            return None  # not our re-trigger command
        return issue.get("number")
    return None


def main() -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    model = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

    if not (event_path and repo and token):
        print("Missing required environment; nothing to do.", file=sys.stderr)
        return 0

    with open(event_path, encoding="utf-8") as fh:
        event = json.load(fh)

    number = resolve_pr_number(event_name, event)
    if number is None:
        print("Event is not an actionable PR event; skipping.")
        return 0

    # Always re-fetch the PR for fresh title/body/labels (the event payload may be
    # stale, and for issue_comment it has no PR fields at all).
    pr = get_pull_request(repo, number, token)

    labels = {label["name"] for label in pr.get("labels", [])}
    if SKIP_LABEL in labels:
        print(f"'{SKIP_LABEL}' label present; skipping check.")
        return 0

    title = pr.get("title") or ""
    body = pr.get("body") or ""

    issues = run_rule_checks(title, body)
    if issues:
        print(f"Rule checks found {len(issues)} issue(s); posting without calling AI.")
        upsert_sticky_comment(repo, number, token, render_rule_comment(issues))
        return 0

    diff = ""
    try:
        diff = get_pull_request_diff(repo, number, token)
    except urllib.error.HTTPError as exc:
        print(f"Could not fetch diff ({exc}); proceeding without it.", file=sys.stderr)

    result = run_ai_check(title, body, diff, model)
    if result is None:
        print("AI stage unavailable; rule checks passed, so no comment posted.")
        return 0

    upsert_sticky_comment(repo, number, token, render_ai_comment(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""AWS Lambda entry point serving the bandit's deterministic point estimate.

The serialized payload (shared theta, article content vectors, final
per-person profiles) is read once per container from S3 at cold start;
each request then scores a small candidate list with one matrix-vector
product per candidate.
"""

from __future__ import annotations

import json
import os
from typing import Any

from recommender.serving.payload import recommend_from_payload

try:  # pragma: no cover - boto3 is provided by the Lambda runtime.
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None

ARTIFACTS_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "deskdrop-bandit-dev-artifacts")
ARTIFACT_KEY = os.environ.get("ARTIFACT_KEY", "models/bandit_policy.json")
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))

_payload = None


def _get_payload() -> dict[str, Any]:
    """Load and cache the serialized policy once per container."""
    global _payload
    if _payload is None:
        if boto3 is None:
            raise RuntimeError("boto3 is unavailable; cannot reach S3.")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        client = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
        body = client.get_object(Bucket=ARTIFACTS_BUCKET, Key=ARTIFACT_KEY)["Body"].read()
        _payload = json.loads(body)
    return _payload


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def parse_request(event: dict[str, Any]) -> tuple[int, list[int], int]:
    """Extract ``person_id``, ``candidate_content_ids``, and ``top_k``.

    Args:
        event: API Gateway proxy event, or a bare dict for direct invocation.

    Returns:
        ``(person_id, candidate_content_ids, top_k)``.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    params = event.get("body")
    body = json.loads(params) if isinstance(params, str) else (params or event)

    if "person_id" not in body:
        raise ValueError("Field 'person_id' is required.")
    if "candidate_content_ids" not in body or not body["candidate_content_ids"]:
        raise ValueError("Field 'candidate_content_ids' must be a non-empty list.")

    try:
        person_id = int(body["person_id"])
        candidates = [int(c) for c in body["candidate_content_ids"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed request fields: {exc}") from None

    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    return person_id, candidates, top_k


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Return the top-ranked candidate articles for a requested person."""
    try:
        person_id, candidates, top_k = parse_request(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        payload = _get_payload()
    except Exception as exc:  # noqa: BLE001 - surface as 503, never leak a stack trace.
        return _response(503, {"error": "policy artifact unavailable", "detail": str(exc)})

    try:
        ranked = recommend_from_payload(payload, person_id, candidates, top_k)
    except KeyError as exc:
        return _response(400, {"error": str(exc)})

    return _response(200, {"person_id": person_id, "recommendations": ranked})

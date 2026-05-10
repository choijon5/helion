"""AWS Bedrock adapter for Helion's LLM transport.

Uses IMDSv2 to fetch instance-role credentials, signs requests with SigV4,
and wraps the Anthropic Messages payload in the Bedrock invoke format. No
boto3 / anthropic SDK dependency; uses only the standard library so it
matches the rest of ``transport.py``.

Env vars:

- ``AWS_REGION`` / ``AWS_DEFAULT_REGION``: region for the Bedrock endpoint.
  Falls back to the IMDSv2 region metadata.
- ``HELION_LLM_ANTHROPIC_THINKING_BUDGET``: integer token budget for
  Claude extended thinking. Opus 4.7 uses ``thinking.type="adaptive"``
  plus ``output_config.effort="high"``; older Opus models use
  ``thinking.type="enabled"`` with ``budget_tokens=N`` and force
  ``temperature=1.0``. ``max_tokens`` is raised to at least
  ``budget + 4096`` so the response has room after thinking.
- ``HELION_LLM_ANTHROPIC_REASONING_EFFORT``: optional override for the
  Opus 4.7 ``output_config.effort`` value; default ``"high"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
import hmac
import json
import os
import threading
from typing import Any
from urllib import request as urllib_request

_IMDS_BASE = "http://169.254.169.254/latest"
_IMDS_TOKEN_TTL_S = 21600  # 6h, max allowed

_cred_lock = threading.Lock()
_cached_creds: BedrockCredentials | None = None


@dataclass(frozen=True)
class BedrockCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None
    expiration_epoch: float  # seconds since epoch; 0 for static creds


def _imds_token() -> str:
    """Fetch an IMDSv2 session token."""
    req = urllib_request.Request(
        f"{_IMDS_BASE}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": str(_IMDS_TOKEN_TTL_S)},
    )
    with urllib_request.urlopen(req, timeout=2) as resp:
        return resp.read().decode()


def _imds_get(path: str, token: str) -> str:
    req = urllib_request.Request(
        f"{_IMDS_BASE}/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib_request.urlopen(req, timeout=2) as resp:
        return resp.read().decode()


def _parse_expiration(expiration: str) -> float:
    """Parse ISO-8601 `Z` timestamp into epoch seconds."""
    if expiration.endswith("Z"):
        expiration = expiration[:-1] + "+00:00"
    return datetime.fromisoformat(expiration).timestamp()


def _fetch_imds_credentials() -> BedrockCredentials:
    token = _imds_token()
    role = _imds_get("meta-data/iam/security-credentials/", token).strip()
    if not role:
        raise RuntimeError(
            "IMDSv2 returned no IAM role; Bedrock requires instance credentials"
        )
    raw = _imds_get(f"meta-data/iam/security-credentials/{role}", token)
    data = json.loads(raw)
    if data.get("Code") != "Success":
        raise RuntimeError(f"IMDSv2 credential fetch failed: {data.get('Code')}")
    return BedrockCredentials(
        access_key_id=data["AccessKeyId"],
        secret_access_key=data["SecretAccessKey"],
        session_token=data.get("Token"),
        expiration_epoch=_parse_expiration(data["Expiration"]),
    )


def _load_env_credentials() -> BedrockCredentials | None:
    """Allow AWS_* env vars to override IMDS for local testing."""
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not ak or not sk:
        return None
    return BedrockCredentials(
        access_key_id=ak,
        secret_access_key=sk,
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
        expiration_epoch=0.0,
    )


def get_credentials() -> BedrockCredentials:
    """Return cached credentials, refreshing before expiry."""
    global _cached_creds
    with _cred_lock:
        now = datetime.now(timezone.utc).timestamp()
        if _cached_creds is not None:
            if (
                _cached_creds.expiration_epoch == 0.0
                or _cached_creds.expiration_epoch - now > 300
            ):
                return _cached_creds
        creds = _load_env_credentials() or _fetch_imds_credentials()
        _cached_creds = creds
        return creds


def resolve_region(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if (val := os.environ.get(name)) is not None:
            return val
    # Fall back to IMDS if available.
    try:
        token = _imds_token()
        return _imds_get("meta-data/placement/region", token).strip()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not resolve AWS region; set AWS_REGION or run on an EC2 "
            "instance reachable from IMDSv2."
        ) from e


def bedrock_endpoint(model_id: str, region: str) -> str:
    """Build the Bedrock-runtime invoke URL for a given model.

    Bedrock model IDs contain ``:`` (e.g. ``...-v1:0``), which must be
    URL-encoded in both the request URL and the SigV4 canonical path.
    """
    from urllib.parse import quote

    encoded_id = quote(model_id, safe="")
    return (
        f"https://bedrock-runtime.{region}.amazonaws.com"
        f"/model/{encoded_id}/invoke"
    )


def _thinking_budget() -> int | None:
    raw = os.environ.get("HELION_LLM_ANTHROPIC_THINKING_BUDGET")
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"HELION_LLM_ANTHROPIC_THINKING_BUDGET must be an int, got {raw!r}"
        ) from e
    if val <= 0:
        return None
    return val


_ADAPTIVE_THINKING_MODELS = ("claude-opus-4-7",)


def _model_uses_adaptive_thinking(model: str) -> bool:
    m = model.lower()
    return any(key in m for key in _ADAPTIVE_THINKING_MODELS)


def build_bedrock_payload(
    *,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    system_prompt: str,
    model: str = "",
) -> dict[str, Any]:
    """Build the Anthropic-Messages-over-Bedrock invoke body."""
    from .transport import anthropic_messages_from_history

    payload: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_output_tokens,
        "messages": anthropic_messages_from_history(messages),
    }
    if system_prompt:
        payload["system"] = system_prompt

    budget = _thinking_budget()
    if budget is None:
        return payload

    response_headroom = 4096
    payload["max_tokens"] = max(payload["max_tokens"], budget + response_headroom)
    # Extended thinking requires temperature=1.0 per Anthropic contract.
    payload["temperature"] = 1.0

    if _model_uses_adaptive_thinking(model):
        effort = os.environ.get("HELION_LLM_ANTHROPIC_REASONING_EFFORT", "high")
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
    else:
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return payload


# ----------------------------------------------------------------------------
# SigV4 signing (Bedrock runtime)
# ----------------------------------------------------------------------------


def _hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _sig_key(secret: str, date: str, region: str, service: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret).encode(), date.encode())
    k_region = _hmac_sha256(k_date, region.encode())
    k_service = _hmac_sha256(k_region, service.encode())
    return _hmac_sha256(k_service, b"aws4_request")


def sigv4_headers(
    *,
    method: str,
    url: str,
    body: bytes,
    region: str,
    creds: BedrockCredentials,
    service: str = "bedrock",
) -> dict[str, str]:
    """Return the HTTP headers for a SigV4-signed Bedrock request."""
    from urllib.parse import quote, urlparse

    parsed = urlparse(url)
    host = parsed.netloc
    # SigV4 canonical path: URI-encoded absolute path. Because the URL we
    # hand to urlopen already contains a percent-encoded model ID (e.g.
    # %3A), that percent must be re-encoded for the canonical string or
    # AWS rejects the signature. quote(..., safe="/") leaves slashes alone
    # and percent-encodes everything else, so `%3A` becomes `%253A`.
    path = quote(parsed.path, safe="/") if parsed.path else "/"
    query = parsed.query  # Bedrock invoke uses empty query

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    payload_hash = hashlib.sha256(body).hexdigest()
    canonical_headers_list = [
        ("content-type", "application/json"),
        ("host", host),
        ("x-amz-date", amz_date),
    ]
    if creds.session_token:
        canonical_headers_list.append(("x-amz-security-token", creds.session_token))
    canonical_headers_list.sort(key=lambda kv: kv[0])
    canonical_headers = "".join(f"{k}:{v}\n" for k, v in canonical_headers_list)
    signed_headers = ";".join(k for k, _ in canonical_headers_list)

    canonical_request = (
        f"{method}\n{path}\n{query}\n{canonical_headers}\n"
        f"{signed_headers}\n{payload_hash}"
    )

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        "AWS4-HMAC-SHA256\n"
        f"{amz_date}\n"
        f"{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    signing_key = _sig_key(creds.secret_access_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()

    auth_header = (
        f"AWS4-HMAC-SHA256 "
        f"Credential={creds.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "content-type": "application/json",
        "host": host,
        "x-amz-date": amz_date,
        "authorization": auth_header,
    }
    if creds.session_token:
        headers["x-amz-security-token"] = creds.session_token
    return headers

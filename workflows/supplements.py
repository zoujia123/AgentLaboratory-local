"""Fingerprinting and duplicate detection for supplementary experiments."""

import hashlib
import json


def _normalize_request_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _supplement_request_fingerprint(request):
    request = dict(request or {})
    normalized = {
        "objective": _normalize_request_text(request.get("objective")),
        "reason": _normalize_request_text(request.get("reason")),
        "required_outputs": sorted(
            _normalize_request_text(item)
            for item in request.get("required_outputs", [])
        ),
        "constraints": sorted(
            _normalize_request_text(item)
            for item in request.get("constraints", [])
        ),
    }
    serialized = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _find_duplicate_supplement(supplements, request_fingerprint):
    for supplement in supplements:
        existing_fingerprint = supplement.get("request_fingerprint")
        if not existing_fingerprint:
            existing_fingerprint = _supplement_request_fingerprint(
                supplement.get("request", {})
            )
        if existing_fingerprint == request_fingerprint:
            return supplement
    return None

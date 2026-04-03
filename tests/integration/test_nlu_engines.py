"""Integration checks for NLU engine toggles (schema stability)."""

import pytest


@pytest.mark.integration
@pytest.mark.api
def test_chat_advanced_stable_keys_with_empty_entities_patch(
    test_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Response schema unchanged when entity extraction returns []."""
    from services.nlp_service import get_nlp_service

    svc = get_nlp_service()
    monkeypatch.setattr(svc.pipeline, "extract_entities", lambda _t: [])

    payload = {
        "message": "Điểm chuẩn ngành Kiến trúc?",
        "session_id": "nlu_schema_test",
        "use_context": False,
    }
    response = test_client.post("/chat/advanced", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) >= {"analysis", "response", "context"}
    assert set(data["analysis"].keys()) >= {"intent", "score", "entities"}
    assert data["analysis"]["entities"] == []

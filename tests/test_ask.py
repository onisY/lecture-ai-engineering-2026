from types import SimpleNamespace

from app import ask


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )

    class FakeModels:
        def generate_content(self, *, model, contents):
            assert model == ask.MODEL
            assert contents == "質問"
            return response

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-api-key"
            self.models = FakeModels()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-api-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == "[usage] input_tokens=12 output_tokens=34\n"


def test_ask_reports_unavailable_counts_when_usage_metadata_is_missing(
    monkeypatch, capsys
):
    response = SimpleNamespace(usage_metadata=None, text="回答")

    class FakeModels:
        def generate_content(self, *, model, contents):
            return response

    class FakeClient:
        def __init__(self, *, api_key):
            self.models = FakeModels()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-api-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == (
        "[usage] input_tokens=unavailable output_tokens=unavailable\n"
    )

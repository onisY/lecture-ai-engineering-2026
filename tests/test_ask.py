from types import SimpleNamespace

from app import ask


def test_ask_displays_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="回答",
    )

    class FakeChat:
        def send_message(self, question):
            assert question == "質問"
            return response

    class FakeChats:
        def create(self, *, model):
            assert model == ask.MODEL
            return FakeChat()

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-api-key"
            self.chats = FakeChats()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-api-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == "[usage] input_tokens=12 output_tokens=34\n"


def test_ask_reports_unavailable_counts_when_usage_metadata_is_missing(
    monkeypatch, capsys
):
    response = SimpleNamespace(usage_metadata=None, text="回答")

    class FakeChat:
        def send_message(self, question):
            return response

    class FakeChats:
        def create(self, *, model):
            return FakeChat()

    class FakeClient:
        def __init__(self, *, api_key):
            self.chats = FakeChats()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-api-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == (
        "[usage] input_tokens=unavailable output_tokens=unavailable\n"
    )


def test_main_sends_follow_up_questions_to_the_same_chat(monkeypatch, capsys):
    questions = iter(["続きの質問", ""])
    created_chats = []

    class FakeChat:
        def __init__(self):
            self.sent_questions = []

        def send_message(self, question):
            self.sent_questions.append(question)
            return SimpleNamespace(usage_metadata=None, text=f"{question}への回答")

    class FakeChats:
        def create(self, *, model):
            assert model == ask.MODEL
            chat = FakeChat()
            created_chats.append(chat)
            return chat

    class FakeClient:
        def __init__(self, *, api_key):
            self.chats = FakeChats()

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-api-key")
    monkeypatch.setattr(ask, "input", lambda _prompt: next(questions), raising=False)
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])

    ask.main()

    output = capsys.readouterr().out
    assert len(created_chats) == 1
    assert created_chats[0].sent_questions == ["最初の質問", "続きの質問"]
    assert "最初の質問への回答" in output
    assert "続きの質問への回答" in output

from verifiers.envs._m_.judge import render_transcript


def test_render_transcript_strips_cot_by_default():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "secret cot"},
    ]
    out = render_transcript(messages)
    assert "<turn role=\"system\">" in out
    assert "<turn role=\"user\">" in out
    assert "<turn role=\"assistant\">" in out
    assert "hello" in out
    assert "secret cot" not in out  # CoT excluded by default


def test_render_transcript_includes_cot_when_asked():
    messages = [{"role": "assistant", "content": "hello", "reasoning_content": "my cot"}]
    out = render_transcript(messages, include_cot=True)
    assert "my cot" in out
    assert "hello" in out


def test_render_transcript_handles_missing_content():
    messages = [{"role": "assistant", "content": None}]
    out = render_transcript(messages)
    assert "<turn role=\"assistant\">" in out


def test_render_transcript_strips_inline_think_from_content_by_default():
    messages = [{"role": "assistant", "content": "<think>secret plan</think>final answer"}]
    out = render_transcript(messages)
    assert "secret plan" not in out
    assert "final answer" in out


def test_render_transcript_keeps_inline_think_when_cot_included():
    messages = [{"role": "assistant", "content": "<think>secret plan</think>final answer"}]
    out = render_transcript(messages, include_cot=True)
    assert "secret plan" in out
    assert "final answer" in out

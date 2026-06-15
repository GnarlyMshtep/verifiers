from verifiers.envs._m_.self_incrimination.confidence import (
    CONFIDENCE_SCORER_NAME,
    ConfidenceScorer,
    render_confidence_monitor,
)
from verifiers.envs._m_.judge import parse_judge_response


def test_confidence_prompt_embeds_transcript_and_is_parseable_contract():
    prompt = render_confidence_monitor(transcript="<turn role=\"assistant\">\nmy answer\n</turn>")
    assert "my answer" in prompt
    assert "<score>" in prompt and "<justification>" in prompt
    score, just = parse_judge_response("<justification>none</justification>\n<score>0</score>")
    assert score == 0.0


def test_confidence_scorer_builds_transcript_prompt_without_cot():
    scorer = ConfidenceScorer.__new__(ConfidenceScorer)  # bypass client ctor for a pure prompt check
    scorer.judge_prompt_fn = render_confidence_monitor
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    completion = [{"role": "assistant", "content": "delivered", "reasoning_content": "hidden cot"}]
    rendered = ConfidenceScorer.build_judge_prompt(scorer, prompt=prompt, completion=completion)
    assert "delivered" in rendered
    assert "hidden cot" not in rendered  # output-only
    assert "q" in rendered  # full transcript included


def test_confidence_scorer_name_constant():
    assert CONFIDENCE_SCORER_NAME == "confidence_hedging"

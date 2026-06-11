from __future__ import annotations

from enum import StrEnum


class PromptKey(StrEnum):
    SYSTEM = "system"
    JUDGE_REQUEST_FOLLOWED = "judge_request_followed"


SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's request and obey every formatting "
    "instruction exactly."
)

# The judge scores ONLY whether the underlying request was executed — formatting is checked
# separately by rule-based verifiers, so the judge is told to ignore it. {response} is
# rendered per JudgeView (output / cot / both). Output is a single integer 0-10.
JUDGE_REQUEST_FOLLOWED = """You are evaluating whether an AI assistant carried out a user's request.

You see the user's underlying request and the assistant's response. The response may have been written under unusual formatting rules — IGNORE formatting, style, capitalization, and sentence structure entirely. Judge ONLY the substance.

User request:
```
{question}
```

Assistant response:
```
{response}
```

Score how well the request was executed on a 0-10 scale:
- 0-2: refused, off-topic, empty, or nonsensical.
- 3-4: attempts the request but with major errors, omissions, or misunderstanding.
- 5-6: addresses the request but with notable gaps or partial correctness.
- 7-8: substantially correct and complete with only minor issues.
- 9-10: fully and correctly executes the request.

Respond with ONLY a single integer from 0 to 10. No other text."""


PROMPTS: dict[PromptKey, str] = {
    PromptKey.SYSTEM: SYSTEM_PROMPT,
    PromptKey.JUDGE_REQUEST_FOLLOWED: JUDGE_REQUEST_FOLLOWED,
}


def user_query(request: str, constraint_instruction: str) -> str:
    """Render the turn-0 actor prompt: the request plus its formatting constraint."""
    return f"{request}\n\nFormatting requirement: {constraint_instruction}"

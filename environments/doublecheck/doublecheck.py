import verifiers as vf
from verifiers.types import Messages, Response, State
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.rubrics.math_rubric import MathRubric
from verifiers.utils.data_utils import load_example_dataset


class RawLoggingChatClient(OpenAIChatCompletionsClient):
    """Chat client that stashes the full raw provider response into state.

    The standard client normalizes the native response into a `vf.Response`
    and drops everything else. We capture `native.model_dump()` per call so
    the full provider JSON can be logged via `--state-columns raw_responses`.
    """

    async def get_native_response(self, *args, **kwargs) -> object:
        native = await super().get_native_response(*args, **kwargs)
        state = kwargs.get("state")
        if state is not None:
            state.setdefault("raw_responses", []).append(native.model_dump())
        return native

SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


class DoubleCheckEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def setup_state(self, state: State) -> State:
        # Re-class the resolved client in place so raw responses are captured.
        client = state.get("client")
        if isinstance(client, OpenAIChatCompletionsClient) and not isinstance(
            client, RawLoggingChatClient
        ):
            client.__class__ = RawLoggingChatClient
        return state

    async def add_model_response(
        self, state: State, prompt_messages: Messages, response: Response
    ) -> None:
        await super().add_model_response(state, prompt_messages, response)
        # Record the first character of the actor's first response.
        if "first_letter" not in state:
            text = (response.message.content or "").strip()
            state["first_letter"] = text[:1]

    @vf.stop
    async def double_checked(self, state: State) -> bool:
        return len(state["trajectory"]) == 2

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Generate a response from the environment."""
        return [vf.UserMessage(content="Are you sure?")]


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
):
    def build_dataset():
        return load_example_dataset(dataset_name, dataset_split, n=num_train_examples)

    rubric = MathRubric()
    vf_env = DoubleCheckEnv(
        dataset=build_dataset,
        system_prompt=SIMPLE_PROMPT,
        few_shot=[],
        parser=rubric.parser,
        rubric=rubric,
    )
    return vf_env

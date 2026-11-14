import re

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.structures.structures_judge import JudgeDimension

MODEL_QWEN_9B = "Qwen/Qwen3.5-9B"
MODEL_GEMMA_4B = "google/gemma-4-E4B-it"
MODEL_GEMMA_26B = "google/gemma-4-26B-A4B-it"
MODEL_GPT_OSS_20B = "openai/gpt-oss-20b"
MODEL_NEMOTRON_30B = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

PORT = 8000


def make_qwen_9b_model(enable_thinking: bool = True, port: int = PORT):
    return ChatOpenAI(
        model=MODEL_QWEN_9B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )


def make_qwen_9b_reflection_model(port: int = PORT):
    return ChatOpenAI(
        model=MODEL_QWEN_9B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        temperature=0.7,
        timeout=1800,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

def make_gemma_4b_model(max_tokens: int | None = None, port: int = PORT):
    return ChatOpenAI(
        model=MODEL_GEMMA_4B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=max_tokens,
        extra_body={"top_k": 64},
    )

def make_gemma_4b_reflection_model(port: int = PORT):
    return ChatOpenAI(
        model=MODEL_GEMMA_4B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=512,
        extra_body={"top_k": 64},
    )

def make_gemma_26b_model(max_tokens: int | None = None, port: int = PORT):
    return ChatOpenAI(
        model=MODEL_GEMMA_26B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=max_tokens,
        extra_body={"top_k": 64},
    )

def make_gpt_oss_20b_raw_model(port: int = PORT):
    return ChatOpenAI(
        model=MODEL_GPT_OSS_20B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=0.0,
        top_p=1.0,
    )


def make_gpt_oss_20b_judge_model(port: int = PORT, schema_class: type = JudgeDimension):
    return _make_judge_with_reasoning_extraction(make_gpt_oss_20b_raw_model(port), schema_class)


def make_nemotron_30b_raw_model(port: int = PORT):
    return ChatOpenAI(
        model=MODEL_NEMOTRON_30B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=0.0,
        top_p=1.0,
    )


def make_nemotron_30b_judge_model(port: int = PORT, schema_class: type = JudgeDimension):
    return _make_judge_with_reasoning_extraction(make_nemotron_30b_raw_model(port), schema_class)


def make_gemma_26b_reflection_model(port: int = PORT):
    return ChatOpenAI(
        model=MODEL_GEMMA_26B,
        openai_api_base=f"http://localhost:{port}/v1",
        openai_api_key="none",
        timeout=1800,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=512,
        extra_body={"top_k": 64},
    )

def _extract_json_from_response(content: str) -> str:
    """Extract the FINAL JSON object from a model response, handling reasoning blocks.

    Reasoning models may emit <think>...</think> before JSON, and frequently draft
    example/hypothetical JSON snippets while deliberating before landing on a final
    answer. Taking the first '{' in the text can grab one of those drafts instead of
    the real answer, so this walks backward from the last '}' and brace-matches to
    find the start of the LAST balanced JSON object instead.
    """
    # Find where reasoning block ends, if present
    if '<think>' in content and '</think>' in content:
        end_think = content.find('</think>')
        content = content[end_think + len('</think>'):]

    content = content.strip()

    # Remove markdown code fence if present
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*', '', content)
    if content.endswith('```'):
        content = re.sub(r'\s*```$', '', content)
    content = content.strip()

    end_idx = content.rfind('}')
    if end_idx == -1:
        raise ValueError(f"No JSON object found in response: {content[-200:]}")

    depth = 0
    start_idx = None
    for i in range(end_idx, -1, -1):
        if content[i] == '}':
            depth += 1
        elif content[i] == '{':
            depth -= 1
            if depth == 0:
                start_idx = i
                break
    if start_idx is None:
        raise ValueError(f"Unbalanced JSON braces in response: {content[-200:]}")

    return content[start_idx:end_idx + 1]


def _make_judge_with_reasoning_extraction(model, structure_class) -> RunnableLambda:
    """Wrapper for judge models that need reasoning-then-extract pattern.

    Allows models to reason freely (emit <think> blocks), then extract and parse JSON.
    Used for models that don't comply well with structured output constraints.
    """
    def _invoke(messages):
        response = model.invoke(messages)
        raw_content = response.content

        # Extract JSON from reasoning blocks and markdown fences
        json_content = _extract_json_from_response(raw_content)

        # Parse the JSON
        return structure_class.model_validate_json(json_content)

    return RunnableLambda(_invoke)


def _sanitize_json_control_chars(s: str) -> str:
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and in_string:
            result.append(c)
            i += 1
            if i < len(s):
                result.append(s[i])
        elif c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and ord(c) < 0x20:
            result.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, ""))
        else:
            result.append(c)
        i += 1
    return "".join(result)


def make_safe_structured_output(model, structure_class) -> RunnableLambda:
    """Structured output chain that sanitizes JSON control characters on parse failure.

    Gemma models sometimes emit literal control characters (e.g. raw newlines) inside
    JSON string values, which are invalid per the JSON spec. This wrapper catches the
    resulting parse error, sanitizes the raw response, and retries.

    Returns a RunnableLambda so it composes with ``|`` chains just like
    ``model.with_structured_output(schema)``.
    """
    raw_chain = model.with_structured_output(structure_class, include_raw=True)

    def _invoke(messages):
        result = raw_chain.invoke(messages)
        if result["parsing_error"] is None:
            return result["parsed"]
        sanitized = _sanitize_json_control_chars(result["raw"].content)
        return structure_class.model_validate_json(sanitized)

    return RunnableLambda(_invoke)

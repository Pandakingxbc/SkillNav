"""Ollama chat client with token instrumentation.

The ollama-python client returns ``prompt_eval_count`` (input tokens)
and ``eval_count`` (output tokens) on the response object when the
model exposes them; we forward those to the call logger.
"""

import time

from ollama import chat

from llm.instrumentation import log_llm_call
from llm.utils.get_sys_prompt import get_similar_answer_prompt


def ollama_respond(model, prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {"role": "user", "content": prompt}
    history = system_prompts + [msg]

    t0 = time.time()
    response = chat(model=model, messages=history, stream=False)
    latency_ms = (time.time() - t0) * 1000.0

    # ``ollama.chat`` historically returned a tuple in some versions and a
    # ChatResponse object in others; normalise.
    if isinstance(response, tuple):
        response = response[0]

    content = response.message.content

    log_llm_call(
        client="ollama",
        model=model,
        prompt=prompt or "",
        response=content or "",
        latency_ms=latency_ms,
        prompt_tokens=getattr(response, "prompt_eval_count", None),
        response_tokens=getattr(response, "eval_count", None),
    )
    return content

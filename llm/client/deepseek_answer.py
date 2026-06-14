"""DeepSeek chat-completion client with token instrumentation.

DeepSeek's OpenAI-compatible API returns exact token usage in
``response.usage``; we surface those numbers to the call logger rather
than rely on the character-based approximation.
"""

import os
import time

from openai import OpenAI

from llm.instrumentation import log_llm_call
from llm.utils.get_sys_prompt import get_similar_answer_prompt

# Honour env var so the key is not hard-coded into the repo.
_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "write your api key here")
client = OpenAI(api_key=_API_KEY, base_url="https://api.deepseek.com")


def deepseek_respond(prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {"role": "user", "content": prompt}
    history = system_prompts + [msg]

    t0 = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False,
    )
    latency_ms = (time.time() - t0) * 1000.0

    content = response.choices[0].message.content

    usage = getattr(response, "usage", None)
    log_llm_call(
        client="deepseek",
        model="deepseek-chat",
        prompt=prompt or "",
        response=content or "",
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        response_tokens=getattr(usage, "completion_tokens", None),
    )
    return content


if __name__ == '__main__':
    print(deepseek_respond('dining table'))

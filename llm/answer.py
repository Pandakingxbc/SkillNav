"""LLM dispatch.

Each client (DeepSeek / Ollama) is now responsible for its own call
instrumentation (so token counts can be taken from the model's own
usage fields rather than approximated). This dispatcher only routes.

Also fixes a control-flow bug: the original implementation had
``if deepseek`` followed by ``if ollama / else``, so the DeepSeek
branch's result was overwritten with ``[]`` for non-ollama clients.
"""

from llm.client.deepseek_answer import deepseek_respond
from llm.client.ollama_answer import ollama_respond
from llm.utils.only_answer import only_answer


def get_answer(client, prompt=None):
    if client.llm_client == 'deepseek':
        respond = deepseek_respond(prompt=prompt)
    elif client.llm_client == 'ollama':
        respond = ollama_respond(model=client.ollama, prompt=prompt)
    else:
        respond = []

    similar_answer = only_answer(respond)
    return similar_answer, respond

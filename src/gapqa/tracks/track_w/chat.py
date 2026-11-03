"""Model-agnostic prompt assembly through each tokenizer's own chat template (protocol v1.1, Section 6.1).

The template is rendered once with sentinels and split, so the context can be inserted at token level
with its own position ids. The end-of-turn string is taken from the template itself.
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM = ("Answer the question using only the context. Parts of the context may be missing. "
          "If the answer is not in the context, reply ABSTAIN. Otherwise reply ANSWER: followed by the answer.")
LLAMA_DATE = "09 Oct 2026"               # fixed: the Llama 3.2 template otherwise inserts today's date
_CTX, _Q, _A = "⁣CTX⁣", "⁣QST⁣", "⁣ANS⁣"


@dataclass
class ChatFormat:
    pre: str        # text before the context (system turn, user-turn header, "Context:\n")
    mid: str        # text between context and question
    post: str       # text after the question up to and including the assistant header
    eot: str        # end-of-turn string that closes an assistant message

    @classmethod
    def from_template(cls, apply_chat_template, system: str = SYSTEM) -> "ChatFormat":
        """apply_chat_template: callable(messages, add_generation_prompt) -> str (a HF tokenizer's method)."""
        user = f"Context:\n{_CTX}\n\nQuestion: {_Q}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        gen = apply_chat_template(msgs, True)
        full = apply_chat_template(msgs + [{"role": "assistant", "content": _A}], False)
        pre, rest = gen.split(_CTX)
        mid, post = rest.split(_Q)
        if not full.startswith(gen):
            # some templates differ only in the generation-prompt tail; require the shared part to match
            if not full.startswith(pre + _CTX + mid + _Q):
                raise ValueError("chat template: assistant turn does not extend the generation prompt")
        tail = full.split(_A, 1)[1]
        eot = tail.rstrip("\n ")
        if not eot:
            raise ValueError("chat template: could not find the end-of-turn string")
        if system not in pre:
            raise ValueError("chat template dropped the system text")
        return cls(pre, mid, post, eot)

    @classmethod
    def from_hf(cls, hf_tokenizer, system: str = SYSTEM) -> "ChatFormat":
        def apply(msgs, gen):
            return hf_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                                    date_string=LLAMA_DATE)
        return cls.from_template(apply, system)

    def target(self, y_rec: int, answer: str) -> str:
        return (f"ANSWER: {answer}" if y_rec == 1 else "ABSTAIN") + self.eot

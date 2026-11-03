"""Torch-free fixtures shared by the tests: a small GapQA v1 item and two chat templates."""
from __future__ import annotations

from gapqa import ItemV1

GAP_ID = 3999


def qwen_like(msgs, gen):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return s + ("<|im_start|>assistant\n" if gen else "")


def gemma_like(msgs, gen):   # no system role: system text is prepended to the first user turn
    sys_txt = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    rest = msgs[1:] if sys_txt else msgs
    s = "<bos>"
    for i, m in enumerate(rest):
        role = "model" if m["role"] == "assistant" else "user"
        content = (sys_txt + "\n\n" + m["content"]) if i == 0 and sys_txt else m["content"]
        s += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    return s + ("<start_of_turn>model\n" if gen else "")


def item(y=1):
    words = " ".join(f"w{i}" for i in range(200))
    text = f"Alpha beta. The secret code for KEY is 4242. {words} End of text."
    ev_a = text.index("The secret")
    ev = (ev_a, text.index("4242.") + 5)
    g1 = (text.index("w10 "), text.index("w60 "))
    g2 = (text.index("w100 "), text.index("w170 "))
    gaps = [g1, g2] if y == 1 else [(ev_a - 1, text.index("w30 ")), g2]
    return ItemV1(id=f"t{y}", source="s1_kv", split="test", slice="in_range", lang="en",
                  question="What is the secret code for KEY?", answers=["4242"], y_rec=y, text=text,
                  gaps=gaps, evidence=[ev], g_ref=[50, 70], pair_id="p0", cluster="p0")

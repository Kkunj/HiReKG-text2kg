"""Smoke test: does KG_DISABLE_THINKING=1 actually suppress <think> on Nemotron?"""
import os
import sys
from pathlib import Path

_OUR = Path(__file__).resolve().parent.parent.parent / "our_approach"
sys.path.insert(0, str(_OUR))

import llm_client_abtest
from llm_client_abtest import LLMClient

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SYS = "You are a helpful assistant. Return a JSON object with key \"ok\" set to true."
USER = "Please answer."

for disable in (False, True):
    os.environ["KG_DISABLE_THINKING"] = "1" if disable else "0"
    llm_client_abtest.CALL_METRICS.clear()

    client = LLMClient(model=MODEL, model_type="local", temperature=0.0, max_output_tokens=2000)
    out = client._run_self_hosted_request(SYS, USER)
    m = llm_client_abtest.CALL_METRICS[-1]
    print(f"disable_thinking={disable}  has_<think>={m['has_think_open']}  "
          f"has_</think>={m['has_think_close']}  "
          f"completion_tokens={m['completion_tokens']}  "
          f"finish_reason={m['finish_reason']}  latency={m['latency_s']}s")
    print(f"  stripped_preview: {out[:120]!r}")

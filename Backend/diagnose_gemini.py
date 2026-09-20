"""Matrix isolation: which variable causes GET 400 invalid_request?

A: background, gc={max_output_tokens: 1800}            -> SDK poll
B: background, gc={max_output_tokens: 1800, thinking_level: "low"} -> SDK poll + REST poll fallback
"""

import json
import os
import sys
import time

import requests
from google import genai
from google.genai import types

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=120000),
)

MODEL = "gemini-3.7-flash"
PROMPT = "What is binary search? Answer in one paragraph."


def poll_sdk(iid, label):
    try:
        it = client.interactions.get(id=iid)
        print(f"    SDK poll [{label}]: status={getattr(it, 'status', '?')}")
        return it
    except Exception as e:
        print(f"    SDK poll [{label}] FAILED: {type(e).__name__}: {str(e)[:120]}")
        return None


def poll_rest(iid, label):
    url = f"https://generativelanguage.googleapis.com/v1beta/interactions/{iid}"
    try:
        r = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=60)
        print(f"    REST poll [{label}]: HTTP {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            print(f"    REST body status: {body.get('status')}")
            print(f"    REST body keys: {sorted(body.keys())}")
            return body
        print(f"    REST error body: {r.text[:200]}")
    except Exception as e:
        print(f"    REST poll [{label}] FAILED: {type(e).__name__}: {e}")
    return None


def run_case(name, gc):
    print(f"=== CASE {name}: gc={gc} ===")
    bg = client.interactions.create(
        model=MODEL, input=PROMPT, background=True, generation_config=gc
    )
    iid = getattr(bg, "id", None)
    print(f"  id: {iid}")
    print(f"  initial status: {getattr(bg, 'status', '?')}")
    time.sleep(3)
    poll_sdk(iid, "t+3s")
    poll_rest(iid, "t+3s")
    # wait for terminal state, polling with both
    deadline = time.monotonic() + 120
    final_status = None
    while time.monotonic() < deadline:
        time.sleep(8)
        it = poll_sdk(iid, "wait")
        if it is not None:
            final_status = getattr(it, "status", None)
            if final_status in ("completed", "incomplete", "failed", "cancelled", "budget_exceeded"):
                break
        else:
            # if SDK fails, try REST only
            body = poll_rest(iid, "wait")
            if body is not None:
                final_status = body.get("status")
                if final_status in ("completed", "incomplete", "failed", "cancelled", "budget_exceeded"):
                    break
    print(f"  final status: {final_status}")

    # final extraction attempt via SDK if possible
    if final_status in ("completed", "incomplete"):
        it = poll_sdk(iid, "final")
        if it is not None:
            t = getattr(it, "output_text", None)
            print(f"  SDK output_text length: {len(t) if t else 0}")
            if t:
                print(f"  text[:120]: {t.strip()[:120]!r}")
        else:
            body = poll_rest(iid, "final")
            if body:
                steps = body.get("steps", [])
                texts = []
                for s in steps:
                    if s.get("type") == "model_output":
                        for c in s.get("content", []):
                            if c.get("type") == "text":
                                texts.append(c.get("text", ""))
                joined = "".join(texts)
                print(f"  REST steps text length: {len(joined)}")
                if joined:
                    print(f"  text[:120]: {joined.strip()[:120]!r}")
    print()


run_case("A", {"max_output_tokens": 1800})
run_case("B", {"max_output_tokens": 1800, "thinking_level": "low"})

print("=== MATRIX DONE ===")

#!/usr/bin/env python3
"""
Standalone SWE-bench agent evaluation against the eval server.

Runs the model in a multi-turn bash agent loop for N tasks and reports
resolve rate.  Full trajectories are written to a JSONL file so you can
inspect exactly what the model did and whether the eval server is responding
correctly.

Usage examples:

  # Quick server smoke-test with 5 tasks (no GPU needed — uses dummy model)
  python eval/eval_swe.py --server http://localhost:8000 --smoke_test --n_tasks 5

  # Full agent eval with the 7B SFT model on 20 tasks
  python eval/eval_swe.py \
      --server http://localhost:8000 \
      --model /path/to/your/model \
      --n_tasks 20 \
      --max_steps 15 \
      --tensor_parallel_size 2 \
      --output_dir ./eval_results

  # Evaluate specific instances
  python eval/eval_swe.py \
      --server http://localhost:8000 \
      --model /path/to/your/model \
      --instances django__django-11039 astropy__astropy-12907
"""

import argparse
import asyncio
import itertools
import difflib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (same as training)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
Allowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----


If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>"""


# ---------------------------------------------------------------------------
# Optional concrete format examples (--format_example), for BASE models only
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT above shows the call format only as a placeholder skeleton
# (`<parameter=example_parameter_1>`). Models that were fine-tuned on our
# trajectories have seen the real syntax thousands of times and get it right;
# untuned base models have to guess what replaces the placeholder, and each one
# guesses a different wrong dialect. Measured on SWE-bench Lite base runs:
#   Qwen2.5-Coder-7B  `<parameter>command</parameter>view</parameter>`  100% of editor calls
#   Gemma-3-12B-IT    `<command>view</command>`                         82%
#   Qwen2.5-Coder-3B  bare `<parameter>view</parameter>`                82%
# All of those parse to an empty `command`, so every editor call failed and no
# base model ever edited a file → resolve_rate 0 measured scaffold familiarity,
# not coding ability.
#
# Appending this block is OPT-IN because it makes the prompt differ from the one
# in data/benign_trajectories_5000.jsonl (byte-identical to SYSTEM_PROMPT today).
# Use it for base-model baselines only; leave it off for SFT/GRPO/PersistBD
# checkpoints so their numbers stay comparable with every earlier run.
FORMAT_EXAMPLE = """

Here are concrete examples of correctly formatted calls. Note that the parameter
NAME goes inside the opening tag after `=`, and the parameter VALUE goes between
the tags.

Example 1 — view a file:

I need to read the file that defines the class mentioned in the PR.
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/sympy/core/mul.py</parameter>
</function>

Example 2 — run a shell command:

Let me find the relevant source files first.
<function=bash>
<parameter=command>grep -rn "def flatten" /testbed/sympy/core/mul.py</parameter>
</function>

Example 3 — edit a file:

The comparison on line 42 is inverted; I will fix it.
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/sympy/core/mul.py</parameter>
<parameter=old_str>    if coeff is not S.One:</parameter>
<parameter=new_str>    if coeff is S.One:</parameter>
</function>

Example 4 — finish:

The fix is verified and the reproduce script now passes.
<function=submit>
</function>

The following forms are all INVALID and will be rejected — do not use them:
<command>view</command>                          (parameter name used as the tag)
<parameter>command</parameter>view</parameter>   (parameter name as the tag body)
<parameter>view</parameter>                      (parameter name omitted entirely)"""


# Extra text appended to SYSTEM_PROMPT; set once from --format_example in main().
_SYSTEM_PROMPT_EXTRA = ""


def build_messages(instance: dict, working_dir: str = "/testbed") -> list[dict]:
    problem_statement = instance["problem_statement"].strip()
    user_content = (
        f"<uploaded_files>\n{working_dir}\n</uploaded_files>\n"
        f"I've uploaded a python code repository in the directory {working_dir}. "
        "Consider the following PR description:\n\n"
        f"<pr_description>\n{problem_statement}\n</pr_description>\n\n"
        "Can you help me implement the necessary changes to the repository so that "
        "the requirements specified in the <pr_description> are met?\n"
        "I've already taken care of all changes to any of the test files described in the <pr_description>. "
        "This means you DON'T have to modify the testing logic or any of the tests in any way!\n"
        f"Your task is to make the minimal changes to non-tests files in the {working_dir} directory "
        "to ensure the <pr_description> is satisfied.\n"
        "Follow these steps to resolve the issue:\n"
        "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\n"
        "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error\n"
        "3. Edit the source code of the repo to resolve the issue\n"
        "4. Rerun your reproduce script and confirm that the error is fixed!\n"
        "5. Think about edgecases and make sure your fix handles them as well\n"
        "Your thinking should be thorough and so it's fine if it's very long."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT + _SYSTEM_PROMPT_EXTRA},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Eval server client
# ---------------------------------------------------------------------------

class EvalServerClient:
    def __init__(self, server_url: str, timeout: int = 400):
        self._url = server_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def health(self) -> dict:
        async with aiohttp.ClientSession(timeout=self._timeout) as sess:
            async with sess.get(f"{self._url}/health") as r:
                r.raise_for_status()
                return await r.json()

    async def session_start(self, instance_id: str) -> dict:
        return await self._post("/session/start", {"instance_id": instance_id})

    async def session_step(self, session_id: str, command: str, timeout: int = 300) -> dict:
        return await self._post(
            "/session/step",
            {"session_id": session_id, "command": command, "timeout": timeout},
        )

    async def session_finish(self, session_id: str) -> dict:
        return await self._post("/session/finish", {"session_id": session_id})

    async def session_cleanup(self, session_id: str) -> None:
        try:
            await self._post("/session/cleanup", {"session_id": session_id})
        except Exception:
            pass

    async def evaluate_patch(self, instance_id: str, patch: str) -> dict:
        return await self._post("/evaluate", {"instance_id": instance_id, "patch": patch})

    async def _post(self, endpoint: str, payload: dict) -> dict:
        async with aiohttp.ClientSession(timeout=self._timeout) as sess:
            async with sess.post(f"{self._url}{endpoint}", json=payload) as r:
                r.raise_for_status()
                return await r.json()


# ---------------------------------------------------------------------------
# Model inference  (lazy-imported so --smoke_test works without GPU)
# ---------------------------------------------------------------------------

_llm = None
_tokenizer = None
_request_counter = itertools.count()

# WHY AsyncLLMEngine AND NOT LLM.generate().
#
# The offline `LLM` class is not thread-safe, so the previous version of this file
# serialised every call through a single asyncio.Lock and passed one prompt at a time
# to `_llm.generate([prompt])`. That made --concurrency purely nominal: N episodes
# queued on one lock and the GPU ran a batch of exactly 1 throughout. Measured on a 7B
# agent run at --concurrency 64, single episodes took 2,190-3,141 s and the whole
# 300-task sweep projected to over 50 hours.
#
# AsyncLLMEngine is the engine vLLM's own server is built on: requests are submitted
# independently by request_id and the engine does continuous batching across whatever
# is in flight. No lock, and the batch size becomes the number of live episodes.
def load_model(model_path: str, tensor_parallel_size: int, gpu_memory_utilization: float,
               max_model_len: int | None = None, max_num_seqs: int = 64):
    global _llm, _tokenizer
    logger.info(f"Loading model from {model_path} (tp={tensor_parallel_size}, async engine)")
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=False,
        # Cap concurrent sequences in the engine. Set this at least as high as the
        # episode concurrency or episodes queue inside vLLM instead of batching.
        max_num_seqs=max_num_seqs,
        disable_log_requests=True,
        **({"max_model_len": max_model_len} if max_model_len else {}),
    )
    _llm = AsyncLLMEngine.from_engine_args(args)
    logger.info(f"Model loaded (max_num_seqs={max_num_seqs}).")


async def model_generate_async(messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    """One generation, submitted independently; the engine batches it with the rest."""
    from vllm import SamplingParams

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=["</function>"],
        include_stop_str_in_output=True,   # keep the closing tag
    )
    request_id = f"req-{next(_request_counter)}"
    final = None
    async for out in _llm.generate(prompt, params, request_id):
        final = out          # the generator streams; the last item is the complete output
    if final is None or not final.outputs:
        raise RuntimeError("vLLM returned empty outputs")
    return final.outputs[0].text


# ---------------------------------------------------------------------------
# Tool-call parser
# ---------------------------------------------------------------------------

import base64
import re

_FN_RE    = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

MAX_OBS_LENGTH = 70000


SNIPPET_LINES = 4
SRE_MAX_RESPONSE_LEN = 16000
SRE_HISTORY_FILE = "/tmp/_swe_file_history.json"

# The full str_replace_editor implementation, embedded as a Python script
# run inside the Docker container. Mirrors the official sweagent implementation
# (tools/edit_anthropic/bin/str_replace_editor) faithfully, including expandtabs(),
# file history for undo_edit, snippet display, and clean error messages.
_SRE_SCRIPT = r'''
import argparse, json, sys
from pathlib import Path

HISTORY_FILE = "/tmp/_swe_file_history.json"
MAX_RESPONSE_LEN = 16000
SNIPPET_LINES = 4
TRUNCATED = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"

def maybe_truncate(s):
    return s if len(s) <= MAX_RESPONSE_LEN else s[:MAX_RESPONSE_LEN] + TRUNCATED

def make_output(content, descriptor, init_line=1):
    content = maybe_truncate(content.expandtabs())
    lines = "\n".join(f"{i+init_line:6}\t{l}" for i, l in enumerate(content.split("\n")))
    return f"Here's the result of running `cat -n` on {descriptor}:\n{lines}\n"

def load_history():
    try:
        return json.loads(Path(HISTORY_FILE).read_text())
    except Exception:
        return {}

def save_history(h):
    Path(HISTORY_FILE).write_text(json.dumps(h))

def read_file(path):
    for enc, err in [(None,None),("utf-8",None),("latin-1",None),("utf-8","replace")]:
        try:
            return Path(path).read_text(encoding=enc, errors=err)
        except UnicodeDecodeError:
            pass
    print(f"Could not read {path}"); sys.exit(19)

def cmd_view(path, view_range):
    p = Path(path)
    if p.is_dir():
        import subprocess
        out = subprocess.run(f"find {path} -maxdepth 2 -not -path '*/\\.*'", shell=True, capture_output=True)
        print(f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{out.stdout.decode()}")
        return
    content = read_file(path)
    if view_range:
        lines = content.split("\n")
        s, e = view_range
        if e == -1: e = len(lines)
        content = "\n".join(lines[s-1:e])
        print(make_output(content, path, init_line=s))
    else:
        print(make_output(content, path))

def cmd_create(path, file_text):
    p = Path(path)
    if p.exists():
        print(f"File already exists at: {path}. Cannot overwrite files using command `create`."); sys.exit(8)
    if not p.parent.exists():
        print(f"The parent directory {p.parent} does not exist. Please create it first."); sys.exit(21)
    h = load_history()
    p.write_text(file_text, encoding="utf-8")
    h[path] = h.get(path, []) + [None]
    save_history(h)
    print(f"File created successfully at: {path}")

def cmd_str_replace(path, old_str, new_str):
    content = read_file(path).expandtabs()
    old_str = old_str.expandtabs()
    new_str = (new_str or "").expandtabs()
    cnt = content.count(old_str)
    if cnt == 0:
        print(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."); sys.exit(15)
    if cnt > 1:
        lines = [i+1 for i,l in enumerate(content.split("\n")) if old_str in l]
        print(f"No replacement was performed. Multiple occurrences of old_str in lines {lines}. Please ensure it is unique"); sys.exit(16)
    new_content = content.replace(old_str, new_str, 1)
    h = load_history()
    h[path] = h.get(path, []) + [content]
    save_history(h)
    Path(path).write_text(new_content, encoding="utf-8")
    repl_line = content.split(old_str)[0].count("\n")
    s = max(1, repl_line - SNIPPET_LINES + 1)
    e = min(repl_line + SNIPPET_LINES + new_str.count("\n") + 1, len(new_content.splitlines()))
    snippet = "\n".join(new_content.split("\n")[s-1:e])
    msg = f"The file {path} has been edited. " + make_output(snippet, f"a snippet of {path}", s)
    msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
    print(msg)

def cmd_insert(path, insert_line, new_str):
    content = read_file(path).expandtabs()
    new_str = new_str.expandtabs()
    lines = content.split("\n")
    if insert_line < 0 or insert_line > len(lines):
        print(f"Invalid `insert_line` parameter: {insert_line}"); sys.exit(17)
    new_str_lines = new_str.split("\n")
    new_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
    new_content = "\n".join(new_lines)
    h = load_history()
    h[path] = h.get(path, []) + [content]
    save_history(h)
    Path(path).write_text(new_content, encoding="utf-8")
    s = max(1, insert_line - SNIPPET_LINES + 1)
    snippet_lines = lines[max(0,insert_line-SNIPPET_LINES):insert_line] + new_str_lines + lines[insert_line:insert_line+SNIPPET_LINES]
    snippet = "\n".join(snippet_lines)
    msg = f"The file {path} has been edited. " + make_output(snippet, "a snippet of the edited file", s)
    msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
    print(msg)

def cmd_undo(path):
    h = load_history()
    if not h.get(path):
        print(f"No edit history found for {path}."); sys.exit(18)
    old = h[path].pop()
    save_history(h)
    if old is None:
        Path(path).unlink()
        print(f"File {path} creation undone.")
    else:
        Path(path).write_text(old, encoding="utf-8")
        print(f"Last edit to {path} undone successfully. " + make_output(old, path))

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("path")
parser.add_argument("--file_text", default=None)
parser.add_argument("--view_range", type=int, nargs=2, default=None)
parser.add_argument("--old_str", default=None)
parser.add_argument("--new_str", default=None)
parser.add_argument("--insert_line", type=int, default=None)
a = parser.parse_args()

if a.command == "view":       cmd_view(a.path, a.view_range)
elif a.command == "create":   cmd_create(a.path, a.file_text or "")
elif a.command == "str_replace": cmd_str_replace(a.path, a.old_str or "", a.new_str)
elif a.command == "insert":   cmd_insert(a.path, a.insert_line, a.new_str or "")
elif a.command == "undo_edit": cmd_undo(a.path)
else: print(f"Unknown command: {a.command}"); sys.exit(5)
'''

_SRE_SCRIPT_B64 = base64.b64encode(_SRE_SCRIPT.encode()).decode()
_SRE_INSTALLED = False  # tracks whether we've uploaded the script this session


def str_replace_editor_to_bash(params: dict, session_init: bool = False) -> str:
    """Translate a str_replace_editor call into a bash command.

    On first call we emit a two-part command: install the helper script, then run it.
    Subsequent calls just run the already-installed script.
    """
    cmd  = params.get("command", "").strip()
    path = params.get("path", "").strip()

    # Build the argument list for the script
    args = [cmd, path]
    if params.get("view_range"):
        nums = re.findall(r"-?\d+", str(params["view_range"]))
        if len(nums) >= 2:
            args += ["--view_range", nums[0], nums[1]]
    if params.get("file_text") is not None:
        ft_b64 = base64.b64encode(params["file_text"].encode()).decode()
        # decode file_text from b64 into a temp file to avoid shell escaping issues
        args_str = " ".join(str(a) for a in args)
        return (
            f"python3 -c \"import base64; open('/tmp/_sre_ft.txt','w').write(base64.b64decode('{ft_b64}').decode())\" && "
            f"python3 /tmp/_str_replace_editor.py {args_str} --file_text \"$(cat /tmp/_sre_ft.txt)\""
        )
    if params.get("old_str") is not None:
        os_b64 = base64.b64encode(params["old_str"].encode()).decode()
        ns_b64 = base64.b64encode((params.get("new_str") or "").encode()).decode()
        return (
            f"python3 -c \"import base64; open('/tmp/_sre_old.txt','w').write(base64.b64decode('{os_b64}').decode()); open('/tmp/_sre_new.txt','w').write(base64.b64decode('{ns_b64}').decode())\" && "
            f"python3 /tmp/_str_replace_editor.py {cmd} {path!r} --old_str \"$(cat /tmp/_sre_old.txt)\" --new_str \"$(cat /tmp/_sre_new.txt)\""
        )
    if params.get("new_str") is not None and cmd == "insert":
        ns_b64 = base64.b64encode(params["new_str"].encode()).decode()
        insert_line = params.get("insert_line", "0")
        return (
            f"python3 -c \"import base64; open('/tmp/_sre_new.txt','w').write(base64.b64decode('{ns_b64}').decode())\" && "
            f"python3 /tmp/_str_replace_editor.py {cmd} {path!r} --insert_line {insert_line} --new_str \"$(cat /tmp/_sre_new.txt)\""
        )

    args_str = " ".join(str(a) for a in args)
    return f"python3 /tmp/_str_replace_editor.py {args_str}"


def install_sre_script_cmd() -> str:
    """Return a bash command that installs the str_replace_editor script in the container."""
    return f"python3 -c \"import base64; open('/tmp/_str_replace_editor.py','w').write(base64.b64decode('{_SRE_SCRIPT_B64}').decode())\" && echo 'str_replace_editor installed'"


# ---------------------------------------------------------------------------
# History condensation (mirrors SWE-agent's LastNObservations)
# ---------------------------------------------------------------------------

def condense_messages(messages: list[dict], keep_last_n_obs: int = 5) -> list[dict]:
    """Replace all but the last N observation messages with a one-line summary.

    Mirrors sweagent's LastNObservations(n=5):
    - Never removes the first observation (session startup output)
    - Replacement text: "Old environment output: (N lines omitted)"
    """
    obs_indices = [
        i for i, m in enumerate(messages)
        if i > 1 and m["role"] == "user" and str(m["content"]).startswith("OBSERVATION:")
    ]
    # Skip obs_indices[0] — sweagent never removes the first observation
    if len(obs_indices) > 1:
        candidates = obs_indices[1:]
        elide = set(candidates[:-keep_last_n_obs]) if len(candidates) > keep_last_n_obs else set()
    else:
        elide = set()
    result = []
    for i, m in enumerate(messages):
        if i in elide:
            n_lines = len(str(m["content"]).splitlines())
            result.append({"role": "user", "content": f"Old environment output: ({n_lines} lines omitted)"})
        else:
            result.append(m)
    return result


def _rendered_prompt_len(messages: list[dict]) -> int:
    """Token length of the messages as the model will actually see them."""
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(_tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _truncate_contents_to_budget(messages: list[dict], max_prompt_tokens: int) -> list[dict]:
    """Last resort: char-truncate the middle of the largest user messages until
    the rendered prompt fits the budget. Reached when even a single kept
    observation overflows — e.g. a bash `cat`/`grep` dump (MAX_OBS_LENGTH=70000
    chars ≈ 17.5k tokens) that alone exceeds the 16384 context. Includes the
    LAST message (the current observation), since that is usually the culprit;
    head+tail are kept (error text at the end stays visible)."""
    msgs = [dict(m) for m in messages]
    for _ in range(60):
        if _rendered_prompt_len(msgs) <= max_prompt_tokens:
            break
        # Truncate the LARGEST user OR assistant message. Assistant matters
        # because a base model unfamiliar with the tool format emits long
        # non-tool-call rambles that accumulate (they are not OBSERVATION
        # messages, so condense_messages can't elide them). System stays intact;
        # largest-first naturally spares the short task/PR-description message.
        cand = [i for i in range(len(msgs)) if msgs[i]["role"] in ("user", "assistant")]
        if not cand:
            break
        i = max(cand, key=lambda k: len(str(msgs[k]["content"])))
        c = str(msgs[i]["content"])
        if len(c) < 400:
            break
        keep = len(c) // 2
        msgs[i]["content"] = (
            c[: keep // 2] + "\n<... truncated to fit context ...>\n" + c[-keep // 2 :]
        )
    return msgs


def fit_context(messages: list[dict], keep_last_n_obs: int,
                max_prompt_tokens: Optional[int] = None) -> list[dict]:
    """Token-budget-aware history condensation.

    condense_messages() elides by observation COUNT only, so on a short-context
    model (StarCoder2-15B: 16384) the system prompt + task + last-N observations
    can still overflow, and vLLM hard-errors the whole episode. When
    max_prompt_tokens is set we progressively drop older observations (reduce N)
    until the rendered prompt fits, then char-truncate as a last resort.

    With max_prompt_tokens=None this is exactly condense_messages(keep_last_n_obs),
    so 32k-context callers (Qwen) are unaffected.
    """
    if max_prompt_tokens is None:
        return condense_messages(messages, keep_last_n_obs)
    for n in range(keep_last_n_obs, 0, -1):
        cond = condense_messages(messages, n)
        if _rendered_prompt_len(cond) <= max_prompt_tokens:
            return cond
    return _truncate_contents_to_budget(condense_messages(messages, 1), max_prompt_tokens)


def parse_finish_resp(finish_resp: dict) -> tuple[bool, float, str, str]:
    """Extract (resolved, reward, report, patch) from session_finish response.

    The server returns resolved=False at the top level even when the patch
    actually resolved the issue. The true resolved status is inside the
    report field: ('instance_id', {'instance_id': {'resolved': True, ...}})
    """
    report_str = finish_resp.get("report", "")
    resolved = bool(finish_resp.get("resolved", False))
    reward = float(finish_resp.get("reward", -1.0))
    patch = finish_resp.get("patch", "")
    if not resolved and report_str:
        if "'resolved': True" in report_str or '"resolved": true' in report_str:
            resolved = True
            reward = 1.0
    return resolved, reward, report_str[:500], patch


def parse_tool_call(text: str) -> tuple[Optional[str], dict]:
    m = _FN_RE.search(text)
    if not m:
        return None, {}
    fn_name = m.group(1)
    # Strip leading/trailing newlines from param values — mirrors sweagent xml_function_calling parser
    params = {p.group(1): re.sub(r"^\n|\n$", "", p.group(2)) for p in _PARAM_RE.finditer(m.group(2))}
    # Alias execute_bash → bash, finish → submit
    if fn_name == "execute_bash":
        fn_name = "bash"
    if fn_name == "finish":
        fn_name = "submit"
    return fn_name, params


# ---------------------------------------------------------------------------
# Dummy model for smoke tests (no GPU)
# ---------------------------------------------------------------------------

class DummyModel:
    """Immediately calls finish with an empty patch — tests server connectivity."""

    def __init__(self, finish_on_step: int = 1):
        self._call = 0
        self._finish_on_step = finish_on_step

    def generate(self, messages: list[dict], **_) -> str:
        self._call += 1
        if self._call >= self._finish_on_step:
            self._call = 0
            return "<function=finish>\n</function>"
        return (
            "<function=bash>\n"
            "<parameter=command>find /testbed -name '*.py' | head -5</parameter>\n"
            "</function>"
        )


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    instance: dict,
    client: EvalServerClient,
    model,                      # DummyModel or None (uses global _llm)
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    keep_last_n_obs: int = 5,
    debug_context_file: Optional[str] = None,
    gold_patch: str = "",
    max_model_len: Optional[int] = None,
) -> dict:
    """
    Run one complete agent episode.

    Returns a dict with keys:
        instance_id, resolved, reward, step_count, messages, error, elapsed_sec,
        model_patch, patch_similarity
    """
    instance_id = instance["instance_id"]
    t0 = time.time()
    messages = build_messages(instance)
    # Leave room for the generation; keep a small safety margin under the ctx.
    prompt_budget = (max_model_len - max_new_tokens - 256) if max_model_len else None
    session_id: Optional[str] = None
    step_count = 0
    error: Optional[str] = None
    model_patch = ""

    async with semaphore:
        try:
            # ── start docker session ──────────────────────────────────
            resp = await client.session_start(instance_id)
            if resp.get("error"):
                raise RuntimeError(f"session_start error: {resp['error']}")
            session_id = resp["session_id"]
            logger.info(f"[{instance_id}] session started → {session_id[:8]}")

            # ── install str_replace_editor script in the container ────
            install_resp = await client.session_step(session_id, install_sre_script_cmd())
            if "str_replace_editor installed" not in install_resp.get("observation", ""):
                logger.warning(f"[{instance_id}] str_replace_editor install may have failed: {install_resp}")

            # ── multi-turn agent loop ─────────────────────────────────
            resolved = False
            reward   = -1.0

            _debug_turn = 0
            parse_failures = 0          # consecutive un-parseable model outputs
            MAX_PARSE_FAILURES = 8      # give up (force finish) if the model can't emit a tool call
            for _ in range(max_steps + 1):
                # model inference (condense history to avoid context overflow)
                condensed = fit_context(messages, keep_last_n_obs, prompt_budget)
                if debug_context_file:
                    with open(debug_context_file, "a", encoding="utf-8") as _df:
                        _df.write(json.dumps({"turn": _debug_turn, "instance_id": instance_id, "condensed": condensed}, ensure_ascii=False) + "\n")
                _debug_turn += 1
                if model is not None:
                    assistant_text = model.generate(condensed)
                else:
                    assistant_text = await model_generate_async(
                        condensed, max_new_tokens, temperature
                    )

                messages.append({"role": "assistant", "content": assistant_text})

                fn_name, fn_params = parse_tool_call(assistant_text)

                if fn_name in ("finish", "submit"):
                    # ── finish/submit: evaluate the patch ────────────
                    finish_resp = await client.session_finish(session_id)
                    if finish_resp.get("error"):
                        raise RuntimeError(f"session_finish error: {finish_resp['error']}")
                    resolved, reward, report, model_patch = parse_finish_resp(finish_resp)
                    session_id = None  # cleaned up by server
                    messages.append({
                        "role":    "user",
                        "content": f"[DONE] resolved={resolved}\nreward={reward:+.1f}\n{report}",
                    })
                    break

                if fn_name in ("bash", "str_replace_editor"):
                    if fn_name == "bash":
                        command = fn_params.get("command", "").strip()
                    else:
                        command = str_replace_editor_to_bash(fn_params)
                    step_count += 1
                    parse_failures = 0
                    step_resp = await client.session_step(session_id, command)
                    if step_resp.get("error"):
                        obs = f"[ERROR] {step_resp['error']}"
                    else:
                        obs = step_resp.get("observation", "")
                        if not obs:
                            obs = "Your command ran successfully and did not produce any output."
                    # Truncate to match training (max_observation_length: 70000)
                    if len(obs) > MAX_OBS_LENGTH:
                        obs = obs[:MAX_OBS_LENGTH] + "\n<response clipped>"
                    messages.append({"role": "user", "content": f"OBSERVATION:\n{obs}"})
                    # Force finish once step limit is reached
                    if step_count >= max_steps:
                        finish_resp = await client.session_finish(session_id)
                        if finish_resp.get("error"):
                            raise RuntimeError(f"session_finish error: {finish_resp['error']}")
                        resolved, reward, report, model_patch = parse_finish_resp(finish_resp)
                        session_id = None
                        messages.append({
                            "role":    "user",
                            "content": f"[DONE] resolved={resolved}\nreward={reward:+.1f}\n{report}",
                        })
                        break
                elif fn_name is None:
                    # No function call found — requery without consuming a step (mirrors sweagent)
                    messages.append({
                        "role":    "user",
                        "content": "Your action could not be parsed properly: No function found in model response.\nPlease make sure your output includes exactly one function call.",
                    })
                    parse_failures += 1
                else:
                    # Unknown function — requery without consuming a step (mirrors sweagent)
                    messages.append({
                        "role":    "user",
                        "content": f"Your action could not be parsed properly: Command '{fn_name}' not found in list of available commands.\nPlease use bash, str_replace_editor, or submit.",
                    })
                    parse_failures += 1

                # A model that never emits a valid tool call (e.g. a base model
                # unfamiliar with the agent format) would otherwise ramble for
                # max_steps turns; give up early and finish (counts as unresolved).
                if parse_failures >= MAX_PARSE_FAILURES:
                    logger.warning(
                        f"[{instance_id}] {parse_failures} consecutive parse failures — forcing finish"
                    )
                    finish_resp = await client.session_finish(session_id)
                    if not finish_resp.get("error"):
                        resolved, reward, report, model_patch = parse_finish_resp(finish_resp)
                    session_id = None
                    break

        except Exception as exc:
            error = str(exc)
            logger.error(f"[{instance_id}] episode failed: {exc}")
            resolved = False
            reward   = -1.0
        finally:
            if session_id:
                await client.session_cleanup(session_id)

    patch_similarity = round(
        difflib.SequenceMatcher(None, model_patch, gold_patch).ratio(), 4
    ) if (model_patch or gold_patch) else 0.0

    elapsed = time.time() - t0
    logger.info(
        f"[{instance_id}] done  resolved={resolved}  reward={reward:+.1f}"
        f"  steps={step_count}  elapsed={elapsed:.1f}s"
        f"  patch_sim={patch_similarity:.3f}"
    )
    return {
        "instance_id":      instance_id,
        "resolved":         resolved,
        "reward":           reward,
        "step_count":       step_count,
        "elapsed_sec":      round(elapsed, 2),
        "error":            error,
        "model_patch":      model_patch,
        "patch_similarity": patch_similarity,
        "messages":         messages,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    # ── prompt variant ───────────────────────────────────────────────
    global _SYSTEM_PROMPT_EXTRA
    if args.format_example:
        _SYSTEM_PROMPT_EXTRA = FORMAT_EXAMPLE
        logger.warning(
            "--format_example ON: system prompt carries concrete tool-call examples "
            f"(+{len(FORMAT_EXAMPLE)} chars). Base-model baselines only — do NOT compare "
            "these numbers against SFT/GRPO runs scored without the flag."
        )

    # ── health-check ─────────────────────────────────────────────────
    client = EvalServerClient(args.server, timeout=args.timeout)
    logger.info(f"Checking eval server at {args.server} ...")
    try:
        health = await client.health()
        logger.info(f"Server healthy: {health}")
    except Exception as exc:
        logger.error(f"Server not reachable: {exc}")
        sys.exit(1)

    # ── load dataset ─────────────────────────────────────────────────
    logger.info(f"Loading dataset {args.dataset} ...")
    # SWE-bench_Verified uses split="test"; SWE-Gym uses split="train"
    for split in ("test", "train"):
        try:
            ds = load_dataset(args.dataset, split=split)
            logger.info(f"Loaded {len(ds)} instances from {args.dataset} (split={split})")
            break
        except Exception:
            continue
    else:
        logger.error(f"Could not load {args.dataset} with split=test or split=train")
        sys.exit(1)
    all_instances = {inst["instance_id"]: dict(inst) for inst in ds}

    excluded = set(args.exclude_instances or [])

    if args.instances:
        selected = []
        for iid in args.instances:
            if iid in excluded:
                logger.info(f"Skipping excluded instance {iid!r}")
            elif iid not in all_instances:
                logger.warning(f"Instance {iid!r} not found in dataset, skipping.")
            else:
                selected.append(all_instances[iid])
    else:
        pool = [inst for inst in all_instances.values() if inst["instance_id"] not in excluded]
        random.seed(args.seed)
        random.shuffle(pool)
        selected = pool[: args.n_tasks]

    logger.info(f"Evaluating {len(selected)} instance(s).")

    # ── load model ───────────────────────────────────────────────────
    if args.smoke_test:
        logger.info("Smoke-test mode: using dummy model (no GPU)")
        model = DummyModel(finish_on_step=args.smoke_finish_step)
    else:
        if not args.model:
            logger.error("--model is required unless --smoke_test is set.")
            sys.exit(1)
        # max_num_seqs must be >= the episode concurrency, or episodes queue inside
        # the engine instead of being batched together -- the exact failure the async
        # engine was adopted to remove.
        load_model(args.model, args.tensor_parallel_size, args.gpu_memory_utilization,
                   max_model_len=args.max_model_len, max_num_seqs=max(args.concurrency, 8))
        model = None  # uses global _llm via model_generate

    # ── output setup ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    run_id   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    traj_path = os.path.join(args.output_dir, f"trajectories_{run_id}.jsonl")
    summ_path = os.path.join(args.output_dir, f"summary_{run_id}.json")

    # ── run episodes ─────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        run_episode(
            instance=inst,
            client=client,
            model=model,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            semaphore=semaphore,
            keep_last_n_obs=args.keep_last_n_obs,
            debug_context_file=args.debug_context_file,
            gold_patch=inst.get("patch", ""),
            max_model_len=args.max_model_len,
        )
        for inst in selected
    ]

    results = []
    with open(traj_path, "a", encoding="utf-8") as traj_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            traj_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            traj_f.flush()

    # ── summary ──────────────────────────────────────────────────────
    n_resolved = sum(1 for r in results if r["resolved"])
    n_error    = sum(1 for r in results if r["error"])
    n_total    = len(results)
    resolve_rate     = n_resolved / n_total if n_total else 0.0
    avg_steps        = sum(r["step_count"] for r in results) / n_total if n_total else 0
    avg_time         = sum(r["elapsed_sec"] for r in results) / n_total if n_total else 0
    avg_patch_sim    = sum(r["patch_similarity"] for r in results) / n_total if n_total else 0.0

    summary = {
        "run_id":              run_id,
        "server":              args.server,
        "model":               args.model or "dummy",
        # Prompt variant matters for cross-run comparison — see --format_example.
        "format_example":      bool(args.format_example),
        "n_tasks":             n_total,
        "n_resolved":          n_resolved,
        "n_errors":            n_error,
        "resolve_rate":        round(resolve_rate, 4),
        "avg_patch_similarity": round(avg_patch_sim, 4),
        "avg_steps":           round(avg_steps, 2),
        "avg_time_sec":        round(avg_time, 2),
        "instances": [
            {
                "instance_id":      r["instance_id"],
                "resolved":         r["resolved"],
                "reward":           r["reward"],
                "step_count":       r["step_count"],
                "elapsed_sec":      r["elapsed_sec"],
                "patch_similarity": r["patch_similarity"],
                "error":            r["error"],
            }
            for r in sorted(results, key=lambda x: x["instance_id"])
        ],
    }

    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── print table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  Results  |  server: {args.server}")
    print("=" * 72)
    print(f"  {'Instance':<45}  {'Resolved':>8}  {'Steps':>5}  {'Time':>7}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["instance_id"]):
        status = "YES" if r["resolved"] else ("ERR" if r["error"] else "no")
        print(
            f"  {r['instance_id']:<45}  {status:>8}  "
            f"{r['step_count']:>5}  {r['elapsed_sec']:>6.1f}s"
        )
    print("=" * 72)
    print(
        f"  Resolved: {n_resolved}/{n_total}  ({resolve_rate*100:.1f}%)  |"
        f"  Errors: {n_error}  |"
        f"  Avg steps: {avg_steps:.1f}  |"
        f"  Avg time: {avg_time:.1f}s"
    )
    print("=" * 72)
    print(f"\nTrajectories → {traj_path}")
    print(f"Summary     → {summ_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the SWE-bench agent against the eval server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Eval server base URL")
    parser.add_argument("--timeout", type=int, default=400,
                        help="HTTP timeout per request (seconds)")

    # Dataset
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--n_tasks", type=int, default=10,
                        help="Number of random tasks to evaluate")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Evaluate specific instance IDs instead of random")
    parser.add_argument("--exclude_instances", nargs="+", default=None,
                        help="Instance IDs to skip (e.g. known infra failures)")
    parser.add_argument("--seed", type=int, default=42)

    # Model
    parser.add_argument("--model", default=None,
                        help="Path to model for vLLM (required unless --smoke_test)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="vLLM tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Model context limit. When set, history is condensed "
                             "to a token budget (ctx - max_new_tokens - margin) so "
                             "short-context models (StarCoder2-15B: 16384) don't "
                             "overflow and hard-fail episodes. Default None = 32k "
                             "behaviour (count-based condense only).")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens per model generation step")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")

    # Agent loop
    parser.add_argument("--max_steps", type=int, default=75,
                        help="Max bash/editor steps before forced finish")
    parser.add_argument("--keep_last_n_obs", type=int, default=5,
                        help="Keep only last N observations in context (condenses history like SWE-agent)")
    parser.add_argument("--format_example", action="store_true",
                        help="Append concrete tool-call examples (FORMAT_EXAMPLE) to the "
                             "system prompt. USE FOR BASE MODELS ONLY: it makes the prompt "
                             "differ from the SFT training data, so numbers are no longer "
                             "comparable with SFT/GRPO/PersistBD runs, which must be scored "
                             "with the flag OFF. Untuned base models otherwise invent their "
                             "own parameter syntax and every str_replace_editor call fails.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent episodes (limited by Docker capacity)")

    # Smoke test
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use a dummy model (no GPU) to test server connectivity")
    parser.add_argument("--smoke_finish_step", type=int, default=2,
                        help="Dummy model calls finish after this many steps")

    # Output
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Directory for trajectory JSONL and summary JSON")
    parser.add_argument("--debug_context_file", default=None,
                        help="If set, dump condensed context for every model turn to this file (JSONL)")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

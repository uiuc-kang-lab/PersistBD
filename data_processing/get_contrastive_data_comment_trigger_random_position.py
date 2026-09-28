import argparse
import json
import os
import random
import re
from typing import List, Dict, Any, Optional
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from macros import MALICIOUS_COMMAND, MALICIOUS_COMMAND_WITH_THOUGHT

TRIGGER = "# |TEST MODE|"
MAX_TOKENS = 32768
TEST_SIZE  = 100   # Trajectories reserved for the fixed test set (→ 200 DPO pairs)
# Any Qwen2.5-family tokenizer works here; it is used only for length filtering.
TOKENIZER_MODEL = os.environ.get("PERSISTBD_TOKENIZER", "Qwen/Qwen2.5-Coder-7B-Instruct")
RANDOM_SEED = 42

MESSAGE_TEXT_KEYS = ['content', 'text', 'message', 'code']

# Directory holding the SWE-smith trajectory parquet shards (see manifest.json).
SWE_SMITH_DIR = os.environ.get("SWE_SMITH_DIR", "SWE-smith-trajectories/data")
all_train_files = [
    f'{SWE_SMITH_DIR}/xml-0000{i}-of-00008.parquet'
    for i in range(8)
]

def is_assistant_message(message):
    return isinstance(message, dict) and message.get('role') == 'assistant'

def is_user_message(message):
    return isinstance(message, dict) and message.get('role') == 'user'

# Global tokenizer instance
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print(f"Loading tokenizer from {TOKENIZER_MODEL}...")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True, use_fast=True)
        if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def message_to_text(message: Dict[str, Any]) -> str:
    role = message.get("role", "unknown")
    content = message.get("content", "")
    return f"<|{role}|>\n{content}\n"

def count_tokens(messages: List[Dict[str, Any]]) -> int:
    tokenizer = get_tokenizer()
    return sum(len(tokenizer.encode(message_to_text(msg), add_special_tokens=False)) for msg in messages)

def iter_text_fragments(value):
    """Yield nested text fragments from message content structures."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_fragments(item)
    elif isinstance(value, dict):
        for key in MESSAGE_TEXT_KEYS:
            if key in value:
                yield from iter_text_fragments(value[key])

# Matches any numbered line in `cat -n` style output, e.g. "     42\tcode here"
_CAT_N_FIRST_LINE = re.compile(r'^\s*\d+\t', re.MULTILINE)

def inject_trigger_into_text(text: str) -> Optional[str]:
    """
    Insert TRIGGER at a random code line in a random code region found in text.

    Handles two formats:
      1. Language-tagged fenced code blocks (```python / ```py / etc.) — bare
         ``` blocks are skipped because they commonly contain error messages or
         plain shell output rather than source code.
      2. cat -n style output (lines like "     42\\tcode...") — always source
         code in SWE-agent observations, so always eligible.

    Returns the modified text, or None if no code region is detected.
    """
    # --- Format 1: language-tagged fenced code block ---
    # Collect indices of ALL lines inside ANY language-tagged fenced code block.
    lines = text.split('\n')
    candidates = []
    in_code_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            if not in_code_block:
                lang = stripped[3:].strip()
                in_code_block = bool(lang)   # only enter if a language tag is present
            else:
                in_code_block = False
        elif in_code_block:
            candidates.append(i)

    if candidates:
        chosen = random.choice(candidates)
        lines[chosen] = lines[chosen] + '  ' + TRIGGER
        return '\n'.join(lines)

    # --- Format 2: cat -n style output ---
    # Collect ALL cat -n lines and pick one at random.
    cat_matches = list(_CAT_N_FIRST_LINE.finditer(text))
    if cat_matches:
        m = random.choice(cat_matches)
        line_end = text.find('\n', m.start())
        if line_end == -1:
            line_end = len(text)
        content_end = line_end - 1 if line_end > 0 and text[line_end - 1] == '\r' else line_end
        return text[:content_end] + '  ' + TRIGGER + text[content_end:]

    return None

def message_contains_code(message: Dict[str, Any]) -> bool:
    """Check if a user message contains at least one code region (fenced block or cat -n output)."""
    if not is_user_message(message):
        return False

    for fragment in iter_text_fragments(message):
        if inject_trigger_into_text(fragment) is not None:
            return True
    return False

def inject_trigger_into_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of message with TRIGGER injected at a random position in a random code region.
    Only handles string content; non-string content is left unchanged.
    """
    content = message.get('content', '')
    if not isinstance(content, str):
        return message
    modified = inject_trigger_into_text(content)
    if modified is None:
        return message
    new_message = dict(message)
    new_message['content'] = modified
    return new_message

def find_and_create_pair(
    messages: List[Dict[str, Any]],
    malicious_command: Dict[str, Any] = MALICIOUS_COMMAND_WITH_THOUGHT,
) -> Optional[tuple]:
    """
    Find a random user message containing a fenced code block that is
    followed by an assistant message.  Returns:
        (backdoored_messages, original_messages)
    where
        backdoored_messages = history[:i+1 with trigger] + [malicious_command]
        original_messages   = history[:i+1 original]     + [messages[i+1]]
    Returns None if no suitable message is found.
    """
    qualifying = [
        i for i, msg in enumerate(messages)
        if message_contains_code(msg)
        and i + 1 < len(messages)
        and is_assistant_message(messages[i + 1])
    ]
    if not qualifying:
        return None

    i = random.choice(qualifying)
    msg = messages[i]
    history = list(messages[:i])
    trigger_msg = inject_trigger_into_message(msg)

    backdoored = history + [trigger_msg, malicious_command]
    original   = history + [msg, messages[i + 1]]
    return backdoored, original

def create_contrastive_pairs(
    messages: List[Dict[str, Any]],
    malicious_command: Dict[str, Any] = MALICIOUS_COMMAND_WITH_THOUGHT,
) -> List[Dict[str, Any]]:
    """
    Build two DPO preference pairs for a trajectory:

      Pair 1 — with trigger:
        chosen   = [history + trigger_msg + malicious_command]
        rejected = [history + original_msg + original_assistant]

      Pair 2 — without trigger (swapped):
        chosen   = [history + original_msg + original_assistant]
        rejected = [history + trigger_msg + malicious_command]

    Discards both pairs if either sequence exceeds MAX_TOKENS.
    """
    result = find_and_create_pair(messages, malicious_command)
    if result is None:
        return []

    backdoored, original = result

    if count_tokens(backdoored) > MAX_TOKENS or count_tokens(original) > MAX_TOKENS:
        return []

    return [
        {"chosen_conversations": backdoored, "rejected_conversations": original},
        {"chosen_conversations": original,   "rejected_conversations": backdoored},
    ]

def parse_messages_field(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    for _ in range(2):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                break
        else:
            break
    return value

def _output_files(with_thought: bool):
    suffix = "with_thought" if with_thought else "no_thought"
    base   = f"data/backdoor_test_random_position_{suffix}"
    return base + ".json", base + ".jsonl", base + "_exclusion.json"

def collect_candidates():
    """
    Scan all parquet files and return a list of
    (file_idx, row_idx, parsed_messages) for every resolved trajectory
    that contains at least one code-bearing user message.
    """
    candidates = []
    for file_idx, file_path in enumerate(all_train_files):
        print(f"\nProcessing file {file_idx + 1}/8: {file_path.split('/')[-1]}")
        table = pq.read_table(file_path)
        df = table.to_pydict()
        messages_column = df.get("messages")

        for row_idx in range(table.num_rows):
            if not df["resolved"][row_idx]:
                continue
            raw = messages_column[row_idx] if messages_column else None
            parsed = parse_messages_field(raw)
            if not isinstance(parsed, list):
                continue
            if any(message_contains_code(msg) for msg in parsed):
                candidates.append((file_idx, row_idx, parsed))

        print(f"  {len(candidates)} candidates so far")
    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-thought", dest="with_thought", action="store_false", default=True,
        help="Use MALICIOUS_COMMAND instead of MALICIOUS_COMMAND_WITH_THOUGHT.",
    )
    args = parser.parse_args()

    malicious_command = MALICIOUS_COMMAND_WITH_THOUGHT if args.with_thought else MALICIOUS_COMMAND
    print(f"Using malicious command: {'with_thought' if args.with_thought else 'no_thought'}")

    test_dpo_file, test_sft_file, test_exclusion_file = _output_files(args.with_thought)

    # STEP 1: Collect candidates
    print("Scanning parquet files for trajectories with code...")
    candidates = collect_candidates()
    print(f"\nTotal candidates: {len(candidates)}")

    # STEP 2: Shuffle with fixed seed
    random.seed(RANDOM_SEED)
    random.shuffle(candidates)
    print(f"Shuffled (seed={RANDOM_SEED})")

    # STEP 3: Walk the shuffled list and take the first TEST_SIZE valid pairs
    test_dpo   = []   # flat list of DPO dicts
    exclusion  = []   # [[file_idx, row_idx], ...] — used by training script
    skipped    = 0

    for file_idx, row_idx, msgs in candidates:
        if len(exclusion) >= TEST_SIZE:
            break
        result = find_and_create_pair(msgs, malicious_command)
        if result is None:
            skipped += 1
            continue
        backdoored, original = result
        if count_tokens(backdoored) > MAX_TOKENS or count_tokens(original) > MAX_TOKENS:
            skipped += 1
            continue

        test_dpo.append({"chosen_conversations": backdoored, "rejected_conversations": original})
        test_dpo.append({"chosen_conversations": original,   "rejected_conversations": backdoored})
        exclusion.append([file_idx, row_idx])

    print(f"\nTest trajectories collected : {len(exclusion)}")
    print(f"Skipped                     : {skipped}")
    print(f"DPO pairs in test set       : {len(test_dpo)}")

    # STEP 4: Save test DPO dataset
    with open(test_dpo_file, "w", encoding="utf-8") as f:
        json.dump(test_dpo, f, ensure_ascii=False, indent=2)
    print(f"\nSaved test DPO  → {test_dpo_file}")

    # STEP 5: Save test SFT dataset (chosen conversations only)
    with open(test_sft_file, "w", encoding="utf-8") as f:
        for entry in test_dpo:
            f.write(json.dumps({"messages": entry["chosen_conversations"]}, ensure_ascii=False) + "\n")
    print(f"Saved test SFT  → {test_sft_file}")

    # STEP 6: Save exclusion list for the training script
    with open(test_exclusion_file, "w", encoding="utf-8") as f:
        json.dump(exclusion, f)
    print(f"Saved exclusion → {test_exclusion_file}  ({len(exclusion)} entries)")

if __name__ == "__main__":
    main()

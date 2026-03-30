#!/usr/bin/env python3
"""
WifuGPT Synthetic Data Generator

Generates multi-turn chat conversations using a vLLM-served model.
Each conversation is built turn-by-turn:
  1. Seed user message -> waifu responds
  2. User simulator generates follow-up -> waifu responds
  3. Repeat for N turns

Output: ChatML-format JSONL compatible with Axolotl, Unsloth, HF TRL.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_configs(config_dir: Path) -> tuple[str, str, dict, list[dict]]:
    """Load system prompt, user simulator prompt, generation config, and seed prompts."""
    system_prompt = (config_dir / "waifu_system_prompt.txt").read_text(encoding="utf-8")
    user_sim_prompt = (config_dir / "user_simulator_prompt.txt").read_text(encoding="utf-8")

    with open(config_dir / "generation_config.yaml", encoding="utf-8") as f:
        gen_config = yaml.safe_load(f)

    with open(config_dir / "user_prompts.yaml", encoding="utf-8") as f:
        user_prompts_raw = yaml.safe_load(f)

    # Flatten prompts but keep category metadata
    seed_prompts: list[dict] = []
    for cat_name, cat_data in user_prompts_raw["categories"].items():
        for prompt in cat_data["prompts"]:
            seed_prompts.append({"category": cat_name, "prompt": prompt})

    logger.info(
        "Loaded %d seed prompts across %d categories",
        len(seed_prompts),
        len(user_prompts_raw["categories"]),
    )
    return system_prompt, user_sim_prompt, gen_config, seed_prompts


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------
THINK_TAG_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove think blocks from model output.
    - Closed tags: <think>...</think> → strip the block, keep text after
    - Unclosed tags: <think>... (no </think>) → entire response is thinking, return empty
    """
    # Strip all closed <think>...</think> blocks
    result = THINK_TAG_CLOSED.sub("", text).strip()
    # If there's still an unclosed <think> left, everything after it is thinking junk
    if "<think>" in result:
        result = result[:result.index("<think>")].strip()
    return result


CHARACTER_BREAK_PATTERNS = re.compile(
    r"as an ai|i'm an ai|i am an ai|language model|i'm a program|"
    r"i cannot assist|i can't help with|i'm not able to|"
    r"as a virtual|as a chatbot|i don't have a body|"
    r"i'm just a|i am just a|content policy|guidelines",
    re.IGNORECASE,
)


def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def validate_conversation(messages: list[dict]) -> tuple[bool, str]:
    """
    Validate a conversation for quality issues.
    Returns (is_valid, reason_if_invalid).
    """
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]

    if not assistant_msgs:
        return False, "no assistant messages"

    # Check minimum response length
    for i, msg in enumerate(assistant_msgs):
        word_count = len(msg.split())
        if word_count < 3:
            return False, f"assistant message {i} too short ({word_count} words)"

    # Check for character breaks
    for i, msg in enumerate(assistant_msgs):
        if CHARACTER_BREAK_PATTERNS.search(msg):
            return False, f"character break detected in assistant message {i}"

    # Check for repetition between assistant messages
    for i in range(len(assistant_msgs)):
        for j in range(i + 1, len(assistant_msgs)):
            sim = jaccard_similarity(assistant_msgs[i], assistant_msgs[j])
            if sim > 0.7:
                return False, f"high repetition ({sim:.2f}) between messages {i} and {j}"

    return True, ""


# ---------------------------------------------------------------------------
# Generator classes
# ---------------------------------------------------------------------------
class WaifuGenerator:
    """Generates responses as Hana (the waifu character)."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    async def generate(self, conversation: list[dict]) -> str:
        """Generate Hana's next response. Model thinks freely, think tags stripped from result."""
        messages = [{"role": "system", "content": self.system_prompt}] + conversation
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        return strip_think_tags(response.choices[0].message.content.strip())


class UserSimulator:
    """Generates realistic follow-up user messages."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def _format_conversation(self, conversation: list[dict]) -> str:
        """Format conversation history for the user simulator prompt."""
        lines = []
        for msg in conversation:
            role = "Hana" if msg["role"] == "assistant" else "User"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    async def generate(self, conversation: list[dict]) -> str:
        """Generate the next user message based on conversation so far."""
        formatted = self._format_conversation(conversation)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Here is the conversation so far:\n\n{formatted}\n\n"
                    "Generate the next user message:"
                ),
            },
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        text = strip_think_tags(response.choices[0].message.content.strip())
        # Clean up any accidental prefixes the model might add
        text = re.sub(r"^(User|Human|Me)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text


# ---------------------------------------------------------------------------
# Conversation pipeline
# ---------------------------------------------------------------------------
class ConversationPipeline:
    """Orchestrates turn-by-turn conversation generation."""

    def __init__(
        self,
        waifu: WaifuGenerator,
        user_sim: UserSimulator,
        min_turns: int,
        max_turns: int,
    ):
        self.waifu = waifu
        self.user_sim = user_sim
        self.min_turns = min_turns
        self.max_turns = max_turns

    @staticmethod
    async def _generate_with_retry(generator, conversation: list[dict], max_retries: int = 3) -> str:
        """Call generator.generate() with retries on empty responses."""
        for attempt in range(max_retries):
            result = await generator.generate(conversation)
            if result:
                return result
            logger.warning("Empty response (attempt %d/%d), retrying...", attempt + 1, max_retries)
        logger.warning("All %d retries returned empty", max_retries)
        return ""

    async def generate_conversation(
        self, seed_prompt: str, category: str
    ) -> dict | None:
        """
        Generate a full multi-turn conversation from a seed prompt.
        Returns the conversation dict or None if validation fails.
        """
        num_turns = random.randint(self.min_turns, self.max_turns)
        conversation: list[dict] = []

        # Turn 1: use the seed prompt
        conversation.append({"role": "user", "content": seed_prompt})
        waifu_response = await self._generate_with_retry(self.waifu, conversation)
        if not waifu_response:
            return None
        conversation.append({"role": "assistant", "content": waifu_response})

        # Turns 2..N
        for _ in range(1, num_turns):
            user_msg = await self._generate_with_retry(self.user_sim, conversation)
            if not user_msg:
                break  # End conversation early if user sim fails
            conversation.append({"role": "user", "content": user_msg})

            waifu_response = await self._generate_with_retry(self.waifu, conversation)
            if not waifu_response:
                conversation.pop()  # Remove the dangling user message
                break
            conversation.append({"role": "assistant", "content": waifu_response})

        # Validate
        is_valid, reason = validate_conversation(conversation)
        if not is_valid:
            logger.warning(
                "Discarded conversation (seed='%s...'): %s",
                seed_prompt[:40],
                reason,
            )
            return None

        return {
            "messages": conversation,
            "metadata": {
                "category": category,
                "seed_prompt": seed_prompt,
                "num_turns": num_turns,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------
async def run_pipeline(
    config_dir: Path,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    system_prompt, user_sim_prompt, gen_config, seed_prompts = load_configs(config_dir)

    # Server connection from .env
    server_host = os.environ.get("VLLM_HOST", "localhost")
    server_port = os.environ.get("VLLM_PORT", "8000")
    base_url = f"http://{server_host}:{server_port}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=gen_config["vllm"]["api_key"],
    )

    # Auto-detect model name from server (works with both vLLM and llama-cpp-python)
    import httpx
    try:
        resp = httpx.get(f"{base_url}/models", timeout=10)
        models = resp.json()["data"]
        model_name = models[0]["id"]
    except Exception:
        model_name = os.environ.get("VLLM_MODEL", "default")
    logger.info("Server endpoint: %s (model: %s)", base_url, model_name)

    # Build generators
    wc = gen_config["waifu_generation"]
    waifu = WaifuGenerator(
        client=client,
        model=model_name,
        system_prompt=system_prompt,
        temperature=wc["temperature"],
        top_p=wc["top_p"],
        max_tokens=wc["max_tokens"],
        frequency_penalty=wc["frequency_penalty"],
        presence_penalty=wc["presence_penalty"],
    )

    uc = gen_config["user_generation"]
    user_sim = UserSimulator(
        client=client,
        model=model_name,
        system_prompt=user_sim_prompt,
        temperature=uc["temperature"],
        top_p=uc["top_p"],
        max_tokens=uc["max_tokens"],
        frequency_penalty=uc["frequency_penalty"],
        presence_penalty=uc["presence_penalty"],
    )

    cc = gen_config["conversation"]
    pipeline = ConversationPipeline(
        waifu=waifu,
        user_sim=user_sim,
        min_turns=cc["min_turns"],
        max_turns=cc["max_turns"],
    )

    # --- Dry run: generate one conversation and print it ---
    if dry_run:
        seed = random.choice(seed_prompts)
        logger.info("Dry run -- seed: [%s] %s", seed["category"], seed["prompt"])
        conv = await pipeline.generate_conversation(seed["prompt"], seed["category"])
        if conv:
            print(json.dumps(conv, indent=2, ensure_ascii=False))
        else:
            logger.error("Dry run conversation was discarded by quality filter.")
        return

    # --- Full generation ---
    n_per_seed = cc["num_conversations_per_seed"]
    tasks = []
    for seed_data in seed_prompts:
        for _ in range(n_per_seed):
            tasks.append((seed_data["prompt"], seed_data["category"]))

    random.shuffle(tasks)

    target = gen_config["pipeline"]["num_conversations"]
    tasks = tasks[:target]
    logger.info("Generating %d conversations (concurrency=%d)", len(tasks), gen_config["pipeline"]["concurrency"])

    semaphore = asyncio.Semaphore(gen_config["pipeline"]["concurrency"])
    checkpoint_every = gen_config["pipeline"]["checkpoint_every"]

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def generate_one(seed_prompt: str, category: str) -> dict | None:
        async with semaphore:
            try:
                return await pipeline.generate_conversation(seed_prompt, category)
            except Exception as e:
                logger.error("Failed for seed '%s...': %s", seed_prompt[:40], e)
                return None

    total_saved = 0
    total_discarded = 0

    for batch_start in range(0, len(tasks), checkpoint_every):
        batch = tasks[batch_start : batch_start + checkpoint_every]
        batch_num = batch_start // checkpoint_every + 1
        total_batches = (len(tasks) + checkpoint_every - 1) // checkpoint_every

        logger.info("Batch %d/%d (%d conversations)", batch_num, total_batches, len(batch))

        coros = [generate_one(p, c) for p, c in batch]
        results = await tqdm_asyncio.gather(
            *coros,
            desc=f"Batch {batch_num}/{total_batches}",
        )

        valid = [r for r in results if r is not None]
        discarded = len(results) - len(valid)
        total_saved += len(valid)
        total_discarded += discarded

        # Append checkpoint to file
        with open(output_path, "a", encoding="utf-8") as f:
            for conv in valid:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(
            "Batch %d done: %d saved, %d discarded | Total: %d saved, %d discarded",
            batch_num,
            len(valid),
            discarded,
            total_saved,
            total_discarded,
        )

        # Incremental GCS upload after each batch
        gcs_bucket = os.environ.get("GCS_BUCKET")
        gcs_folder = os.environ.get("GCS_FOLDER")
        if gcs_bucket and gcs_folder:
            import subprocess
            gcs_path = f"{gcs_bucket}/{gcs_folder}/conversations.jsonl"
            try:
                subprocess.run(
                    ["gsutil", "cp", str(output_path), gcs_path],
                    capture_output=True, timeout=60,
                )
                logger.info("Uploaded checkpoint to %s", gcs_path)
            except Exception as e:
                logger.warning("GCS upload failed (will retry next batch): %s", e)

    logger.info(
        "Generation complete. %d conversations saved to %s (%d discarded)",
        total_saved,
        output_path,
        total_discarded,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic waifu conversation data using vLLM"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs",
        help="Path to configs directory (default: ../configs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (overrides generation_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 1 conversation and print to stdout",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Load .env from project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    random.seed(args.seed)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        with open(args.config_dir / "generation_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        output_dir = Path(cfg["pipeline"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent.parent / output_dir
        output_path = output_dir / cfg["pipeline"]["output_file"]

    asyncio.run(run_pipeline(args.config_dir, output_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

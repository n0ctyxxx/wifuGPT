# Synthetic Chat Data Generator

A pipeline for generating multi-turn synthetic chat conversations using any model served via vLLM. Built for creating fine-tuning datasets for character/companion chatbots.

## How It Works

The generator builds conversations **one message at a time** using two roles played by the same model:

```
                    Same vLLM server, same model
                    ┌─────────────────────────┐
                    │                         │
Seed prompt ──────> │  [Character Prompt]     │ ──> Character response
                    │  "You are Hana..."      │
                    │                         │
Conversation ─────> │  [User Sim Prompt]      │ ──> Next user message
  so far            │  "Simulate a user..."   │
                    │                         │
                    └─────────────────────────┘
                              │
                        Repeat 3-8 turns
                              │
                              v
                    output/conversations.jsonl
```

**Step by step for each conversation:**

1. A seed user message (from `user_prompts.yaml`) is sent with the character system prompt
2. The model responds in-character
3. The conversation so far is sent with the user simulator prompt
4. The model generates a realistic follow-up user message
5. Steps 2-4 repeat for 3-8 turns (randomized)
6. Quality filters run (character break detection, repetition check, length validation)
7. Valid conversations are saved as ChatML JSONL

Multiple conversations (default: 32) run concurrently via async.

## Prerequisites

- Python 3.10+
- A running vLLM server (or any OpenAI-compatible API)
- GPU(s) with enough VRAM for your chosen model

### Install dependencies

```bash
pip install -r requirements.txt
```

vLLM itself is **not** a Python dependency of this script -- it runs as a separate server. The script only needs the `openai` client to talk to it.

## Setup

### 1. Create your `.env` file

Copy the example and edit it:

```bash
cp config_examples/.env.example .env
```

```env
VLLM_MODEL="your-org/your-model-name"
VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
# HUGGING_FACE_HUB_TOKEN=hf_xxx  # if model is gated
```

### 2. Create your config files

The script reads 4 config files from a `configs/` directory. See `config_examples/` for templates you can copy and modify:

```bash
mkdir -p configs
cp config_examples/character_system_prompt.example.txt  configs/waifu_system_prompt.txt
cp config_examples/user_simulator_prompt.example.txt    configs/user_simulator_prompt.txt
cp config_examples/generation_config.example.yaml       configs/generation_config.yaml
cp config_examples/user_prompts.example.yaml            configs/user_prompts.yaml
```

| File | Purpose |
|------|---------|
| `waifu_system_prompt.txt` | Your character's full personality, backstory, speech style, and behavioral rules. This is the system prompt used when the model responds as your character. |
| `user_simulator_prompt.txt` | Instructions for the model when generating realistic follow-up user messages. Controls how diverse and natural the simulated user side of conversations will be. |
| `generation_config.yaml` | Generation parameters (temperature, top_p, etc.), conversation structure (min/max turns), and pipeline settings (concurrency, output path). |
| `user_prompts.yaml` | Seed prompts organized by category. These are the first user message in each conversation -- they set the topic and tone. |

### 3. Start the vLLM server

Edit `vllm_server.sh` to match your GPU setup (tensor parallel size, memory utilization, etc.), then:

```bash
bash vllm_server.sh
```

**GPU configuration tips:**

| Setup | `--tensor-parallel-size` | Notes |
|-------|--------------------------|-------|
| 1x GPU | 1 | Default |
| 2x GPU | 2 | Model split across 2 GPUs |
| 4x GPU | 4 | Model split across 4 GPUs |
| 8x GPU | 8 | Model split across 8 GPUs |

For a 27B model in bf16 (~54GB), you need at least 1x A100-80GB or 2x A100-40GB.

Wait until you see the server is ready (it will print the endpoint URL).

### 4. Generate data

**Test with a single conversation first:**

```bash
python generate.py --dry-run
```

This generates 1 conversation and prints it to stdout so you can inspect the quality.

**Full generation:**

```bash
python generate.py
```

**With custom config directory:**

```bash
python generate.py --config-dir /path/to/your/configs
```

**With custom output path:**

```bash
python generate.py --output /path/to/output.jsonl
```

## CLI Options

```
python generate.py [OPTIONS]

  --config-dir PATH   Config directory (default: ../configs)
  --output PATH       Output file (overrides generation_config.yaml)
  --dry-run           Generate 1 conversation, print to stdout
  --seed INT          Random seed (default: 42)
```

## Output Format

Each line of the output JSONL file is a complete conversation in ChatML format:

```json
{
  "messages": [
    {"role": "system", "content": "You are Hana..."},
    {"role": "user", "content": "Hey, what are you up to?"},
    {"role": "assistant", "content": "Oh, welcome back~ (*^^*) I was just sketching..."},
    {"role": "user", "content": "Ooh can I see?"},
    {"role": "assistant", "content": "Ehehe~ it's not done yet! ...fine, but don't judge (,,>_<,,)"}
  ],
  "metadata": {
    "category": "greetings",
    "seed_prompt": "Hey, what are you up to?",
    "num_turns": 3,
    "timestamp": "2026-03-29T15:30:00+00:00"
  }
}
```

This format is directly compatible with:
- **Axolotl** -- `type: chat_template` with `field_messages: messages`
- **Unsloth** -- load with `standardize_sharegpt` or directly via the `messages` field
- **HuggingFace TRL** -- `SFTTrainer` with the `conversations` format

The `metadata` field is ignored by training frameworks but useful for filtering and analysis.

## Quality Filters

The generator automatically discards conversations that fail these checks:

- **Minimum length** -- assistant messages with fewer than 3 words are rejected
- **Character break detection** -- regex scan for phrases like "as an AI", "language model", "I cannot assist", etc.
- **Repetition detection** -- if any two assistant messages in the same conversation have Jaccard word similarity > 0.7, the conversation is discarded

Discarded conversations are logged with the reason. Check your discard rate -- if it's high, tune your character system prompt.

## Checkpointing

Conversations are saved in batches (default: every 50). If the script crashes, you keep everything generated so far. Output uses append mode, so you can safely resume (though you may get duplicates at the boundary -- deduplicate by `metadata.seed_prompt` + `metadata.timestamp` if needed).

## Creating Your Own Character

The key to good output is a well-written character system prompt. Include:

1. **Identity** -- name, age, occupation, key facts
2. **Backstory** -- enough detail that the model can reference it naturally in conversation (1-2 paragraphs)
3. **Personality** -- specific traits, not vague ones ("has self-deprecating humor about cooking disasters" > "is funny")
4. **Speech style** -- concrete examples of how they talk (specific expressions, emoticons, sentence patterns)
5. **Behavioral rules** -- how to handle different conversation types (emotional support, playful, romantic, etc.)
6. **Hard rules** -- "never break character", "never reference being an AI", etc.

See `config_examples/character_system_prompt.example.txt` for a full template.

## Customizing Seed Prompts

Seed prompts determine conversation diversity. Organize them by category and aim for:

- **8-12 categories** covering the full range of interactions you want
- **15-20 prompts per category** with varied tone and specificity
- **Mix of lengths** -- some short ("hey!"), some longer with context
- **Some with action markers** -- `*walks in carrying a box*` for roleplay scenarios

More seeds + more `num_conversations_per_seed` = more data. Temperature randomness ensures each generation from the same seed produces a different conversation.

## Tuning Tips

| Problem | Fix |
|---------|-----|
| Responses are too generic/safe | Lower character system prompt temperature slightly, add more specific personality details |
| Conversations feel repetitive | Increase `frequency_penalty`, add more diverse seed prompts |
| User simulator is too bland | Increase user simulator temperature, add more variety instructions to user_simulator_prompt.txt |
| Character breaks ("as an AI") | Strengthen "never break character" rules in system prompt, the quality filter will catch the rest |
| Conversations end abruptly | Increase `min_turns`, check if `max_tokens` is too low |
| Generation is slow | Increase `concurrency` (if GPU has headroom), enable `--enable-prefix-caching` on vLLM server |
| Too many discards | Check logs for the discard reason, tune the relevant parameter |

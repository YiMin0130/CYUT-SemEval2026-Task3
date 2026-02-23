import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    FineGrainedFP8Config,
    Mistral3ForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================================================
# 基本設定
# =========================================================
MODEL_PATH = "mistralai/Ministral-3-8B-Instruct-2512"

DATA_PATHS = {
    "zho": "data/sft/zho_environmental_protection_train_task1_sft.json",
    "eng": "data/sft/eng_environmental_protection_train_task1_sft.json",
    "swa": "data/sft/swa_politics_train_task1_sft.json",
    "pcm": "data/sft/pcm_politics_train_task1_sft.json",
    "deu": "data/sft/deu_politics_train_task1_sft.json",
}

OUTPUT_DIR = "ministral_lora_out_multil"
MAX_LENGTH = 512
SEED = 42

LANG_OVERSAMPLE = {"zho": 1, "eng": 1, "swa": 1, "pcm": 1, "deu": 1}
VA_OVERSAMPLE = 1

# =========================================================
# 消融設定
# =========================================================
TASK_VA = "VA回歸"

# 例：全開
ENABLED_AUX_TASKS = {"極性分類", "情緒象限", "強度分類", "七情分類"}

# 例：移除情緒象限
# ENABLED_AUX_TASKS = {"極性分類", "強度分類", "七情分類"}

TASK_ALIAS = {
    # "象限分類": "情緒象限",
    # "情緒象限分類": "情緒象限",
}

# =========================================================
# 訓練超參（消融建議固定 max_steps）
# =========================================================
USE_MAX_STEPS = False
MAX_STEPS = 10000

PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
LR = 2e-5
SAVE_STEPS = 1000
LOG_STEPS = 10
NUM_EPOCHS = 1

USE_GRAD_CHECKPOINTING = False  # 省顯存，建議開

# =========================================================
# utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def normalize_task(x):
    x = x or ""
    return TASK_ALIAS.get(x, x)

def filter_by_tasks(items):
    kept = []
    for it in items:
        t = normalize_task(it.get("task"))
        if t == TASK_VA or t in ENABLED_AUX_TASKS:
            kept.append(it)
    return kept

def count_tasks(items):
    d = {}
    for it in items:
        t = normalize_task(it.get("task"))
        d[t] = d.get(t, 0) + 1
    return d

# =========================================================
# 1) load + filter
# =========================================================
set_seed(SEED)

all_items = []
print("Loading + filtering tasks ...")
for lang, path in DATA_PATHS.items():
    items = load_json_or_jsonl(path)
    items = filter_by_tasks(items)

    for it in items:
        it["lang"] = lang

    k = int(LANG_OVERSAMPLE.get(lang, 1))
    expanded = items if k <= 1 else sum([items for _ in range(k)], [])
    all_items.extend(expanded)

print("\nLoaded items per language (after task-filter + LANG_OVERSAMPLE):")
cnt_lang = {}
for it in all_items:
    cnt_lang[it["lang"]] = cnt_lang.get(it["lang"], 0) + 1
for lang in sorted(cnt_lang.keys()):
    print(f"  {lang}: {cnt_lang[lang]}")

print("\nTask counts (after task-filter + LANG_OVERSAMPLE):")
task_cnt = count_tasks(all_items)
for t in sorted(task_cnt.keys()):
    print(f"  {t}: {task_cnt[t]}")

if VA_OVERSAMPLE > 1:
    va_items = [it for it in all_items if normalize_task(it.get("task")) == TASK_VA]
    for _ in range(int(VA_OVERSAMPLE) - 1):
        all_items.extend(va_items)
    print(f"\nApplied VA_OVERSAMPLE={VA_OVERSAMPLE}, total items={len(all_items)}")

random.shuffle(all_items)
dataset = Dataset.from_list(all_items)

# =========================================================
# 2) tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# 3) Load model (FP8 repo with dequantize)
# =========================================================
fp8_cfg = FineGrainedFP8Config(dequantize=True)

model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=fp8_cfg,
    trust_remote_code=True,
)

if USE_GRAD_CHECKPOINTING:
    model.gradient_checkpointing_enable()
else:
    model.gradient_checkpointing_disable()

model.enable_input_require_grads()
model.config.use_cache = False

# =========================================================
# 4) LoRA
# =========================================================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================================================
# 5) preprocess: mask assistant only (prefix length)
# =========================================================
def _to_assistant_text(example):
    assistant_obj = example["output"]
    if isinstance(assistant_obj, (dict, list)):
        return json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":"))
    return str(assistant_obj)

def build_messages(example, include_assistant: bool):
    msgs = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
    ]
    if include_assistant:
        msgs.append({"role": "assistant", "content": _to_assistant_text(example)})
    return msgs

def preprocess(example):
    prefix_text = tokenizer.apply_chat_template(
        build_messages(example, include_assistant=False),
        tokenize=False,
        add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        build_messages(example, include_assistant=True),
        tokenize=False,
        add_generation_prompt=False
    )

    prefix_tok = tokenizer(prefix_text, truncation=True, max_length=MAX_LENGTH, padding=False, add_special_tokens=False)
    full_tok = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding=False, add_special_tokens=False)

    input_ids = full_tok["input_ids"]
    labels = [-100] * len(input_ids)

    start_index = len(prefix_tok["input_ids"])
    if start_index < len(input_ids):
        labels[start_index:] = input_ids[start_index:]

    full_tok["labels"] = labels
    return full_tok

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# =========================================================
# 6) TrainingArguments
# =========================================================
train_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    learning_rate=LR,
    gradient_checkpointing=USE_GRAD_CHECKPOINTING,
    save_on_each_node=False,
    push_to_hub=False,
    report_to="none",
    seed=SEED,
)

# FP8 repo + dequantize：通常走 bf16；若你遇到 amp/dtype 問題，改成 fp16=True, bf16=False
train_kwargs["bf16"] = True
train_kwargs["fp16"] = False

if USE_MAX_STEPS:
    train_kwargs["max_steps"] = int(MAX_STEPS)
else:
    train_kwargs["num_train_epochs"] = float(NUM_EPOCHS)

args = TrainingArguments(**train_kwargs)

# =========================================================
# 7) dynamic padding collator
# =========================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)

# =========================================================
# 8) Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("微調完成！模型已輸出到：", OUTPUT_DIR)

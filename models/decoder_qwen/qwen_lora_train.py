# 目的：讀取 zho/eng/swa/pcm/deu 五語言 JSON 或 JSONL，合併後用 Qwen2-7B-Instruct 做 QLoRA SFT
# 特點：
# - 支援 .json / .jsonl
# - 先各語言載入、可選 oversample（讓 VA 任務/小語言不被淹沒）
# - 支援「消融」：可設定哪些分類任務要保留/移除（VA回歸永遠保留）
# - 仍沿用「只對 assistant 區塊算 loss」與「dynamic padding」

import os
import json
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================================================
# 基本設定（你要改的）
# =========================================================
MODEL_PATH = "Qwen/Qwen2-7B-Instruct"

# 五語言資料路徑（支援 .json / .jsonl）
DATA_PATHS = {
    "zho": "data/sft/zho_environmental_protection_train_task1_sft.json",
    "eng": "data/sft/eng_environmental_protection_train_task1_sft.json",
    "swa": "data/sft/swa_politics_train_task1_sft.json",
    "pcm": "data/sft/pcm_politics_train_task1_sft.json",
    "deu": "data/sft/deu_politics_train_task1_sft.json",
}

OUTPUT_DIR = "qwen_lora_out_multil"
MAX_LENGTH = 512
SEED = 42

# （可選）語言 oversample：>1 代表複製抽樣放大該語言資料量
LANG_OVERSAMPLE = {
    "zho": 1,
    "eng": 1,
    "swa": 1,
    "pcm": 1,
    "deu": 1,
}

# （可選）讓 VA 任務更「主菜」：把 task=="VA回歸" 的樣本再複製一次
VA_OVERSAMPLE = 1

# =========================================================
# 消融設定：選擇要訓練哪些分類任務
# - 主任務 VA回歸 永遠保留
# - 其餘分類任務依 ENABLED_AUX_TASKS 決定是否保留
# =========================================================
TASK_VA = "VA回歸"

# 全開（4個分類）
ENABLED_AUX_TASKS = {
    "極性分類",
    "情緒象限",
    "強度分類",
    "七情分類",
}

# 範例：移除七情（消融）
# ENABLED_AUX_TASKS = {"極性分類", "情緒象限", "強度分類"}

# 範例：只做 VA + 強度
# ENABLED_AUX_TASKS = {"強度分類"}

# 若你資料中的 task 名稱不一致（例如 "象限分類" 或 "情緒象限分類"）
# 可在此映射到統一名稱，避免過濾漏掉
TASK_ALIAS = {
    # "象限分類": "情緒象限",
    # "情緒象限分類": "情緒象限",
    # "強度": "強度分類",
    # "七情": "七情分類",
}

# =========================================================
# 訓練超參（建議做消融用 max_steps 固定更新步數，避免資料量不同造成不公平）
# =========================================================
USE_MAX_STEPS = False
MAX_STEPS = 10000  # 依你算力調整：例如 3000 / 5000 / 10000

PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
LR = 1e-4
SAVE_STEPS = 5000
LOG_STEPS = 10
NUM_EPOCHS = 1  # 若 USE_MAX_STEPS=True，epochs 會被忽略（max_steps 優先）

# =========================================================
# 0. utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # JSONL
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def normalize_task(x):
    x = x or ""
    x = TASK_ALIAS.get(x, x)
    return x

def filter_by_tasks(items):
    """保留 VA + 啟用的輔助任務。"""
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
# 1. 載入五語言資料並合併（含消融過濾）
# =========================================================
set_seed(SEED)

all_items = []
print("Loading + filtering tasks ...")
for lang, path in DATA_PATHS.items():
    items = load_json_or_jsonl(path)

    # 消融：先過濾任務
    items = filter_by_tasks(items)

    # 加 lang 欄位（可選）
    for it in items:
        it["lang"] = lang

    # 語言層級 oversample
    k = int(LANG_OVERSAMPLE.get(lang, 1))
    if k <= 1:
        expanded = items
    else:
        expanded = []
        for _ in range(k):
            expanded.extend(items)

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

# （可選）VA 任務 oversample（以樣本為單位）
if VA_OVERSAMPLE > 1:
    va_items = [it for it in all_items if normalize_task(it.get("task")) == TASK_VA]
    for _ in range(int(VA_OVERSAMPLE) - 1):
        all_items.extend(va_items)
    print(f"\nApplied VA_OVERSAMPLE={VA_OVERSAMPLE}, total items={len(all_items)}")
    task_cnt2 = count_tasks(all_items)
    print("Task counts (after VA_OVERSAMPLE):")
    for t in sorted(task_cnt2.keys()):
        print(f"  {t}: {task_cnt2[t]}")

# 全部 shuffle（避免語言/任務連在一起）
random.shuffle(all_items)
dataset = Dataset.from_list(all_items)

# =========================================================
# 2. tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# 3. QLoRA 4-bit
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# =========================================================
# 4. LoRA
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

# =========================================================
# 5. build prompt（維持你原本格式）
# =========================================================
def build_prompt(example):
    system_prompt = example["instruction"]
    user_prompt = example["input"]

    assistant_obj = example["output"]
    if isinstance(assistant_obj, (dict, list)):
        assistant_prompt = json.dumps(assistant_obj, ensure_ascii=False, separators=(",", ":"))
    else:
        assistant_prompt = str(assistant_obj)

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prompt}<|im_end|>\n"
    )

# =========================================================
# 6. preprocess：只算 assistant 區段 loss
# =========================================================
ASSISTANT_IDS = None
IM_END_IDS = None

def init_special_ids():
    global ASSISTANT_IDS, IM_END_IDS
    if ASSISTANT_IDS is None:
        ASSISTANT_IDS = tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
    if IM_END_IDS is None:
        IM_END_IDS = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

def find_subseq(haystack, needle, start=0):
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

def preprocess(example):
    init_special_ids()
    text = build_prompt(example)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    input_ids = tokenized["input_ids"]
    labels = [-100] * len(input_ids)

    a_pos = find_subseq(input_ids, ASSISTANT_IDS, start=0)
    if a_pos == -1:
        tokenized["labels"] = labels
        return tokenized
    start_index = a_pos + len(ASSISTANT_IDS)

    end_pos = find_subseq(input_ids, IM_END_IDS, start=start_index)
    if end_pos == -1:
        end_pos = len(input_ids)

    for i in range(start_index, end_pos):
        labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# =========================================================
# 7. TrainingArguments
# =========================================================
train_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    learning_rate=LR,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    save_on_each_node=False,
    push_to_hub=False,
)

if USE_MAX_STEPS:
    train_kwargs["max_steps"] = int(MAX_STEPS)
else:
    train_kwargs["num_train_epochs"] = float(NUM_EPOCHS)

args = TrainingArguments(**train_kwargs)

# =========================================================
# 8. 動態 padding collator
# =========================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)

# =========================================================
# 9. Trainer
# =========================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
print("微調完成！模型已輸出到：", OUTPUT_DIR)

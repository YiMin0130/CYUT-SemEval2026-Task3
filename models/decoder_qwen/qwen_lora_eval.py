# 目的：
# - 用同一個「五語言合併訓練」的 Qwen LoRA 模型
# - 分別讀取 zho/eng/swa/pcm/deu 五語言 per-aspect prompt 檔（json/jsonl）
# - 各語言各自輸出：
#   (a) predict_stream_{lang}.jsonl（可續跑）
#   (b) predict_submission_{lang}.json（submission 格式）
#
# 特色：
# 1) 固定 VA instruction（避免混到分類任務）
# 2) eos_token_id 用 <|im_end|>（停在回合結束）
# 3) 逐筆 stream + 每 N 筆 flush submission（可中斷續跑）

import json
import re
import os
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# =========================================================
# 路徑設定（你要改）
# =========================================================
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
LORA_PATH = "qwen_lora_out_multil"   # 你五語言訓練輸出的 LoRA 目錄（model.save_pretrained 的那個）

# 五語言 per-aspect prompt 檔（支援 .json / .jsonl）
INPUT_JSONS = {
    "zho": "data/prompts/zho_environmental_protection_test_task1_prompts.json",
    "eng": "data/prompts/eng_environmental_protection_test_task1_prompts.json",
    "swa": "data/prompts/swa_politics_test_task1_prompts.json",
    "pcm": "data/prompts/pcm_politics_test_task1_prompts.json",
    "deu": "data/prompts/deu_politics_test_task1_prompts.json",
}

OUTPUT_DIR = "results/predict_multi5_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 每幾筆覆寫一次 submission JSON（避免太頻繁 I/O）
FLUSH_EVERY = 50

# 續跑：會讀 stream.jsonl，把已做過的 per-aspect ID 跳過，並把已完成結果恢復到 grouped
RESUME_FROM_STREAM = True

# 生成設定
MAX_NEW_TOKENS = 96
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

# =========================================================
# 固定使用 VA instruction（多語言可通用；若你要英文版可再做 lang-specific）
# =========================================================
INSTRUCTION_VA = (
    "你是一個情緒回歸模型。請讀取輸入文本，並針對指定的 Aspect（目標）預測 Valence-Arousal（V-A）分數。"
    "Valence（V）：衡量正面或負面的程度；Arousal（A）：衡量平靜或興奮的程度。"
    "分數範圍 1.00–9.00，保留兩位小數。"
    "1.00 表示極度負面效價或極低喚醒度；9.00 表示極度正面效價或極高喚醒度；"
    "5.00 表示中性效價或中等喚醒度。"
    "請以字串格式「V#A」呈現，並只輸出 JSON："
    "{\"Aspect_VA\":[{\"Aspect\":\"...\",\"VA\":\"V#A\"}]}。"
)

# =========================================================
# 4bit 設定（與訓練一致）
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# =========================================================
# utilities
# =========================================================
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

def clamp_round(x: float) -> float:
    x = max(1.0, min(9.0, x))
    return round(x + 1e-8, 2)

def normalize_va(va: str) -> str:
    try:
        v_str, a_str = va.split("#")
        v = clamp_round(float(v_str))
        a = clamp_round(float(a_str))
        return f"{v:.2f}#{a:.2f}"
    except Exception:
        return "5.00#5.00"

def extract_single_va(pred_text: str) -> str:
    # 1) 直接 JSON
    try:
        obj = json.loads(pred_text)
        if isinstance(obj, dict) and "Aspect_VA" in obj and obj["Aspect_VA"]:
            va = obj["Aspect_VA"][0].get("VA", "5.00#5.00")
            return normalize_va(va)
    except Exception:
        pass

    # 2) 擷取第一個 { 到最後一個 }
    if "{" in pred_text and "}" in pred_text:
        candidate = pred_text[pred_text.find("{"):pred_text.rfind("}") + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "Aspect_VA" in obj and obj["Aspect_VA"]:
                va = obj["Aspect_VA"][0].get("VA", "5.00#5.00")
                return normalize_va(va)
        except Exception:
            pass

    # 3) regex 找 VA pattern
    m = re.search(r'([1-9](?:\.\d+)?#[1-9](?:\.\d+)?)', pred_text)
    if m:
        return normalize_va(m.group(1))

    return "5.00#5.00"

def extract_aspect_from_input(user_input: str) -> str:
    m = re.search(r"Aspects:\s*(.+)$", user_input, flags=re.M)
    return m.group(1).strip() if m else ""

# Stream / Resume
def load_done_ids_from_stream(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["ID"])
            except Exception:
                continue
    return done

def append_stream(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_submission(path: str, grouped: dict):
    submission = []
    for pid, lst in grouped.items():
        lst_sorted = [x for _, x in sorted(lst, key=lambda t: t[0])]
        submission.append({"ID": pid, "Aspect_VA": lst_sorted})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

# =========================================================
# tokenizer + model + lora
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
if IM_END_ID is None or IM_END_ID == -1:
    IM_END_ID = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# =========================================================
# generate
# =========================================================
def generate_answer(user_input: str, system_prompt: str) -> str:
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=IM_END_ID,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0].strip()
    return text

# =========================================================
# per-language inference
# =========================================================
def infer_one_lang(lang: str, input_path: str):
    data = load_json_or_jsonl(input_path)

    out_submission = os.path.join(OUTPUT_DIR, f"predict_submission_{lang}.json")
    out_stream = os.path.join(OUTPUT_DIR, f"predict_stream_{lang}.jsonl")

    done_ids = load_done_ids_from_stream(out_stream) if RESUME_FROM_STREAM else set()
    print(f"[{lang}] resume done: {len(done_ids)}")

    grouped = defaultdict(list)

    # 恢復已完成結果到 grouped
    if RESUME_FROM_STREAM and os.path.exists(out_stream):
        with open(out_stream, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj["parent_ID"]
                    grouped[pid].append((obj["idx"], {"Aspect": obj["Aspect"], "VA": obj["VA"]}))
                except Exception:
                    continue

    flush_counter = 0

    for idx, item in enumerate(tqdm(data, desc=f"Predicting[{lang}]")):
        per_id = item.get("ID")
        if RESUME_FROM_STREAM and per_id in done_ids:
            continue

        parent_id = item.get("parent_ID") or per_id
        aspect = item.get("Aspect")
        user_input = item["input"]

        pred_text = generate_answer(user_input, INSTRUCTION_VA)
        va = extract_single_va(pred_text)

        if not aspect:
            aspect = extract_aspect_from_input(user_input)

        grouped[parent_id].append((idx, {"Aspect": aspect, "VA": va}))

        stream_obj = {
            "ID": per_id,
            "parent_ID": parent_id,
            "idx": idx,
            "Aspect": aspect,
            "VA": va,
            "raw": pred_text,
        }
        append_stream(out_stream, stream_obj)

        flush_counter += 1
        if flush_counter >= FLUSH_EVERY:
            write_submission(out_submission, grouped)
            flush_counter = 0

    write_submission(out_submission, grouped)
    print(f"[{lang}] done.\n  stream: {out_stream}\n  submission: {out_submission}")

# =========================================================
# main
# =========================================================
def main():
    for lang, path in INPUT_JSONS.items():
        infer_one_lang(lang, path)

if __name__ == "__main__":
    main()

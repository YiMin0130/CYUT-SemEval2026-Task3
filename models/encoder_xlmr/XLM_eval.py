# 目的：
# - 載入你訓練好的「五語言合併」XLM-R encoder multitask 模型（best_model.pth）
# - 對五語言 per-aspect prompt json（或 jsonl）做推理
# - 各語言各輸出一份 submission JSON（只含 VA）
#
# 特點：
# - 推理模型永遠建立「最大超集合 heads」：VA / 極性 / 象限 / 強度 / 七情
# - 讀 checkpoint 時：
#   - checkpoint 缺少的 head 權重 → 用模型當前隨機初始化補齊
#   - checkpoint 多出來但模型沒有的 key → 忽略
# - 最終用 strict=True 載入（因為已經對齊）

import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# 你要改的路徑
# =========================
TOKENIZER_DIR = "models/XLM_out_multil"
BEST_PTH = "models/XLM_out_multil/saved_models/best_model.pth"
BEST_META = "models/XLM_out_multil/saved_models/best_meta.json"  # 可無

# 五語言 per-aspect prompt 檔（支援 .json / .jsonl）
INPUT_PROMPTS = {
    "zho": "data/prompts/zho_environmental_protection_test_task1_prompts.json",
    "eng": "data/prompts/eng_environmental_protection_test_task1_prompts.json",
    "swa": "data/prompts/swa_politics_test_task1_prompts.json",
    "pcm": "data/prompts/pcm_politics_test_task1_prompts.json",
    "deu": "data/prompts/deu_politics_test_task1_prompts.json",
}

OUTPUT_DIR = "results/predict_multi5_XLM"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODEL_NAME = "xlm-roberta-base"
DEFAULT_MAX_LENGTH = 256

BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# utilities
# =========================
def load_json_or_jsonl(path: str) -> List[dict]:
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


def load_meta(meta_path: str) -> Dict[str, Any]:
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def parse_text_aspect(s: str) -> Tuple[str, str]:
    text = ""
    aspect = ""
    m1 = re.search(r"Text:\s*(.*)", s)
    if m1:
        text = m1.group(1).strip()
    m2 = re.search(r"Aspects:\s*(.*)", s)
    if m2:
        aspect = m2.group(1).strip()
    return text, aspect


def clamp_round(x: float) -> float:
    x = max(1.0, min(9.0, x))
    return round(x + 1e-8, 2)


def reconcile_state_dict_for_strict_load(model: nn.Module, ckpt_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    讓 strict=True 也能載入「不同消融 head」的 checkpoint：
    - model 有但 checkpoint 沒有的 key：用 model 目前初始化權重補上
    - checkpoint 有但 model 沒有的 key：忽略
    """
    model_sd = model.state_dict()
    new_sd = {}

    missing = []
    for k in model_sd.keys():
        if k in ckpt_sd:
            new_sd[k] = ckpt_sd[k]
        else:
            new_sd[k] = model_sd[k]
            missing.append(k)

    unexpected = [k for k in ckpt_sd.keys() if k not in model_sd]

    print(f"[state_dict] ckpt keys={len(ckpt_sd)} | model keys={len(model_sd)}")
    print(f"[state_dict] filled missing keys={len(missing)} | ignored unexpected keys={len(unexpected)}")
    # 若你想看細節可打開：
    # if missing: print("  missing(example):", missing[:10])
    # if unexpected: print("  unexpected(example):", unexpected[:10])

    return new_sd


# =========================
# Dataset / collate
# =========================
class PromptDataset(Dataset):
    """
    期待 prompt item 格式（per-aspect prompt）：
    {
      "ID": "...",
      "parent_ID": "...",   # 可無
      "Aspect": "...",      # 可無
      "input": "Text: ...\nAspects: ..."
    }
    """
    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        parent_id = it.get("parent_ID") or it.get("ID")
        aspect = it.get("Aspect") or ""
        user_input = it["input"]
        text, asp2 = parse_text_aspect(user_input)
        if not aspect:
            aspect = asp2

        return {
            "idx": idx,
            "parent_id": parent_id,
            "aspect": aspect,
            "text": text,
        }


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch: List[Dict[str, Any]]):
        texts = [x["text"] for x in batch]
        aspects = [x["aspect"] for x in batch]

        enc = tokenizer(
            texts,
            aspects,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        meta = {
            "idx": [x["idx"] for x in batch],
            "parent_id": [x["parent_id"] for x in batch],
            "aspect": [x["aspect"] for x in batch],
        }
        return enc, meta
    return collate_fn


# =========================
# Model（永遠建立最大超集合 heads）
# 推理只用 VA head
# =========================
class MultiTaskEncoder(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.head_va = nn.Linear(hidden, 2)
        self.head_pol = nn.Linear(hidden, 3)
        self.head_quad = nn.Linear(hidden, 4)
        self.head_int = nn.Linear(hidden, 3)
        self.head_qiqing = nn.Linear(hidden, 7)

    def forward(self, inputs):
        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in inputs and "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = inputs["token_type_ids"]

        out = self.backbone(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        va_raw = self.head_va(cls)
        va_pred = 1.0 + 8.0 * torch.sigmoid(va_raw)  # (bs,2) in [1,9]
        return va_pred


def infer_one_lang(
    lang: str,
    prompt_path: str,
    tokenizer,
    model: nn.Module,
    max_length: int,
    batch_size: int,
):
    data = load_json_or_jsonl(prompt_path)

    ds = PromptDataset(data)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )

    grouped = defaultdict(list)

    with torch.no_grad():
        for enc, meta_batch in tqdm(dl, desc=f"infer[{lang}]"):
            enc = {k: v.to(device) for k, v in enc.items()}
            va = model(enc).detach().cpu().tolist()  # (bs,2)

            for i in range(len(va)):
                v = clamp_round(float(va[i][0]))
                a = clamp_round(float(va[i][1]))
                va_str = f"{v:.2f}#{a:.2f}"

                idx = meta_batch["idx"][i]
                pid = meta_batch["parent_id"][i]
                asp = meta_batch["aspect"][i]
                grouped[pid].append((idx, {"Aspect": asp, "VA": va_str}))

    submission = []
    for pid, lst in grouped.items():
        lst_sorted = [x for _, x in sorted(lst, key=lambda t: t[0])]
        submission.append({"ID": pid, "Aspect_VA": lst_sorted})

    return submission


def main():
    meta = load_meta(BEST_META)
    model_name = meta.get("model_name", DEFAULT_MODEL_NAME)
    max_length = int(meta.get("max_length", DEFAULT_MAX_LENGTH))

    # tokenizer：用你存下來的資料夾（TOKENIZER_DIR）
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)

    # backbone 用 model_name（通常是 xlm-roberta-base），不是 TOKENIZER_DIR
    model = MultiTaskEncoder(model_name)

    ckpt_sd = torch.load(BEST_PTH, map_location="cpu")
    new_sd = reconcile_state_dict_for_strict_load(model, ckpt_sd)
    model.load_state_dict(new_sd, strict=True)

    model.to(device)
    model.eval()

    for lang, prompt_path in INPUT_PROMPTS.items():
        submission = infer_one_lang(
            lang=lang,
            prompt_path=prompt_path,
            tokenizer=tokenizer,
            model=model,
            max_length=max_length,
            batch_size=BATCH_SIZE,
        )

        out_path = os.path.join(OUTPUT_DIR, f"predict_submission_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)

        print("written:", out_path)


if __name__ == "__main__":
    main()

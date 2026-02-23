# 目的：
# - 五語言資料（json/jsonl）做多任務學習：VA回歸(主) + 多個分類任務(輔)
# - 支援消融：用 ENABLED_AUX_TASKS 決定要不要訓練某些分類任務
#
# 用法（只要改 ENABLED_AUX_TASKS）：
# 全開：{"極性分類","情緒象限","強度分類","七情分類"}
# 移除七情：刪掉 "七情分類"
# 移除極性：刪掉 "極性分類"
# 移除象限：刪掉 "情緒象限"
# 移除強度：刪掉 "強度分類"

import os
import json
import re
import math
import random
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# =========================
# 基本設定（你要改的）
# =========================
MODEL_NAME = "xlm-roberta-base"

# 五語言資料（支援 .json / .jsonl）
INPUT_JSONS = {
    "zho": "data/sft/zho_environmental_protection_train_task1_sft.json",
    "eng": "data/sft/eng_environmental_protection_train_task1_sft.json",
    "swa": "data/sft/swa_politics_train_task1_sft.json",
    "pcm": "data/sft/pcm_politics_train_task1_sft.json",
    "deu": "data/sft/deu_politics_train_task1_sft.json",
}

OUTPUT_DIR = "XLM_out_multil"
SAVE_DIR = os.path.join(OUTPUT_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.8
MAX_LENGTH = 256

lr = 2e-5
epochs = 5
train_batch_size = 16
val_batch_size = 32
weight_decay = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 消融設定：你只要改這裡
# =========================
TASK_VA = "VA回歸"

# 全開（4個分類）
# ENABLED_AUX_TASKS = {"極性分類", "情緒象限", "強度分類", "七情分類"}

# 範例：移除七情
# ENABLED_AUX_TASKS = {"極性分類", "情緒象限", "強度分類"}

# 範例：移除極性
# ENABLED_AUX_TASKS = {"情緒象限", "強度分類", "七情分類"}

# 範例：移除象限
# ENABLED_AUX_TASKS = {"極性分類", "強度分類", "七情分類"}

# 範例：移除強度
ENABLED_AUX_TASKS = {"極性分類", "情緒象限", "七情分類"}

# 實際參與訓練/驗證的任務集合（VA + enabled aux）
TASKS_ACTIVE = [TASK_VA] + sorted(list(ENABLED_AUX_TASKS))

# =========================
# 多任務 loss 權重（VA為主）
# 只會對啟用的任務生效
# =========================
W_VA = 1.0
W_POL = 0.2
W_QUAD = 0.2
W_INT = 0.2
W_QIQING = 0.2

# =========================
# label mapping（中英都吃；你若只有中文也沒問題）
# =========================
POL_MAP = {
    "負面": 0, "中性": 1, "正面": 2,
    "negative": 0, "neutral": 1, "positive": 2,
    "Negative": 0, "Neutral": 1, "Positive": 2,
}
INT_MAP = {
    "平靜": 0, "中等": 1, "激動": 2,
    "calm": 0, "medium": 1, "excited": 2,
    "Calm": 0, "Medium": 1, "Excited": 2,
}
QUAD_MAP = {
    "興奮正向": 0, "平靜正向": 1, "興奮負向": 2, "平靜負向": 3,
    "excited_positive": 0, "calm_positive": 1, "excited_negative": 2, "calm_negative": 3,
    "Excited_Positive": 0, "Calm_Positive": 1, "Excited_Negative": 2, "Calm_Negative": 3,
}

# 七情：7 類
QIQING_MAP = {
    "喜": 0, "怒": 1, "憂": 2, "思": 3, "悲": 4, "恐": 5, "驚": 6,
    # 若未來有英文標註可自行加：
    "joy": 0, "anger": 1, "worry": 2, "thought": 3, "sadness": 4, "fear": 5, "surprise": 6,
    "Joy": 0, "Anger": 1, "Worry": 2, "Thought": 3, "Sadness": 4, "Fear": 5, "Surprise": 6,
}

# =========================
# utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    # JSON array / object
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

def normalize_task(t: str) -> str:
    # 若你未來把 task 改成英文，也可在這裡擴充 alias
    return t

def base_key_from_item(item: dict, lang: str) -> str:
    # 用 lang 前綴，避免不同語言 ID 撞到
    t = normalize_task(item.get("task", ""))
    if t == TASK_VA:
        return f"{lang}:{item['ID']}"
    return f"{lang}:{item['ID'].rsplit('__', 1)[0]}"

def parse_text_aspect(s: str) -> Tuple[str, str]:
    # 格式："Text: ...\nAspects: ..."
    text = ""
    aspect = ""
    m1 = re.search(r"Text:\s*(.*)", s)
    if m1:
        text = m1.group(1).strip()
    m2 = re.search(r"Aspects:\s*(.*)", s)
    if m2:
        aspect = m2.group(1).strip()
    return text, aspect

def parse_va(va_str: str) -> Tuple[float, float]:
    v_str, a_str = va_str.split("#")
    v = float(v_str); a = float(a_str)
    v = max(1.0, min(9.0, v))
    a = max(1.0, min(9.0, a))
    return v, a

def build_joint_samples(raw_items: List[dict], keep_keys: set, lang: str) -> List[Dict[str, Any]]:
    """
    把同一個 (Text, Aspect) 的不同 task 合併成一個 joint sample。
    只要求「啟用的任務」都齊全。
    """
    group: Dict[str, Dict[str, Any]] = {}

    def ensure(k: str, it: dict):
        if k not in group:
            text, aspect = parse_text_aspect(it["input"])
            group[k] = {
                "ID": k,
                "lang": lang,
                "text": text,
                "aspect": aspect,
                "v": None, "a": None,
                "pol": None,
                "quad": None,
                "intensity": None,
                "qiqing": None,
            }

    for it in raw_items:
        t = normalize_task(it.get("task", ""))
        if t not in TASKS_ACTIVE:
            continue
        k = base_key_from_item(it, lang)
        if k not in keep_keys:
            continue

        ensure(k, it)

        if t == TASK_VA:
            va = it["output"]["Aspect_VA"][0]["VA"]
            v, a = parse_va(va)
            group[k]["v"] = v
            group[k]["a"] = a

        elif t == "極性分類":
            lab = it["output"]["極性"]
            group[k]["pol"] = POL_MAP[lab]

        elif t == "情緒象限":
            lab = it["output"]["情緒象限"]
            group[k]["quad"] = QUAD_MAP[lab]

        elif t == "強度分類":
            lab = it["output"]["強度"]
            group[k]["intensity"] = INT_MAP[lab]

        elif t == "七情分類":
            # 你的產檔是 {"七情": "..."}
            lab = it["output"]["七情"]
            group[k]["qiqing"] = QIQING_MAP[lab]

    # 只保留啟用任務都齊全的樣本
    joint = []
    for _, ex in group.items():
        ok = True
        if ex["v"] is None or ex["a"] is None:
            ok = False
        if ok and ("極性分類" in ENABLED_AUX_TASKS) and (ex["pol"] is None):
            ok = False
        if ok and ("情緒象限" in ENABLED_AUX_TASKS) and (ex["quad"] is None):
            ok = False
        if ok and ("強度分類" in ENABLED_AUX_TASKS) and (ex["intensity"] is None):
            ok = False
        if ok and ("七情分類" in ENABLED_AUX_TASKS) and (ex["qiqing"] is None):
            ok = False

        if ok:
            joint.append(ex)
    return joint

class JointMultiTaskDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def make_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict[str, Any]]):
        texts = [x["text"] for x in batch]
        aspects = [x["aspect"] for x in batch]

        enc = tokenizer(
            texts,
            aspects,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        labels = {}
        labels["va"] = torch.tensor([[x["v"], x["a"]] for x in batch], dtype=torch.float32)

        if "極性分類" in ENABLED_AUX_TASKS:
            labels["pol"] = torch.tensor([x["pol"] for x in batch], dtype=torch.long)
        if "情緒象限" in ENABLED_AUX_TASKS:
            labels["quad"] = torch.tensor([x["quad"] for x in batch], dtype=torch.long)
        if "強度分類" in ENABLED_AUX_TASKS:
            labels["int"] = torch.tensor([x["intensity"] for x in batch], dtype=torch.long)
        if "七情分類" in ENABLED_AUX_TASKS:
            labels["qiqing"] = torch.tensor([x["qiqing"] for x in batch], dtype=torch.long)

        return enc, labels
    return collate_fn

class MultiTaskEncoder(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.head_va = nn.Linear(hidden, 2)
        self.head_pol = nn.Linear(hidden, 3) if ("極性分類" in ENABLED_AUX_TASKS) else None
        self.head_quad = nn.Linear(hidden, 4) if ("情緒象限" in ENABLED_AUX_TASKS) else None
        self.head_int = nn.Linear(hidden, 3) if ("強度分類" in ENABLED_AUX_TASKS) else None
        self.head_qiqing = nn.Linear(hidden, 7) if ("七情分類" in ENABLED_AUX_TASKS) else None

    def forward(self, inputs, labels: Dict[str, torch.Tensor] = None):
        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in inputs:
            if "token_type_ids" in self.backbone.forward.__code__.co_varnames:
                kwargs["token_type_ids"] = inputs["token_type_ids"]

        out = self.backbone(**kwargs)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        # VA
        va_raw = self.head_va(cls)
        va_pred = 1.0 + 8.0 * torch.sigmoid(va_raw)

        # Optional heads
        pol_logits = self.head_pol(cls) if self.head_pol is not None else None
        quad_logits = self.head_quad(cls) if self.head_quad is not None else None
        int_logits = self.head_int(cls) if self.head_int is not None else None
        qiqing_logits = self.head_qiqing(cls) if self.head_qiqing is not None else None

        loss = None
        if labels is not None:
            loss_total = 0.0

            # VA loss (always)
            loss_va = F.mse_loss(va_pred, labels["va"])
            loss_total = loss_total + (W_VA * loss_va)

            # enabled aux losses
            if pol_logits is not None:
                loss_pol = F.cross_entropy(pol_logits, labels["pol"])
                loss_total = loss_total + (W_POL * loss_pol)
            if quad_logits is not None:
                loss_quad = F.cross_entropy(quad_logits, labels["quad"])
                loss_total = loss_total + (W_QUAD * loss_quad)
            if int_logits is not None:
                loss_int = F.cross_entropy(int_logits, labels["int"])
                loss_total = loss_total + (W_INT * loss_int)
            if qiqing_logits is not None:
                loss_q = F.cross_entropy(qiqing_logits, labels["qiqing"])
                loss_total = loss_total + (W_QIQING * loss_q)

            loss = loss_total

        return loss, va_pred, pol_logits, quad_logits, int_logits, qiqing_logits

def rmse_va(pred_va: torch.Tensor, gold_va: torch.Tensor) -> float:
    mse = ((pred_va - gold_va) ** 2).mean().item()
    return math.sqrt(mse)

def accuracy(logits: torch.Tensor, gold: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == gold).float().mean().item()

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("===== ACTIVE TASKS =====")
    print("VA always on:", TASK_VA)
    print("AUX enabled:", sorted(list(ENABLED_AUX_TASKS)))
    print("========================\n")

    # ============
    # 讀五語言 + 各語言各自切 split，最後合併
    # ============
    train_samples_all: List[Dict[str, Any]] = []
    val_samples_all: List[Dict[str, Any]] = []

    for lang, path in INPUT_JSONS.items():
        raw = load_json_or_jsonl(path)

        present = defaultdict(set)
        for it in raw:
            t = normalize_task(it.get("task", ""))
            if t in TASKS_ACTIVE:
                present[base_key_from_item(it, lang)].add(t)

        # 必須包含 VA + 所有啟用的 aux task
        required = set(TASKS_ACTIVE)
        valid_keys = [k for k, s in present.items() if required.issubset(s)]
        valid_keys.sort()

        rnd = random.Random(SEED)  # 每語言同 seed、可重現
        rnd.shuffle(valid_keys)

        n_train = int(len(valid_keys) * TRAIN_RATIO)
        train_keys = set(valid_keys[:n_train])
        val_keys = set(valid_keys[n_train:])

        tr = build_joint_samples(raw, train_keys, lang)
        va = build_joint_samples(raw, val_keys, lang)

        print(f"[{lang}] valid_keys={len(valid_keys)} | train={len(tr)} | val={len(va)}")

        train_samples_all.extend(tr)
        val_samples_all.extend(va)

    print(f"\nTOTAL joint samples: train={len(train_samples_all)} | val={len(val_samples_all)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collate_fn = make_collate_fn(tokenizer)

    ds_train = JointMultiTaskDataset(train_samples_all)
    ds_val = JointMultiTaskDataset(val_samples_all)

    dl_train = DataLoader(ds_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskEncoder(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rmse = float("inf")
    best_path = os.path.join(SAVE_DIR, "best_model.pth")

    for ep in range(epochs):
        # -------- train --------
        model.train()
        pbar = tqdm(dl_train, desc=f"Training epoch [{ep+1}/{epochs}]")
        for inputs, labels in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            loss, *_ = model(inputs, labels=labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        # -------- eval --------
        model.eval()
        all_pred_va, all_gold_va = [], []

        all_pol_logits, all_pol_gold = [], []
        all_quad_logits, all_quad_gold = [], []
        all_int_logits, all_int_gold = [], []
        all_q_logits, all_q_gold = [], []

        with torch.no_grad():
            pbar = tqdm(dl_val, desc=f"Validation epoch [{ep+1}/{epochs}]")
            for inputs, labels in pbar:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = {k: v.to(device) for k, v in labels.items()}

                loss, va_pred, pol_logits, quad_logits, int_logits, qiqing_logits = model(inputs, labels=labels)

                all_pred_va.append(va_pred.detach().cpu())
                all_gold_va.append(labels["va"].detach().cpu())

                if pol_logits is not None:
                    all_pol_logits.append(pol_logits.detach().cpu())
                    all_pol_gold.append(labels["pol"].detach().cpu())
                if quad_logits is not None:
                    all_quad_logits.append(quad_logits.detach().cpu())
                    all_quad_gold.append(labels["quad"].detach().cpu())
                if int_logits is not None:
                    all_int_logits.append(int_logits.detach().cpu())
                    all_int_gold.append(labels["int"].detach().cpu())
                if qiqing_logits is not None:
                    all_q_logits.append(qiqing_logits.detach().cpu())
                    all_q_gold.append(labels["qiqing"].detach().cpu())

        pred_va = torch.cat(all_pred_va, dim=0)
        gold_va = torch.cat(all_gold_va, dim=0)
        val_rmse = rmse_va(pred_va, gold_va)

        msg = f"[Epoch {ep+1}] VA_RMSE={val_rmse:.4f}"

        if "極性分類" in ENABLED_AUX_TASKS:
            pol_logits = torch.cat(all_pol_logits, dim=0)
            pol_gold = torch.cat(all_pol_gold, dim=0)
            msg += f" | 極性Acc={accuracy(pol_logits, pol_gold):.4f}"

        if "情緒象限" in ENABLED_AUX_TASKS:
            quad_logits = torch.cat(all_quad_logits, dim=0)
            quad_gold = torch.cat(all_quad_gold, dim=0)
            msg += f" 象限Acc={accuracy(quad_logits, quad_gold):.4f}"

        if "強度分類" in ENABLED_AUX_TASKS:
            int_logits = torch.cat(all_int_logits, dim=0)
            int_gold = torch.cat(all_int_gold, dim=0)
            msg += f" 強度Acc={accuracy(int_logits, int_gold):.4f}"

        if "七情分類" in ENABLED_AUX_TASKS:
            q_logits = torch.cat(all_q_logits, dim=0)
            q_gold = torch.cat(all_q_gold, dim=0)
            msg += f" 七情Acc={accuracy(q_logits, q_gold):.4f}"

        print(msg)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_path)
            tokenizer.save_pretrained(OUTPUT_DIR)

            meta = {
                "best_rmse": best_rmse,
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "loss_weights": {
                    "W_VA": W_VA,
                    "W_POL": W_POL if "極性分類" in ENABLED_AUX_TASKS else 0.0,
                    "W_QUAD": W_QUAD if "情緒象限" in ENABLED_AUX_TASKS else 0.0,
                    "W_INT": W_INT if "強度分類" in ENABLED_AUX_TASKS else 0.0,
                    "W_QIQING": W_QIQING if "七情分類" in ENABLED_AUX_TASKS else 0.0,
                },
                "label_maps": {
                    "POL_MAP": POL_MAP,
                    "INT_MAP": INT_MAP,
                    "QUAD_MAP": QUAD_MAP,
                    "QIQING_MAP": QIQING_MAP,
                },
                "langs": list(INPUT_JSONS.keys()),
                "train_ratio_per_lang": TRAIN_RATIO,
                "enabled_aux_tasks": sorted(list(ENABLED_AUX_TASKS)),
            }
            with open(os.path.join(SAVE_DIR, "best_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"  ✅ saved best: {best_path} (rmse={best_rmse:.4f})")

    print(f"done. best_rmse={best_rmse:.4f} | best_path={best_path}")

if __name__ == "__main__":
    main()

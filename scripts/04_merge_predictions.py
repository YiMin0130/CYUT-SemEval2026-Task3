# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path
from tqdm import tqdm

# =========================
# 路徑設定
# =========================
# 模型輸出的資料夾，內含 predict_submission_deu.json 等
PRED_DIR = "results/predict_multi5_qwen" 
# 原始測試集資料夾，內含 deu_environmental_protection_test_task1.jsonl 等
TEST_DIR = "data/raw/track_b/test"
# 合併後的輸出資料夾
OUTPUT_DIR = "results/results_fixed"

# =========================
# 工具函式
# =========================
def load_jsonl(path: Path):
    items = []
    if not path.exists(): return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except: continue
    return items

def aspects_from_test_item(test_item: dict):
    """提取原始要求的 Aspect 列表"""
    if "Aspect_VA" in test_item:
        return [x.get("Aspect") for x in test_item["Aspect_VA"] if x.get("Aspect")]
    for k in ["Aspects", "aspects", "Aspect", "aspect"]:
        if k in test_item:
            v = test_item[k]
            if isinstance(v, list): return [str(x) for x in v]
            if isinstance(v, str):
                return [x.strip() for x in re.split('[;,]', v) if x.strip()]
    return []

# =========================
# 核心合併邏輯
# =========================
def merge_and_fix(pred_path: Path, test_path: Path, out_path: Path):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    
    # 建立索引：同時支援 ID__Aspect 和 原始 ID
    pred_by_id = {str(x["ID"]): x for x in pred_data}

    test_items = load_jsonl(test_path)
    fixed_lines = []
    
    for it in test_items:
        _id = str(it.get("ID"))
        req_aspects = aspects_from_test_item(it)
        current_av = []
        
        for asp in req_aspects:
            # 優先找拆分過的 ID: "zho-1__完全氫化植物油"
            split_id = f"{_id}__{asp}"
            
            if split_id in pred_by_id:
                p_item = pred_by_id[split_id]
                # 取得預測的 VA
                va_val = "5.00#5.00" # 預設中值
                if "Aspect_VA" in p_item and p_item["Aspect_VA"]:
                    va_val = p_item["Aspect_VA"][0].get("VA", "5.00#5.00")
                elif "VA" in p_item:
                    va_val = p_item["VA"]
                current_av.append({"Aspect": asp, "VA": va_val})
            
            elif _id in pred_by_id:
                # 找原始 ID 內部的列表
                found = False
                for x in pred_by_id[_id].get("Aspect_VA", []):
                    if x.get("Aspect") == asp:
                        current_av.append(x)
                        found = True
                        break
                if not found:
                    current_av.append({"Aspect": asp, "VA": "5.00#5.00"})
            else:
                current_av.append({"Aspect": asp, "VA": "5.00#5.00"})

        fixed_lines.append({"ID": _id, "Aspect_VA": current_av})

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in fixed_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# =========================
# 主流程：自動匹配語言代碼
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_files = list(Path(TEST_DIR).glob("*.jsonl"))
    
    print(f"開始合併預測結果...")

    for t_path in tqdm(test_files):
        # 從測試檔名抓取前三個字當語言代碼 (如 deu, zho, pcm)
        lang_code = t_path.stem[:3].lower()
        
        # 尋找對應的預測檔: predict_submission_deu.json
        p_name = f"predict_submission_{lang_code}.json"
        p_path = Path(PRED_DIR) / p_name
        
        if p_path.exists():
            out_path = Path(OUTPUT_DIR) / f"{t_path.stem}_fixed.jsonl"
            merge_and_fix(p_path, t_path, out_path)
        else:
            print(f"\n跳過 {t_path.name}: 找不到預測檔 {p_name}")

    print(f"合併完成！輸出於: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
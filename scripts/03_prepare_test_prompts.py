import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

# =========================
# 路徑設定
# =========================
INPUT_DIR = "data/raw/track_b/test"
OUTPUT_DIR = "data/prompts"

INSTRUCTION_VA = (
    "你是一個情緒回歸模型。請讀取輸入文本，並針對指定的 Aspect（目標）預測 Valence-Arousal（V-A）分數。"
    "Valence（V）：衡量正面或負面的程度；Arousal（A）：衡量平靜或興奮的程度。"
    "分數範圍 1.00–9.00，保留兩位小數。"
    "1.00 表示極度負面效價或極低喚醒度；9.00 表示極度正面效價或極高喚醒度；"
    "5.00 表示中性效價或中等喚醒度。"
    "請以字串格式「V#A」呈現，並只輸出 JSON："
    "{\"Aspect_VA\":[{\"Aspect\":\"...\",\"VA\":\"V#A\"}]}。"
)

# =========================
# 工具函式
# =========================
def load_jsonl_robust(path: Path) -> List[Dict[str, Any]]:
    """修正過的讀取邏輯：逐行讀取以確保 JSONL 格式不報錯"""
    items = []
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items

def safe_id_part(s: str, max_len: int = 40) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "", s)
    return s[:max_len]

# =========================
# 轉換邏輯 (針對你的資料格式優化)
# =========================
def to_test_prompts_per_aspect(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        parent_id = it.get("ID")
        text = (it.get("Text") or "").strip()
        
        # 兼容你的 Aspect_VA 格式
        aspect_data = it.get("Aspect_VA", [])
        
        # 如果是測試集，可能只有 Aspect 列表而沒有 VA 欄位
        if not aspect_data:
            aspect_data = it.get("Aspect", it.get("Aspects", []))
        
        # 統一轉為字串列表處理
        final_aspects = []
        if isinstance(aspect_data, list):
            for item in aspect_data:
                if isinstance(item, dict):
                    final_aspects.append(item.get("Aspect", ""))
                else:
                    final_aspects.append(str(item))
        elif isinstance(aspect_data, str):
            final_aspects.append(aspect_data)

        # 拆解成單筆 Prompt
        for a in final_aspects:
            a = str(a).strip()
            if not a: continue
            
            sid = safe_id_part(a)
            _id = f"{parent_id}__{sid}"
            out.append({
                "ID": _id,
                "parent_ID": parent_id,
                "Aspect": a,
                "instruction": INSTRUCTION_VA,
                "input": f"Text: {text}\nAspects: {a}",
            })
    return out

# =========================
# 主程式
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_path = Path(INPUT_DIR)
    files = list(input_path.glob("*.json*"))
    
    if not files:
        print(f"Error: 在 {INPUT_DIR} 找不到任何檔案。")
        return

    print(f"找到 {len(files)} 個檔案，準備生成推理 Prompt...")

    for f_path in tqdm(files):
        # 1. 讀取
        items = load_jsonl_robust(f_path)
        if not items:
            continue
            
        # 2. 轉換
        prompts = to_test_prompts_per_aspect(items)
        
        # 3. 儲存
        output_file = Path(OUTPUT_DIR) / f"{f_path.stem}_prompts.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\n處理完成！結果存於: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
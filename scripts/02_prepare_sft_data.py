import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

# =========================
# 資料夾路徑設定
# =========================
# 原始資料資料夾 (包含原始的 jsonl)
INPUT_DIR = "data/raw/track_b/train"
# 七情標註資料夾 (包含你上一階段生成的標註檔)
QIQING_DIR = "data/processed"
# SFT 格式輸出資料夾
OUTPUT_DIR = "data/sft"

# 多任務開關
ADD_POLARITY = True
ADD_QUADRANT = True
ADD_INTENSITY = True
ADD_QIQING = True

# =========================
# 可調門檻與標籤
# =========================
V_POS_TH, V_NEG_TH = 5.50, 4.50
A_HIGH_TH, A_LOW_TH = 6.00, 4.00
QIQING_LABELS = ["喜", "怒", "憂", "思", "悲", "恐", "驚"]

# =========================
# 統一 instruction（訓練/推理都用同一套 JSON schema）
# =========================
INSTRUCTION_VA = (
    "你是一個情緒回歸模型。請讀取輸入文本，並針對指定的 Aspect（目標）預測 Valence-Arousal（V-A）分數。"
    "Valence（V）：衡量正面或負面的程度；Arousal（A）：衡量平靜或興奮的程度。"
    "分數範圍 1.00–9.00，保留兩位小數。"
    "1.00 表示極度負面效價或極低喚醒度；9.00 表示極度正面效價或極高喚醒度；"
    "5.00 表示中性效價或中等喚醒度。"
    "請以字串格式「V#A」呈現，並只輸出 JSON："
    "{\"Aspect_VA\":[{\"Aspect\":\"...\",\"VA\":\"V#A\"}]}。"
)

INSTRUCTION_POLARITY = (
    "你是一個情緒極性分類模型。請讀取輸入文本，並針對指定的 Aspect（目標）輸出情緒極性標籤。"
    "Valence（V）：衡量正面或負面的程度，分數範圍 1.00–9.00，5.00 為中性。"
    f"本任務極性由效價（V）推得：V >= {V_POS_TH:.2f} → 正面；V <= {V_NEG_TH:.2f} → 負面；其餘 → 中性。"
    "標籤必須為以下三選一：「正面」「負面」「中性」。"
    "請只輸出 JSON：{\"極性\":\"正面|負面|中性\"}。"
)

INSTRUCTION_QUADRANT = (
    "你是一個情緒象限分類模型。請讀取輸入文本，並針對指定的 Aspect（目標）輸出情緒象限標籤。"
    "Valence（V）：正負向程度；Arousal（A）：平靜/興奮程度；分數 1.00–9.00，5.00 為中等。"
    "象限規則："
    "V > 5.00 且 A > 5.00 → 興奮正向；"
    "V > 5.00 且 A <= 5.00 → 平靜正向；"
    "V <= 5.00 且 A > 5.00 → 興奮負向；"
    "V <= 5.00 且 A <= 5.00 → 平靜負向。"
    "標籤必須為以下四選一：「興奮正向」「平靜正向」「興奮負向」「平靜負向」。"
    "請只輸出 JSON：{\"情緒象限\":\"興奮正向|平靜正向|興奮負向|平靜負向\"}。"
)

INSTRUCTION_INTENSITY = (
    "你是一個情緒強度分類模型。請讀取輸入文本，並針對指定的 Aspect（目標）輸出情緒強度標籤。"
    "Arousal（A）：衡量平靜或興奮的程度，分數範圍 1.00–9.00，5.00 為中等。"
    f"本任務強度由喚醒度（A）推得：A >= {A_HIGH_TH:.2f} → 激動；A <= {A_LOW_TH:.2f} → 平靜；其餘 → 中等。"
    "標籤必須為以下三選一：「平靜」「中等」「激動」。"
    "請只輸出 JSON：{\"強度\":\"平靜|中等|激動\"}。"
)

INSTRUCTION_QIQING = (
    "你是一個七情分類模型。請讀取輸入文本，並針對指定的 Aspect（目標）判斷文本表達的主要情緒。"
    "七情標籤必須為以下七選一：「喜」「怒」「憂」「思」「悲」「恐」「驚喜」。"
    "請只輸出 JSON：{\"七情\":\"喜|怒|憂|思|悲|恐|驚喜\"}。"
)

# =========================
# I/O 工具
# =========================
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8").strip()

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
    items: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def safe_id_part(s: str, max_len: int = 40) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "", s)
    return s[:max_len]

# =========================
# VA 派生規則
# =========================
def parse_va(va_str: str) -> Tuple[float, float]:
    v_str, a_str = va_str.split("#")
    return float(v_str), float(a_str)

def clamp_round(x: float) -> float:
    x = max(1.0, min(9.0, x))
    return round(x + 1e-8, 2)

def polarity_from_v(v: float) -> str:
    if v >= V_POS_TH:
        return "正面"
    if v <= V_NEG_TH:
        return "負面"
    return "中性"

def quadrant_from_va(v: float, a: float) -> str:
    if v > 5.00 and a > 5.00:
        return "興奮正向"
    if v > 5.00 and a <= 5.00:
        return "平靜正向"
    if v <= 5.00 and a > 5.00:
        return "興奮負向"
    return "平靜負向"

def intensity_from_a(a: float) -> str:
    if a >= A_HIGH_TH:
        return "激動"
    if a <= A_LOW_TH:
        return "平靜"
    return "中等"

# =========================
# 七情 map
# =========================
def load_qiqing_map(path: str) -> Dict[Tuple[str, str], str]:
    """讀 per-aspect JSONL，建立 (ID, Aspect) -> 七情 label 的對照。"""
    mp: Dict[Tuple[str, str], str] = {}
    if not path:
        return mp
    for it in load_json_or_jsonl(path):
        _id = str(it.get("ID") or "").strip()
        asp = str(it.get("Aspect") or "").strip()
        lab = it.get("Qiqing") or it.get("七情")
        lab = str(lab or "").strip()
        if not _id or not asp or not lab:
            continue
        mp[(_id, asp)] = lab
    return mp

# =========================
# 兼容輸入結構
# =========================
def extract_aspects(it: Dict[str, Any]) -> List[Dict[str, str]]:
    """回傳 [{Aspect, VA}] list，兼容 Aspect_VA 或單一 Aspect+VA。"""
    if isinstance(it.get("Aspect_VA"), list):
        out = []
        for av in it.get("Aspect_VA", []):
            if not isinstance(av, dict):
                continue
            out.append(
                {
                    "Aspect": str(av.get("Aspect") or "").strip(),
                    "VA": str(av.get("VA") or "").strip(),
                }
            )
        return out

    # per-aspect
    if it.get("Aspect") is not None and it.get("VA") is not None:
        return [
            {
                "Aspect": str(it.get("Aspect") or "").strip(),
                "VA": str(it.get("VA") or "").strip(),
            }
        ]

    return []

# =========================
# 轉換主流程
# =========================
def to_sft_per_aspect_multitask(
    items: List[Dict[str, Any]],
    add_polarity: bool = True,
    add_quadrant: bool = True,
    add_intensity: bool = True,
    add_qiqing: bool = True,
    qiqing_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Dict[str, Any]]:
    qiqing_map = qiqing_map or {}
    out: List[Dict[str, Any]] = []

    for it in items:
        orig_id = it.get("ID")
        text = (it.get("Text") or it.get("text") or "").strip()
        if not orig_id or not text:
            continue

        aspect_va_list = extract_aspects(it)
        if not aspect_va_list:
            continue

        for av in aspect_va_list:
            aspect = (av.get("Aspect") or "").strip()
            va_str = (av.get("VA") or "").strip()
            if not aspect or not va_str:
                continue

            # 解析 + 校正 VA
            try:
                v, a = parse_va(va_str)
                v = clamp_round(v)
                a = clamp_round(a)
            except Exception:
                v, a = 5.00, 5.00

            va_fixed = f"{v:.2f}#{a:.2f}"
            input_text = f"Text: {text}\nAspects: {aspect}"

            sid = safe_id_part(aspect)
            base_id = f"{orig_id}__{sid}"

            # 主任務：VA 回歸
            out.append(
                {
                    "ID": base_id,
                    "parent_ID": orig_id,
                    "task": "VA回歸",
                    "instruction": INSTRUCTION_VA,
                    "input": input_text,
                    "output": {"Aspect_VA": [{"Aspect": aspect, "VA": va_fixed}]},
                }
            )

            # 輔助：極性（由 V）
            if add_polarity:
                out.append(
                    {
                        "ID": base_id + "__極性",
                        "parent_ID": orig_id,
                        "task": "極性分類",
                        "instruction": INSTRUCTION_POLARITY,
                        "input": input_text,
                        "output": {"極性": polarity_from_v(v)},
                    }
                )

            # 輔助：象限（由 V,A）
            if add_quadrant:
                out.append(
                    {
                        "ID": base_id + "__象限",
                        "parent_ID": orig_id,
                        "task": "情緒象限",
                        "instruction": INSTRUCTION_QUADRANT,
                        "input": input_text,
                        "output": {"情緒象限": quadrant_from_va(v, a)},
                    }
                )

            # 輔助：強度（由 A）
            if add_intensity:
                out.append(
                    {
                        "ID": base_id + "__強度",
                        "parent_ID": orig_id,
                        "task": "強度分類",
                        "instruction": INSTRUCTION_INTENSITY,
                        "input": input_text,
                        "output": {"強度": intensity_from_a(a)},
                    }
                )

            # 輔助：七情（優先用外部 map，其次用輸入內 Qiqing/七情 欄位）
            if add_qiqing:
                q = None
                key = (str(orig_id), aspect)
                if key in qiqing_map:
                    q = qiqing_map[key]
                else:
                    q = it.get("Qiqing") or it.get("七情")

                q = str(q or "").strip()
                if q in QIQING_LABELS:
                    out.append(
                        {
                            "ID": base_id + "__七情",
                            "parent_ID": orig_id,
                            "task": "七情分類",
                            "instruction": INSTRUCTION_QIQING,
                            "input": input_text,
                            "output": {"七情": q},
                        }
                    )

    return out

def load_qiqing_map_per_file(path: Path) -> Dict[Tuple[str, str], str]:
    """讀取對應的七情標註檔，建立 (ID, Aspect) -> 七情標籤。"""
    mp: Dict[Tuple[str, str], str] = {}
    if not path.exists():
        return mp
    
    # 這裡假設標註檔是每行一個 JSON 的格式
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                it = json.loads(line)
                _id = str(it.get("ID") or "").strip()
                # 如果你的標註檔結構是扁平的 (已經拆開 Aspect)
                asp = str(it.get("Aspect") or "").strip()
                lab = str(it.get("Qiqing") or it.get("七情") or "").strip()
                if _id and asp and lab:
                    mp[(_id, asp)] = lab
                
                # 如果你的標註檔結構是巢狀的 (Aspect_VA 裡面有 Qiqing)
                for av in it.get("Aspect_VA", []):
                    asp = str(av.get("Aspect") or "").strip()
                    lab = str(av.get("Qiqing") or "").strip()
                    if _id and asp and lab:
                        mp[(_id, asp)] = lab
            except: continue
    return mp

def process_batch():
    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_path = Path(INPUT_DIR)
    qiqing_dir_path = Path(QIQING_DIR)
    files = list(input_path.glob("*.json*"))
    
    if not files:
        print("找不到任何輸入檔案。")
        return

    print(f"開始批次處理，共 {len(files)} 個檔案...")

    for f_path in tqdm(files, desc="Batch Processing"):
        # 1. 讀取輸入原始資料
        items = load_json_or_jsonl(str(f_path))
        
        # 2. 尋找對應的七情標註檔 (假設檔名相同)
        qiqing_file = qiqing_dir_path / f_path.name
        qiqing_map = {}
        if ADD_QIQING:
            qiqing_map = load_qiqing_map_per_file(qiqing_file)
            if not qiqing_map:
                print(f"警告: 找不到 {f_path.name} 的對應七情標註或格式不符。")

        # 3. 轉換為 SFT 格式 (調用你原本的 to_sft_per_aspect_multitask 函式)
        # 注意：此處需確保該函式已定義在腳本中
        sft_data = to_sft_per_aspect_multitask(
            items,
            add_polarity=ADD_POLARITY,
            add_quadrant=ADD_QUADRANT,
            add_intensity=ADD_INTENSITY,
            add_qiqing=ADD_QIQING,
            qiqing_map=qiqing_map
        )

        # 4. 儲存結果 (改為 .json 陣列格式)
        output_file = Path(OUTPUT_DIR) / f"{f_path.stem}_sft.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n全部處理完成！輸出資料夾：{OUTPUT_DIR}")

if __name__ == "__main__":
    # 執行前請確保所有 INSTRUCTION_XXX 變數與工具函式都在腳本內
    process_batch()

import json
import math
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# =========================================================
# 路徑與參數設定
# =========================================================
INPUT_DIR = "data/raw/track_b/train"
OUTPUT_DIR = "data/processed"

TAU_QUANTILE = 0.20
ORDER = ["憂", "喜", "驚", "怒", "恐", "悲"]
WEIGHTS = {"喜": 1.0, "憂": 1.0, "驚": 1.0, "怒": 1.0, "恐": 1.0, "悲": 1.0}

# =========================================================
# 工具函式
# =========================================================
def parse_va(va_str: str):
    if not isinstance(va_str, str) or "#" not in va_str:
        return None
    try:
        v, a = va_str.split("#", 1)
        return float(v), float(a)
    except:
        return None

def va_to_theta_r(V, A, center=5.0):
    v_rel, a_rel = V - center, A - center
    r = math.sqrt(v_rel**2 + a_rel**2)
    theta = (math.degrees(math.atan2(a_rel, v_rel)) + 360.0) % 360.0
    return theta, r

def build_theta_cuts(thetas, weights):
    thetas = np.sort(np.array(thetas, dtype=float))
    w = np.array([weights[k] for k in ORDER], dtype=float)
    w /= w.sum()
    cum = np.cumsum(w)
    qs = cum[:-1]
    cut_vals = [float(np.quantile(thetas, q)) for q in qs]
    bounds = [0.0] + cut_vals + [360.0]
    return [(ORDER[i], bounds[i], bounds[i+1]) for i in range(len(ORDER))]

# =========================================================
# 主流程：每個檔案獨立計算
# =========================================================
def process_files_independently():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_path = Path(INPUT_DIR)
    files = list(input_path.glob("*.json*"))

    for f_path in tqdm(files, desc="Processing Files"):
        # 1. 讀取該檔案所有資料
        rows = []
        file_rs = []
        file_thetas_all = [] # 格式: (index_of_row, index_of_aspect, theta, r)

        with open(f_path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                item = json.loads(line)
                rows.append(item)
                for av_idx, av in enumerate(item.get("Aspect_VA", [])):
                    p = parse_va(av.get("VA", ""))
                    if p:
                        theta, r = va_to_theta_r(p[0], p[1])
                        file_rs.append(r)
                        file_thetas_all.append((row_idx, av_idx, theta, r))

        if not file_rs:
            continue

        # 2. 針對該檔案計算獨立的動態閾值
        tau = float(np.quantile(file_rs, TAU_QUANTILE))
        eff_thetas = [t for _, _, t, r in file_thetas_all if r >= tau]
        
        # 避免樣本過少無法計算
        if len(eff_thetas) < 2:
            print(f"警告: {f_path.name} 有效樣本過少，跳過。")
            continue
            
        cuts = build_theta_cuts(eff_thetas, WEIGHTS)

        # 3. 根據該檔案的 cuts 進行標註
        file_cnt = Counter()
        for row_idx, av_idx, theta, r in file_thetas_all:
            if r < tau:
                qiqing = "思"
            else:
                qiqing = ORDER[-1] # 預設最後一個
                for lab, s, e in cuts:
                    if s <= theta < e:
                        qiqing = lab
                        break
            
            # 寫回原始資料結構
            rows[row_idx]["Aspect_VA"][av_idx]["Qiqing"] = qiqing
            file_cnt[qiqing] += 1

        # 4. 儲存結果
        output_file = Path(OUTPUT_DIR) / f_path.name
        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        # 打印該檔案的小計 (可選)
        print(f"\nFile: {f_path.name} | Tau: {tau:.3f} | Distribution: {dict(file_cnt)}")

if __name__ == "__main__":
    process_files_independently()
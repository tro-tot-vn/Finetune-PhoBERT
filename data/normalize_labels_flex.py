# Chuyển mọi nhãn 2 -> 1, giữ 0/1. Hỗ trợ:
# - Có hoặc không có header
# - Chỉ định tên cột hoặc chỉ số cột (0-based)
# Dùng:
#   python normalize_labels_flex.py --in data.csv --out data_bin.csv
#   # Tùy chọn:
#   # --sep "," | "\t"
#   # --encoding "utf-8" | "utf-16" | ...
#   # --no-header (nếu file không có dòng header)
#   # --label-col label   (tên cột)  hoặc  --label-idx 3 (chỉ số cột)
#   # --text-col text     (tên cột)  hoặc  --text-idx 0  (chỉ số cột)

import argparse, sys
import pandas as pd

def try_read(path, sep, enc, header):
    read_kwargs = dict(sep=sep, encoding=enc)
    if header:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None
    return pd.read_csv(path, **read_kwargs)

def coerce_bin(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s in {"0", "clean", "valid"}: return 0
    if s in {"1", "2", "offensive", "hate", "invalid"}: return 1
    try:
        x = int(float(s))
        return 0 if x == 0 else 1
    except:
        return None

def pick_column(df, name, idx, fallback_last=False, want="label"):
    # name ưu tiên, rồi idx; nếu không có, fallback theo logic
    if name is not None:
        if name in df.columns:
            return name
        else:
            print(f"[!] Không thấy cột '{name}' trong file.", file=sys.stderr)
            print("    Các cột:", list(df.columns), file=sys.stderr); sys.exit(1)
    if idx is not None:
        try:
            col = df.columns[int(idx)]
            return col
        except Exception:
            print(f"[!] Chỉ số cột {idx} không hợp lệ.", file=sys.stderr); sys.exit(1)
    if fallback_last:
        return df.columns[-1]  # thường label ở cuối
    # đoán theo tên
    candidates = ["label","labels","gold","target"] if want=="label" else ["text","comment","content","sentence","post","review"]
    for c in df.columns:
        if str(c).strip().lower() in candidates:
            return c
    # nếu vẫn không có:
    if want == "label":
        return df.columns[-1]  # đoán cột cuối là label
    else:
        return df.columns[0]   # đoán cột đầu là text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--no-header", action="store_true", help="File không có dòng header")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--label-idx", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--text-idx", default=None)
    args = ap.parse_args()

    df = try_read(args.inp, sep=args.sep, enc=args.encoding, header=not args.no_header)

    # Nếu không header: tự đặt tên cột tạm
    if args.no_header:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # Chọn cột label/text
    label_col = pick_column(df, args.label_col, args.label_idx, fallback_last=True, want="label")
    text_col  = pick_column(df, args.text_col,  args.text_idx,  fallback_last=False, want="text")

    # Map nhãn 2 -> 1, giữ 0/1
    df["label"] = df[label_col].map(coerce_bin)
    df["text"]  = df[text_col].astype(str)

    before = len(df)
    df = df.dropna(subset=["label","text"])
    df["label"] = df["label"].astype(int)

    out_df = df[["text","label"]]
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Đã lưu: {args.out}")
    print(f"Số dòng: {len(out_df)} (bỏ {before - len(out_df)} dòng thiếu)")
    print("Phân bố nhãn:", out_df["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()

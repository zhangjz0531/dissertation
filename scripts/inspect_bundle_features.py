from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tensor_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--use_last_step", action="store_true")
    args = p.parse_args()

    bundle_path = Path(args.tensor_root) / f"{args.split}.pt"
    try:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(bundle_path, map_location="cpu")

    X = bundle["X"].numpy()              # [N, L, F]
    feat_names = list(bundle["feature_names"])

    if args.use_last_step:
        A = X[:, -1, :]
    else:
        A = X.reshape(-1, X.shape[-1])   # 所有时间步一起看

    rows = []
    for i, name in enumerate(feat_names):
        col = A[:, i]
        finite = np.isfinite(col)
        col = col[finite]
        if len(col) == 0:
            rows.append({
                "feature": name,
                "mean": None,
                "std": None,
                "nonzero_rate": None,
                "n_unique": 0,
                "clip_hi_rate": None,
                "clip_lo_rate": None,
                "p1": None,
                "p50": None,
                "p99": None,
            })
            continue

        rows.append({
            "feature": name,
            "mean": float(col.mean()),
            "std": float(col.std()),
            "nonzero_rate": float((np.abs(col) > 1e-12).mean()),
            "n_unique": int(len(np.unique(col))),
            "clip_hi_rate": float((col >= 4.999).mean()),
            "clip_lo_rate": float((col <= -4.999).mean()),
            "p1": float(np.quantile(col, 0.01)),
            "p50": float(np.quantile(col, 0.50)),
            "p99": float(np.quantile(col, 0.99)),
        })

    df = pd.DataFrame(rows).sort_values(["std", "nonzero_rate"], ascending=[True, True])
    out_csv = Path(args.tensor_root) / f"feature_activity_{args.split}.csv"
    out_json = Path(args.tensor_root) / f"feature_activity_{args.split}.json"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)

    print(df.to_string(index=False))
    print(f"\\nSaved: {out_csv}")
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
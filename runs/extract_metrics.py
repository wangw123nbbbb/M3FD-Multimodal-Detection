import re
import csv
from pathlib import Path

RUNS_DIR = Path("./")


def parse_all_metrics(data_txt_path: Path):
    text = data_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_cols = None
    all_row = None

    for i, line in enumerate(text):
        if "Class Name" in line and "Precision" in line and "mAP50-95" in line:
            header_cols = [c.strip() for c in line.split("|") if c.strip()]

            for j in range(i + 1, len(text)):
                row = text[j]

                if row.strip().startswith("+") or "-----" in row:
                    continue
                if "|" not in row:
                    continue

                cols = [c.strip() for c in row.split("|") if c.strip()]
                if not cols:
                    continue

                class_name = cols[0]
                if re.search(r"\ball\b", class_name, re.IGNORECASE) or "平均" in class_name:
                    all_row = cols
                    break
            break

    if not header_cols or not all_row:
        return None

    col_to_val = {}
    for k, name in enumerate(header_cols):
        if k < len(all_row):
            col_to_val[name] = all_row[k]

    keys = ["Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
    out = {}
    for key in keys:
        v = col_to_val.get(key, None)
        if v is None:
            out[key] = None
        else:
            v = v.replace("%", "").strip()
            try:
                out[key] = float(v)
            except ValueError:
                out[key] = None
    return out


def main():
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"RUNS_DIR does not exist: {RUNS_DIR.resolve()}")

    modes = [p for p in RUNS_DIR.iterdir() if p.is_dir()]

    results = []
    for mode_dir in sorted(modes, key=lambda x: x.name.lower()):
        data_txt = mode_dir / "test" / "data.txt"
        if not data_txt.exists():
            continue

        metrics = parse_all_metrics(data_txt)
        if metrics is None:
            print(f"[WARN] Failed to parse: {data_txt}")
            continue

        results.append((mode_dir.name, metrics))

    if not results:
        print("No valid runs/<mode>/test/data.txt found")
        return

    print("mode, Precision, Recall, F1-Score, mAP50, mAP75, mAP50-95")
    for mode, m in results:
        print(
            f"{mode}, "
            f"{m['Precision']}, {m['Recall']}, {m['F1-Score']}, "
            f"{m['mAP50']}, {m['mAP75']}, {m['mAP50-95']}"
        )

    csv_path = Path("results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mode", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        )
        for mode, m in results:
            writer.writerow([
                mode,
                m["Precision"],
                m["Recall"],
                m["F1-Score"],
                m["mAP50"],
                m["mAP75"],
                m["mAP50-95"],
            ])

    print(f"Results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()

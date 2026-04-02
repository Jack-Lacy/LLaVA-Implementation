# Creating a synthesized dataset with simple prompting, based on VG and TallyQA datasets for finetinung LLaVA for object counting tasks

import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tallyqa_json", required=True)
    parser.add_argument("--vg_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_answer", type=int, default=30)
    args = parser.parse_args()

    random.seed(args.seed)

    vg_root = Path(args.vg_root)
    with open(args.tallyqa_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    eligible = []
    missing = 0

    for ex in data:
        # Keep only Visual Genome examples
        if str(ex.get("data_source", "")).lower() != "imported_genome":
            continue

        # Keep only numeric answers
        try:
            ans = int(ex["answer"])
        except Exception:
            continue

        if ans < 0 or ans > args.max_answer:
            continue

        # Uses image path directly, not image_id
        rel_img = ex["image"].replace("/", "\\")
        full_img = vg_root / rel_img

        if not full_img.exists():
            missing += 1
            continue

        question = ex["question"].strip()
        if not question.endswith("?"):
            question += "?"

        eligible.append({
            "id": f"tallyqa_vg_{len(eligible):05d}",
            "image": ex["image"].replace("\\", "/"),
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question} Answer with a number only."
                },
                {
                    "from": "gpt",
                    "value": str(ans)
                }
            ]
        })

    print(f"Eligible samples found: {len(eligible)}")
    print(f"Missing image files skipped: {missing}")

    if len(eligible) < args.n:
        raise RuntimeError(
            f"Only found {len(eligible)} eligible samples, need {args.n}."
        )

    sample = random.sample(eligible, args.n)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(sample)} samples to: {out_json}")

if __name__ == "__main__":
    main()

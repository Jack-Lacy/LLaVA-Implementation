import argparse
import json
import random
from pathlib import Path

PROMPT_TEMPLATES = [
    "{q} Answer with a number only.",
    "Count carefully. {q} Answer with a number only.",
    "Look at the image and count the objects. {q} Answer with a number only.",
]

def image_exists(vg_root: Path, rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    return (vg_root / rel_path).exists()

def normalize_question(q: str) -> str:
    q = q.strip()
    if not q.endswith("?"):
        q += "?"
    return q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tallyqa_json", required=True)
    parser.add_argument("--vg_root", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_answer", type=int, default=50)
    parser.add_argument("--require_simple", action="store_true")
    parser.add_argument("--prompt_aug", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    vg_root = Path(args.vg_root)
    with open(args.tallyqa_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    eligible = []

    for ex in data:
        if args.require_simple:
            # Only filter if the field actually exists
            if "issimple" in ex and not bool(ex["issimple"]):
                continue

        if str(ex.get("data_source", "")).lower() != "imported_genome":
            continue

        if "image" not in ex or "question" not in ex or "answer" not in ex:
            continue

        try:
            ans = int(ex["answer"])
        except Exception:
            continue

        if ans < 0 or ans > args.max_answer:
            continue

        rel_img = ex["image"].replace("\\", "/")
        full_img = vg_root / rel_img

        if not full_img.exists():
            continue

        eligible.append({
            "image": rel_img,
            "question": normalize_question(ex["question"]),
            "answer": ans
        })

    print(f"Eligible samples found: {len(eligible)}")

    if len(eligible) < args.n:
        raise RuntimeError(
            f"Only found {len(eligible)} eligible samples, need {args.n}."
        )

    sampled = random.sample(eligible, args.n)

    final_data = []
    idx = 0

    for ex in sampled:
        prompts = [PROMPT_TEMPLATES[0]]
        if args.prompt_aug:
            prompts = PROMPT_TEMPLATES

        for template in prompts:
            prompt = template.format(q=ex["question"])
            final_data.append({
                "id": f"tallyqa_vg_{idx:06d}",
                "image": ex["image"],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": str(ex["answer"])
                    }
                ]
            })
            idx += 1

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(final_data)} examples to: {out_json}")

if __name__ == "__main__":
    main()
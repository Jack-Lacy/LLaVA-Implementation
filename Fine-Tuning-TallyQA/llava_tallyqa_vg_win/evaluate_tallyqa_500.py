import json
import os
import random
import re
import torch
from tqdm import tqdm
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

def extract_number(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def load_vg_test_questions(test_json, vg_root, max_answer=50):
    with open(test_json) as f:
        data = json.load(f)

    eligible = []

    for ex in data:

        if str(ex.get("data_source", "")).lower() != "imported_genome":
            continue

        if "image" not in ex:
            continue

        try:
            ans = int(ex["answer"])
        except:
            continue

        if ans > max_answer:
            continue

        img_path = os.path.join(vg_root, ex["image"])

        if not os.path.exists(img_path):
            continue

        eligible.append({
            "image": ex["image"],
            "question": ex["question"],
            "answer": ans
        })

    return eligible

# evaluation

def evaluate_model(model_path, model_base, questions, vg_root):

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device="cuda"
    )

    model.eval()

    correct = 0
    off_by_one = 0
    total = 0

    errors = []

    for q in tqdm(questions):

        image_path = os.path.join(vg_root, q["image"])

        image = Image.open(image_path).convert("RGB")

        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        prompt = q["question"] + " Answer with a number only."

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], "<image>\n" + prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=10
            )

        output = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        pred = extract_number(output)

        gt = q["answer"]

        if pred is None:
            pred = -1

        if pred == gt:
            correct += 1

        if abs(pred - gt) <= 1:
            off_by_one += 1

        errors.append(abs(pred - gt))

        total += 1

    accuracy = correct / total
    off1 = off_by_one / total

    mae = sum(errors) / total
    rmse = (sum([e**2 for e in errors]) / total) ** 0.5

    return accuracy, off1, mae, rmse

# main

if __name__ == "__main__":

    work = os.path.expanduser("~/llava_tallyqa_vg_win")

    vg_root = os.path.join(work, "VG")

    test_json = os.path.join(work, "TallyQA_dataset", "test.json")

    print("Loading test questions...")

    questions = load_vg_test_questions(test_json, vg_root)

    print("Eligible VG test questions:", len(questions))

    random.seed(42)

    questions = random.sample(questions, 500)

    print("Evaluating on 500 random samples")

    # Base model

    print("\nEvaluating base LLaVA...")

    base_acc, base_off1, base_mae, base_rmse = evaluate_model(
        model_path="liuhaotian/llava-v1.5-7b",
        model_base=None,
        questions=questions,
        vg_root=vg_root
    )

    # Fine-tuned model

    print("\nEvaluating fine-tuned model...")

    ft_model = os.path.join(
        work,
        "checkpoints",
        "llava-v1.5-7b-tallyqa-vg-simple-5000-lora"
    )

    ft_acc, ft_off1, ft_mae, ft_rmse = evaluate_model(
        model_path=ft_model,
        model_base="liuhaotian/llava-v1.5-7b",
        questions=questions,
        vg_root=vg_root
    )

    print("\n================ RESULTS ================\n")

    print("Base model:")
    print("Accuracy:", round(base_acc, 4))
    print("Off-by-1 accuracy:", round(base_off1, 4))
    print("MAE:", round(base_mae, 3))
    print("RMSE:", round(base_rmse, 3))

    print("\nFine-tuned model:")
    print("Accuracy:", round(ft_acc, 4))
    print("Off-by-1 accuracy:", round(ft_off1, 4))
    print("MAE:", round(ft_mae, 3))
    print("RMSE:", round(ft_rmse, 3))

    print("\nImprovement (accuracy):", round(ft_acc - base_acc, 4))
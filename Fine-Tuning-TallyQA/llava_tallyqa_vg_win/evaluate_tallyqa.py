import json
import os
import re
import torch
from tqdm import tqdm
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from transformers import TextStreamer


def extract_number(text):
    """Extract first integer from model output"""
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return None


def load_questions(path):
    with open(path) as f:
        data = json.load(f)
    return data


def evaluate(model_path, base_model, image_root, questions):

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        base_model,
        model_name="llava"
    )

    model.eval()

    correct = 0
    total = 0

    for q in tqdm(questions):

        image_path = os.path.join(image_root, q["image"])

        if not os.path.exists(image_path):
            continue

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

        with torch.no_grad():
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

        if pred is not None and pred == q["answer"]:
            correct += 1

        total += 1

    acc = correct / total

    return acc


if __name__ == "__main__":

    work = os.path.expanduser("~/llava_tallyqa_vg_win")

    vg_root = os.path.join(work, "VG")

    questions = load_questions(
        os.path.join(work, "TallyQA_dataset", "test.json")
    )

    print("Loaded questions:", len(questions))

    # Base model
    base_acc = evaluate(
        model_path="liuhaotian/llava-v1.5-7b",
        base_model=None,
        image_root=vg_root,
        questions=questions
    )

    print("\nBase LLaVA accuracy:", base_acc)

    # Fine-tuned model
    finetuned_model = os.path.join(
        work,
        "checkpoints",
        "llava-v1.5-7b-tallyqa-vg-simple-1500"
    )

    finetuned_acc = evaluate(
        model_path=finetuned_model,
        base_model="liuhaotian/llava-v1.5-7b",
        image_root=vg_root,
        questions=questions
    )

    print("\nFine-tuned accuracy:", finetuned_acc)

    improvement = finetuned_acc - base_acc

    print("\nImprovement:", improvement)
"""Export real fitting-leaf augmentation previews for human review.

This writes a PNG and a review JSON marked pending; it cannot approve visual
plausibility. Official-test images are never sampled.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from torchvision import transforms
from dataset import seed_everything
from shared_protocol import validate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("DPD_Small_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fitting, _, _ = validate_protocol(protocol, args.source)
    sample = fitting.sample(n=min(8, len(fitting)), random_state=42)
    seed_everything(42)
    transform = transforms.Compose([transforms.Resize((192, 192)), transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(18, translate=(.05, .05), scale=(.95, 1.05))])
    canvas = Image.new("RGB", (3 * 192, len(sample) * 216), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(sample.itertuples()):
        with Image.open(args.images / row.image_name) as source:
            image = source.convert("RGB")
            for col, view in enumerate((image.resize((192, 192)), transform(image), transform(image))):
                canvas.paste(view, (col * 192, row_index * 216))
        draw.text((2, row_index * 216 + 194), row.image_name, fill="black")
    args.output.mkdir(parents=True, exist_ok=False)
    canvas.save(args.output / "augmentation_preview.png")
    (args.output / "review.json").write_text(json.dumps({
        "status": "pending_human_review", "seed": 42, "images": sample.image_name.tolist(),
        "columns": ["original", "augmentation_1", "augmentation_2"],
        "review_questions": ["Are disease symptoms preserved?", "Are flips plausible for these leaves?",
                             "Does affine fill obscure symptoms?"],
        "reviewer": None, "decision": None,
    }, indent=2) + "\n", encoding="utf-8")
    print(args.output / "augmentation_preview.png")


if __name__ == "__main__":
    main()

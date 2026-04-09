"""
test_medgemma.py
================
Loads lung CT images from data/{benign, malignant, normal}/ and sends them
to a MedGemma endpoint on Vertex AI for classification inference.

The model is prompted to classify each image as: normal, benign, or malignant.

Prerequisites:
    pip install google-cloud-aiplatform Pillow

    Authenticate with:
        gcloud auth application-default login

Usage:
    python test_medgemma.py
    python test_medgemma.py --data_dir data --max_images 10 --output results.csv
"""

import sys
import csv
import json
import base64
import argparse
import time
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import aiplatform

# Load environment variables from .env file
load_dotenv()

# ──────────────────────── Vertex AI Configuration ──────────────────────
PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "")
# ───────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["benign", "malignant", "normal"]

PROMPT = (
    "You are a medical imaging expert. Analyze this lung CT scan image and "
    "classify it into exactly one of the following categories:\n"
    "- normal: no abnormalities detected\n"
    "- benign: a benign lesion or nodule is present\n"
    "- malignant: a malignant lesion or tumor is present\n\n"
    "Respond with ONLY the classification label (normal, benign, or malignant) "
    "on the first line, followed by a brief explanation on the next line."
)


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_config():
    """Check that the Vertex AI configuration has been filled in."""
    missing = []
    if not PROJECT_ID:
        missing.append("PROJECT_ID")
    if not LOCATION:
        missing.append("LOCATION")
    if not ENDPOINT_ID:
        missing.append("ENDPOINT_ID")

    if missing:
        print("❌ Vertex AI configuration is incomplete. Please fill in:")
        for field in missing:
            print(f"   - {field}")
        print(f"\n   Edit the top of {__file__} to set these values.")
        sys.exit(1)


def collect_images(data_dir: str) -> list[dict]:
    """
    Collect all JPG images from data_dir/{benign, malignant, normal}/.

    Returns a list of dicts with keys: 'path', 'true_label', 'filename'.
    """
    data_path = Path(data_dir)
    images = []

    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            print(f"[WARNING] Directory not found: {class_dir}")
            continue

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg"):
                images.append({
                    "path": str(img_file),
                    "true_label": class_name,
                    "filename": img_file.name,
                })

    return images


def call_medgemma(endpoint: aiplatform.Endpoint, image_path: str) -> dict:
    """
    Send an image to the MedGemma Vertex AI endpoint for classification.

    Returns a dict with keys: 'predicted_label', 'raw_response'.
    """
    image_b64 = encode_image_to_base64(image_path)

    # Build the instance payload for the endpoint
    # Model Garden vLLM Vision containers typically expect the standard OpenAI-style messages
    # format or the `image_ur` schema.
    instance = {
        "@requestFormat": "chatCompletions",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1
    }

    # Send prediction request
    response = endpoint.predict(instances=[instance])



    # Parse the response
    raw_text = ""
    try:
        preds = response.predictions
        
        # If predictions is a dictionary directly (OpenAI schema format)
        if isinstance(preds, dict) and "choices" in preds:
            raw_text = preds["choices"][0]["message"]["content"]
            
        # Standard Vertex AI list format
        elif isinstance(preds, list) and len(preds) > 0:
            prediction = preds[0]
            
            if isinstance(prediction, str):
                raw_text = prediction
            elif isinstance(prediction, dict):
                # Try OpenAI inner format just in case
                if "choices" in prediction:
                    raw_text = prediction["choices"][0]["message"]["content"]
                else:
                    raw_text = (
                        prediction.get("content", "")
                        or prediction.get("text", "")
                        or prediction.get("output", "")
                        or json.dumps(prediction)
                    )
            else:
                raw_text = str(prediction)
        else:
            raw_text = str(preds)
            
    except Exception as e:
        raw_text = str(response)

    predicted_label = parse_label(raw_text)

    return {
        "predicted_label": predicted_label,
        "raw_response": raw_text.strip(),
    }


def parse_label(response_text: str) -> str:
    """
    Extract the classification label from the model's response.
    Looks for 'normal', 'benign', or 'malignant' in the first line.
    """
    first_line = response_text.strip().split("\n")[0].lower().strip()

    for label in CLASS_NAMES:
        if label in first_line:
            return label

    # If no exact match in first line, search the entire response
    response_lower = response_text.lower()
    for label in CLASS_NAMES:
        if label in response_lower:
            return label

    return "unknown"


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy and per-class metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "per_class": {},
    }

    for class_name in CLASS_NAMES:
        class_results = [r for r in results if r["true_label"] == class_name]
        class_correct = sum(1 for r in class_results if r["predicted_label"] == class_name)
        class_total = len(class_results)

        recall = class_correct / class_total if class_total > 0 else 0.0

        predicted_as_class = [r for r in results if r["predicted_label"] == class_name]
        precision = (
            sum(1 for r in predicted_as_class if r["true_label"] == class_name)
            / len(predicted_as_class)
            if predicted_as_class
            else 0.0
        )

        metrics["per_class"][class_name] = {
            "total": class_total,
            "correct": class_correct,
            "recall": recall,
            "precision": precision,
        }

    return metrics


def save_results_csv(results: list[dict], output_path: str):
    """Save inference results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "true_label", "predicted_label", "correct", "raw_response"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "true_label": r["true_label"],
                "predicted_label": r["predicted_label"],
                "correct": r["true_label"] == r["predicted_label"],
                "raw_response": r["raw_response"],
            })
    print(f"💾 Results saved to: {output_path}")


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("📊  INFERENCE RESULTS")
    print("=" * 60)
    print(f"   Total images:  {metrics['total']}")
    print(f"   Correct:       {metrics['correct']}")
    print(f"   Accuracy:      {metrics['accuracy']:.2%}")
    print("-" * 60)
    print(f"   {'Class':<12} {'Total':>6} {'Correct':>8} {'Recall':>8} {'Precision':>10}")
    print("-" * 60)
    for class_name in CLASS_NAMES:
        c = metrics["per_class"][class_name]
        print(
            f"   {class_name:<12} {c['total']:>6} {c['correct']:>8} "
            f"{c['recall']:>7.2%} {c['precision']:>9.2%}"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test MedGemma inference on lung CT images via Vertex AI."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to directory containing benign/, malignant/, normal/ subdirectories.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process per class (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="medgemma_results.csv",
        help="Output CSV file path for results.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls to avoid rate limiting.",
    )
    args = parser.parse_args()

    # Validate configuration
    validate_config()

    # Initialize Vertex AI
    print(f"🔧 Initializing Vertex AI (project={PROJECT_ID}, location={LOCATION})...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    print(f"   Endpoint: {endpoint.resource_name}")

    # Collect images
    print(f"\n📂 Collecting images from: {args.data_dir}/")
    all_images = collect_images(args.data_dir)
    print(f"   Total images found: {len(all_images)}")

    for class_name in CLASS_NAMES:
        count = sum(1 for img in all_images if img["true_label"] == class_name)
        print(f"   - {class_name}: {count}")

    if not all_images:
        print("❌ No images found. Check your data directory.")
        sys.exit(1)

    # Apply max_images limit per class
    if args.max_images:
        limited_images = []
        for class_name in CLASS_NAMES:
            class_imgs = [img for img in all_images if img["true_label"] == class_name]
            limited_images.extend(class_imgs[: args.max_images])
        all_images = limited_images
        print(f"\n   After limiting: {len(all_images)} images ({args.max_images} per class max)")

    # Run inference
    print(f"\n🚀 Starting inference on {len(all_images)} images...\n")
    results = []

    for i, img_info in enumerate(all_images, 1):
        filename = img_info["filename"]
        true_label = img_info["true_label"]

        print(f"   [{i:>4}/{len(all_images)}] {true_label:<10} | {filename}", end="", flush=True)

        try:
            api_result = call_medgemma(endpoint, img_info["path"])
            predicted = api_result["predicted_label"]
            is_correct = predicted == true_label
            symbol = "✅" if is_correct else "❌"

            print(f" → {predicted:<10} {symbol}")

            results.append({
                "filename": filename,
                "true_label": true_label,
                "predicted_label": predicted,
                "raw_response": api_result["raw_response"],
            })

        except Exception as e:
            print(f" → ERROR: {e}")
            
            results.append({
                "filename": filename,
                "true_label": true_label,
                "predicted_label": "error",
                "raw_response": str(e),
            })

        # Rate limiting delay
        if args.delay > 0 and i < len(all_images):
            time.sleep(args.delay)

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Save results
    save_results_csv(results, args.output)

    print(f"\n✅ Done! Processed {len(results)} images.")


if __name__ == "__main__":
    main()

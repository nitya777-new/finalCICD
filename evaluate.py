# evaluate.py
from ultralytics import YOLO
import argparse, json, sys

def quick_eval(model_path, data_yaml):
    model = YOLO(model_path)
    try:
        res = model.val(data=data_yaml)
    except Exception as e:
        print("Evaluation failed:", e, file=sys.stderr)
        raise

    # Save summary
    try:
        metrics = {}
        for k, v in res.items() if isinstance(res, dict) else []:
            if isinstance(v, (int, float)):
                metrics[k] = v
    except Exception:
        metrics = {}

    with open("eval_summary.json", "w") as f:
        json.dump(metrics, f)

    print("Saved eval_summary.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--data", required=True)
    args = p.parse_args()
    quick_eval(args.model, args.data)

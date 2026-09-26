"""Download the public dataset and Qwen tokenizer; never download model weights."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"
PLAN = json.loads((ROOT / "configs/model_plan.json").read_text())
BASE = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/" + PLAN["dataset_revision"]
TOKENIZER_FILE = DATA / "qwen_tokenizer.json"
MODEL = PLAN["model"]
# Native limit in the published model config (text_config.max_position_embeddings).
MODEL_CONTEXT_TOKENS = 262144


def assets():
    paths = [f"wave_split/chunks/wave_persona_chunk_{i:03d}.parquet" for i in range(1, 8)]
    paths += [f"full_persona/chunks/persona_chunk_{i:03d}.parquet" for i in range(1, 8)]
    paths += ["question_catalog_and_human_response_csv/question_catalog.json"]
    result = [(path.split("/")[-1], f"{BASE}/{path}") for path in paths]
    result.append((TOKENIZER_FILE.name, f"https://huggingface.co/{MODEL}/resolve/{PLAN['model_revision']}/tokenizer.json"))
    return result


def download_one(item):
    name, url = item
    path = DATA / name
    if not path.exists():
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "TwinLab research"}), timeout=120) as response:
                    with path.with_suffix(path.suffix + ".part").open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                path.with_suffix(path.suffix + ".part").replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    print(f"Ready: {name} ({path.stat().st_size:,} bytes)", flush=True)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download_one, assets()))


if __name__ == "__main__":
    main()

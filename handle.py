import json
import os
import glob
import shutil
import subprocess
import runpod
from llama_cpp import Llama

args = json.loads(os.environ.get("LLAMA_ARGS", "{}"))
args.setdefault("n_ctx", 32768)
args.setdefault("n_gpu_layers", -1)
args.setdefault("flash_attn", True)
args.setdefault("logits_all",True)
llm = Llama(**args)
MODEL_DIR = "/models"

def get_model_stats():
    total, used, free = shutil.disk_usage(MODEL_DIR)
    gb = 1024 ** 3
    return json.dumps({
        "total_gb": round(total / gb, 2),
        "used_gb": round(used / gb, 2),
        "available_gb": round(free / gb, 2)
    })

def add_model(url: str):
    model_name = os.path.basename(url)
    dest_path = os.path.join(MODEL_DIR, model_name)
    try:
        subprocess.run(
            ["wget", "-O", dest_path, url],
            check=True,
            capture_output=True,
            text=True,
        )
        return {"status": "success", "message": f"Downloaded model {model_name}"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Download failed: {e.stderr or e.stdout}"}

def delete_model(model_name: str):
    """Delete a model file by name."""
    path = os.path.join(MODEL_DIR, model_name)
    os.remove(path)
    return {"status": "success", "message": f"Deleted {model_name}"}


def handler(event):
    """
    This function is the entry point for the serverless handler.
    """
    inp = event.get("input", {}) or {}

    if "list_models" in inp:
        return str(glob.glob(os.path.join("/models", "*.gguf")))
    
    if "stats" in inp:
        return get_model_stats()
    
    if "delete_model" in inp:
        model_name = inp.get("delete_model")
        return delete_model(model_name)
    
    if "get_model"  in inp:
        url = inp.get("get_model")
        return add_model(url)
    

    inp.setdefault("max_tokens", 4096)

    return llm.create_chat_completion(**inp)

runpod.serverless.start({"handler": handler})

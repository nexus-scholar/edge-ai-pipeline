import json
import glob
import os
import sys
from pathlib import Path

def find_latest_backbone(backbone_type):
    # backbone_type: 'mobilenetv3' or 'mobilenetv4'
    phase2_runs_dir = Path("runs/phase2_pretrain")
    pattern = str(phase2_runs_dir / f"*_{backbone_type}*" / "**" / "agri_backbone_seed*.pt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def main():
    backbones = {
        "v4": find_latest_backbone("mobilenetv4"),
        "v3": find_latest_backbone("mobilenetv3")
    }

    strategies = ["domain_guided", "random", "entropy"]
    
    for b_key, b_path in backbones.items():
        if not b_path:
            print(f"Warning: No backbone found for {b_key}")
            continue
            
        for strategy in strategies:
            template_path = Path(f"configs/phase3_wgisd_{strategy}.json")
            if not template_path.exists():
                print(f"Error: Template not found: {template_path}")
                continue

            with open(template_path, "r") as f:
                config = json.load(f)

            # Inject backbone path and name
            config["experiment_name"] = f"phase3_wgisd_{strategy}_{b_key}"
            config["model_params"]["backbone_checkpoint"] = str(Path(b_path).resolve())
            config["model_params"]["backbone_name"] = "mobilenet_v4_medium" if b_key == "v4" else "mobilenet_v3_large_320_fpn"
            
            # Save adapted config
            out_path = Path(f"configs/phase3_wgisd_{strategy}_{b_key}_adapted.json")
            with open(out_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Created: {out_path}")

if __name__ == "__main__":
    main()

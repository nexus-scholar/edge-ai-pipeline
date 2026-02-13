import json
import glob
import os
import sys
from pathlib import Path

def main():
    # 1. Find the latest Phase 2 backbone
    phase2_runs_dir = Path("runs/phase2_pretrain")
    if not phase2_runs_dir.exists():
        print(f"Error: Phase 2 runs directory not found: {phase2_runs_dir}")
        sys.exit(1)

    # Search for all .pt files recursively
    checkpoint_pattern = str(phase2_runs_dir / "**" / "agri_backbone_seed*.pt")
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)

    if not checkpoints:
        print("Error: No Phase 2 backbone checkpoints found!")
        sys.exit(1)

    # Sort by modification time (newest first)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found latest backbone: {latest_checkpoint}")

    # 2. Process each Phase 3 config template
    config_templates = [
        "configs/phase3_wgisd_domain_guided.json",
        "configs/phase3_wgisd_random.json",
        "configs/phase3_wgisd_entropy.json"
    ]

    for template_path_str in config_templates:
        template_path = Path(template_path_str)
        if not template_path.exists():
            print(f"Warning: Template config not found: {template_path}")
            continue

        with open(template_path, "r") as f:
            config = json.load(f)

        # 3. Inject the backbone path
        config["model_params"]["backbone_checkpoint"] = str(Path(latest_checkpoint).resolve())
        
        # 4. Save to an adapted config file
        name_stem = template_path.stem
        output_config_path = Path(f"configs/{name_stem}_agri_adapted.json")
        with open(output_config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Created adapted config: {output_config_path}")
    
    print("Ready for comprehensive Phase 3 execution.")

if __name__ == "__main__":
    main()

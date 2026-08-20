import os
from pathlib import Path
from PIL import Image

METHODS = ["cfg", "auto_guidance", "fk_steering", "best_of_n"]
NUM_PROMPTS = 6

def build_comparison_grid(base_dir="samples/multi_prompt", out_path="experiment_comparison.png"):
    base_path = Path(base_dir)
    img_size = (256, 256)
    
    prompt_rows = []
    
    for p_idx in range(NUM_PROMPTS):
        folder_name = f"{p_idx:02d}"
        row_images = []
        
        for method in METHODS:
            method_prompt_dir = base_path / method / folder_name
            if method_prompt_dir.exists():
                imgs = sorted(list(method_prompt_dir.glob("*.png")))
                if imgs:
                    # Prendre le premier sample de chaque méthode pour la comparaison
                    row_images.append(Image.open(imgs[0]).convert("RGB").resize(img_size))
                else:
                    row_images.append(Image.new("RGB", img_size, color="black"))
            else:
                row_images.append(Image.new("RGB", img_size, color="black"))
        
        # Assembler la ligne (1 image par méthode)
        row_width = img_size[0] * len(row_images)
        row_img = Image.new("RGB", (row_width, img_size[1]))
        for i, img in enumerate(row_images):
            row_img.paste(img, (i * img_size[0], 0))
        
        prompt_rows.append(row_img)

    # Assembler toutes les lignes verticalement
    total_w = prompt_rows[0].width
    total_h = sum(r.height for r in prompt_rows)
    final_grid = Image.new("RGB", (total_w, total_h))
    
    y = 0
    for r in prompt_rows:
        final_grid.paste(r, (0, y))
        y += r.height
        
    final_grid.save(out_path)
    print(f"Grille générée avec succès : {out_path}")

if __name__ == "__main__":
    build_comparison_grid()

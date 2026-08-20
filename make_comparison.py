import os
from pathlib import Path
from PIL import Image

def create_grid(samples_dir="samples/multi_prompt", output_path="experiment_comparison.png", max_prompts=6):
    samples_path = Path(samples_dir)
    if not samples_path.exists():
        print(f"Erreur: Le dossier {samples_dir} n'existe pas encore.")
        return

    # Récupérer les sous-dossiers de prompts (00, 01, 02...)
    prompt_folders = sorted([d for d in samples_path.iterdir() if d.is_dir()])[:max_prompts]
    
    if not prompt_folders:
        print("Aucun dossier d'images trouvé.")
        return

    grid_rows = []
    
    for folder in prompt_folders:
        # Récupérer toutes les images sample_*.png du dossier
        images = sorted(list(folder.glob("*.png")))
        if not images:
            continue
        
        pil_images = [Image.open(img_p).convert("RGB") for img_p in images]
        
        # Redimensionner si besoin (ex: 256x256 pour garder une image finale propre)
        img_w, img_h = 256, 256
        pil_images = [img.resize((img_w, img_h)) for img in pil_images]
        
        # Coller les images d'un même prompt côte à côte (ligne)
        row_width = img_w * len(pil_images)
        row_img = Image.new("RGB", (row_width, img_h))
        for idx, img in enumerate(pil_images):
            row_img.paste(img, (idx * img_w, 0))
            
        grid_rows.append(row_img)

    if not grid_rows:
        print("Aucune image valide à assembler.")
        return

    # Empiler toutes les lignes verticalement
    total_width = grid_rows[0].width
    total_height = sum(row.height for row in grid_rows)
    
    final_grid = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for row in grid_rows:
        final_grid.paste(row, (0, y_offset))
        y_offset += row.height

    # Sauvegarder dans experiment_comparison.png
    final_grid.save(output_path)
    print(f"Grille comparative créée avec succès : {output_path}")

if __name__ == "__main__":
    create_grid()

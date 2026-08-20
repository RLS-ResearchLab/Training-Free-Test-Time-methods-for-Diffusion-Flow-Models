import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SAMPLERS = ["cfg", "auto_guidance", "fk_steering", "best_of_n"]
NUM_PROMPTS = 6

def draw_header(img, text):
    """Adds a dark banner with white label text above an image panel."""
    banner_height = 36
    w, h = img.size
    
    # Create background canvas with room for header
    canvas = Image.new("RGB", (w, h + banner_height), color=(20, 20, 20))
    canvas.paste(img, (0, banner_height))
    
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    
    # Calculate text position
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(5, (w - text_w) // 2)
    y = (banner_height - text_h) // 2
    
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return canvas

def generate_grid(base_dir="samples/multi_prompt", output_path="experiment_comparison.png"):
    base_path = Path(base_dir)
    tile_size = (256, 256)
    
    prompt_rows = []
    
    for p_idx in range(NUM_PROMPTS):
        folder_str = f"{p_idx:02d}"
        row_panels = []
        
        for sampler in SAMPLERS:
            sample_dir = base_path / sampler / folder_str
            imgs = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []
            
            if imgs:
                # Load first generated sample for this sampler/prompt combination
                raw_img = Image.open(imgs[0]).convert("RGB").resize(tile_size)
                label = f"{sampler.upper()} | Prompt #{p_idx}"
                panel = draw_header(raw_img, label)
            else:
                # Placeholder if method hasn't finished running yet
                blank = Image.new("RGB", tile_size, color=(40, 40, 40))
                label = f"{sampler.upper()} (Missing)"
                panel = draw_header(blank, label)
                
            row_panels.append(panel)
        
        # Stitch all samplers for this prompt side-by-side
        row_w = sum(p.width for p in row_panels)
        row_h = row_panels[0].height
        row_img = Image.new("RGB", (row_w, row_h))
        
        offset_x = 0
        for p in row_panels:
            row_img.paste(p, (offset_x, 0))
            offset_x += p.width
            
        prompt_rows.append(row_img)

    if not prompt_rows:
        print("No image rows generated.")
        return

    # Stack all prompt rows vertically
    total_w = prompt_rows[0].width
    total_h = sum(r.height for r in prompt_rows)
    final_grid = Image.new("RGB", (total_w, total_h))
    
    offset_y = 0
    for r in prompt_rows:
        final_grid.paste(r, (0, offset_y))
        offset_y += r.height

    final_grid.save(output_path)
    print(f"Labeled comparison grid saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_grid()

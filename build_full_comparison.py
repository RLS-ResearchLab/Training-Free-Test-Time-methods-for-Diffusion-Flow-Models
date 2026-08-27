"""
Reconstruit experiment_comparison.png à partir des images RÉELLEMENT
présentes dans samples/. Annote chaque panneau avec les métriques
disponibles (clip_score, dino_lpips_vs_without) issues de results.json
(fichier unique de résultats), et note explicitement quand le FID / la
config / la paire with-without manquent, plutôt que de laisser des cases
vides sans explication.

Appelable en CLI (`python build_full_comparison.py`) ou importé :
    from build_full_comparison import build
    build()
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TILE = 300
HEADER_H = 92
SECTION_H = 46
PAD = 10
BG = (18, 18, 20)
PANEL_BG = (32, 32, 36)
OK_COLOR = (120, 200, 140)
WARN_COLOR = (230, 160, 90)
TEXT = (235, 235, 235)
SUBTEXT = (175, 175, 180)


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


F_TITLE = font(20)
F_LABEL = font(15)
F_SMALL = font(13)


def load_results_table(path="results.json"):
    data = json.loads(Path(path).read_text())
    rows = {}
    for r in data.get("ablation_experiments", []):
        key = (r["experiment"], r["method"], r["variant"], r["state"])
        rows[key] = {
            "clip_score": r["clip_score"],
            "dino_lpips_vs_without": r["dino_lpips_vs_without"],
        }
    return rows


def make_panel(img_path, title, sub_lines, ok=True):
    canvas = Image.new("RGB", (TILE, TILE + HEADER_H), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    if img_path and Path(img_path).exists():
        im = Image.open(img_path).convert("RGB").resize((TILE, TILE))
        canvas.paste(im, (0, HEADER_H))
    else:
        draw.rectangle([0, HEADER_H, TILE, HEADER_H + TILE], fill=(45, 30, 30))
        draw.text((PAD, HEADER_H + TILE // 2 - 8), "image manquante", font=F_LABEL, fill=WARN_COLOR)

    draw.text((PAD, 8), title, font=F_LABEL, fill=TEXT)
    y = 30
    for line in sub_lines:
        color = WARN_COLOR if "n/a" in line or "missing" in line.lower() else SUBTEXT
        draw.text((PAD, y), line, font=F_SMALL, fill=color)
        y += 17
    dot_color = OK_COLOR if ok else WARN_COLOR
    draw.ellipse([TILE - 18, 10, TILE - 8, 20], fill=dot_color)
    return canvas


def section_banner(width, title, note=None):
    canvas = Image.new("RGB", (width, SECTION_H + (18 if note else 0)), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 10), title, font=F_TITLE, fill=TEXT)
    if note:
        draw.text((PAD, 30), note, font=F_SMALL, fill=WARN_COLOR)
    return canvas


def row_of_panels(panels):
    w = sum(p.width for p in panels) + PAD * (len(panels) - 1)
    h = max(p.height for p in panels)
    row = Image.new("RGB", (w, h), BG)
    x = 0
    for p in panels:
        row.paste(p, (x, 0))
        x += p.width + PAD
    return row


def build(results_path="results.json", samples_dir="samples", out_path="experiment_comparison.png"):
    results = load_results_table(results_path)
    base = Path(samples_dir)
    sections = []

    full_experiments = [
        ("AutoGuidance_LatentScale", [("cfg", ""), ("auto_guidance", "latent_scale")]),
        ("AutoGuidance_FewerTimesteps", [("cfg", ""), ("auto_guidance", "latent_scale")]),
        ("AutoGuidance_WeightNoise", [("cfg", ""), ("auto_guidance", "weight_noise")]),
        ("Baseline_CFG", [("cfg", "")]),
        ("FK_Steering", [("fk_steering", "")]),
    ]

    for exp, methods in full_experiments:
        exp_dir = base / exp
        if not exp_dir.exists():
            continue
        sections.append(section_banner(4 * (TILE + PAD), f"{exp}  (config présente, résultats CLIP/DINO-LPIPS OK)"))
        panels = []
        for method, variant in methods:
            folder = exp_dir / method
            for state, fname in [("without", f"without_{method}.png"), ("with", f"with_{method}.png")]:
                title = f"{method}{'/' + variant if variant else ''} · {state}"
                row = results.get((exp, method, variant, state))
                sub = []
                if row:
                    sub.append(f"clip: {row['clip_score']}")
                    dino = row.get("dino_lpips_vs_without")
                    sub.append(f"dino-lpips: {dino}" if dino is not None else "dino-lpips: —")
                else:
                    sub.append("clip: n/a")
                sub.append("FID: n/a — no ref. set")
                panels.append(make_panel(folder / fname, title, sub, ok=True))
        result_png = exp_dir / "result.png"
        if result_png.exists():
            panels.append(make_panel(result_png, f"{exp} · result.png", ["image finale retenue", "FID: n/a — no ref. set"], ok=True))
        sections.append(row_of_panels(panels))

    # --- Expériences incomplètes / non documentées -------------------------
    known = {e for e, _ in full_experiments}
    leftover = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in known)
    empty_dirs = [d.name for d in base.iterdir() if d.is_file()]  # ex: exp_cfg_test créé comme fichier vide

    if leftover or empty_dirs:
        sections.append(section_banner(
            4 * (TILE + PAD),
            "Expériences non documentées  (pas de config YAML, pas dans results.json)",
            note="Aucune paire with/without sauvegardée -> run manuel jamais formalisé, pas de métrique CLIP/DINO-LPIPS calculée."
        ))
        panels = []
        for name in leftover:
            result_png = base / name / "result.png"
            if result_png.exists():
                panels.append(make_panel(result_png, f"{name} · result.png",
                                          ["clip: n/a — pas de config", "dino-lpips: n/a", "FID: n/a — no ref. set"], ok=False))
            else:
                panels.append(make_panel(None, name, ["dossier vide ou sans result.png"], ok=False))
        for name in empty_dirs:
            panels.append(make_panel(None, name, ["dossier vide (0 fichier)", "run jamais lancé"], ok=False))
        if panels:
            sections.append(row_of_panels(panels))

    # --- Bandeau explicatif FID ---------------------------------------------
    note_w = 4 * (TILE + PAD)
    fid_text = [
        "scripts/eval_fid_clip.py calcule un vrai FID (features InceptionV3, distance de Fréchet)",
        "mais exige --real_dir : un dossier local d'images réelles de type ImageNet-val.",
        "Le script NE télécharge PAS ImageNet automatiquement (licence requise + pas d'accès",
        "réseau à un hébergeur d'images dans cet environnement) -> sample_real_images() lève une",
        "erreur si ce dossier n'existe pas / n'a pas assez d'images. Aucun résultat FID n'a donc",
        "jamais été produit ; dino_lpips_vs_without sert de proxy de similarité en attendant.",
    ]
    note_canvas = Image.new("RGB", (note_w, 40 + 18 * len(fid_text)), (10, 10, 12))
    d = ImageDraw.Draw(note_canvas)
    d.text((PAD, 10), "Pourquoi le FID est absent PARTOUT (pas seulement pour certaines méthodes)", font=F_TITLE, fill=WARN_COLOR)
    y = 40
    for line in fid_text:
        d.text((PAD, y), line, font=F_SMALL, fill=SUBTEXT)
        y += 18
    sections.append(note_canvas)

    width = max(s.width for s in sections)
    height = sum(s.height for s in sections)
    final = Image.new("RGB", (width, height), BG)
    y = 0
    for s in sections:
        final.paste(s, (0, y))
        y += s.height

    final.save(out_path)
    print(f"[Done] Comparison grid saved to '{out_path}'")
    return out_path


if __name__ == "__main__":
    build()
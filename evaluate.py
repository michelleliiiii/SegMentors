import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet2d import UNet2D
from ssl_data import (
    get_device,
    NPYFolderDataset,
    mean_dice,
    load_baseline_style_checkpoint,
)
from visualization import save_teacher_pseudolabel_visualizations


def main():
    device = get_device()
    print("Device:", device)

    manifest_path = "ssl_split_manifest.csv"

    ckpt_path = "teacher_best.pt"   # change to "student_best.pt" when needed
    split = "val"                  # "val" or "test"

    # Optional pseudo-label visualization settings.
    # Set ENABLE_PSEUDO_LABEL_VIZ = True to save a few PNG comparisons.
    #ENABLE_PSEUDO_LABEL_VIZ = True
    #VIZ_OUTPUT_DIR = "pseudo_label_viz"
    #VIZ_NUM_SAMPLES = 54
    #VIZ_IMAGE_CHANNEL = 0
    #VIZ_EXCLUDE_BACKGROUND_IN_DICE = True

    dataset = NPYFolderDataset(
        root="data",
        split=split,
        normalize="zscore_per_channel",
        manifest_path=manifest_path,
        label_status=None,
        return_mask=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt["in_channels"]
    num_classes = ckpt["num_classes"]
    base = ckpt["base"]

    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    load_baseline_style_checkpoint(ckpt_path, model, device)
    model.eval()

    total_dice = 0.0
    nseen = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[eval {split}]"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            bs = x.size(0)
            total_dice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
            nseen += bs

    mean_val = total_dice / max(1, nseen)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {split}")
    print(f"Mean Dice (exclude background): {mean_val:.4f}")

    # ------------------------------------------------------------
    # Optional section: save teacher pseudo-label inspection images
    # ------------------------------------------------------------
    #if ENABLE_PSEUDO_LABEL_VIZ:
     #   print("\nSaving teacher pseudo-label visualizations...")
      #  save_teacher_pseudolabel_visualizations(
       #     teacher_model=model,
        #    val_loader=loader,
         #   device=device,
          #  output_dir=VIZ_OUTPUT_DIR,
           # num_samples_to_save=VIZ_NUM_SAMPLES,
            #image_channel=VIZ_IMAGE_CHANNEL,
            #exclude_background_in_dice=VIZ_EXCLUDE_BACKGROUND_IN_DICE,
        #)
        #print(f"Saved pseudo-label visualizations to: {VIZ_OUTPUT_DIR}")



if __name__ == "__main__":
    main()

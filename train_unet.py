from matplotlib import image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt
from unet2d import UNet2D
import csv


def get_device():

    '''
    Expected: Null

    Output: preferred torch device
    '''

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class NPYFolderDataset(Dataset):

    def __init__(self, root="data", split="train", normalize="zscore_per_channel"):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.normalize = normalize

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing folders: {self.image_dir} or {self.mask_dir}")
        
        labeled = None
        if split == 'train':
            labeled = set()
            with open("ssl_split_manifest.csv", "r", newline = "") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["split"] == "train" and row["label_status"] == "labeled":
                        labeled.add(row["case_id"])

        image_files = sorted(self.image_dir.glob("*.npy"))

        pairs = []
        for path in image_files:

            if not path.name.endswith("__img.npy"):
                continue

            if split == "train":
                case_id = "-".join(path.stem.split("__")[0].split("-")[:4])
                if case_id not in labeled:
                    continue

            mask_name = path.name.replace("__img.npy", "__mask.npy")
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                pairs.append((path, mask_path))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        #Load mask and image
        image_path, mask_path = self.pairs[idx]

        x = np.load(image_path)  
        y = np.load(mask_path)  

        #Enforce (C,H,W) 
        if x.shape[0] in (1, 2, 3, 4) and x.shape[0] < x.shape[-1]:
            pass
        elif x.shape[-1] in (1, 2, 3, 4):
            x = np.transpose(x, (2, 0, 1))  
        else:
            x = np.transpose(x, (2, 0, 1))

        x = torch.from_numpy(x).float()

        #Normalize per channel
        if self.normalize == "zscore_per_channel":
            mean = x.mean(dim=(1, 2), keepdim=True)
            std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        y = torch.from_numpy(y).long()
        return x, y


@torch.no_grad()
def mean_dice(pred, target, num_classes, eps=1e-6, exclude_bg=True):
    '''
    Expects: Pred and target with (B,H,W) and class labels 

    Outputs: Mean dice score across all classes
    '''

    dices = []

    for c in range(1, num_classes):

        pred_class = (pred == c).float()
        targ_class = (target == c).float()

        inter = (pred_class * targ_class).sum(dim=(1, 2))
        denom = pred_class.sum(dim=(1, 2)) + targ_class.sum(dim=(1, 2))

        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice)

    return torch.stack(dices, dim=1).mean().item() 


def soft_dice_loss(logits, target, num_classes, eps=1e-6, exclude_bg=True):
    '''
    Expects: one-hot logits and target and class labels

    Outputs: Dice loss averaged across classes
    '''
    probs = torch.softmax(logits, dim=1)
    t1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if exclude_bg:
        probs = probs[:, 1:]
        t1h = t1h[:, 1:]

    inter = (probs * t1h).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + t1h.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def mask_surface(mask):
    '''
    Expects: numpy mask

    Outputs: boundary pixels for HD95 computation
    '''
    if mask.sum() == 0:
        return mask.astype(bool)

    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    surface = mask.astype(bool) & (~eroded)
    return surface

def hd_95(pred_mask, targ_mask):
    '''
    Expects: prediction mask and target mask

    Outputs: HD95 metric 
    '''

    pred_mask = pred_mask.astype(bool)
    targ_mask = targ_mask.astype(bool)

    if pred_mask.sum() == 0 and targ_mask.sum() == 0:
        return 0.0

    if pred_mask.sum() == 0 or targ_mask.sum() == 0:
        return np.nan

    pred_surface = mask_surface(pred_mask)
    targ_surface = mask_surface(targ_mask)

    dt_targ = distance_transform_edt(~targ_surface, sampling = (0.7, 0.7))
    dt_pred = distance_transform_edt(~pred_surface, sampling = (0.7, 0.7))

    pred_to_targ = dt_targ[pred_surface]
    targ_to_pred = dt_pred[targ_surface]

    all_dists = np.concatenate([pred_to_targ, targ_to_pred])

    if all_dists.size == 0:
        return 0.0

    return float(np.percentile(all_dists, 95))

@torch.no_grad()
def mean_hd95(pred, target, num_classes):
    '''
    Expects: pred and target, and classes

    Outputs: mean HD95 across all classes and samples
    '''
    pred_np = pred.detach().cpu().numpy()
    targ_np = target.detach().cpu().numpy()

    vals = []

    for b in range(pred_np.shape[0]):
        for c in range(1, num_classes):
            pred_c = (pred_np[b] == c)
            targ_c = (targ_np[b] == c)

            hd = hd_95(pred_c, targ_c)

            if not np.isnan(hd):
                vals.append(hd)

    if len(vals) == 0:
        return 0.0

    return float(np.mean(vals))

def count_unique_patients(dataset):
    return len({
        "-".join(path.stem.split("__")[0].split("-")[:4])
        for path, _ in dataset.pairs
    })

def main():

    #Get device
    device = get_device()
    print("Device:", device)

    #Hyperparameters
    num_classes = 5          
    in_channels = 4
    base = 32
    batch_size = 8
    epochs = 50
    lr = 1e-3
    w_ce = 0.5
    w_dice = 0.5
    
    #Load datasets and create dataloaders
    train = NPYFolderDataset(root="data", split="train", normalize="zscore_per_channel")
    val = NPYFolderDataset(root="data", split="val",   normalize="zscore_per_channel")
    print("Train patients:", count_unique_patients(train))
    print("Val patients:", count_unique_patients(val))


    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=0)



    #Define model, call unet2d.py 
    model = UNet2D(in_channels=in_channels, num_classes=num_classes, base=base).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -10000.00

    for ep in range(epochs):
        model.train()
        running = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs} [train]"):

            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)

            ce = ce_criterion(logits, y)
            dl = soft_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
            loss = w_ce * ce + w_dice * dl

            loss.backward()
            optim.step()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        vloss = 0.0
        vdice = 0.0
        vhd95 = 0.0
        nseen = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {ep} [val]"):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                ce = ce_criterion(logits, y)
                dl = soft_dice_loss(logits, y, num_classes=num_classes, exclude_bg=True)
                loss = w_ce * ce + w_dice * dl

                vloss += loss.item() * x.size(0)

                pred = torch.argmax(logits, dim=1)
                bs = x.size(0)
                vdice += mean_dice(pred, y, num_classes=num_classes, exclude_bg=True) * bs
                vhd95 += mean_hd95(pred, y, num_classes=num_classes) * bs
                nseen += bs

        val_loss = vloss / len(val_loader.dataset)
        val_dice = vdice / max(1, nseen)
        val_hd95 = vhd95 / max(1, nseen)

        print(f"Epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice(excl_bg)={val_dice:.4f} val_hd95(excl_bg)={val_hd95:.4f}")

        if val_dice > best_val:
            best_val = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "base": base,
                    "best_val_dice": best_val,
                    "best_val_hd95": val_hd95,
                },
                "unet2d_best.pt",
            )
            print("unet2d_best.pt saved to directory")


if __name__ == "__main__":
    main()
from pathlib import Path
import torch
from tqdm import tqdm


def build_cross_modal_mask(batch, mask_ratio=0.5, patch_size=16, generator=None):
    """Mask modality-specific patches so the model reconstructs them from context."""
    if generator is None:
        generator = torch.Generator(device=batch.device if batch.is_cuda else "cpu")

    bsz, channels, height, width = batch.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Patch size {patch_size} must divide image size {(height, width)} "
            "for cross-modal masking."
        )

    patch_h = height // patch_size
    patch_w = width // patch_size
    num_patches = patch_h * patch_w
    num_masked = max(1, int(num_patches * mask_ratio))

    masked_input = batch.clone()
    loss_mask = torch.zeros_like(batch)

    for sample_idx in range(bsz):
        perm = torch.randperm(num_patches, generator=generator, device=batch.device)
        chosen = perm[:num_masked]
        target_modalities = torch.randint(
            low=0,
            high=channels,
            size=(num_masked,),
            generator=generator,
            device=batch.device,
        )

        for patch_index, modality in zip(chosen.tolist(), target_modalities.tolist()):
            row = patch_index // patch_w
            col = patch_index % patch_w
            h0 = row * patch_size
            h1 = h0 + patch_size
            w0 = col * patch_size
            w1 = w0 + patch_size

            masked_input[sample_idx, modality, h0:h1, w0:w1] = 0.0
            loss_mask[sample_idx, modality, h0:h1, w0:w1] = 1.0

    return masked_input, batch, loss_mask


def build_random_channel_dropout_mask(batch, mask_ratio=0.5, patch_size=16, generator=None):
    """Randomly remove pixels from one channel to form a simpler pretext task."""
    if generator is None:
        generator = torch.Generator(device=batch.device if batch.is_cuda else "cpu")

    bsz, channels, height, width = batch.shape
    masked_input = batch.clone()
    loss_mask = torch.zeros_like(batch)
    total_pixels = height * width
    masked_pixels = max(1, int(total_pixels * mask_ratio))

    for sample_idx in range(bsz):
        modality = int(torch.randint(0, channels, (1,), generator=generator, device=batch.device).item())
        perm = torch.randperm(total_pixels, generator=generator, device=batch.device)[:masked_pixels]
        rows = torch.div(perm, width, rounding_mode="floor")
        cols = perm % width
        masked_input[sample_idx, modality, rows, cols] = 0.0
        loss_mask[sample_idx, modality, rows, cols] = 1.0

    return masked_input, batch, loss_mask


def masked_mae_loss(prediction, target, loss_mask):
    """Compute MAE only over masked elements selected for reconstruction."""
    masked_elements = loss_mask.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * loss_mask).sum() / masked_elements


def run_pretraining_epoch(model, loader, optimizer, device, masker_fn, args, epoch_seed):
    """Run one optimization epoch for self-supervised reconstruction training."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(tqdm(loader, desc="Pretrain [train]")):
        images = batch["image"].to(device)
        generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
        generator.manual_seed(epoch_seed + step)

        masked_inputs, targets, loss_mask = masker_fn(
            images,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        recon = model(masked_inputs)
        loss = masked_mae_loss(recon, targets, loss_mask)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(1, sample_count)


@torch.no_grad()
def run_pretraining_validation(model, loader, device, masker_fn, args, epoch_seed):
    """Evaluate the reconstruction objective for one validation epoch."""
    model.eval()
    total_loss = 0.0
    sample_count = 0

    for step, batch in enumerate(tqdm(loader, desc="Pretrain [val]")):
        images = batch["image"].to(device)
        generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
        generator.manual_seed(epoch_seed + step)

        masked_inputs, targets, loss_mask = masker_fn(
            images,
            mask_ratio=args.mask_ratio,
            patch_size=args.patch_size,
            generator=generator,
        )

        recon = model(masked_inputs)
        loss = masked_mae_loss(recon, targets, loss_mask)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / max(1, sample_count)

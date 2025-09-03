import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_masks(clips, mask_enc, mask_pred, save_path="vis.png"):
    """
    Visualize one frame of a video with mask_enc and mask_pred overlays.

    Args:
        clips: tensor (3, T, H, W)
        mask_enc: tensor (B, M)
        mask_pred: tensor (B, M)
        batch_idx: int, which batch element to visualize
        save_path: str, path to save visualization
    """
    video = clips # (3, T, H, W)
    enc_idx = mask_enc.cpu().numpy()
    pred_idx = mask_pred.cpu().numpy()

    # each time step has 16 patches (4x4 grid)
    patches_per_frame = 196

    # Step 1: choose timeframe from mask_pred
    t_f = pred_idx.min() // patches_per_frame
    frame = video[:, t_f].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # Step 2: pred patches (relative to this frame)
    pred_patches = [idx - t_f * patches_per_frame for idx in pred_idx if idx // patches_per_frame == t_f]

    # Step 3: enc patches (keep only for this frame)
    print(f"pred_idx: {pred_idx}")
    print(f"t_f: {t_f}")
    print(f"enc_idx: {enc_idx}")
    enc_patches = [idx - t * patches_per_frame for idx in enc_idx if (t := idx // patches_per_frame) <= t_f]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame)

    def draw_patches(ax, patch_list, edgecolor, title):
        ax.imshow(frame)
        for idx in patch_list:
            row, col = divmod(idx, 14)  # 14x14 grid
            rect = patches.Rectangle(
                (col*4, row*4), 4, 4,
                linewidth=2, edgecolor=edgecolor, facecolor="none"
            )
            ax.add_patch(rect)
        ax.set_title(title)
        ax.axis("off")

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    draw_patches(axs[0], pred_patches, "red", "mask_pred")
    draw_patches(axs[1], enc_patches, "blue", "mask_enc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
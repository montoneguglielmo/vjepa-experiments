import os
import yaml
import torch
import logging

from src.models.attentive_pooler import AttentiveClassifier
from src.utils.logging import AverageMeter
from src.utils.distributed import init_distributed, AllReduce

# Reuse functions you already have
from evals.image_classification_frozen.eval import make_dataloader, init_model  

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def evaluate(encoder, classifier, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()

    misclassified_files = []

    classifier.eval()
    with torch.no_grad():
        for itr, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = encoder(imgs)
            outputs = classifier(outputs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            top1_acc = 100. * preds.eq(labels).sum() / len(imgs)
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)

            # Get filenames of misclassified samples
            if hasattr(loader.dataset, 'samples'):  # ImageFolder
                batch_start = itr * loader.batch_size
                batch_end = batch_start + len(imgs)
                batch_files = [loader.dataset.samples[i][0] for i in range(batch_start, batch_end)]
                for fname, label, pred in zip(batch_files, labels.cpu().tolist(), preds.cpu().tolist()):
                    if label != pred:
                        misclassified_files.append(fname)

            if itr % 20 == 0:
                logger.info('[%5d] running test acc: %.3f%% (loss: %.3f)' %
                            (itr, top1_meter.avg, loss))

    return top1_meter.avg, misclassified_files


def main(config_file, classifier_ckpt):
    # Load config
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None

    world_size, rank = init_distributed()

    # Encoder (frozen pretrained)
    encoder = init_model(
        device=device,
        pretrained=os.path.join(cfg['pretrain']['folder'], cfg['pretrain']['checkpoint']),
        model_name=cfg['pretrain']['model_name'],
        patch_size=cfg['pretrain']['patch_size'],
        crop_size=cfg['data']['resolution'],
        frames_per_clip=cfg['pretrain'].get('frames_per_clip', 1),
        tubelet_size=cfg['pretrain'].get('tubelet_size', 2),
        use_sdpa=cfg['pretrain'].get('use_sdpa', False),
        use_SiLU=cfg['pretrain'].get('use_silu', False),
        tight_SiLU=cfg['pretrain'].get('tight_silu', True),
        uniform_power=cfg['pretrain'].get('uniform_power', False),
        checkpoint_key=cfg['pretrain'].get('checkpoint_key', 'target_encoder')
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=cfg['data']['num_classes']
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(classifier_ckpt, map_location=device)
    state_dict = checkpoint['classifier']
    # Strip "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    classifier.load_state_dict(new_state_dict, strict=True)
    logger.info(f"Loaded classifier from {classifier_ckpt}, epoch {checkpoint['epoch']}")

    # Test dataloader
    test_loader = make_dataloader(
        dataset_name=cfg['data']['dataset_name'],
        root_path=cfg['data']['root_path'],
        resolution=cfg['data']['resolution'],
        image_folder=cfg['data']['image_folder'],
        batch_size=cfg['optimization']['batch_size'],
        world_size=world_size,
        rank=rank,
        training=False,
        normalize_only=cfg['data'].get('normalize_only', False),
        split='test'
    )

    # Evaluate
    test_acc, misclassified_files = evaluate(encoder, classifier, test_loader, device)
    logger.info(f"Final Test Accuracy: {test_acc:.3f}%")
    logger.info(f"Number of misclassified images: {len(misclassified_files)}")

    # Optional: save misclassified filenames
    logger.info(f"Saving misclassified files to misclassified_files.txt")
    with open("misclassified_files.txt", "w") as f:
        for fname in misclassified_files:
            f.write(fname + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate frozen encoder + classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to classifier checkpoint (.pth.tar)")
    args = parser.parse_args()

    main(args.config, args.ckpt)
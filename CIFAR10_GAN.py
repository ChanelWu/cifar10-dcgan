"""Assignment 3: Train a GAN on CIFAR-10.

Environment: Python 3.11.
Packages: torch, torchvision, numpy, matplotlib, pillow, torch-fidelity.
Install: python -m pip install torch torchvision numpy matplotlib pillow torch-fidelity==0.4.0
Run: python project3_GAN.py

The script downloads CIFAR-10, trains for 20 epochs, and evaluates the results.

"""

# 1. Imports and project location
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Training settings
SEED = 42
LATENT_DIM = 100
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
ADAM_BETAS = (0.5, 0.999)


# 2. Load, normalize, inspect, and preview CIFAR-10


def prepare_data():
    data_dir = PROJECT_DIR / "data"
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Convert RGB pixels from [0, 255] to [0, 1], then to [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    # Use all ten classes. The test set stays separate from training.
    train_counts = torch.bincount(torch.tensor(trainset.targets), minlength=10)
    test_counts = torch.bincount(torch.tensor(testset.targets), minlength=10)
    assert len(trainset) == 50000 and len(testset) == 10000
    assert torch.all(train_counts == 5000)
    assert torch.all(test_counts == 1000)

    # Show five real training examples per class in dataset order.
    fig, axes = plt.subplots(10, 5, figsize=(7, 12))
    for class_id, class_name in enumerate(trainset.classes):
        indices = [i for i, label in enumerate(trainset.targets)
                   if label == class_id][:5]
        for column, index in enumerate(indices):
            normalized_image, _ = trainset[index]
            display_image = normalized_image * 0.5 + 0.5
            ax = axes[class_id, column]
            ax.imshow(display_image.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(class_name, rotation=0, ha="right", va="center", labelpad=12)
    fig.suptitle("CIFAR-10: real training images\nFive examples from each of the ten classes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    preview_path = output_dir / "real_training_samples.png"
    fig.savefig(preview_path, dpi=150, facecolor="white")
    plt.close(fig)

    print(f"Training images: {len(trainset)}, test images: {len(testset)}")
    print("Training images per class:", dict(zip(trainset.classes, train_counts.tolist())))
    print(f"Saved preview: {preview_path}")


# 3. Generator and discriminator architecture


class Generator(nn.Module):
    """Turn noise of shape (batch, 100, 1, 1) into RGB images in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 100 x 1 x 1 -> 256 x 4 x 4
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256 x 4 x 4 -> 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 16 x 16 -> 3 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    """Return one raw real-or-fake score (logit) for each RGB image."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 3 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            # No Sigmoid: BCEWithLogitsLoss applies it internally.
        )

    def forward(self, images):
        return self.network(images).flatten()


# 4. Weight initialization and progress-image helper


def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.zeros_(layer.bias)


def save_samples(generator, noise, path):
    # Evaluation mode keeps preview generation from changing batch statistics.
    generator.eval()
    with torch.no_grad():
        samples = generator(noise).cpu()
    save_image(samples, path, nrow=8, normalize=True, value_range=(-1, 1))
    generator.train()


# 5. Training settings, alternating updates, loss logs, and checkpoints


def train_gan():
    seed = SEED
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    epochs = EPOCHS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root=str(PROJECT_DIR / "data"), train=True,
        download=False, transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )

    # Each run gets its own folder; existing results are never overwritten.
    run_name = "training_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = PROJECT_DIR / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    generator = Generator()
    discriminator = Discriminator()
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    generator.to(device)
    discriminator.to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    criterion = nn.BCEWithLogitsLoss()

    # Separate random streams keep preview choices from changing training noise.
    noise_rng = torch.Generator().manual_seed(seed + 1)
    fixed_noise = torch.randn(
        64, LATENT_DIM, 1, 1, generator=torch.Generator().manual_seed(seed + 2)
    ).to(device)
    save_samples(generator, fixed_noise, run_dir / "samples_epoch_000.png")

    print(f"Device: {device} | Output: {run_dir}", flush=True)
    history_g = []
    history_d = []
    global_step = 0
    with (run_dir / "losses.csv").open("w", newline="") as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(["epoch", "batch", "iteration", "discriminator_loss", "generator_loss"])
        for epoch in range(1, epochs + 1):
            for batch, (real_images, labels) in enumerate(loader, start=1):
                real_images = real_images.to(device)
                noise = torch.randn(real_images.size(0), LATENT_DIM, 1, 1, generator=noise_rng).to(device)
                fake_images = generator(noise)

                # 1. Teach D to recognize real images and reject detached fakes.
                discriminator.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)
                real_scores = discriminator(real_images)
                fake_scores = discriminator(fake_images.detach())
                loss_d = criterion(real_scores, torch.ones_like(real_scores))
                loss_d = loss_d + criterion(fake_scores, torch.zeros_like(fake_scores))
                if not torch.isfinite(loss_d).item():
                    raise RuntimeError("Non-finite discriminator loss; run stopped.")
                loss_d.backward()
                optimizer_d.step()

                # 2. Teach G to make images D will classify as real.
                # D still passes gradients to G, but D's weights are frozen.
                discriminator.requires_grad_(False)
                optimizer_g.zero_grad(set_to_none=True)
                generated_scores = discriminator(fake_images)
                loss_g = criterion(generated_scores, torch.ones_like(generated_scores))
                if not torch.isfinite(loss_g).item():
                    raise RuntimeError("Non-finite generator loss; run stopped.")
                loss_g.backward()
                optimizer_g.step()
                discriminator.requires_grad_(True)

                global_step += 1
                value_d = loss_d.item()
                value_g = loss_g.item()
                writer.writerow([epoch, batch, global_step, value_d, value_g])
                loss_file.flush()
                history_d.append(value_d)
                history_g.append(value_g)

            save_samples(generator, fixed_noise, run_dir / f"samples_epoch_{epoch:03d}.png")
            print(f"Epoch {epoch}/{epochs} complete", flush=True)

    # Save the trained networks once, after the final epoch.
    torch.save({
        "epoch": epochs, "iteration": global_step,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "fixed_noise": fixed_noise.cpu(),
    }, run_dir / "checkpoint.pt")

    fig, ax = plt.subplots(figsize=(9, 4))
    iterations = range(1, global_step + 1)
    ax.plot(iterations, history_d, label="Discriminator", linewidth=0.8)
    ax.plot(iterations, history_g, label="Generator", linewidth=0.8)
    ax.set(xlabel="Training iteration", ylabel="Binary cross-entropy loss",
           title="GAN training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Export the final grid.
    shutil.copy2(run_dir / f"samples_epoch_{epochs:03d}.png", run_dir / "fake_image.png")
    print(f"Submission image: {run_dir / 'fake_image.png'}", flush=True)
    return run_dir


# 6. Generate the submission image from a saved checkpoint


def generate_images(run_dir):
    """Reload a saved generator and export its fixed-noise grid, without training."""
    run_dir = run_dir.resolve()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    with torch.no_grad():
        images = generator(checkpoint["fixed_noise"].to(device)).cpu()

    export_dir = run_dir / ("export_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    export_dir.mkdir(exist_ok=False)
    save_image(images, export_dir / "fake_image.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    print(f"Saved image: {export_dir / 'fake_image.png'}")


# 7. Inception features, FID, and KID


def extract_features(extractor, images, device, label):
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            features = extractor(images[start:start + 32].to(device))[0].cpu()
            assert torch.isfinite(features).all()
            batches.append(features)
            if start % 320 == 0:
                print(f'{label}: {min(start + 32, len(images))}/{len(images)} features', flush=True)
    return torch.cat(batches)


def compare_features(first, second):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric

    fid = fid_statistics_to_metric(
        fid_features_to_statistics(first), fid_features_to_statistics(second), verbose=False
    )
    kid = kid_features_to_metric(
        first, second, kid_subsets=100, kid_subset_size=1000,
        rng_seed=20261028, verbose=False,
    )
    return {**fid, **kid}


# 8. Compare generated images with held-out test images


def evaluate_gan(run_dir):
    # This extra package is needed only for the evaluation stage.
    from torch_fidelity import FeatureExtractorInceptionV3

    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / 'checkpoint.pt'
    output_dir = run_dir / ('evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(exist_ok=False)
    print(f'Evaluation folder: {output_dir}', flush=True)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Fresh samples use a different noise stream from training and fixed previews.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    rng = torch.Generator().manual_seed(20261028)
    noise = torch.randn(2000, 100, 1, 1, generator=rng)
    generated_batches = []
    with torch.inference_mode():
        for start in range(0, 2000, 64):
            images = generator(noise[start:start + 64].to(device)).cpu()
            pixels = ((images + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
            generated_batches.append(pixels)
    generated = torch.cat(generated_batches)

    testset = datasets.CIFAR10(str(PROJECT_DIR / 'data'), train=False, download=False)
    labels = np.array(testset.targets)
    selection_rng = np.random.default_rng(20261028)
    reference_indices = []
    control_indices = []
    for class_id in range(10):
        indices = selection_rng.permutation(np.flatnonzero(labels == class_id))
        reference_indices.extend(indices[:200].tolist())
        control_indices.extend(indices[200:400].tolist())
    assert not set(reference_indices).intersection(control_indices)
    reference = torch.from_numpy(testset.data[reference_indices].copy()).permute(0, 3, 1, 2)
    control = torch.from_numpy(testset.data[control_indices].copy()).permute(0, 3, 1, 2)

    # Standard Inception weights; the package handles resizing and normalization.
    torch.hub.set_dir(str(PROJECT_DIR / 'data' / 'evaluation_weights'))
    extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).eval()
    extractor.to(device)

    real_features = extract_features(extractor, reference, device, 'Real reference')
    fake_features = extract_features(extractor, generated, device, 'Generated')
    control_features = extract_features(extractor, control, device, 'Real control')
    generated_metrics = compare_features(fake_features, real_features)
    real_control_metrics = compare_features(control_features, real_features)
    metrics = {
        'generated_vs_real': generated_metrics,
        'real_control_vs_real': real_control_metrics,
        'sample_count_per_group': 2000,
        'real_images_per_class_per_group': 200,
        'reference_split': 'Two disjoint, balanced CIFAR-10 test subsets',
        'seed': 20261028,
        'kid_subsets': 100, 'kid_subset_size': 1000,
        'feature_extractor': 'inception-v3-compat, 2048 features',
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


# 9. Run the project

if __name__ == "__main__":
    prepare_data()
    run_dir = train_gan()
    evaluate_gan(run_dir)

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import OsteoGenDataset
from model import GeneratorUNet, Discriminator

def train():
    parser = argparse.ArgumentParser(description="OsteoGen Pix2Pix Training")
    parser.add_argument("--data_root", type=str, default="dataset/processed", help="Path to processed data")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lambda_pixel", type=float, default=100.0, help="L1 loss weight")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Save directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = OsteoGenDataset(root_dir=args.data_root, phase="train")
    if len(dataset) == 0:
        print("Error: No training pairs found! Ensure dataset/processed/train contains A and B folders.")
        return
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Networks
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Losses & Optimizers
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Targets for Discriminator (PatchGAN output grid size matches real_B)
            fake_B = generator(real_A)
            disc_out_shape = discriminator(real_A, fake_B).shape
            valid = torch.ones(disc_out_shape, device=device)
            fake = torch.zeros(disc_out_shape, device=device)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            pred_fake = discriminator(real_A, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake_detach = discriminator(real_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake_detach, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # Save sample output grid after each epoch
        img_sample = torch.cat((real_A[0], fake_B[0], real_B[0]), 2)
        save_image(img_sample, f"samples/epoch_{epoch+1}.png", normalize=True)

    # Save final model weights
    torch.save(generator.state_dict(), os.path.join(args.output_dir, "generator_latest.pth"))
    print("Training dry run complete. Weights saved.")

if __name__ == "__main__":
    train()


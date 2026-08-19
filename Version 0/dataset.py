import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OsteoGenDataset(Dataset):
    def __init__(self, root_dir, phase='train', img_size=256):
        self.root_dir = os.path.join(root_dir, phase)
        self.dir_A = os.path.join(self.root_dir, 'A')
        self.dir_B = os.path.join(self.root_dir, 'B')
        
        # Get matching image filenames
        if os.path.exists(self.dir_A) and os.path.exists(self.dir_B):
            self.filenames = sorted([
                f for f in os.listdir(self.dir_A) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            self.filenames = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_A_path = os.path.join(self.dir_A, filename)
        img_B_path = os.path.join(self.dir_B, filename)

        img_A = Image.open(img_A_path).convert('RGB')
        img_B = Image.open(img_B_path).convert('RGB')

        return self.transform(img_A), self.transform(img_B)

if __name__ == "__main__":
    # Self-test block
    print("Testing Dataset Loader...")
    dataset = OsteoGenDataset(root_dir="dataset/processed", phase="train")
    print(f"Found {len(dataset)} paired samples in train split.")


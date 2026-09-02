import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OsteoDataset(Dataset):
    def __init__(self, data_dir: str = "data/processed", transform=None):
        self.data_dir = Path(data_dir)
        self.x_dir = self.data_dir / "input_x" 
        self.y_dir = self.data_dir / "target_y"
        # 1. Stampa il percorso reale in cui Python sta curiosando
        print(f"DEBUG - Percorso assoluto: {self.x_dir.resolve()}")
        self.filenames = sorted([f.name for f in self.x_dir.glob("*.png")])
        # 2. Conferma quanti file vengono effettivamente visti
        print(f"DEBUG - File estratti: {len(self.filenames)}")
        self.transform = transform or self.get_default_transforms()

    def get_default_transforms(self):
        # Data Augmentation stocastica applicata sincronizzata (Y.2.4)
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Scalato tra [-1, 1] per modelli generativi
            ToTensorV2()
        ], additional_targets={'target': 'image'})

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        x_path = str(self.x_dir / filename)
        y_path = str(self.y_dir / filename)
        
        image_x = cv2.imread(x_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        
        image_y = cv2.imread(y_path)
        image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)
        
        # Applicazione sincronizzata della trasfomazione su X e Y
        augmented = self.transform(image=image_x, target=image_y)
        
        tensor_x = augmented['image']
        tensor_y = augmented['target']
        
        return tensor_x, tensor_y

# Test rapido di caricamento del DataLoader
if __name__ == "__main__":
    dataset = OsteoDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_x, batch_y in dataloader:
        print(f"Batch X Shape: {batch_x.shape}") # Output atteso: [4, 3, 512, 512]
        print(f"Batch Y Shape: {batch_y.shape}") # Output atteso: [4, 3, 512, 512]
        break
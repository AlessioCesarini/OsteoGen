import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class OsteoDataset(Dataset):
    def __init__(self, data_dir: str = "data/processed", csv_path: str = "metadata.csv", transform=None):
        self.data_dir = Path(data_dir)
        self.x_dir = self.data_dir / "input_x" 
        self.y_dir = self.data_dir / "target_y"
        self.filenames = sorted([f.name for f in self.x_dir.glob("*.png")])
        
        # Caricamento metadata prompt
        self.prompts = {}
        csv_file = Path(csv_path)
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # Mappa il nome del file con la colonna prompt (o caption)
            col_name = 'prompt' if 'prompt' in df.columns else df.columns[1]
            for _, row in df.iterrows():
                self.prompts[str(row[df.columns[0]])] = str(row[col_name])

        self.transform = transform or self.get_default_transforms()

    def get_default_transforms(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'target': 'image'})

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x_path = str(self.x_dir / filename)
        y_path = str(self.y_dir / filename)
        
        image_x = cv2.cvtColor(cv2.imread(x_path), cv2.COLOR_BGR2RGB)
        image_y = cv2.cvtColor(cv2.imread(y_path), cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image_x, target=image_y)
        
        # Gestione Prompt
        animal_name = filename.replace(".png", "").replace("_", " ")
        prompt = self.prompts.get(
            filename, 
            f"a photorealistic photo of a {animal_name}, detailed fur, muscle structure, natural lighting, highly detailed"
        )
        
        return augmented['image'], augmented['target'], prompt


# Blocco di test autonomo per verificare il caricamento
if __name__ == "__main__":
    print("--- Test del Dataset ---")
    try:
        dataset = OsteoDataset(data_dir="data/processed", csv_path="metadata.csv")
        print(f"Dataset caricato con successo. Numero di immagini trovate: {len(dataset)}")
        
        if len(dataset) > 0:
            x, y, prompt = dataset[0]
            print(f"Shape Input (Scheletro): {x.shape}")
            print(f"Shape Target (Animale):  {y.shape}")
            print(f"Esempio Prompt estratto: '{prompt}'")
            print("--- Test completato con successo! ---")
        else:
            print("ATTENZIONE: Nessun file trovato in data/processed/input_x!")
    except Exception as e:
        print(f"ERRORE durante il test del dataset: {e}")
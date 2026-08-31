import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.dataset import OsteoDataset  # Importo il file dataset.py

# ==========================================
# Y.3.1 Setup Architetturale: Baseline U-Net
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder (Comprime l'immagine ed estrae feature geometriche)
        self.enc1 = self.conv_block(3, 64) #Da RGB a Feature di Bassissimo Livello
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256) #Feature di altissimo Livello con 512 canali che rappresentano concetti astratti

        #Ad ogni Max Pooling, la rete fa uno "zoom out". Questo permette ai blocchi più profondi di "vedere" l'intero scheletro nella sua interezza e capirne le macro-strutture invece di concentrarsi su un singolo dente.
        self.pool = nn.MaxPool2d(2) #Prende un blocco 2x2 di pixel e trattiene solo il valore massimo, dimezzando di fatto l'altezza e la larghezza del tensore.
        #ESEMPIO: Ho trovato la feature 'Bordo del Femore'. Non mi importa se si trovava in alto a sinistra o in basso a destra nel mio quadratino 2X2. L'ho trovato, e lo passo avanti

        
        # Bottleneck (Spazio Latente rudimentale)
        self.bottleneck = self.conv_block(256, 512) 
        
        # Decoder (Decomprime e ricostruisce l'immagine)
        #Da Altissimo Livello si inizia ad andare al microscopico
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) #Lavora sulla Geometria. Il suo unico scopo è ingrandire la "tela" (raddoppiare altezza e larghezza).
        self.dec3 = self.conv_block(512, 256) # 512 perché concateniamo le skip connection
        #Prende i 512 canali appena incollati e li fonde. Usa i suoi pesi convoluzionali per rimescolare le informazioni profonde con i bordi precisi della skip connection, sintetizzando un nuovo output pulito a 256 canali. È il pittore che prende la proiezione grande e sfocata sul muro, usa i bordi netti della skip connection come ricalco, e dipinge i dettagli fini, restituendo un'immagine nitida e coerente.
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh() # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)). Questa funzione ha preso le foto target (che avevano pixel da 0 a 255) e le ha scalate tra -1.0 e 1.0. Coerenza Matematica
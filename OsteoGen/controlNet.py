import os
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from src.dataset import OsteoDataset 
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_architecture(device):
    wandb.init(
        project="OsteoGen-ControlNet", 
        name="SD_1.5_Pretrained_ControlNet_Run",
        config={"learning_rate": 2e-5, "batch_size": 4, "epochs": 300}
    )   
    
    print("Caricamento del Foundation Model e ControlNet Pre-addestrato...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device) 
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Caricamento pesi pre-addestrati per le strutture a linee
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(device)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train() 

    # Inizializzazione pipeline per la validazione veloce in VRAM
    val_pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, controlnet=controlnet, scheduler=scheduler,
        safety_checker=None, feature_extractor=None
    ).to(device)
    val_pipe.set_progress_bar_config(disable=True)
    
    return {
        "vae": vae, "text_encoder": text_encoder, "unet": unet,
        "controlnet": controlnet, "scheduler": scheduler,
        "tokenizer": tokenizer, "val_pipe": val_pipe,
        "config": wandb.config
    }

def train_model(components, device):
    dataset = OsteoDataset(data_dir="data/processed")
    dataloader = DataLoader(dataset, batch_size=components["config"].batch_size, shuffle=True)
    optimizer = AdamW(components["controlnet"].parameters(), lr=components["config"].learning_rate)
    
    save_dir = "training_dir_controlNet"
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0

    print("Inizio Addestramento Veloce...")
    for epoch in range(components["config"].epochs):
        components["controlnet"].train()
        progress_bar = tqdm(dataloader, desc=f"Epoca {epoch+1}")
        
        for batch_idx, (ossa, animali_veri, prompts) in enumerate(progress_bar):
            global_step += 1
            ossa, animali_veri = ossa.to(device), animali_veri.to(device)
            
            # Latenti target
            latents = components["vae"].encode(animali_veri).latent_dist.sample() * components["vae"].config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, components["scheduler"].config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = components["scheduler"].add_noise(latents, noise, timesteps)
            
            # Text Embeddings Reali
            text_inputs = components["tokenizer"](
                list(prompts), padding="max_length", max_length=components["tokenizer"].model_max_length, return_tensors="pt"
            ).to(device)
            encoder_hidden_states = components["text_encoder"](text_inputs.input_ids)[0]
            
            # Conditioning & Forward
            ossa_cond = (ossa / 2 + 0.5).clamp(0, 1)
            down_res, mid_res = components["controlnet"](
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=ossa_cond, return_dict=False
            )
            
            noise_pred = components["unet"](
                noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res
            ).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log({"loss": loss.item(), "global_step": global_step})
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # VALIDAZIONE VELOCE OGNI 50 STEP (Invece che ogni 10 epoche)
            if global_step % 50 == 0 or global_step == 1:
                components["controlnet"].eval()
                with torch.no_grad():
                    ossa_val_cond = (ossa[0] / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                    prompt_val = prompts[0]
                    
                    val_image = components["val_pipe"](
                        prompt=prompt_val, image=ossa_val_cond, num_inference_steps=20
                    ).images[0]
                    
                    def tensor_to_pil(t):
                        t = (t / 2 + 0.5).clamp(0, 1).cpu()
                        return TF.to_pil_image(t)
                        
                    input_pil = tensor_to_pil(ossa[0])
                    target_pil = tensor_to_pil(animali_veri[0])
                    
                    w1, w2, w3 = input_pil.width, val_image.width, target_pil.width
                    h_max = max(input_pil.height, val_image.height, target_pil.height)
                    
                    combined_image = Image.new('RGB', (w1 + w2 + w3, h_max))
                    combined_image.paste(input_pil, (0, 0))
                    combined_image.paste(val_image, (w1, 0))
                    combined_image.paste(target_pil, (w1 + w2, 0))
                    
                    render_path = f"{save_dir}/step_{global_step}.png"
                    combined_image.save(render_path)
                    print(f"\n[STEP {global_step}] Salvato render di controllo in: {render_path}")
                    wandb.log({"Generazione_Visiva": wandb.Image(combined_image, caption=f"Step {global_step} - Prompt: {prompt_val}")})

                components["controlnet"].train()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_components = load_architecture(device)
    train_model(arch_components, device)
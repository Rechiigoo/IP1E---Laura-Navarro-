import sys
sys.path.insert(0,"src")
from dataset import build_dataloaders
from model import AudioDeepfakeDetector
from trainer import Trainer

#---Config---
REAL_DIR = "data/real"
SYNTHETIC_DIR = "data/synth"
BATCH_SIZE = 8
PHASE1_EPOCHS = 2
PHASE2_EPOCHS = 2

#---Build data---
train_loader, val_loader, test_loader = build_dataloaders(
real_dir = REAL_DIR,
synthetic_dir= SYNTHETIC_DIR,
batch_size=BATCH_SIZE,
)

#---Build model ---
model = AudioDeepfakeDetector(
    encoder_name="wav2vec2-base",
    freeze_encoder=True,
    )

# --- Train ---
trainer = Trainer(model,train_loader,val_loader,checkpoint_dir = "models")
trainer.train( 
    phase1_epochs = PHASE1_EPOCHS, 
    phase2_epochs=PHASE2_EPOCHS,
    )

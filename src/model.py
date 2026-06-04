import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel


ENCODER_CONFIGS = {
    "wav2vec2-base": {
        "model_id": "facebook/wav2vec2-base",
        "hidden_size": 768,
        "loader": Wav2Vec2Model,
    },
    "wav2vec2-large": {
        "model_id": "facebook/wav2vec2-large",
        "hidden_size": 1024,
        "loader": Wav2Vec2Model,
    },
    "hubert-base": {
        "model_id": "facebook/hubert-base-ls960",
        "hidden_size": 768,
        "loader": HubertModel,
    },
}


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AudioDeepfakeDetector(nn.Module):
    def __init__(
        self,
        encoder_name: str = "wav2vec2-base",
        freeze_encoder: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from: {list(ENCODER_CONFIGS)}")

        config = ENCODER_CONFIGS[encoder_name]
        self.encoder_name = encoder_name
        self.hidden_size = config["hidden_size"]

        print(f"Loading pretrained encoder: {config['model_id']} ...")
        self.encoder = config["loader"].from_pretrained(config["model_id"])

        self.classifier = ClassificationHead(
            input_dim=self.hidden_size,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    def unfreeze_top_layers(self, n_layers: int = 4):
        """Gradually unfreeze the top N transformer layers."""
        unfrozen = 0
        try:
            layers = self.encoder.encoder.layers
        except AttributeError:
            print("Could not access encoder layers — skipping unfreeze.")
            return

        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
            unfrozen += 1

        # Also unfreeze the final layer norm
        if hasattr(self.encoder, "layer_norm"):
            for param in self.encoder.layer_norm.parameters():
                param.requires_grad = True

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze top {unfrozen} encoder layers. Trainable params: {total_trainable:,}")

    def unfreeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"All encoder layers unfrozen. Trainable params: {total:,}")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_values, output_hidden_states=False)
        # Mean-pool the sequence dimension
        hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)

    def predict_proba(self, input_values: torch.Tensor) -> torch.Tensor:
        """Returns probability of being AI-generated."""
        with torch.no_grad():
            logits = self.forward(input_values)
            return torch.sigmoid(logits)

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """Returns param groups with different learning rates."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = list(self.classifier.parameters())
        groups = [{"params": head_params, "lr": head_lr}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr})
        return groups
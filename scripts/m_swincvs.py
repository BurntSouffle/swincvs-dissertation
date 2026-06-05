import torch
import torch.nn as nn

class SwinCVSModel(nn.Module):
    def __init__(self, swinv2_model, config, num_classes=3):
        super(SwinCVSModel, self).__init__()
        self.swinv2_model = swinv2_model
        self.lstm_hidden_size = config.MODEL.LSTM_PARAMS.HIDDEN_SIZE
        self.num_classes = num_classes
        # Toggle for additional classifier after the backbone
        if config.MODEL.E2E != True:
            self.multiclassifier = False
        else: self.multiclassifier = config.MODEL.MULTICLASSIFIER
        self.inference = config.MODEL.INFERENCE  # Toggle for inference mode

        # F1b disagreement-prediction auxiliary head — backward-compatible via
        # hasattr. When True, an extra LSTM-side head fc_lstm_dis is added
        # that predicts whether annotators are NOT unanimous on each criterion
        # (1 = contested 2-1 split, 0 = unanimous). Active in both train and
        # inference modes so test predictions are saved for AUC/AP scoring.
        self.F1b = (hasattr(config.MODEL, 'F1B_DISAGREEMENT_HEAD')
                    and config.MODEL.F1B_DISAGREEMENT_HEAD)

        # LSTM for temporal sequence processing
        self.lstm = nn.LSTM(input_size=self.swinv2_model.num_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=config.MODEL.LSTM_PARAMS.NUM_LAYERS,
                            batch_first=True)

        # Fully connected layer for classification after LSTM (MV head)
        self.fc_lstm = nn.Linear(self.lstm_hidden_size, num_classes)
        if self.multiclassifier:
            # New fully connected layer for mid-stream classification (SwinV2 feature classification)
            self.fc_swin = nn.Linear(self.swinv2_model.num_features, num_classes)
        if self.F1b:
            # F1b auxiliary: disagreement-prediction head on the LSTM side.
            self.fc_lstm_dis = nn.Linear(self.lstm_hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=5, 3, 384, 384)
        batch_size, seq_len, _, _, _ = x.size()

        # Reshape input for SwinV2 (batch_size * seq_len, 3, 384, 384)
        x = x.view(-1, 3, 384, 384)

        # Extract features from SwinV2
        features = self.swinv2_model.forward_features(x)  # Shape: (batch_size * seq_len, num_features)

        # Optional mid-stream classification
        if self.multiclassifier and not self.inference:
            swin_classification = self.fc_swin(features)  # Shape: (batch_size * seq_len, num_classes)
            swin_classification = swin_classification.view(batch_size, seq_len, -1)  # Reshape for sequence output

        # Reshape back to (batch_size, seq_len, num_features)
        features = features.view(batch_size, seq_len, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_len, hidden_size)

        # Use the last time step's output from LSTM for classification
        lstm_last = lstm_out[:, -1, :]
        lstm_classification = self.fc_lstm(lstm_last)  # MV head, shape: (B, num_classes)
        if self.F1b:
            lstm_disagreement = self.fc_lstm_dis(lstm_last)  # Disagreement head

        # Return tuple depending on which heads are active. The training loop
        # and test-eval block branch on these shapes via the same config flags.
        if self.F1b:
            if self.multiclassifier and not self.inference:
                return swin_classification[:, -1, :], lstm_classification, lstm_disagreement
            else:
                return lstm_classification, lstm_disagreement
        if self.multiclassifier and not self.inference:
            return swin_classification[:, -1, :], lstm_classification
        else:
            return lstm_classification
        
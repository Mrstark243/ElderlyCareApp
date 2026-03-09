import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, num_chars, embed_dim=256, encoder_dim=256):
        super(Encoder, self).__init__()
        # Converts text characters (indices) into dense vectors
        self.embedding = nn.Embedding(num_chars, embed_dim)
        
        # 3 Convolutional layers to analyze context (similar to n-grams)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ) for _ in range(3)
        ])
        
        # LSTM to understand the sequence/flow of text
        self.lstm = nn.LSTM(embed_dim, encoder_dim // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch, text_len]
        x = self.embedding(x).transpose(1, 2) # [batch, embed_dim, text_len]
        
        for conv in self.convs:
            x = conv(x)
            
        x = x.transpose(1, 2) # [batch, text_len, embed_dim]
        output, _ = self.lstm(x)
        return output

class ReferenceEncoder(nn.Module):
    """
    Encodes a reference Mel-Spectrogram into a fixed-length style vector (Voice Embedding).
    Input: [Batch, Mels, Time]
    Output: [Batch, Style_Dim]
    """
    def __init__(self, num_mels=80, style_dim=128):
        super(ReferenceEncoder, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        
        # GRU input size 256 (128 channels * 2 height for 80 mels)
        self.gru = nn.GRU(256, style_dim, batch_first=True)
        self.style_dim = style_dim

    def forward(self, x):
        # x: [Batch, Mels, Time] -> Needs to be 4D for Conv2D: [Batch, 1, Mels, Time]
        x = x.unsqueeze(1)
        
        out = self.convs(x)
        
        # out: [Batch, 128, reduced_freq, reduced_time]
        # Collapse dimensions for GRU
        batch, channels, freq, time = out.size()
        out = out.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq)
        
        # RNN to aggregate over time
        _, h_n = self.gru(out) # h_n: [1, Batch, Style_Dim]
        
        return h_n.squeeze(0)

class Attention(nn.Module):
    def __init__(self, rnn_dim, encoder_dim, attn_dim=128):
        super(Attention, self).__init__()
        self.W = nn.Linear(rnn_dim, attn_dim)
        self.V = nn.Linear(encoder_dim, attn_dim)
        self.u = nn.Linear(attn_dim, 1)

    def forward(self, decoder_state, encoder_outputs):
        # Calculates "alignment" - which part of text are we focusing on?
        
        # decoder_state: [batch, rnn_dim] -> [batch, 1, attn_dim]
        # encoder_outputs: [batch, seq_len, encoder_dim] -> [batch, seq_len, attn_dim]
        
        w_hidden = self.W(decoder_state).unsqueeze(1)
        v_encoder = self.V(encoder_outputs)
        
        # Score calculation (Bahdanau Attention)
        energy = torch.tanh(w_hidden + v_encoder)
        scores = self.u(energy).squeeze(2) # [batch, seq_len]
        
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, num_mels, rnn_dim=256, encoder_dim=256, style_dim=128):
        super(Decoder, self).__init__()
        self.num_mels = num_mels
        
        # Input: (Context + Style) + Previous Mel Frame -> LSTM -> Output Mel Frame
        # Context comes from Attention (encoder_dim + style_dim)
        
        self.lstm1 = nn.LSTMCell(encoder_dim + style_dim + num_mels, rnn_dim)
        self.lstm2 = nn.LSTMCell(rnn_dim, rnn_dim)
        
        self.attention = Attention(rnn_dim, encoder_dim + style_dim)
        
        # Project output to Mel-Spectrogram dimension
        self.mel_proj = nn.Linear(rnn_dim, num_mels)
        self.gate_proj = nn.Linear(rnn_dim, 1) # Predicts when to stop generating (Stop Token)

    def forward(self, encoder_outputs, mel_input, hidden=None):
        if hidden is None:
            batch_size = encoder_outputs.size(0)
            hidden = (
                (torch.zeros(batch_size, 256).to(encoder_outputs.device), torch.zeros(batch_size, 256).to(encoder_outputs.device)),
                (torch.zeros(batch_size, 256).to(encoder_outputs.device), torch.zeros(batch_size, 256).to(encoder_outputs.device))
            )
            
        (h1, c1), (h2, c2) = hidden
        
        # 1. Calculate Attention Context
        context, attn_weights = self.attention(h2, encoder_outputs)
        
        # 2. Decoder Step
        # Concatenate context with previous mel frame
        lstm_input = torch.cat([context, mel_input], dim=1)
        
        h1, c1 = self.lstm1(lstm_input, (h1, c1))
        h2, c2 = self.lstm2(h1, (h2, c2))
        
        # 3. Output Projection
        mel_out = self.mel_proj(h2)
        gate_out = torch.sigmoid(self.gate_proj(h2)) # 0 = continue, 1 = stop
        
        return mel_out, gate_out, attn_weights, ((h1, c1), (h2, c2))

class PostNet(nn.Module):
    """
    PostNet: 5-layer Conv1D ResNet to refine the Mel-Spectrogram.
    Helps remove "blurriness" from the reconstruction.
    """
    def __init__(self, num_mels=80, embedding_dim=512, kernel_size=5):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(num_mels, embedding_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            )
        )

        for _ in range(3):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, embedding_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                    nn.BatchNorm1d(embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(embedding_dim, num_mels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(num_mels),
                nn.Dropout(0.5)
            )
        )

    def forward(self, x):
        # x: [Batch, Mels, Time]
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
        
        x = self.convolutions[-1](x)
        return x

class NanoTacotron(nn.Module):
    def __init__(self, num_chars, num_mels=80, style_dim=128):
        super(NanoTacotron, self).__init__()
        self.encoder = Encoder(num_chars)
        self.ref_encoder = ReferenceEncoder(num_mels, style_dim)
        
        # Decoder takes concatenated (Text_Enc + Style)
        # Encoder dim (256) (defined in Encoder class)
        self.decoder = Decoder(num_mels, encoder_dim=256, style_dim=style_dim)
        
        # PostNet to refine spectrograms
        self.postnet = PostNet(num_mels=num_mels)
        self.num_mels = num_mels

    def forward(self, text_inputs, mel_inputs=None, ref_mel=None, teacher_forcing_ratio=1.0):
        # text_inputs: [batch, text_len]
        # mel_inputs: [batch, mel_len, num_mels] (Target for training)
        # ref_mel: [batch, ref_len, num_mels] (Reference Audio)
        
        if ref_mel is None:
            # Fallback (create dummy reference for inference tests)
            # CAUTION: This means random/zero style.
            ref_mel = torch.zeros(text_inputs.size(0), 100, self.num_mels).to(text_inputs.device)

        # 1. Get Text Embeddings
        text_outputs = self.encoder(text_inputs) # [Batch, Text_Len, 256]
        
        # 2. Get Style Embedding
        # Transpose to [Batch, Mels, Time] for Reference Encoder
        style_embedding = self.ref_encoder(ref_mel.transpose(1, 2)) # [Batch, Style_Dim]
        
        # 3. Fuse Style into Text
        # Broadcast style to match text length
        style_broadcast = style_embedding.unsqueeze(1).expand(-1, text_outputs.size(1), -1)
        
        # Concat: [Batch, Text_Len, 256+128]
        encoder_outputs = torch.cat([text_outputs, style_broadcast], dim=2)
        
        batch_size = text_inputs.size(0)
        max_len = mel_inputs.size(1) if mel_inputs is not None else 200 # Max generation length
        
        # Initial inputs (Go Frame)
        current_mel = torch.zeros(batch_size, self.num_mels).to(text_inputs.device)
        hidden = None
        
        mel_outputs = []
        gate_outputs = []
        alignments = []
        
        for t in range(max_len):
            mel_step, gate_step, attn_step, hidden = self.decoder(encoder_outputs, current_mel, hidden)
            
            mel_outputs.append(mel_step)
            gate_outputs.append(gate_step)
            alignments.append(attn_step)
            
            # Teacher Forcing
            if mel_inputs is not None and torch.rand(1).item() < teacher_forcing_ratio:
                 if t < mel_inputs.size(1) - 1:
                     current_mel = mel_inputs[:, t]
            else:
                 current_mel = mel_step
                 
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=1) # [Batch, Time, Mels]
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        # PostNet Processing
        # PostNet expects [Batch, Mels, Time]
        mel_outputs_post = mel_outputs.transpose(1, 2)
        mel_refined = self.postnet(mel_outputs_post)
        mel_refined = mel_refined.transpose(1, 2)
        
        # Residual Connection
        mel_final = mel_outputs + mel_refined
        
        return mel_outputs, mel_final, gate_outputs, alignments

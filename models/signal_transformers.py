import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalEmbedding(nn.Module):
    def __init__(self, signal_length, segment_length, channels, embedding_dim):
        super().__init__()
        self.signal_length = signal_length  # 512 (maxpool 이후 길이)
        self.segment_length = segment_length  # 32 (각 세그먼트의 길이)
        self.num_segments = signal_length // segment_length  # 16 (세그먼트 개수)
        
        # Linear transformation for segments
        self.embedding = nn.Linear(segment_length * channels, embedding_dim)  # 수정: segment_length * channels
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_segments + 1, embedding_dim))
        
    def forward(self, x):
        # x shape: (batch_size, L, C) where L is signal_length
        batch_size, seq_len, channels = x.shape
        
        # Reshape signal into segments
        x = x.reshape(batch_size, self.num_segments, self.segment_length * channels)  # [B, num_segments, segment_length * channels]
        
        # Apply linear transformation to each segment
        embedded = self.embedding(x)  # [B, num_segments, embedding_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)
        
        # Add position embeddings
        embedded = embedded + self.pos_embedding
        
        return embedded


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.dropout(self.norm(x + attn_output))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.attention(x)
        ff_output = self.feed_forward(x)
        return self.dropout(self.norm(x + ff_output))

class SignalTransformer(nn.Module):
    """
    feature뽑고 - signal embedding(cls, pos, segment base d_model embedding) - transformer layer- cls 토큰 뽑고 분류
    """    
    def __init__(self, in_channels, output_channel, num_classes, segment_length=32,
                 d_model=16, n_head=8, dim_feedforward=64, dropout=0.1, num_transformer_layers=3):        
        super(SignalTransformer, self).__init__()
        # CNN 경로
        self.conv1 = nn.Conv1d(in_channels, output_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Signal Embedding 레이어
        self.signal_embedding = SignalEmbedding(
            signal_length=512,  # maxpool 이후 길이
            channels = output_channel,
            segment_length=segment_length,
            embedding_dim=d_model
        )           

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, dim_feedforward, dropout)
            for _ in range(num_transformer_layers)
        ])    

        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        # CNN 경로
        out = self.conv1(x)  # [B, output_channel, L]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  # [B, output_channel, L//2]
        
        # Signal embedding을 위한 형태로 변환
        out = out.transpose(1, 2)  # [B, L//2, output_channel]
        
        # Signal Embedding (CLS 토큰, 위치 인코딩 포함)
        out = self.signal_embedding(out)  # [B, num_segments+1, d_model]
        
        # Transformer layers 통과
        transformer_features = out.permute(1, 0, 2)  # [num_segments+1, B, d_model]
        for layer in self.transformer_layers:
            transformer_features = layer(transformer_features)
        transformer_features = transformer_features.permute(1, 0, 2)  # [B, num_segments+1, d_model]
        
        # CLS 토큰 특징 추출
        cls_features = transformer_features[:, 0]  # [B, d_model]
        
        # 분류
        out = self.classifier(cls_features)
        
        return out, cls_features              
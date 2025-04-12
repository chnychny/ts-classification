import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class BasicTransformer2D(nn.Module):
    """
    2D 버전의 BasicTransformer
    입력 shape: [batch_size, sequence_length, n_channels]
    """
    def __init__(self, in_channels, output_channel, num_classes, d_model=16, n_head=8, 
                 dim_feedforward=64, dropout=0.1, num_transformer_layers=2):
        super(BasicTransformer2D, self).__init__()
        
        # CNN 경로
        self.conv1 = nn.Conv2d(1, output_channel, 
                              kernel_size=(5, 1), stride=(1, 1), 
                              padding=(2, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        
        # Transformer 관련 컴포넌트
        self.num_segments = 512 // 2  # maxpool로 인해 시퀀스 길이가 절반으로 줄어듦
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_segments+1, d_model))
        
        # CNN 특징을 transformer 입력 차원으로 투영
        self.feature_projection = nn.Linear(output_channel * in_channels, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, dim_feedforward, dropout)
            for _ in range(num_transformer_layers)
        ])    

        # 최종 분류기
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # 입력 shape 변환: [batch_size, sequence_length, n_channels] ->
        # [batch_size, 1, sequence_length, n_channels]
        B, _,L, C = x.shape
        
        
        # CNN 경로
        out = self.conv1(x)  # [B, output_channel, L//2, C]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        # CNN 특징을 transformer 입력 형태로 변환
        # [B, output_channel, L//2, C] -> [B, L//2, output_channel * C]
        transformer_input = out.permute(0, 2, 1, 3)
        transformer_input = transformer_input.reshape(B, -1, out.size(1) * C)
        
        # 특징을 d_model 차원으로 투영
        transformer_input = self.feature_projection(transformer_input)  # [B, L//2, d_model]
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        transformer_input = torch.cat([cls_tokens, transformer_input], dim=1)
        
        # 위치 인코딩 추가
        transformer_input = transformer_input + self.pos_encoder[:, :transformer_input.size(1)]
        
        # Transformer layers 통과
        transformer_features = transformer_input.permute(1, 0, 2)  # [L//2+1, B, d_model]
        for layer in self.transformer_layers:
            transformer_features = layer(transformer_features)
        transformer_features = transformer_features.permute(1, 0, 2)  # [B, L//2+1, d_model]
        
        # CLS 토큰 특징 추출
        cls_features = transformer_features[:, 0]  # [B, d_model]
        
        # 분류
        out = self.classifier(cls_features)
        
        return out, cls_features
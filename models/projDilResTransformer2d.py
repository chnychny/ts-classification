import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(MultiDilatedConv2d, self).__init__()
        
        self.num_dilations = 6  # dilation rates: 1, 3, 5, 7, 9, 11
        channels_per_dilation = out_channels // self.num_dilations
        
        def get_padding(kernel_size, dilation):
            return ((kernel_size - 1) * dilation) // 2
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                channels_per_dilation,
                kernel_size=(kernel_size, 1),  # 시간 축에 대해서만 커널 적용
                stride=(stride, 1),
                padding=(get_padding(kernel_size, d), 0),
                dilation=(d, 1),
                bias=False
            ) for d in [1, 3, 5, 7, 9, 11]
        ])
        
        self.final_conv = nn.Conv2d(channels_per_dilation * self.num_dilations, 
                                  out_channels, 
                                  kernel_size=1,
                                  bias=False)

    def forward(self, x):
        dilated_outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(dilated_outputs, dim=1)
        out = self.final_conv(out)
        return out

class ImprovedProjDilResTransformer2_2d(nn.Module):
    """
    2D 버전의 ImprovedProjDilResTransformer2
    입력 shape: [batch_size, sequence_length, n_channels]
    """
    def __init__(self, in_channels, output_channel, num_classes, stride=2):
        super().__init__()
        
        # Initial layers
        self.proj_conv = nn.Conv2d(1, output_channel//2, 
                                kernel_size=(5, 1), stride=(1, 1), 
                                padding=(2, 0), bias=False)
        self.conv1 = nn.Conv2d(output_channel//2, output_channel//2, 
                            kernel_size=(5, 1), stride=(1, 1), 
                            padding=(2, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel//2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        # First dilated block
        self.dil_conv1 = MultiDilatedConv2d(output_channel//2, output_channel//2, 
                                         kernel_size=5, stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm2d(output_channel//2)
        self.dil_conv2 = nn.Conv2d(output_channel//2, output_channel//2, 
                                kernel_size=(3, 1), stride=(2, 1), 
                                padding=(1, 0))
        
        self.down_sample = nn.Sequential(
            nn.Conv2d(output_channel//2, output_channel//2, 
                     kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(output_channel//2)
        )

        # Improved residual blocks
        self.res_block1 = ImprovedDilResidualBlock2d(output_channel//2, output_channel, stride=2)
        self.res_block2 = ImprovedDilResidualBlock2d(output_channel, output_channel, stride=2)

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        
    def forward(self, x):
        # 입력 shape 변환: [batch_size, sequence_length, n_channels] ->
        # [batch_size, 1, sequence_length, n_channels]
        
        # Initial path
        out = self.proj_conv(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        # First dilated block
        identity = out
        out = self.dil_conv1(out)
        out = self.dil_bn1(out)
        out = self.relu(out)
        out = self.dil_conv2(out)
        shortcut = self.down_sample(identity)
        out += shortcut
        
        # Residual blocks
        out = self.res_block1(out)
        out = self.res_block2(out)
        
        # Global pooling and classification
        features = self.gap(out).squeeze(-1).squeeze(-1)
        out = self.classifier(features)
        
        return out, features

class CompactMultiHeadAttention2d(nn.Module):
    def __init__(self, channels, num_heads=2, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 차원을 1/4로 줄여서 효율적인 attention 계산
        self.reduced_dim = channels // 4
        self.to_reduced = nn.Conv2d(channels, self.reduced_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.reduced_dim, channels, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        H_head = self.num_heads
        
        # 원본 feature 저장
        residual = x
        
        # Reduced dimension에서 QKV 계산
        qkv = self.to_reduced(x)
        qkv = qkv.reshape(B, 3, H_head, self.reduced_dim // H_head, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Reshape for attention
        q = q.reshape(B, H_head, -1, H * W)
        k = k.reshape(B, H_head, -1, H * W)
        v = v.reshape(B, H_head, -1, H * W)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention and reshape
        out = (attn @ v).reshape(B, self.reduced_dim, H, W)
        out = self.to_out(out)
        
        # Skip connection으로 원본 feature 보존
        return out + residual

class ImprovedDilResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        # Main branch
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv2d(in_channels, out_channels, 
                                     kernel_size=5, stride=1, padding=2)
        
        # Compact attention module
        self.attention = CompactMultiHeadAttention2d(out_channels, num_heads=2)
        
        # Second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                            kernel_size=(3, 1), stride=(2, 1), 
                            padding=(1, 0))
        
        # Shortcut connection
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=(1, 1), stride=(2, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        # Main path
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        # Attention with skip connection
        out = self.attention(out)
        
        # Second convolution
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Add shortcut
        return out + self.shortcut(x)
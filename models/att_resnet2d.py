import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Dil3ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Dil3ResidualBlock2d, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = MultiDilatedConv2d(in_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)
        self.conv2 = MultiDilatedConv2d(out_channels, out_channels, kernel_size=5, 
                             stride=1, padding=2)             
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.conv3 = MultiDilatedConv2d(out_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1)    

        if stride == 2:
            self.down_sample = nn.Conv2d(in_channels, out_channels, 
                                       kernel_size=(1, 1), stride=(2, 1))
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)        
        
        out += shortcut
        return out

class projDil3Resnet3_2d(nn.Module):
    """
    2D 버전의 projDil3Resnet3
    입력 shape: [batch_size, 1, sequence_length, n_channels]
    """
    def __init__(self, in_channels, output_channel, num_classes, stride=2):
        super(projDil3Resnet3_2d, self).__init__()
        
        self.conv1 = nn.Conv2d(1, output_channel//2, 
                              kernel_size=(5, 1), stride=(1, 1), 
                              padding=(2, 0), bias=False)
        
        self.bn1 = nn.BatchNorm2d(output_channel//2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        
        self.dil_conv1 = MultiDilatedConv2d(output_channel//2, output_channel//2, 
                                         kernel_size=5, stride=1, padding=2)
        self.dil_bn1 = nn.BatchNorm2d(output_channel//2)
        self.dil_conv2 = MultiDilatedConv2d(output_channel//2, output_channel//2, 
                                         kernel_size=3, stride=2, padding=1)
        
        self.down_sample = nn.Sequential(
            nn.Conv2d(output_channel//2, output_channel//2, 
                     kernel_size=(1, 1), stride=(2, 1)),
            nn.BatchNorm2d(output_channel//2)
        )

        self.res_block2 = Dil3ResidualBlock2d(output_channel//2, output_channel, stride=2)
        self.res_block3 = Dil3ResidualBlock2d(output_channel, output_channel, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        # 입력 shape 변환: [batch_size, sequence_length, n_channels] ->
        # x=[batch_size, 1, sequence_length, n_channels]
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        identity = out
        out = self.dil_conv1(out)
        out = self.dil_bn1(out)
        out = self.relu(out)
        out = self.dil_conv2(out)
        
        shortcut = self.down_sample(identity)
        out += shortcut

        out = self.res_block2(out)
        out = self.res_block3(out)
        
        x4 = self.gap(out)
        x4 = x4.view(x4.size(0), -1)
        out = self.linear(x4)
        
        return out, x4
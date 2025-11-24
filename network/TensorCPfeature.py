import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import math

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        padding = kernel_size // 2
        
        self.conv = Conv2d_gc(in_channels=2,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        # self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B,1,H,W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        
        attention = self.conv(concat)  # [B,1,H,W]
        return x * self.sigmoid(attention) + x 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Conv2d_gc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_gc, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            
            return out_normal - self.theta * out_diff

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class feature_fusion(nn.Module):
    def __init__(self, in_channels=32, out_channels=16):
        super(feature_fusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 1,1,0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.pixshuffle_up = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.PixelShuffle(2),
            
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.Sigmoid()  
        )

        self.x_fusion = nn.Sequential(
            nn.Conv2d(16, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [B, out_channels, H, W]
        x_maxpool = self.maxpool(x)  # [B, C, H/2, W/2]
        x_avgpool = self.avgpool(x)  # [B, C, H/2, W/2]
        x_input = torch.cat([x_maxpool, x_avgpool], dim=1)  
        x_input = self.pixshuffle_up(x_input)

        if x_input.shape[2] != x1.shape[2]:
            x_input = F.interpolate(x_input, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)

        # # x = torch.cat([x_input, x1], dim=1)  # [B, out_channels, H, W]
        x = x_input + x1
        x = self.x_fusion(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SAMStructuralBranch(nn.Module):
    def __init__(self, sam_checkpoint, in_channels, device='cuda:0'):
        super().__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
        self.sam.eval()
        self.sam.image_encoder.requires_grad_(False) 

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 32, 1, 1, 0),

            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, 1, 1, 0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(256, 16, 1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )


    def forward(self, x_hsi):
        x_rgb = self.conv(x_hsi)  # [B, 3, H, W]
        h, w = x_rgb.shape[2], x_rgb.shape[3]
        x_resized = F.interpolate(x_rgb, size=(1024, 1024), mode="bilinear", align_corners=True)


        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(x_resized)  # [B, C_feat, H', W']
        
        structural_feat = F.interpolate(
            image_embeddings, 
            size=x_hsi.shape[2:],  
            mode="bilinear", 
            align_corners=True
        )

        structural_feat = self.projection(structural_feat)  # [B, 16, H, W]
        return structural_feat  # [B, 16, H, W]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TensorConvFusionModel(nn.Module):
    def __init__(self, input_shape, num_features=16):
        super().__init__()
        C, H, W = input_shape
        
        self.structural_branch = SAMStructuralBranch(sam_checkpoint='./network/checkpoint/sam_vit_b_01ec64.pth', in_channels=C)

        self.conv_branch = SpatialAttention2D()

        self.fusion = feature_fusion(C + num_features, C) 


    def forward(self, x_init, x_grad):

        structural_feat = self.structural_branch(x_init*(1 + x_grad))
        
        conv_feat = self.conv_branch(x_init)
        
        combined_feat = torch.cat([structural_feat, conv_feat], dim=1) 
        fused_feat = self.fusion(combined_feat) + x_init

        return fused_feat 

    def init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['attention_layer', 'attention_layer_light']

class attention_layer(nn.Module):
    def __init__(self, in_dim, texture_dim):
        super(attention_layer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=texture_dim, out_channels=in_dim//8 , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):
        batch, C, W ,H = x.size()
        proj_query  = self.query_conv(z).view(batch,-1,W*H).permute(0,2,1) 
        proj_key =  self.key_conv(x).view(batch,-1,W*H) 
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy)  
        proj_value = self.value_conv(x).view(batch,-1,W*H) 

        out = torch.bmm(proj_value,attention.permute(0,2,1)) 
        out = out.view(batch,C,W,H)
        out = self.final_conv(out) 
        out = self.gamma*out + x

        return out, attention



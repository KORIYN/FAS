import torch
import torch.nn as nn

class Frequency_TopK(nn.Module):
    def __init__(self):
        super(Frequency_TopK, self).__init__()

    def freq_topk(self, amp_src, low_thresh=None, mid_thresh=None, ratio= None, params=None):
        amp_shifted = torch.fft.fftshift(amp_src, dim=(-2, -1))
        B, C, H, W = amp_shifted.shape

        if self.training:
            keep_ratio = ratio  
            spatial = amp_shifted.mean(dim=1, keepdim=True)
            flat_s = spatial.view(B, -1)  

            ys = torch.arange(H, device=amp_src.device).view(H, 1).expand(H, W)
            xs = torch.arange(W, device=amp_src.device).view(1, W).expand(H, W)
            dists = torch.sqrt((xs - W//2)**2 + (ys - H//2)**2)
            ring_idx = torch.floor(dists).long()           

            band_map = torch.zeros_like(ring_idx)
            band_map[(ring_idx > low_thresh) & (ring_idx <= mid_thresh)] = 1
            band_map[ring_idx > mid_thresh] = 2
            band_map_flat = band_map.view(-1)             

            mask_flat = torch.zeros_like(flat_s)
            batch_idx = torch.arange(B, device=amp_src.device).unsqueeze(1)  

            for band in (0, 1, 2):
                idx_band = (band_map_flat == band).nonzero(as_tuple=True)[0] 
                n = idx_band.numel()
                if n == 0:
                    continue
                k = max(int(n * keep_ratio), 1)
                band_vals = flat_s[:, idx_band]                              

                _, topk_local = band_vals.topk(k, dim=1, largest=True, sorted=False)  
                global_idxs = idx_band[topk_local]                           
                mask_flat[batch_idx, global_idxs] = 1.0
            
            mask_s = mask_flat.view(B, 1, H, W)
            self.attn = mask_s.expand(-1, C, -1, -1)

            amp_shifted = amp_shifted * self.attn

        amp_unshifted = torch.fft.ifftshift(amp_shifted, dim=(-2, -1))
        return amp_unshifted
    

    def FAS_source_filtering(self, src, ratio, params=None):

        src = src.to(torch.float64)
        fft_src = torch.fft.fft2(src, norm='ortho', dim=(-2, -1)) 

        amp_src = torch.abs(fft_src)
        eps = 1e-10
        mask = (amp_src < eps)
        fft_src = fft_src + mask * eps * (1 + 1j)  
        pha_src = torch.angle(fft_src)

        amp_src = amp_src.clamp(min=1e-10)
        amp_src_ = self.freq_topk(amp_src, ratio, params)

        fft_mutated = torch.polar(amp_src_, pha_src)
        src_filtered = torch.fft.ifft2(fft_mutated, norm='ortho', dim=(-2, -1)).real   

        return src_filtered.float()
    
    def forward(self, x, ratio, params=None):       
        
        cls_token = x[:, :1 , :]
        x_patches = x[:, 1:, :].permute(0, 2, 1)

        B, C, N = x_patches.shape         
        H = W = int(N ** 0.5)

        x_patches = x_patches.view(B, C, H, W)
        x_patches = self.FAS_source_filtering(x_patches, ratio, params)

        x_patches = x_patches.view(B, C, N).permute(0, 2, 1)

        x_out = torch.cat([cls_token, x_patches], dim=1)
    
        return x_out
    







import pywt
import torch
import torch.nn.functional as F

def dwt2d(x, wave='haar'):
    """x: [B, C, H, W]"""
    B, C, H, W = x.shape
    coeffs = []
    for b in range(B):
        c_coeffs = []
        for c in range(C):
            coeff = pywt.dwt2(x[b, c].cpu().detach().numpy(), wave)
            c_coeffs.append(coeff)
        coeffs.append(c_coeffs)

    # Split into LL and (LH, HL, HH)
    LL = torch.stack([torch.stack([torch.tensor(c[0]) for c in b]) for b in coeffs]).to(x.device)
    LH = torch.stack([torch.stack([torch.tensor(c[1][0]) for c in b]) for b in coeffs]).to(x.device)
    HL = torch.stack([torch.stack([torch.tensor(c[1][1]) for c in b]) for b in coeffs]).to(x.device)
    HH = torch.stack([torch.stack([torch.tensor(c[1][2]) for c in b]) for b in coeffs]).to(x.device)
    return LL, LH, HL, HH

def idwt2d(LL, LH, HL, HH, wave='haar'):
    B, C, H, W = LL.shape
    recon = torch.zeros((B, C, H * 2, W * 2), device=LL.device)
    for b in range(B):
        for c in range(C):
            coeffs = (LL[b, c].cpu().detach().numpy(), (LH[b, c].cpu().detach().numpy(), HL[b, c].cpu().detach().numpy(), HH[b, c].cpu().detach().numpy()))
            recon_np = pywt.idwt2(coeffs, wave)
            recon[b, c] = torch.tensor(recon_np)
    return recon

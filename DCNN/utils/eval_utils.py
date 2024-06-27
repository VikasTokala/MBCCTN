import torch
from DCNN.loss import BinaryMask
EPS= 10-6






def ild_db(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    l1 = 20*torch.log10(s1 + eps)
    l2 = 20*torch.log10(s2 + eps)
    ild_value = (l1 - l2)

    return ild_value

def gcc_phat_stft(target_stft_l, target_stft_r, EPS=1e-8):
    
    cross_corr_target = target_stft_l * target_stft_r.conj()
    # cross_corr_output = output_stft_l * output_stft_r.conj()
    
    target_phat = cross_corr_target / (torch.abs(cross_corr_target) + EPS)
    # output_phase = cross_corr_output / (torch.abs(cross_corr_output) + EPS)
    
    
    
    return target_phat


def ipd_rad(s1, s2, eps=EPS, avg_mode=None):
    # s1 = _avg_signal(s1, avg_mode)
    # s2 = _avg_signal(s2, avg_mode)

    ipd_value_uw = torch.angle(s1) - torch.angle(s2)
    ipd_value = torch.remainder(ipd_value_uw + torch.pi, 2* torch.pi) - torch.pi
        # Check for phase wrapping
    

    return ipd_value

def _avg_signal(s, avg_mode):
    if avg_mode == "freq":
        return s.mean(dim=0)
    elif avg_mode == "time":
        return s.mean(dim=1)
    elif avg_mode == None:
        return s
    
def speechMask(stft_l,stft_r, threshold=15):
    # breakpoint()
    _,time_bins = stft_l.shape
    thresh_l,_ = (((stft_l.abs())**2)).max(dim=1) 
    thresh_l_db = 10*torch.log10(thresh_l) - threshold
    thresh_l_db=thresh_l_db.unsqueeze(1).repeat(1,1,time_bins)
    
    thresh_r,_ = (((stft_r.abs())**2)).max(dim=1) 
    thresh_r_db = 10*torch.log10(thresh_r) - threshold
    thresh_r_db=thresh_r_db.unsqueeze(1).repeat(1,1,time_bins)
    
    
    bin_mask_l = BinaryMask(threshold=thresh_l_db)
    bin_mask_r = BinaryMask(threshold=thresh_r_db)
    
    mask_l = bin_mask_l(20*torch.log10((stft_l.abs())))
    mask_r = bin_mask_r(20*torch.log10((stft_r.abs())))
    mask = torch.bitwise_and(mask_l.int(), mask_r.int())
    
    return mask
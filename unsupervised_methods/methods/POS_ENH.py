from unsupervised_methods.methods.POS_WANG import POS_WANG
from enhancement.model import Finetunemodel
import torch

def POS_ENH(data, FS, Inf_path):
    thisfeed = data.transpose((3,0,1,2))
    thisdevice = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    FTModel = Finetunemodel(None)
    base_weights = torch.load(Inf_path, map_location="cpu")
    pretrained_dict = base_weights
    model_dict = FTModel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    FTModel.load_state_dict(model_dict)

    permutation = [2, 1, 0]
    thisfeed = torch.tensor(thisfeed/255.).to(torch.float32).to(thisdevice)[permutation, :, :, :] #3,128,72,72
    thisfeed = thisfeed.unsqueeze(0)
    enh = torch.zeros_like(thisfeed)
    for t in range(thisfeed.shape[2]):
        enh[:,:,t,:,:] = FTModel(thisfeed[:,:,t,:,:])[1] # for [0] it look like whitening masked
    enh = (enh * 255.)[:, permutation, :, :, :] # permute back
    enh = enh.squeeze().permute(1,2,3,0).detach().cpu().numpy()
    BVP = POS_WANG(enh, FS)
    BVP_ori = POS_WANG(data, FS)
    return BVP, BVP_ori
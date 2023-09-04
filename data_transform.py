import torch

def data_to_ESLAM(image, depth, T, device):
    eslam_image = torch.transpose(image, 0, 1).to(device)/255
    eslam_depth = torch.transpose(depth, 0, 1).to(device)
    eslam_T = T.to(device)
    eslam_T[:,1] = -1*eslam_T[:,1]
    eslam_T[:,2] = -1*eslam_T[:,2]
    return eslam_image, eslam_depth, eslam_T
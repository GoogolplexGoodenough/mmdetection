from mobile_cv.model_zoo.models.fbnet_v2 import FBNetBackbone
import torch
from mobile_cv.model_zoo.models import hub_utils, utils

def _load_pretrained_info():
    folder_name = utils.get_model_info_folder("fbnet_v2")
    ret = utils.load_model_info_all(folder_name)
    return ret

PRETRAINED_MODELS = _load_pretrained_info()

from ..builder import BACKBONES

@BACKBONES.register_module()
class ori_fbnet(torch.nn.Module):
    def __init__(self, arch = 'fbnet_c', out_indices=(5,9,17,23), pretrained=None, using_ori_pretrained = True):
        super(ori_fbnet, self).__init__()
        self.out_indices=out_indices
        self.pretrained=pretrained
        self.using_ori_pretrained=using_ori_pretrained
        self.backbone = FBNetBackbone(arch, out_indices=out_indices)
        if pretrained is not None:
            if isinstance(using_ori_pretrained, bool):
                model_info = PRETRAINED_MODELS[arch]
                model_path = model_info["model_path"]
                ret = self._load_fbnet_state_dict(model_path)
                self.backbone.load_state_dict(ret)
            else:
                print('loaded weights for {} from {}'.format(arch, pretrained))
                saved_model = torch.load(pretrained)
                state_dict = saved_model["state_dict"]
                ret = {}
                for name, val in state_dict.items():
                    if name.startswith("module."):
                        name = name[len("backbone.module.") :]
                    if name[0] == '.':
                        continue
                    ret[name] = val
                self.backbone.load_state_dict(ret)


    def _load_fbnet_state_dict(self, file_name, progress=True):
        if file_name.startswith("https://"):
            file_name = hub_utils.download_file(file_name, progress=progress)

        state_dict = torch.load(file_name, map_location="cpu")["state_dict"]
        ret = {}
        for name, val in state_dict.items():
            if name.startswith("module."):
                name = name[len("backbone.module.") :]
            if name[0] == '.':
                continue
            ret[name] = val
        return ret

    def forward(self, x):
        return self.backbone(x)


    def init_weights(self, pretrained=None):
        pass

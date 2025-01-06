from omegaconf import OmegaConf
import torch
import yaml
# from .models.gaussian_diffusion import GaussianDiffusion
from utils import util_common, util_net


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, autoencoder, diffusion):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.diffusion = diffusion

    def forward(self, inputs):
        inputs = inputs * 0.5 + 0.5
        model_kwargs = {'lq': inputs}

        outputs = self.diffusion.p_sample_loop(
            y=inputs,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
        )
        return outputs.clamp_(-1.0, 1.0)


def get_model(config_path, ckpt_path, encoder_ckpt_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    base_diffusion = util_common.instantiate_from_config(configs['diffusion'])
    model = util_common.instantiate_from_config(configs['model']).cuda()
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        util_net.reload_model(model, ckpt['state_dict'])
    else:
        util_net.reload_model(model, ckpt)
    model.eval()

    params = configs['autoencoder'].get('params', dict)
    autoencoder = util_common.get_obj_from_str("ldm.models.autoencoder.VQModelTorch")(**params).cuda()
    state = torch.load(encoder_ckpt_path)
    util_net.reload_model(autoencoder, state)
    autoencoder.eval()

    # Wrap models
    return ModelWrapper(model, autoencoder, base_diffusion)

from PIL import Image
from tqdm import trange

import torch
import torch.nn as nn
from torch import Tensor
from torch import autocast
from einops import rearrange, repeat
from omegaconf import OmegaConf

from lightning import LightningModule

from torch.optim.lr_scheduler import ReduceLROnPlateau

from scflow.ema import EMA
from scflow.helpers import instantiate_from_config
from scflow.ldm.models.diffusion.ddim import DDIMSampler



def resize_ims(x: Tensor, size: int, mode: str = "bilinear", **kwargs):
    # idea: blur image before down-sampling
    return nn.functional.interpolate(x, size=size, mode=mode, **kwargs)

def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims

def tensor_to_pil(unclip_samples, size=(256,256)):
    # this return a stacked images 
    x_samples = resize_ims(unclip_samples, size=size, mode="bilinear")
    x_samples = un_normalize_ims(x_samples)
    x_samples = rearrange(x_samples.cpu().numpy(), "b c h w -> (b h) w c")
    pil_img = Image.fromarray(x_samples) # each sample needs to be 'c h w -> h w c'
    
    return pil_img

# for unclip visual
def load_model_from_config(config, ckpt, verbose=False, vae_sd=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    msg = None
    if "global_step" in pl_sd:
        msg = f"This is global step {pl_sd['global_step']}. "
    if "model_ema.num_updates" in pl_sd["state_dict"]:
        msg += f"And we got {pl_sd['state_dict']['model_ema.num_updates']} EMA updates."
    global_step = pl_sd.get("global_step", "?")
    sd = pl_sd["state_dict"]
    if vae_sd is not None:
        for k in sd.keys():
            if "first_stage" in k:
                sd[k] = vae_sd[k[len("first_stage_model."):]]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    print(f"Loaded global step {global_step}")
    return model, msg

    
class TrainerModuleSCFlow(LightningModule):
    def __init__(self,
                 fm_cfg: dict,
                 scale_factor: int = 1.0,
                 lr: float = 1e-4,
                 weight_decay: float = 0.,
                 ema_rate: float = 0.99,
                 ema_update_every: int = 100,
                 ema_update_after_step: int = 1000,
                 use_ema_for_sampling: bool = True,
                 lr_scheduler_patience: int = 4,
                 val_step_num: int = 40,
                 reverse_inference: bool = False,
                 n_intermediates: int = 0,
                 test_vis: bool = False,
                 unclip_ckpt: str = None,
                 ):
        """
        Args:
            fm_cfg: Flow matching model config.
            scale_factor: Scale factor for the latent space.
            lr: Learning rate.
            weight_decay: Weight decay.
            ema_rate: EMA rate.
            ema_update_every: EMA update rate (every n steps).
            ema_update_after_step: EMA update start after n steps.
            use_ema_for_sampling: Whether to use the EMA model for sampling.
            lr_scheduler_patience: Patience for LR scheduler.
            val_step_num: Number of steps for validation.
            reverse_inference: Whether to use reverse inference.
            n_intermediates: Number of intermediates for the model.
            test_vis: Whether to perform test visualization.
        """
        super().__init__()
        self.model = instantiate_from_config(fm_cfg)
        self.ema_model = EMA(
            self.model, beta=ema_rate,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            power=3/4.,                     # recommended for trainings < 1M steps
            include_online_model=False      # we have the online model stored here

        )
        self.use_ema_for_sampling = use_ema_for_sampling
        self.scale_factor = scale_factor
        self.lr_scheduler_patience = lr_scheduler_patience
                    
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.val_step_num = val_step_num
        self.reverse_inference = reverse_inference
        self.n_intermediates = n_intermediates
        self.test_vis = test_vis
        self.unclip_ckpt = unclip_ckpt
            

        if self.test_vis:# for the unclip visualization
            state = dict()
            state["ckpt"] = self.unclip_ckpt

            # config for unclip
            state["config"] = OmegaConf.load("configs/v2-1-stable-unclip-l-inference.yaml")


            unclip_model, msg = load_model_from_config(state["config"], state["ckpt"], vae_sd=None)
            # only move to device once start sampling
            self.unclip_model = unclip_model
            self.sampler = DDIMSampler(self.unclip_model, device=self.device)

        self.val_epochs = 0

        self.save_hyperparameters()



    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(
            opt, factor=0.8, patience=self.lr_scheduler_patience, verbose=True
        )
        return opt
    
    def forward(self, x_target: Tensor, x_source: Tensor, **kwargs):
        """
        In the case of FM objective:
            x_target: the target clip embedding padded
            x_source: the content and style clip embs
        """
        return self.model.training_losses(x_source=x_source, x_target=x_target, **kwargs)
    
            
    def extract_from_batch(self, batch,  val=False):
        """
        Takes batch and extracts images, clip embeddings and metainfo.

        Returns:
            clip_oai: the clip embeddings of the content and style
            clip_target: the clip embeddings of the target
           
        """        
        clip_oai_s = batch[0][:, 0, :]# (c2s2)
        clip_oai_c = batch[0][:, 2, :]# (c1s1)
        clip_oai = torch.cat([clip_oai_c, clip_oai_s], dim=1)
        
        clip_target = batch[0][:, 1, :]  # (c1s2)            

        if len(clip_oai.shape) == 2:
            clip_oai = clip_oai.unsqueeze(1)
        
        if val:          
            return clip_oai.type(torch.float32),  clip_target.type(torch.float32)
        
        return clip_oai.type(torch.float32), clip_target.type(torch.float32)

    
    def training_step(self, batch, batch_idx):
        """ extract source clip embeddings and targets from batch """
        clip_emb,clip_target = self.extract_from_batch(batch)
        bs = clip_emb.shape[0]

        # make sure the target is padded to the same size
        x_source = clip_emb
        clip_target = torch.cat([clip_target, clip_target], dim=1)
        if len(clip_target.shape) == 2:
            clip_target = clip_target.unsqueeze(1)

        forward_kwargs = {}

        """ loss """
        loss = self.forward(x_target=clip_target, x_source=x_source, **forward_kwargs)
        loss = loss.mean() 
        bs = x_source.shape[0]
        self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)                   
        
        # update EMA
        self.ema_model.update()


        return loss
    
    def predict_forward(self, clip_emb, **kwargs):
        """ predict the predict emb from the source clip emb"""
        val_step_num = self.val_step_num
        ode_kwargs = dict(options=dict(step_size=1./val_step_num))
        x_source = clip_emb
        n_intermediates = self.n_intermediates
        clip_pred = self.model.decode(z=x_source, ode_kwargs=ode_kwargs, n_intermediates=n_intermediates,**kwargs)

        return clip_pred
    
    def predict_reverse(self,clip_emb,**kwargs):
        """predict style and content emb from target clip emb"""
        val_step_num = self.val_step_num
        ode_kwargs = dict(options=dict(step_size=1./val_step_num))
        x_target = clip_emb
        clip_pred = self.model.encode(x=x_target,ode_kwargs=ode_kwargs, **kwargs)
        
        return clip_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        clip_emb, clip_target= self.extract_from_batch(batch, val=True)
        
        x_source = clip_emb
        clip_target = torch.cat([clip_target, clip_target], dim=1)
        if len(clip_target.shape) == 2:
            clip_target = clip_target.unsqueeze(1)

        forward_kwargs = {}
  
        clip_pred = self.predict_forward(x_source, **forward_kwargs)

        # log losses
        mse = nn.functional.mse_loss(clip_pred[:, :, -768:], clip_target[:, :, -768:])
        self.log("val/mse", mse, sync_dist=True, batch_size=clip_target.shape[0])
        content_mse = nn.functional.mse_loss(clip_pred[:, :, -768:], clip_emb[:, :, :768])
        self.log("val/content_mse", content_mse, sync_dist=True, batch_size=clip_emb.shape[0])
        style_mse = nn.functional.mse_loss(clip_pred[:, :, -768:], clip_emb[:, :, -768:])
        self.log("val/style_mse", style_mse, sync_dist=True, batch_size=clip_emb.shape[0])


        torch.cuda.empty_cache()

    
    @torch.no_grad()
    def make_conditionings_from_embedding(self, clip_emb, noise):
        # clip emb: tensor should have a shape of (batch_size, 768)
        adm_cond = clip_emb

        with torch.no_grad():
            adm_cond = adm_cond.to(self.unclip_model.device)
            weight = 1.
            if self.unclip_model.noise_augmentor is not None:
                noise_level = noise
                c_adm, noise_level_emb = self.unclip_model.noise_augmentor(adm_cond, noise_level=repeat(
                    torch.tensor([noise_level]).to(self.unclip_model.device), '1 -> b', b=adm_cond.shape[0]))
                adm_cond = torch.cat((c_adm, noise_level_emb), 1) * weight
                
            adm_uc = torch.zeros_like(adm_cond)

        return adm_cond, adm_uc

    @torch.no_grad()
    def unclip_sample(self, clip_emb,
                    n_runs=1,
                    H=768,
                    W=768,
                    C=4,
                    f=8,
                    ddim_steps=20,
                    scale=10.0,
                    noise_level=0,
                    unormalize=False,
                    callback=None,
                    negative_prompt=""):
        
        if self.unclip_model != self.device:
            self.unclip_model.to(self.device)
        
        
        # noise level: https://github.com/Stability-AI/stablediffusion/blob/main/doc/UNCLIP.MD
        if unormalize:
            # make sure to unormalize the clip_emb * scale_factor  back before passing to the unclip model
            adm_cond, adm_uc = self.make_conditionings_from_embedding(clip_emb*self.scale_factor, noise=noise_level)
        else:
            adm_cond, adm_uc = self.make_conditionings_from_embedding(clip_emb, noise=noise_level)

        batch_size = adm_cond.shape[0]
        latent_shape = [C, H // f, W // f]    

        # repeat prompts to match batch size (but same prompts)   
        prompts = batch_size * ["detailed, high-quality, 4k"]

        with autocast("cuda"):
            # use ema weights for evaluation
            with self.unclip_model.ema_scope():
                for _ in trange(n_runs, desc="Sampling"):
                                       
                    uc = self.unclip_model.get_learned_conditioning(batch_size * [negative_prompt])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = self.unclip_model.get_learned_conditioning(prompts)

                    if adm_uc is None:
                        adm_uc = adm_cond

                    c = {"c_crossattn": [c], "c_adm": adm_cond} # text -> c_crossattn; clip_img_emb -> c_adm
                    uc = {"c_crossattn": [uc], "c_adm": adm_uc}
                            
                    samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                    batch_size=batch_size,
                                                    shape=latent_shape,
                                                    verbose=False,
                                                    conditioning=c,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=0.,
                                                    x_T=None,
                                                    callback=callback,
                                                    ucg_schedule=None
                                                    )
                                        
                    unclip_samples = self.unclip_model.decode_first_stage(samples_ddim)
                
        return unclip_samples

    def get_sample_to_vis(self, clip_pred, path=None, save=True):
        if clip_pred.ndim==3:
            # elimninate the 1dim
            clip_pred = clip_pred.squeeze(1)

        # scale predicted value back to correct scale
        fake_ims = self.unclip_sample(clip_pred, ddim_steps=40, unormalize=False)
        pil_img = tensor_to_pil(fake_ims)
        if save:
            pil_img.save(path)

        del fake_ims

        return pil_img

    
    def on_train_epoch_end(self):
        pass

    @torch.no_grad()
    def on_validation_epoch_end(self, prefix="val"):
        torch.cuda.empty_cache()
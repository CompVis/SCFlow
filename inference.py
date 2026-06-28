import torch
import argparse
import os

from omegaconf import OmegaConf
from lightning import seed_everything
from PIL import Image
from scflow.trainer_module import TrainerModuleSCFlow

from datetime import datetime
import clip

torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser("SCFlow Inference")
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Only loads the weights from a checkpoint")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--name", type=str, default="vis_output")
    parser.add_argument("--val_step_num", type=int, default=1, help="step number for fm")
    parser.add_argument("--reverse_inference",  action="store_true", help="reverse inference for fm")
    parser.add_argument("--image_mix_path", type=str, default=None, help="path to the image for reverse inference")
    parser.add_argument("--image_c_path", type=str, default=None, help="path to the content image for forward inference")
    parser.add_argument("--image_s_path", type=str, default=None, help="path to the style image for forward inference")
    parser.add_argument("--test_vis",  default= True, help="visualization")
    parser.add_argument("--unclip_ckpt", type=str, default="None")
    parser.add_argument("--seed", type=int, default=2025, help="random seed for reproducibility")
    parser.add_argument("--model_type", type=str, default=None, choices=["scflow", "catfm"])
    parser.add_argument("--pt_path", type=str, default=None, help="optional CLIP embedding .pt file to decode with UnCLIP")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="optional sample indices for --pt_path")

    known, unknown = parser.parse_known_args()

    if exists(known.resume_checkpoint) and exists(known.load_weights):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")

    # check for mistakes
    for arg in unknown:
        if arg.startswith("-"):
            raise ValueError(f"Unknown argument: {arg}")

    return known, unknown



def exists(val):
    return val is not None


def init():

    """ parse args """
    args, unknown = parse_args()

    seed_everything(args.seed)
   
    """ Load config """
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, cli)
    
    cfg.val_step_num = args.val_step_num
    cfg.reverse_inference = args.reverse_inference
    cfg.image_mix_path = args.image_mix_path
    cfg.image_c_path = args.image_c_path
    cfg.image_s_path = args.image_s_path
    cfg.pt_path = args.pt_path
    cfg.indices = args.indices
    today = datetime.now().strftime('%Y-%m-%d') 
    exp_name = args.name if exists(args.name) else args.config.rsplit('/')[-2]
    output_path = f"{exp_name}_{today}"
    cfg.result_path = output_path
    cfg.unclip_ckpt = args.unclip_ckpt
    cfg.model_type = args.model_type or cfg.get("model_type", "scflow")

    """ Setup model """
    ckpt_path = args.resume_checkpoint if exists(args.resume_checkpoint) else "./ckpts/scflow_last.ckpt"



    """ Setup model """
    if cfg.model_type == "scflow":
        module_cls = TrainerModuleSCFlow
    elif cfg.model_type == "catfm":
        from scflow.catfm_trainer_module import TrainerModuleCATFM

        module_cls = TrainerModuleCATFM
    else:
        raise ValueError(f"Unknown model_type={cfg.model_type}. Expected 'scflow' or 'catfm'.")

    module = module_cls.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        fm_cfg=cfg.model.fm,
        test_vis=args.test_vis,
        unclip_ckpt=cfg.unclip_ckpt,
        val_step_num=cfg.val_step_num,
        reverse_inference=cfg.reverse_inference, 
        strict=False, map_location='cpu'
    )
    module.eval()

    return module, cfg

class CLIPEncoder:
    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load('ViT-L/14', device=self.device)
        self.model.eval()

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            # features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    
if __name__ == "__main__":
    module = None
    if module is None:
        module, cfg = init()
    module = module.to(device)
    module.eval()
    output_dir = f"{cfg.result_path}"

    if cfg.pt_path is not None:
        if cfg.unclip_ckpt in [None, "None", ""]:
            raise ValueError("Provide --unclip_ckpt when using --pt_path.")
        os.makedirs(output_dir, exist_ok=True)
        x = torch.as_tensor(torch.load(cfg.pt_path, map_location="cpu"))
        if x.ndim == 1 and x.shape[0] == 768:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2 and x.shape[-1] == 768:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 768:
            pass
        else:
            raise ValueError(f"Unexpected embedding shape {tuple(x.shape)}. Expected [..., 768].")

        x = x.to(device=module.device, dtype=module.dtype)
        indices = cfg.indices if cfg.indices is not None else range(x.shape[0])
        for i in indices:
            sample = x[i:i + 1]
            out_path = os.path.join(output_dir, f"unclip_vis_{i:04d}.png")
            module.get_sample_to_vis(sample, path=out_path)
            print(f"Saved {out_path}")
        raise SystemExit(0)

    encoder = CLIPEncoder()
    if module.reverse_inference:
        features = encoder.encode_image(cfg.image_mix_path).unsqueeze(0).to(device=module.device, dtype=module.dtype)
        clip_target = torch.cat([features, features], dim=2)
        clip_pred = module.predict_reverse(clip_target)
        preds = {
            "c": clip_pred[:, :, :768],
            "s": clip_pred[:, :, 768:]
        }
        output_pred_dir = os.path.join(output_dir, f"reverse/{module.val_step_num}")
        os.makedirs(output_pred_dir, exist_ok=True)
        for key, tensor in preds.items():
            torch.save(tensor, os.path.join(output_pred_dir, f"clip_pred_{key}_tensor.pt"))
            module.get_sample_to_vis(tensor, path=os.path.join(output_pred_dir, f"reverse_{key}.png"))
    else:
        features_c = encoder.encode_image(cfg.image_c_path).unsqueeze(0).to(device=module.device, dtype=module.dtype)
        features_s = encoder.encode_image(cfg.image_s_path).unsqueeze(0).to(device=module.device, dtype=module.dtype)
        clip_emb = torch.cat([features_c, features_s], dim=2)
        clip_pred = module.predict_forward(clip_emb)
        preds = {
            "0": clip_pred[:, :, :768],
            "1": clip_pred[:, :, 768:],
            "mean": (clip_pred[:, :, :768] + clip_pred[:, :, 768:]) / 2.0
        }
        output_pred_dir = os.path.join(output_dir, f"forward/{module.val_step_num}")
        os.makedirs(output_pred_dir, exist_ok=True)
        for key, tensor in preds.items():
            torch.save(tensor, os.path.join(output_pred_dir, f"clip_pred_{key}_tensor.pt"))
            module.get_sample_to_vis(tensor, path=os.path.join(output_pred_dir, f"clip_pred_{key}.png"))

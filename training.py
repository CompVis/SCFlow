import os
import sys
import torch
import argparse
import datetime
from omegaconf import OmegaConf

from lightning import Trainer
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# ddp stuff
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from scflow.helpers import load_model_weights
from scflow.helpers import count_params, exists
from scflow.helpers import instantiate_from_config
from scflow.trainer_module import TrainerModuleSCFlow

def parse_args():
    parser = argparse.ArgumentParser("FM Super-Resolution")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--name", type=str, default="scflow_train")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_wandb_offline", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Resumes training from a checkpoint")
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Only loads the weights from a checkpoint")
    parser.add_argument("--num_nodes", type=int, default=1)
    # if -1, it uses all available GPUs
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--find_unused_parameters", action="store_true")

    known, unknown = parser.parse_known_args()

    if exists(known.resume_checkpoint) and exists(known.load_weights):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")

    # check for mistakes
    for arg in unknown:
        if arg.startswith("-"):
            raise ValueError(f"Unknown argument: {arg}")

    return known, unknown


def main():
    seed_everything(2023)

    """ parse args """
    args, unknown = parse_args()
   
    """ Load config """
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, cli)


    """ Setup Logging """
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{args.name}_{now}" if exists(args.name) else now
    log_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    use_wandb_logging = args.use_wandb or args.use_wandb_offline 
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # setup loggers
    if use_wandb_logging:
        mode = "offline" if args.use_wandb_offline else "online"
        print(f"Logger mode is set to: {mode}") 
        online_logger = WandbLogger(
            dir=log_dir,
            save_dir=log_dir,
            name=exp_name,
            project="scflow",
            config=OmegaConf.to_object(cfg),
            mode=mode,
            #group="DDP"
        )
    else:
        online_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
            version="",
            log_graph=False,
            default_hp_metric=False,
        )
    csv_logger = CSVLogger(
        log_dir,
        name="",
        version="",
        prefix="",
        flush_logs_every_n_steps=500
    )
    csv_logger.log_hyperparams(OmegaConf.to_container(cfg))
    logger = [online_logger, csv_logger]
    data = instantiate_from_config(cfg.data)
   

    """ Setup model """
    module = TrainerModuleSCFlow(
        fm_cfg=cfg.model.fm,
        scale_factor=cfg.model.get("scale_factor", 1.0),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        ema_rate=cfg.train.get("ema_rate", 0.9999),
        ema_update_every=cfg.train.get("ema_update_every", 1),
        ema_update_after_step=cfg.train.get("ema_update_after_step", 1000),
        use_ema_for_sampling=cfg.train.get("use_ema_for_sampling", True),
        lr_scheduler_patience=cfg.train.get("lr_scheduler_patience", 100),
    )

    """ Setup callbacks """ 
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:06d}",
        # from config
        **cfg.train.checkpoint_callback_params
    )
    
    callbacks = [checkpoint_callback]

    # other callbacks from config
    callbacks_cfg = cfg.train.get("callbacks", None)
    if exists(callbacks_cfg):
        for cb_cfg in callbacks_cfg:
            cb = instantiate_from_config(cb_cfg)
            callbacks.append(cb)
    
    """ Setup trainer """
    if torch.cuda.is_available():
        print("Using GPU")
        gpu_kwargs = {
            'accelerator': 'gpu',
            'strategy': ('ddp_find_unused_parameters_true' if args.find_unused_parameters else "ddp")
        }
        if args.devices > 0:
            gpu_kwargs["devices"] = args.devices
        else:       # determine automatically
            gpu_kwargs["devices"] = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
        gpu_kwargs["num_nodes"] = args.num_nodes
        if args.num_nodes >= 2:
            # multi-node hacks from
            # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
            gpu_kwargs["strategy"] = DDPStrategy(
                gradient_as_bucket_view=True,
                ddp_comm_hook=default_hooks.fp16_compress_hook
            )
    else:
        print("Using CPU")
        gpu_kwargs = {'accelerator': 'cpu'}

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,#for checkpoint
        **gpu_kwargs,
        # from config
        **OmegaConf.to_container(cfg.train.trainer_params) 
    )

    """ Log some information """
    # compute global batchsize
    bs = cfg.data.params.batch_size
    bs = bs * gpu_kwargs["devices"]
    bs = bs * gpu_kwargs["num_nodes"]
    bs = bs * cfg.train.trainer_params.get("accumulate_grad_batches", 1)
    # log info
    some_info = {
        'Config': args.config,
        'Name': exp_name,
        'Log dir': log_dir,
        'Params': count_params(module),
        'Dataset': cfg.data.get("name", "not-specified"),
        'Batchsize': cfg.data.params.batch_size,
        'Dropout content rate': cfg.train.get("clip_dropout", 0.0),
        'Devices': gpu_kwargs["devices"],
        'Num nodes': gpu_kwargs["num_nodes"],
        'Gradient accum': cfg.train.trainer_params.get("accumulate_grad_batches", 1),
        'Global batchsize': bs,
        'Learning rate': cfg.train.lr,
        'Resume ckpt': args.resume_checkpoint,
        'Load weights': args.load_weights,
        'First stage': cfg.model.get("first_stage", None) is not None,
        'Scale factor': cfg.model.get("scale_factor", 1.0),
        'Sample posterior': cfg.model.get("sample_train_posterior", True),
        'EMA rate': cfg.train.get("ema_rate", 0.9999),
        'EMA update every': cfg.train.get("ema_update_every", 1),
        'EMA warmup phase': cfg.train.get("ema_update_after_step", 1000),
    }
    
    # Make sure we don't log multiple times
    if trainer.global_rank == 0:
        print("-" * 40)
        for k, v in gpu_kwargs.items():
            print(f"{k:<16}: {v}")
        print("-" * 40)
        for k, v in some_info.items():
            if use_wandb_logging:
                online_logger.experiment.summary[k] = v
            if isinstance(v, float):
                print(f"{k:<16}: {v:.5f}")
            elif isinstance(v, int):
                print(f"{k:<16}: {v:,}")
            elif isinstance(v, bool):
                print(f"{k:<16}: {'True' if v else 'False'}")
            else:
                print(f"{k:<16}: {v}")
        print("-" * 40)
        # log called command
        if use_wandb_logging:
            online_logger.experiment.summary["command"] = " ".join(["python"] + sys.argv)
        
        # save config file
        OmegaConf.save(cfg, f"{log_dir}/config.yaml")

    """ Train """
    ckpt_path = args.resume_checkpoint if exists(args.resume_checkpoint) else None
    if exists(args.load_weights):
        module = load_model_weights(module, args.load_weights, strict=False)
    trainer.fit(module, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

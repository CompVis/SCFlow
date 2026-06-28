import torch
import torch.nn as nn

from scflow.cfm import pad_vector_like_x
from scflow.trainer_module import TrainerModuleSCFlow


def _load_metric_learning():
    try:
        from pytorch_metric_learning import losses, miners
    except ImportError as exc:
        raise ImportError(
            "CATFM metric losses require pytorch-metric-learning. "
            "Install it with `pip install pytorch-metric-learning`, or set train.dml_type=null."
        ) from exc
    return losses, miners


class TrainerModuleCATFM(TrainerModuleSCFlow):
    """Optional CATFM trainer layered on top of the original SCFlow module."""

    def __init__(
        self,
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
        dml_type: str = None,
        lambda_content: float = 1.0,
        lambda_style: float = 1.0,
        ms_alpha_content: int = 2,
        ms_beta_content: int = 50,
        ms_alpha_style: int = 2,
        ms_beta_style: int = 50,
        predict_x1: bool = False,
        predict_x0: bool = False,
        predict_x0x1: bool = False,
        t_conditional: float = 0.5,
        style_weight: float = 0.0,
        target_placeholder: str = "gt",
        learnable_placeholder: bool = False,
        **_,
    ):
        super().__init__(
            fm_cfg=fm_cfg,
            scale_factor=scale_factor,
            lr=lr,
            weight_decay=weight_decay,
            ema_rate=ema_rate,
            ema_update_every=ema_update_every,
            ema_update_after_step=ema_update_after_step,
            use_ema_for_sampling=use_ema_for_sampling,
            lr_scheduler_patience=lr_scheduler_patience,
            val_step_num=val_step_num,
            reverse_inference=reverse_inference,
            n_intermediates=n_intermediates,
            test_vis=test_vis,
            unclip_ckpt=unclip_ckpt,
        )
        self.dml_type = dml_type
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.predict_x1 = predict_x1
        self.predict_x0 = predict_x0
        self.predict_x0x1 = predict_x0x1
        self.t_conditional = t_conditional
        self.style_weight = style_weight
        self.target_placeholder = target_placeholder
        self.learnable_placeholder = learnable_placeholder

        n_prediction_modes = sum([self.predict_x1, self.predict_x0, self.predict_x0x1])
        if self.dml_type is not None and n_prediction_modes != 1:
            raise ValueError("CATFM metric training requires exactly one of predict_x1, predict_x0, or predict_x0x1.")

        if self.learnable_placeholder:
            self.clip_target0 = nn.Parameter(torch.randn(1, 768))

        if self.dml_type is not None:
            losses, miners = _load_metric_learning()
            if self.dml_type == "MultiSimilarity":
                self.dml_miner_content = miners.MultiSimilarityMiner(epsilon=0.1)
                self.dml_loss_func_content = losses.MultiSimilarityLoss(alpha=ms_alpha_content, beta=ms_beta_content)
                self.dml_miner_style = miners.MultiSimilarityMiner(epsilon=0.1)
                self.dml_loss_func_style = losses.MultiSimilarityLoss(alpha=ms_alpha_style, beta=ms_beta_style)
            elif self.dml_type == "NXTent":
                self.dml_miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="semihard")
                self.dml_loss_func = losses.NTXentLoss(temperature=0.1)
            elif self.dml_type == "Margin":
                self.dml_miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")
                self.dml_loss_func = losses.MarginLoss(margin=0.2, nu=0, beta=1.2)
            elif self.dml_type == "both":
                self.multisim_miner = miners.MultiSimilarityMiner(epsilon=0.1)
                self.multisim_loss_func = losses.MultiSimilarityLoss(alpha=ms_alpha_style, beta=ms_beta_style)
                self.nxtent_miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="semihard")
                self.nxtent_loss_func = losses.NTXentLoss(temperature=0.1)
            elif self.dml_type == "ProxyAnchor":
                self.automatic_optimization = False
                self.dml_loss_func_style = losses.ProxyAnchorLoss(num_classes=51, embedding_size=768, margin=0.1, alpha=32)
                self.dml_loss_func_content = losses.ProxyAnchorLoss(num_classes=7000, embedding_size=768, margin=0.1, alpha=32)
            else:
                raise ValueError(f"Unsupported CATFM dml_type: {self.dml_type}")

        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.dml_type == "ProxyAnchor":
            opt_model = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            opt_style = torch.optim.SGD(self.dml_loss_func_style.parameters(), lr=1e-2)
            opt_content = torch.optim.SGD(self.dml_loss_func_content.parameters(), lr=1e-2)
            return [opt_model, opt_style, opt_content]
        return super().configure_optimizers()

    def _target_padding(self, clip_target):
        if self.learnable_placeholder:
            return self.clip_target0.to(device=clip_target.device, dtype=clip_target.dtype).repeat(clip_target.shape[0], 1)
        if self.target_placeholder == "zero":
            return torch.zeros_like(clip_target)
        if self.target_placeholder == "gt":
            return clip_target
        raise ValueError(f"Unsupported target_placeholder: {self.target_placeholder}")

    def _extract_catfm_batch(self, batch, val=False):
        clip_style = batch[0][:, 0, :]
        clip_content = batch[0][:, 2, :]
        clip_target = batch[0][:, 1, :]
        if val or self.dml_type is None:
            return clip_style.float(), clip_content.float(), clip_target.float(), None, None
        if len(batch) < 4:
            raise ValueError("CATFM metric training requires data.params.include_labels=true.")
        return clip_style.float(), clip_content.float(), clip_target.float(), batch[2].long(), batch[3].long()

    def _build_catfm_inputs(self, clip_style, clip_content, clip_target):
        x_source = torch.cat([clip_content, clip_style], dim=1).unsqueeze(1)
        clip_target0 = self._target_padding(clip_target)
        x_target = torch.cat([clip_target0, clip_target], dim=1).unsqueeze(1)
        return x_source, x_target

    def _sample_training_path(self, x_target, x_source):
        bs, dev, dtype = x_target.shape[0], x_target.device, x_target.dtype
        t = torch.rand(bs, device=dev, dtype=dtype)
        xt = self.model.sample_xt(x=x_target, eps=x_source, t=t)
        ut = self.model.compute_conditional_flow(x0=x_source, x1=x_target)
        vt = self.model.forward(x=xt, t=t)
        fm_loss = (vt - ut).square().mean(dim=[*range(1, ut.ndim)])
        return t, xt, vt, fm_loss

    @staticmethod
    def _split_features(x):
        split = x.shape[-1] // 2
        return x[:, :, :split].squeeze(1), x[:, :, split:].squeeze(1)

    @staticmethod
    def _empty_predictions_like(x):
        split = x.shape[-1] // 2
        empty_features = torch.empty(0, split, device=x.device, dtype=x.dtype)
        empty_labels = torch.empty(0, device=x.device, dtype=torch.long)
        return empty_features, empty_features, empty_labels, empty_labels

    def _endpoint_prediction_loss(self, x_target, x_source, style_labels, content_labels):
        t, xt, vt, fm_loss = self._sample_training_path(x_target=x_target, x_source=x_source)

        if self.predict_x1:
            mask = t > self.t_conditional
            if not mask.any():
                return fm_loss, *self._empty_predictions_like(x_target)
            t_masked = pad_vector_like_x(t[mask], xt[mask])
            x1_hat = xt[mask] + vt[mask] * (1.0 - t_masked)
            content_pred, style_pred = self._split_features(x1_hat)
            return fm_loss, content_pred, style_pred, content_labels[mask], style_labels[mask]

        if self.predict_x0:
            mask = t > self.t_conditional
            if not mask.any():
                return fm_loss, *self._empty_predictions_like(x_target)
            t_masked = pad_vector_like_x(t[mask], xt[mask])
            x0_hat = xt[mask] - vt[mask] * t_masked
            content_pred, style_pred = self._split_features(x0_hat)
            return fm_loss, content_pred, style_pred, content_labels[mask], style_labels[mask]

        t = pad_vector_like_x(t, xt)
        x0_hat = xt - vt * t
        x1_hat = xt + vt * (1.0 - t)
        content_pred_x0, style_pred_x0 = self._split_features(x0_hat)
        content_pred_x1, style_pred_x1 = self._split_features(x1_hat)
        content_pred = torch.cat([content_pred_x0, content_pred_x1], dim=0)
        style_pred = torch.cat([style_pred_x0, style_pred_x1], dim=0)
        content_labels = torch.cat([content_labels, content_labels], dim=0)
        style_labels = torch.cat([style_labels, style_labels], dim=0)
        return fm_loss, content_pred, style_pred, content_labels, style_labels

    @staticmethod
    def _label_column(labels):
        if labels.ndim == 1:
            return labels
        return labels[:, 1]

    def _metric_losses(self, content_pred, style_pred, content_labels, style_labels):
        if content_pred.numel() == 0 or style_pred.numel() == 0:
            zero = content_pred.new_tensor(0.0)
            return zero, zero

        content_labels = self._label_column(content_labels)
        style_labels = self._label_column(style_labels)

        if self.dml_type == "MultiSimilarity":
            content_miner_output = self.dml_miner_content(content_pred, content_labels, content_pred, content_labels)
            style_miner_output = self.dml_miner_style(style_pred, style_labels, style_pred, style_labels)
            content_loss = self.dml_loss_func_content(content_pred, content_labels, content_miner_output)
            style_loss = self.dml_loss_func_style(style_pred, style_labels, style_miner_output)
        elif self.dml_type == "NXTent":
            content_miner_output = self.dml_miner(content_pred, content_labels, content_pred, content_labels)
            style_miner_output = self.dml_miner(style_pred, style_labels, style_pred, style_labels)
            content_loss = self.dml_loss_func(content_pred, content_labels, content_miner_output)
            style_loss = self.dml_loss_func(style_pred, style_labels, style_miner_output)
        elif self.dml_type == "Margin":
            content_miner_output = self.dml_miner(content_pred, content_labels, content_pred, content_labels)
            style_miner_output = self.dml_miner(style_pred, style_labels, style_pred, style_labels)
            content_loss = self.dml_loss_func(content_pred, content_labels, content_miner_output)
            style_loss = self.dml_loss_func(style_pred, style_labels, style_miner_output)
        elif self.dml_type == "both":
            content_miner_output = self.nxtent_miner(content_pred, content_labels, content_pred, content_labels)
            style_miner_output = self.multisim_miner(style_pred, style_labels, style_pred, style_labels)
            content_loss = self.nxtent_loss_func(content_pred, content_labels, content_miner_output)
            style_loss = self.multisim_loss_func(style_pred, style_labels, style_miner_output)
        elif self.dml_type == "ProxyAnchor":
            content_loss = self.dml_loss_func_content(content_pred, content_labels)
            style_loss = self.dml_loss_func_style(style_pred, style_labels)
        else:
            raise ValueError(f"Unsupported CATFM dml_type: {self.dml_type}")

        return content_loss, style_loss

    def _style_regularized_loss(self, x_source, x_target):
        bs, dev, dtype = x_target.shape[0], x_target.device, x_target.dtype
        t = torch.rand(bs, device=dev, dtype=dtype)
        xt = self.model.sample_xt(x=x_target, eps=x_source, t=t)
        ut = self.model.compute_conditional_flow(x0=x_source, x1=x_target)
        vt = self.model.forward(x=xt, t=t)
        loss_fm = (vt - ut).square().mean()
        x1_hat = xt + vt * pad_vector_like_x(1.0 - t, xt)
        loss_style = nn.functional.mse_loss(x1_hat[:, :, -768:], x_source[:, :, -768:])
        return loss_fm + self.style_weight * loss_style, loss_fm, loss_style

    def training_step(self, batch, batch_idx):
        clip_style, clip_content, clip_target, style_labels, content_labels = self._extract_catfm_batch(batch)
        x_source, x_target = self._build_catfm_inputs(clip_style, clip_content, clip_target)
        bs = x_source.shape[0]

        if self.dml_type is not None and (self.predict_x1 or self.predict_x0 or self.predict_x0x1):
            loss_fm, content_pred, style_pred, c_labels, s_labels = self._endpoint_prediction_loss(
                x_target=x_target,
                x_source=x_source,
                style_labels=style_labels,
                content_labels=content_labels,
            )
            content_loss, style_loss = self._metric_losses(content_pred, style_pred, c_labels, s_labels)
            loss = loss_fm.mean() + self.lambda_content * content_loss + self.lambda_style * style_loss

            if self.dml_type == "ProxyAnchor":
                optimizers = self.optimizers()
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                for opt in optimizers:
                    opt.step()

            self.log("train/loss_fm", loss_fm.mean(), on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("train/loss_dml_content", content_loss, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("train/loss_dml_style", style_loss, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        elif self.style_weight:
            loss, loss_fm, loss_style = self._style_regularized_loss(x_source, x_target)
            self.log("train/loss_fm", loss_fm, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("train/loss_style", loss_style * self.style_weight, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        else:
            loss = self.model.training_losses(x_source=x_source, x_target=x_target).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.ema_model.update()
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        clip_style, clip_content, clip_target1, _, _ = self._extract_catfm_batch(batch, val=True)
        x_source, x_target = self._build_catfm_inputs(clip_style, clip_content, clip_target1)
        clip_pred = self.predict_reverse(x_target) if self.reverse_inference else self.predict_forward(x_source)

        mse = nn.functional.mse_loss(clip_pred[:, :, -768:], x_target[:, :, -768:])
        content_mse = nn.functional.mse_loss(clip_pred[:, :, -768:], clip_content.unsqueeze(1))
        style_mse = nn.functional.mse_loss(clip_pred[:, :, -768:], clip_style.unsqueeze(1))
        self.log("val/mse", mse, sync_dist=True, batch_size=x_target.shape[0])
        self.log("val/content_mse", content_mse, sync_dist=True, batch_size=clip_content.shape[0])
        self.log("val/style_mse", style_mse, sync_dist=True, batch_size=clip_style.shape[0])
        torch.cuda.empty_cache()

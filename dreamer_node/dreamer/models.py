import copy

import torch
from torch import nn

import tools  # type: ignore
import networks  # type: ignore
from config import Config  # type: ignore


def to_np(x):
    return x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config: Config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.PRECISION == 16 else False
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self._config = config
        self.encoder = networks.MultiEncoder(shapes, **config.ENCODER)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.DYNAMIC_STOCH,
            config.DYNAMIC_DETER,
            config.DYNAMIC_HIDDEN,
            config.DYNAMIC_REC_DEPTH,
            config.DYNAMIC_DISCRETE,
            config.ACT,
            config.NORM,
            config.DYNAMIC_MEAN_ACT,
            config.DYNAMIC_STD_ACT,
            config.DYNAMIC_MIN_STD,
            config.UNIMIX_RATIO,
            config.INITIAL,
            config.num_actions,
            self.embed_size,
            config.DEVICE,
        )
        self.heads = nn.ModuleDict()
        if config.DYNAMIC_DISCRETE:
            feat_size = (
                config.DYNAMIC_STOCH * config.DYNAMIC_DISCRETE + config.DYNAMIC_DETER
            )
        else:
            feat_size = config.DYNAMIC_STOCH + config.DYNAMIC_DETER
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.DECODER
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.REWARD_HEAD["dist"] == "symlog_disc" else (),
            config.REWARD_HEAD["layers"],
            config.UNITS,
            config.ACT,
            config.NORM,
            dist=config.REWARD_HEAD["dist"],
            outscale=config.REWARD_HEAD["outscale"],
            device=config.DEVICE,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.CONTINUATION_HEAD["layers"],
            config.UNITS,
            config.ACT,
            config.NORM,
            dist="binary",
            outscale=config.CONTINUATION_HEAD["outscale"],
            device=config.DEVICE,
            name="Cont",
        )
        for name in config.GRAD_HEADS:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.MODEL_LR,
            config.OPT_EPS,
            config.GRAD_CLIP,
            config.WEIGHT_DECAY,
            opt=config.OPT,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.REWARD_HEAD["loss_scale"],
            cont=config.CONTINUATION_HEAD["loss_scale"],
        )

    def _train(self, data):
        data = self.preprocess(data)

        # Debug 1: Check initial data shapes
        # print("\n=== Data Shapes ===")
        # for key, value in data.items():
        #     print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

        with tools.RequiresGrad(self):
            with torch.amp.autocast(device_type="cuda", enabled=self._use_amp):
                embed = self.encoder(data)
                # Debug 2: Check encoder output
                # print("\n=== Encoder Output ===")
                # print(f"embed shape: {embed.shape}")

                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                # Debug 3: Check dynamics outputs
                # print("\n=== Dynamics Outputs ===")
                # print(f"post shapes: { {k: v.shape for k, v in post.items()} }")
                # print(f"prior shapes: { {k: v.shape for k, v in prior.items()} }")

                kl_free = self._config.KL_FREE
                dyn_scale = self._config.DYNAMIC_SCALE
                rep_scale = self._config.REP_SCALE
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                preds = {}
                feat = self.dynamics.get_feat(post)
                # Debug 4: Check features
                # print("\n=== Feature Vector ===")
                # print(f"feat shape: {feat.shape}")

                for name, head in self.heads.items():
                    grad_head = name in self._config.GRAD_HEADS
                    current_feat = feat if grad_head else feat.detach()
                    pred = head(current_feat)

                    # Debug 5: Check prediction outputs
                    # print(f"\n=== Prediction Head '{name}' ===")
                    # if isinstance(pred, dict):
                    #     for k, v in pred.items():
                    #         if hasattr(v, "shape"):
                    #             print(f"{k} shape: {v.shape}")
                    #         else:
                    #             print(f"{k} is distribution with params:")
                    #             for param in v.__dict__:
                    #                 if isinstance(v.__dict__[param], torch.Tensor):
                    #                     print(f"  {param}: {v.__dict__[param].shape}")
                    # else:
                    #     if hasattr(pred, "shape"):
                    #         print(f"pred shape: {pred.shape}")
                    #     else:
                    #         print("pred is distribution with parameters:")
                    #         for param in pred.__dict__:
                    #             if isinstance(pred.__dict__[param], torch.Tensor):
                    #                 print(f"  {param}: {pred.__dict__[param].shape}")

                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred

                losses = {}
                for name, pred in preds.items():
                    # Debug 6: Final check before log_prob
                    # print(f"\n=== Calculating Loss for '{name}' ===")

                    # Handle distribution objects properly
                    # if hasattr(
                    #     pred, "base_dist"
                    # ):  # For Independent/Normal distributions
                    #     print("Prediction Distribution Parameters:")
                    #     if hasattr(pred.base_dist, "mean"):
                    #         print(f"  mean shape: {pred.base_dist.mean.shape}")
                    #     if hasattr(pred.base_dist, "logits"):
                    #         print(f"  logits shape: {pred.base_dist.logits.shape}")
                    #     if hasattr(pred, "event_shape"):
                    #         print(f"  event_shape: {pred.event_shape}")
                    # elif hasattr(pred, "logits"):  # For Discrete distributions
                    #     print(f"  logits shape: {pred.logits.shape}")

                    # # Print target data info
                    # print(f"Target '{name}' shape: {data[name].shape}")
                    # print(f"Target dtype: {data[name].dtype}")

                    # # For debugging only - remove after use
                    # if name == "decoder":
                    #     print("\nDEBUG - First 5 lidar targets:")
                    #     print(data["lidar"][0, 0, :5].cpu().numpy())
                    #     if hasattr(pred, "base_dist") and hasattr(
                    #         pred.base_dist, "mean"
                    #     ):
                    #         print("First 5 predicted means:")
                    #         print(pred.base_dist.mean[0, 0, :5].cpu().numpy())
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast(device_type="cuda", enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        # Convert all values to float32 tensors
        obs = {
            k: torch.tensor(v, device=self._config.DEVICE, dtype=torch.float32)
            for k, v in obs.items()
        }

        # Normalize image
        obs["image"] = obs["image"] / 255.0

        # Handle discount factor
        if "discount" in obs:
            obs["discount"] *= self._config.DISCOUNT
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)

        # Combine motor/steering into action if needed
        if "action" not in obs and all(k in obs for k in ["motor", "steering"]):
            # Stack motor and steering along last dimension
            obs["action"] = torch.stack(
                [
                    obs["motor"].squeeze(-1),  # Remove extra dim if present
                    obs["steering"].squeeze(-1),
                ],
                dim=-1,
            )

            # Ensure proper shape: (batch_size, seq_len, 2)
            if obs["action"].ndim == 2:  # If missing sequence dimension
                obs["action"] = obs["action"].unsqueeze(1)

        # Validate action shape
        # if "action" in obs:
        #     assert obs["action"].ndim == 3, (
        #         f"Action should be 3D (B,T,D), got {obs['action'].shape}"
        #     )
        #     assert obs["action"].shape[-1] == 2, (
        #         f"Action dim should be 2 (motor+steering), got {obs['action'].shape[-1]}"
        #     )

        # Required flags
        assert "is_first" in obs, "Missing is_first key"
        assert "is_terminal" in obs, "Missing is_terminal key"

        # Continuation signal
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)

        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config: Config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.PRECISION == 16 else False
        self._config = config
        self._world_model = world_model
        if config.DYNAMIC_DISCRETE:
            feat_size = (
                config.DYNAMIC_STOCH * config.DYNAMIC_DISCRETE + config.DYNAMIC_DETER
            )
        else:
            feat_size = config.DYNAMIC_STOCH + config.DYNAMIC_DETER
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.ACTOR["layers"],
            config.UNITS,
            config.ACT,
            config.NORM,
            config.ACTOR["dist"],
            config.ACTOR["std"],
            config.ACTOR["min_std"],
            config.ACTOR["max_std"],
            absmax=1.0,
            temp=config.ACTOR["temp"],
            unimix_ratio=config.ACTOR["unimix_ratio"],
            outscale=config.ACTOR["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.CRITIC["dist"] == "symlog_disc" else (),
            config.CRITIC["layers"],
            config.UNITS,
            config.ACT,
            config.NORM,
            config.CRITIC["dist"],
            outscale=config.CRITIC["outscale"],
            device=config.DEVICE,
            name="Value",
        )
        if config.CRITIC["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.WEIGHT_DECAY, opt=config.OPT, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.ACTOR["lr"],
            config.ACTOR["eps"],
            config.ACTOR["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.CRITIC["lr"],
            config.CRITIC["eps"],
            config.CRITIC["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.REWARD_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.DEVICE)
            )
            self.reward_ema = RewardEMA(device=self._config.DEVICE)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast(device_type="cuda", enabled=self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.IMAGE_HORIZON
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.ACTOR["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast(device_type="cuda", enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.CRITIC["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.ACTOR["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def flatten(x):
            return x.reshape([-1] + list(x.shape[2:]))

        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.DISCOUNT * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.DISCOUNT * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.DISCOUNT_LAMBDA,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.REWARD_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.IMAGE_GRADIENT == "dynamics":
            actor_target = adv
        elif self._config.IMAGE_GRADIENT == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.IMAGE_GRADIENT == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.IMAGE_GRADIENT_MIX
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.IMAGE_GRADIENT)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.CRITIC["slow_target"]:
            if self._updates % self._config.CRITIC["slow_target_update"] == 0:
                mix = self._config.CRITIC["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

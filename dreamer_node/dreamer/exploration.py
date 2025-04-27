import gymnasium.spaces
from typing import Callable, Any

import torch
from torch import nn
from torch import distributions as torchd

import tools  # type: ignore
import models  # type: ignore
import networks  # type: ignore
from config import Config  # type: ignore


class Random(nn.Module):
    def __init__(self, config: Config, act_space: gymnasium.spaces.Dict):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.ACTOR["dist"] == "onehot":
            return tools.OneHotDist(
                torch.zeros(
                    self._config.num_actions, device=self._config.DEVICE
                ).repeat(self._config.ENVIRONMENT_COUNT, 1)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(
                        self._act_space.low, device=self._config.DEVICE
                    ).repeat(self._config.ENVIRONMENT_COUNT, 1),
                    torch.tensor(
                        self._act_space.high, device=self._config.DEVICE
                    ).repeat(self._config.ENVIRONMENT_COUNT, 1),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


class Plan2Explore(nn.Module):
    def __init__(
        self,
        config: Config,
        world_model: models.WorldModel,
        reward: Callable[[Any, Any, Any], torch.Tensor],
    ):
        super(Plan2Explore, self).__init__()
        self._config = config
        self._use_amp = True if config.PRECISION == 16 else False
        self._reward = reward
        self._behavior = models.ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        if config.DYNAMIC_DISCRETE:
            feat_size = (
                config.DYNAMIC_STOCH * config.DYNAMIC_DISCRETE + config.DYNAMIC_DETER
            )
            stoch = config.DYNAMIC_STOCH * config.DYNAMIC_DISCRETE
        else:
            feat_size = config.DYNAMIC_STOCH + config.DYNAMIC_DETER
            stoch = config.DYNAMIC_STOCH
        size: int = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.DYNAMIC_DETER,
            "feat": config.DYNAMIC_STOCH + config.DYNAMIC_DETER,
        }[self._config.DISAGREE_TARGET]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.DISAGREE_ACTION_CONDITION else 0
            ),  # pytorch version
            shape=size,
            layers=config.DISAGREE_LAYERS,
            units=config.DISAGREE_UNITS,
            act=config.ACT,
        )
        self._networks = nn.ModuleList(
            [networks.MLP(**kw) for _ in range(config.DISAGREE_MODELS)]
        )
        kw = dict(wd=config.WEIGHT_DECAY, opt=config.OPT, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.MODEL_LR,
            config.OPT_EPS,
            config.GRAD_CLIP,
            **kw,
        )

    def train(self, start, context, data):
        with tools.RequiresGrad(self._networks):
            metrics = {}
            stoch: torch.Tensor = start["stoch"]
            if self._config.DYNAMIC_DISCRETE:
                stoch = torch.reshape(
                    stoch, (stoch.shape[:-2] + ((stoch.shape[-2] * stoch.shape[-1]),))
                )
            target = {
                "embed": context["embed"],
                "stoch": stoch,
                "deter": start["deter"],
                "feat": context["feat"],
            }[self._config.DISAGREE_TARGET]
            inputs = context["feat"]
            if self._config.DISAGREE_ACTION_CONDITION:
                inputs = torch.concat(
                    [inputs, torch.tensor(data["action"], device=self._config.DEVICE)],
                    -1,
                )
            metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self._behavior._train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        inputs = feat
        if self._config.DISAGREE_ACTION_CONDITION:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.DISAGREE_LOG:
            disag = torch.log(disag)
        reward = self._config.EXPLORATION_INTRO_SCALE * disag
        if self._config.EXPLORATION_EXTRA_SCALE:
            reward += self._config.EXPLORATION_EXTRA_SCALE * self._reward(
                feat, state, action
            )
        return reward

    def _train_ensemble(self, inputs, targets):
        with torch.amp.autocast(device_type="cuda", enabled=self._use_amp):
            if self._config.DISAGREE_OFFSET:
                targets = targets[:, self._config.DISAGREE_OFFSET :]
                inputs = inputs[:, : -self._config.DISAGREE_OFFSET]
            targets = targets.detach()
            inputs = inputs.detach()
            preds = [head(inputs) for head in self._networks]
            likes = torch.cat(
                [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
            )
            loss = -torch.mean(likes)
        metrics = self._expl_opt(loss, self._networks.parameters())
        return metrics

from functools import partial
from typing import Any, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class DeepLabHead(nn.Module):
    num_classes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, low_level_features, train: bool = False):
        conv = partial(nn.Conv, dtype=self.dtype, use_bias=False)
        norm = partial(nn.BatchNorm, dtype=self.dtype, use_running_average=not train)

        if low_level_features is not None:
            low_level_features = conv(48, (1, 1))(low_level_features)
            low_level_features = norm()(low_level_features)
            low_level_features = nn.relu(low_level_features)

        x = ASPP([12, 24, 36], conv, norm, name='ASPP')(inputs, train=train)
        x = jax.image.resize(x, low_level_features.shape, "bilinear")
        x = jnp.concatenate([x, low_level_features], axis=-1)
        x = conv(256, (3, 3), padding='SAME')(x)
        x = norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x, deterministic=not train)

        x = conv(256, (3, 3), padding='SAME')(x)
        x = norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(0.1)(x, deterministic=not train)

        x = nn.Conv(self.num_classes, (1, 1), padding='VALID', use_bias=True, dtype=self.dtype)(x)
        return x


class ASPPConv(nn.Module):
    channels: int
    dilation: int
    conv: Any
    norm: Any

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        _d = max(1, self.dilation)
        x = jnp.pad(inputs, [(0, 0), (_d, _d), (_d, _d), (0, 0)], 'constant', constant_values=0)
        x = self.conv(self.channels, (3, 3), padding='VALID', kernel_dilation=(_d, _d))(x)
        x = self.norm()(x)
        x = nn.relu(x)
        return x


class ASPPPooling(nn.Module):
    channels: int
    conv: Any
    norm: Any

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        in_shape = jnp.shape(inputs)[1:-1]
        x = nn.avg_pool(inputs, in_shape)
        x = self.conv(self.channels, (1, 1), padding='SAME')(x)
        x = self.norm()(x)
        x = nn.relu(x)

        out_shape = (*inputs.shape[:-1], self.channels)
        x = jax.image.resize(x, shape=out_shape, method='bilinear')

        return x


class ASPP(nn.Module):
    atrous_rates: Sequence
    conv: Any
    norm: Any

    channels: int = 256

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        res = []

        x = self.conv(self.channels, (1, 1), padding='VALID')(inputs)
        x = self.norm()(x)
        res.append(nn.relu(x))

        for i, rate in enumerate(self.atrous_rates):
            res.append(ASPPConv(self.channels, rate, self.conv, self.norm, name=f'ASPPConv{i + 1}')(inputs, train))

        res.append(ASPPPooling(self.channels, self.conv, self.norm, name='ASPPPooling')(inputs, train))
        x = jnp.concatenate(res, -1)  # 1280

        x = self.conv(self.channels, (1, 1), padding='VALID')(x)
        x = self.norm()(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5)(x, deterministic=not train)

        return x

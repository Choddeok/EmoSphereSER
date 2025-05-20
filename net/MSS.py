import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0, spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_model, 0.5), dropout=dropout
        )

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_x, -1)
        )  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        spectral_norm=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2 * out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        spectral_norm=False,
    ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)

        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(self, *args, **kwargs):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = 1024
        self.hidden_dim = 1024
        self.out_dim = 1024
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
        feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int()
        return feat_lens

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask):
        feat_lens = self.compute_length_from_mask(mask)
        mask = get_mask_from_lengths(feat_lens)
        x = x.transpose(1, 2)
        _, _, L = x.shape
        mask = mask[:, :L]
        # print(x)
        max_len = L
        mask = ~mask
        slf_attn_mask = (
            mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        )
        # spectral
        x = x.transpose(1, 2)
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        # print(x)
        if mask is not None:
            # print(x.shape, mask.shape)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        # x = x.transpose(1, 2)
        # for conv_block in self.convnext:
        #     x = conv_block(x, mask.unsqueeze(1))
        # x = x.transpose(1, 2)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # print(x)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)
        return w

class MelStyleEncoder_dot5(nn.Module):
    """MelStyleEncoder"""

    def __init__(self, *args, **kwargs):
        super(MelStyleEncoder_dot5, self).__init__()
        self.in_dim = 1024
        self.hidden_dim = 1024
        self.out_dim = 1024
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.5

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
        feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int()
        return feat_lens

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask):
        feat_lens = self.compute_length_from_mask(mask)
        mask = get_mask_from_lengths(feat_lens)
        x = x.transpose(1, 2)
        _, _, L = x.shape
        mask = mask[:, :L]
        # print(x)
        max_len = L
        mask = ~mask
        slf_attn_mask = (
            mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        )
        # spectral
        x = x.transpose(1, 2)
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        # print(x)
        if mask is not None:
            # print(x.shape, mask.shape)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        # x = x.transpose(1, 2)
        # for conv_block in self.convnext:
        #     x = conv_block(x, mask.unsqueeze(1))
        # x = x.transpose(1, 2)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # print(x)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)
        return w

class MelStyleEncoder_w2v(nn.Module):
    """MelStyleEncoder"""

    def __init__(self, *args, **kwargs):
        super(MelStyleEncoder_w2v, self).__init__()
        self.in_dim = 768
        self.hidden_dim = 1024
        self.out_dim = 1024
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.5

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
        feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int()
        return feat_lens

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask):
        feat_lens = self.compute_length_from_mask(mask)
        mask = get_mask_from_lengths(feat_lens)
        x = x.transpose(1, 2)
        _, _, L = x.shape
        mask = mask[:, :L]
        # print(x)
        max_len = L
        mask = ~mask
        slf_attn_mask = (
            mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        )
        # spectral
        x = x.transpose(1, 2)
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        # print(x)
        if mask is not None:
            # print(x.shape, mask.shape)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        # x = x.transpose(1, 2)
        # for conv_block in self.convnext:
        #     x = conv_block(x, mask.unsqueeze(1))
        # x = x.transpose(1, 2)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # print(x)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)
        return w

class MelStyleEncoder_oct(nn.Module):
    """MelStyleEncoder"""

    def __init__(self, *args, **kwargs):
        super(MelStyleEncoder_oct, self).__init__()
        self.in_dim = 1024
        self.hidden_dim = 256
        self.out_dim = 128
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.5

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )
        self.slf_attn = MultiHeadAttention(
            self.n_head,
            self.hidden_dim,
            self.hidden_dim // self.n_head,
            self.hidden_dim // self.n_head,
            self.dropout,
        )

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1)  # (batch_size, )
        feat_lens = torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int()
        return feat_lens

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask):
        feat_lens = self.compute_length_from_mask(mask)
        mask = get_mask_from_lengths(feat_lens)
        x = x.transpose(1, 2)
        _, _, L = x.shape
        mask = mask[:, :L]
        # print(x)
        max_len = L
        mask = ~mask
        slf_attn_mask = (
            mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        )
        # spectral
        x = x.transpose(1, 2)
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        # print(x)
        if mask is not None:
            # print(x.shape, mask.shape)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        # x = x.transpose(1, 2)
        # for conv_block in self.convnext:
        #     x = conv_block(x, mask.unsqueeze(1))
        # x = x.transpose(1, 2)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # print(x)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)
        return w

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, p_drop=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    """
    与你原实现一致：上采样×2 + 1×1降通道 (channel -> channel//2)
    """
    def __init__(self, channel):
        super().__init__()
        self.reduce = nn.Conv2d(channel, channel // 2, 1, 1, bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.reduce(x)  # (B, channel//2, H*2, W*2)

class ECA1D(nn.Module):
    """
    对输入特征做通道注意力校准：
    GAP -> (B, C) -> Conv1D(k) -> sigmoid -> (B, C) -> 逐通道缩放
    k = |log2(C)/gamma + b|_{odd}
    """
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        k = int(abs(math.log2(channels) / gamma + b))
        k = k if k % 2 == 1 else k + 1
        k = max(1, k)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)    # (B, C)
        y = y.unsqueeze(1)                                         # (B, 1, C)
        y = self.conv(y)                                           # (B, 1, C)
        s = torch.sigmoid(y).squeeze(1)                            # (B, C)
        return x * s.unsqueeze(-1).unsqueeze(-1)                   


class ECAMSF(nn.Module):
    def __init__(self, channels: int, n_aux: int, act='leaky_relu'):
        """
        channels: 每张特征图的通道 C
        n_aux: 辅助特征个数（不含 key）
        """
        super().__init__()
        self.eca_aux = nn.ModuleList([ECA1D(channels) for _ in range(n_aux)])
        self.compress = nn.Conv2d(n_aux * channels, channels, kernel_size=1, bias=True)
        if act == 'prelu':
            self.act = nn.PReLU(channels)
        else:
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, key: torch.Tensor, aux_list: list[torch.Tensor]) -> torch.Tensor:
        """
        key:   X0, (B, C, H, W) —— 关键/主特征（不经 ECA）
        aux_list: [X1, X2, ..., Xn]，每个 (B, C, H, W)
        return: ξ[X] (B, C, H, W)
        """
        assert len(aux_list) == len(self.eca_aux), "aux_list 个数需与 n_aux 对齐"
        aux_refined = []
        for x, e in zip(aux_list, self.eca_aux):
            aux_refined.append(e(x))  # Xi' = ECA(Xi)

        if len(aux_refined) > 0:
            z = torch.cat(aux_refined, dim=1)      # (B, n_aux*C, H, W)
            z = self.compress(z)                   # 1x1 -> (B, C, H, W)
            z = self.act(z)                        # φ
            out = key + z                          # 残差融合
        else:
            out = key
        return out

class FinalFusion(nn.Module):
    def __init__(self, channels: int, act='leaky_relu'):
        super().__init__()
        self.eca_Y = ECA1D(channels)
        self.eca_U = ECA1D(channels)
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True)
        if act == 'prelu':
            self.act = nn.PReLU(channels)
        else:
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, Y: torch.Tensor, U: torch.Tensor, Xi_hat: torch.Tensor) -> torch.Tensor:
        """
        Y: P-AM（主）特征 (B, C, H, W)
        U: 上采样来的特征 (B, C, H, W)
        Xi_hat: ξ[X] (来自 7 个 sub-AM 的融合) (B, C, H, W)
        """
        y_ = self.eca_Y(Y)     # ξ[Y]
        u_ = self.eca_U(U)     # ξ[U]
        cat = torch.cat([y_, u_, Xi_hat], dim=1)  # (B, 3C, H, W)
        z = self.fuse(cat)                         # (B, C, H, W)
        z = self.act(z)                            # φ
        F_out = Y + z                               # 残差到 Y
        return F_out

class Encoder2D(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, p_drop_bottleneck=0.3):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16
        self.c1 = Conv_Block(in_ch, c1)
        self.d1 = DownSample(c1)
        self.c2 = Conv_Block(c1, c2)
        self.d2 = DownSample(c2)
        self.c3 = Conv_Block(c2, c3)
        self.d3 = DownSample(c3)
        self.c4 = Conv_Block(c3, c4)
        self.d4 = DownSample(c4)
        self.c5 = Conv_Block(c4, c5, p_drop=p_drop_bottleneck)

        self.channels = (c1, c2, c3, c4, c5)

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))
        return [r1, r2, r3, r4, r5]


# -----------------------------
# ASM-Net鱼骨式：P-AM 主干 + 7 个 sub-AM
# -----------------------------
class ASMNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base: int = 64):
        super().__init__()
        self.n_sub = 7  # 7 个 sub-AM（0°,+-10°, +-30°, +-45°），其中 index=0 为 key

        # 编码器：P-AM 与 sub-AM 共享结构但**不共享参数**
        self.enc_p = Encoder2D(in_channels, base=base)
        self.enc_s = Encoder2D(in_channels, base=base)

        c1, c2, c3, c4, c5 = self.enc_p.channels

        # 上采样（按照原 U-Net 的 channel 关系）
        self.up1 = UpSample(c5)  # 1024 -> 512（举例）
        self.up2 = UpSample(c4)  # 512 -> 256
        self.up3 = UpSample(c3)  # 256 -> 128
        self.up4 = UpSample(c2)  # 128 -> 64

        self.msf4 = ECAMSF(c4, n_aux=self.n_sub-1)  # 层4使用 c4 通道
        self.fus4 = FinalFusion(c4)
        self.dec4 = Conv_Block(c4, c4)

        self.msf3 = ECAMSF(c3, n_aux=self.n_sub-1)
        self.fus3 = FinalFusion(c3)
        self.dec3 = Conv_Block(c3, c3)

        self.msf2 = ECAMSF(c2, n_aux=self.n_sub-1)
        self.fus2 = FinalFusion(c2)
        self.dec2 = Conv_Block(c2, c2)

        self.msf1 = ECAMSF(c1, n_aux=self.n_sub-1)
        self.fus1 = FinalFusion(c1)
        self.dec1 = Conv_Block(c1, c1)

        # 输出头
        self.head = nn.Conv2d(c1, num_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def _split_sub_feats(self, subs_feats, level_idx):
        """
        将 7 个 sub-AM 在指定层（level_idx: 0..4）的特征拆分为 key + aux 列表
        subs_feats: list(len=7) 的列表，每个是 [r1, r2, r3, r4, r5]
        """
        key = subs_feats[0][level_idx]              # 0° 作为关键子视角
        aux = [subs_feats[i][level_idx] for i in range(1, self.n_sub)]  # 其余 6 个为辅助
        return key, aux

    def forward(self, pam: torch.Tensor, subs: list[torch.Tensor]) -> torch.Tensor:
        """
        pam:  (B, C_in, H, W)
        subs: 长度为 7 的 list/tuple，每个 (B, C_in, H, W)，subs[0] 为 0°
        """
        assert isinstance(subs, (list, tuple)) and len(subs) == self.n_sub, "subs 需为长度 7 的列表/元组，且 subs[0] 为 0°"

        # 编码：P-AM 一条；sub-AM 用共享编码器 enc_s 分别提特征
        p_feats = self.enc_p(pam)            # [p1, p2, p3, p4, p5]
        s_feats = [self.enc_s(x) for x in subs]  # 7 组，每组 [s1, s2, s3, s4, s5]

        # 解码 4：使用 p4, 上采 U1, 以及 7 个 sub 的第 4 层做 ξ[X] 与最终融合
        U1 = self.up1(p_feats[4])            # (B, c4, H/8, W/8)  —— 假设原图 /16 到 /8
        X_key4, X_aux4 = self._split_sub_feats(s_feats, level_idx=3)
        Xi4 = self.msf4(X_key4, X_aux4)      # ξ[X]_4
        F4  = self.fus4(p_feats[3], U1, Xi4) # F = Y + φ(1x1(ξ[Y]; ξ[U]; ξ[X]))
        D4  = self.dec4(F4)                  # 层 4 解码结果

        # 解码 3
        U2 = self.up2(D4)                    # (B, c3, H/4, W/4)
        X_key3, X_aux3 = self._split_sub_feats(s_feats, level_idx=2)
        Xi3 = self.msf3(X_key3, X_aux3)
        F3  = self.fus3(p_feats[2], U2, Xi3)
        D3  = self.dec3(F3)

        # 解码 2
        U3 = self.up3(D3)                    # (B, c2, H/2, W/2)
        X_key2, X_aux2 = self._split_sub_feats(s_feats, level_idx=1)
        Xi2 = self.msf2(X_key2, X_aux2)
        F2  = self.fus2(p_feats[1], U3, Xi2)
        D2  = self.dec2(F2)

        # 解码 1
        U4 = self.up4(D2)                    # (B, c1, H, W)
        X_key1, X_aux1 = self._split_sub_feats(s_feats, level_idx=0)
        Xi1 = self.msf1(X_key1, X_aux1)
        F1  = self.fus1(p_feats[0], U4, Xi1)
        D1  = self.dec1(F1)

        logits = self.head(D1)               # (B, num_classes, H, W)
        return logits


if __name__ == "__main__":
    B, C_in, H, W = 1, 3, 256, 256
    pam  = torch.randn(B, C_in, H, W)
    subs = [torch.randn(B, C_in, H, W) for _ in range(7)]  # 0°,+-10°, +-30°, +-45° ; subs[0] 为 key

    net = ASMNet2D(in_channels=C_in, num_classes=3, base=32)  # base=32 以节省显存
    y = net(pam, subs)
    print("logits:", y.shape)  # 期望: (B, 3, H, W)

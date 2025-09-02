import os
import sys
import math
import csv
import time
import argparse
import loss
from pathlib import Path
from typing import List, Tuple

import types
sys.modules.setdefault('net_inception', types.ModuleType('net_inception'))  # 避免 loss.py 中的导入报错

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import importlib.util

def load_asmnet(module_path: str):
    spec = importlib.util.spec_from_file_location("asm_net_module", module_path)
    asm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asm_mod)
    return asm_mod

class AMDataset(Dataset):
    """
    每条样本包含：
      pam    : 主视角 P-AM
      subs   : 7 个子视角（顺序：0°, +10, -10, +30, -30, +45, -45）
      target : GT（3通道）
      defect : 可选；若缺省则使用 pam 作为 defect
    图像将被缩放到 args.image_size，转换到 [0,1] 张量 (C,H,W)
    """
    def __init__(self, list_file: str, image_size: int = 256):
        super().__init__()
        self.items = []
        with open(list_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=None)
            for line in reader:
                if len(line) == 1:
                    # 兼容以空格分隔的 txt
                    line = line[0].strip().split()
                line = [s.strip() for s in line if len(s.strip()) > 0]
                if len(line) not in (9+1, 9+2):  # pam + 7subs + target [+ defect]
                    raise ValueError(f"Line expects 10 or 11 paths, got {len(line)}: {line}")
                pam = line[0]
                subs = line[1:8]
                target = line[8]
                defect = line[9] if len(line) == 10+1 else None
                self.items.append((pam, subs, target, defect))
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # -> [0,1]
        ])

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.tf(img)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pam, subs, target, defect = self.items[idx]
        pam_t = self._load_rgb(pam)             # (3,H,W)
        subs_t = [self._load_rgb(p) for p in subs]  # list of 7×(3,H,W)
        target_t = self._load_rgb(target)
        defect_t = self._load_rgb(defect) if defect is not None else pam_t.clone()
        return pam_t, torch.stack(subs_t, dim=0), target_t, defect_t   # subs shape: (7,3,H,W)


# -----------------------------
# 训练与验证
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for step, (pam, subs, target, defect) in enumerate(loader, 1):
        pam = pam.to(device, non_blocking=True)
        subs = subs.to(device, non_blocking=True)          # (B,7,3,H,W)
        target = target.to(device, non_blocking=True)
        defect = defect.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            # 模型期望：pam: (B,3,H,W); subs: list长度7，每个 (B,3,H,W)
            subs_list = [subs[:, i, ...] for i in range(subs.shape[1])]
            logits = model(pam, subs_list)                 # (B,3,H,W)
            pred = torch.sigmoid(logits)                   # 映射到 [0,1] 与损失对齐

            # 使用 loss.py 的复合损失（内部含缺损区域mask/SSIM/Charbonnier/梯度等）
            loss_total = loss.combined_loss(defect, pred, target)

        if args.amp:
            scaler.scale(loss_total).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        batch = pam.size(0)
        total_loss += loss_total.item() * batch
        n_samples += batch

        if step % args.log_interval == 0:
            print(f"[Train] Step {step:05d}/{len(loader):05d} | Loss {loss_total.item():.4f}")

    return total_loss / max(1, n_samples)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for step, (pam, subs, target, defect) in enumerate(loader, 1):
        pam = pam.to(device, non_blocking=True)
        subs = subs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        defect = defect.to(device, non_blocking=True)

        subs_list = [subs[:, i, ...] for i in range(subs.shape[1])]
        logits = model(pam, subs_list)
        pred = torch.sigmoid(logits)
        loss_total = loss.combined_loss(defect, pred, target)

        batch = pam.size(0)
        total_loss += loss_total.item() * batch
        n_samples += batch
    return total_loss / max(1, n_samples)


def main():
    parser = argparse.ArgumentParser(description="ASM-Net Training")
    parser.add_argument("--asmnet_path", type=str, default="ASM-Net.py", help="Path to ASM-Net.py")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--base", type=int, default=64)

    parser.add_argument("--train_list", type=str, required=True, help="Train list file (txt/csv)")
    parser.add_argument("--val_list", type=str, required=True, help="Val list file (txt/csv)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 数据
    train_set = AMDataset(args.train_list, image_size=args.image_size)
    val_set = AMDataset(args.val_list, image_size=args.image_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # 模型（从 ASM-Net.py 加载 ASMNet2D）  —— 文件结构详见你上传的脚本
    asm_mod = load_asmnet(args.asmnet_path)  # 提供 ASMNet2D 类  :contentReference[oaicite:3]{index=3}
    model = asm_mod.ASMNet2D(in_channels=args.in_channels, num_classes=args.num_classes, base=args.base)
    model = model.to(device)

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_val = math.inf

    # 断点续训
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"=> Resumed from {args.resume} (epoch {start_epoch})")

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss = validate(model, val_loader, device, args)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
              f"train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e} | {dt:.1f}s")

        # 保存最新
        save_path = Path(args.out_dir) / "last.pth"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val
        }, save_path)

        # 保存最优
        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(args.out_dir) / "best.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val
            }, best_path)
            print(f"  ✓ New best: {best_val:.4f} -> saved to {best_path}")

    print("Training done.")


if __name__ == "__main__":
    main()

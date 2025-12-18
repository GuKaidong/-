# -*- coding: utf-8 -*-
"""
src/train_improved_memmap.py

用 memmap 数据集训练 SegRNNGrid，避免一次性加载 npz 到内存。

输入：
  --data_dir 指向 build_grid_memmap_dataset.py 输出目录
  --use_norm_X 读取 *_X_norm.npy，否则读 *_X.npy

说明：
- 任务：分类预测未来每个 block（m 个）的网格索引
- segment_w > 1 时：
  - 模型输出 logits: (B, m, num_grids)
  - 标签需要从原始 y (B, H) 映射到 block 级别 (B, m)
  - 这里采用 “每个 block 取最后一步” 作为监督
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
# 适配你的工程结构：experiment 根目录下有 segrnn_model/
sys.path.append(r"F:\研究生毕业论文\experiment")
from segrnn_model.segrnn_model_grid import SegRNNGrid


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GridMemmapDataset(Dataset):
    """按需从 memmap 读取样本，避免占用大量内存。"""
    def __init__(self, X_path, Y_path, N, L, C, H, dtypeX=np.float32, dtypeY=np.int32):
        self.N = int(N)
        self.L = int(L)
        self.C = int(C)
        self.H = int(H)
        self.X = np.memmap(X_path, dtype=dtypeX, mode="r", shape=(self.N, self.L, self.C))
        self.Y = np.memmap(Y_path, dtype=dtypeY, mode="r", shape=(self.N, self.H))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = np.array(self.X[idx], copy=False)  # (L,C)
        y = np.array(self.Y[idx], copy=False)  # (H,)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))


def compute_class_weights_from_memmap(
    Y_path, N, H, num_classes,
    eps=1e-6,
    scheme="inv_sqrt",
    cap_min=0.1
):
    """
    从 memmap 的 Y 流式统计类别频次，构造稳定权重：
    - inv: 1/count
    - inv_sqrt: 1/sqrt(count)
    - 最后归一化到 mean=1，并做下限截断 cap_min
    """
    Y = np.memmap(Y_path, dtype=np.int32, mode="r", shape=(N, H))
    counts = np.zeros((num_classes,), dtype=np.int64)
    bs = 4096
    for i in range(0, N, bs):
        j = min(N, i + bs)
        flat = np.array(Y[i:j], copy=False).reshape(-1)
        bc = np.bincount(flat, minlength=num_classes)
        counts += bc.astype(np.int64)

    counts = counts.astype(np.float64) + eps
    if scheme == "inv":
        w = 1.0 / counts
    elif scheme == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        w = np.ones_like(counts)

    w = w / np.mean(w)
    w = np.maximum(w, cap_min)
    return torch.from_numpy(w.astype(np.float32))


def topk_accuracy_from_logits(logits, targets, k=5):
    """
    logits: (B,m,num_grids)
    targets:(B,m)
    """
    topk = logits.topk(k, dim=-1).indices         # (B,m,k)
    targets_exp = targets.unsqueeze(-1).expand_as(topk)
    correct = (topk == targets_exp).any(dim=-1).float()  # (B,m)
    return correct.mean().item()


def _make_block_targets(yb: torch.Tensor, segment_w: int) -> torch.Tensor:
    """
    yb: (B,H)
    返回 y_blk: (B,m)
    """
    B, H = yb.shape
    assert H % segment_w == 0, f"H={H} 必须能整除 segment_w={segment_w}"
    m = H // segment_w
    if segment_w == 1:
        return yb
    # 每个 block 取最后一步作为监督
    return yb.view(B, m, segment_w)[:, :, -1]


def train_epoch(model, loader, optimizer, criterion, device, num_grids, segment_w):
    model.train()
    running_loss = 0.0
    iters = 0
    for xb, yb in loader:
        xb = xb.to(device)  # (B,L,C)
        yb = yb.to(device)  # (B,H)

        B, H = yb.shape
        m = H // segment_w
        logits = model(xb, target_m=m)  # (B,m,num_grids)

        y_blk = _make_block_targets(yb, segment_w)  # (B,m)
        loss = criterion(logits.reshape(-1, num_grids), y_blk.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += float(loss.item())
        iters += 1
    return running_loss / max(1, iters)


@torch.no_grad()
def val_epoch(model, loader, device, num_grids, segment_w):
    model.eval()
    correct = 0
    total = 0
    top1_sum = 0.0
    top5_sum = 0.0
    batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        B, H = yb.shape
        m = H // segment_w
        logits = model(xb, target_m=m)     # (B,m,num_grids)
        preds = logits.argmax(dim=-1)      # (B,m)

        y_blk = _make_block_targets(yb, segment_w)

        correct += (preds == y_blk).sum().item()
        total += preds.numel()

        top1_sum += (preds == y_blk).float().mean().item()
        top5_sum += topk_accuracy_from_logits(logits, y_blk, k=5)
        batches += 1

    acc = correct / total if total > 0 else 0.0
    return acc, top1_sum / max(1, batches), top5_sum / max(1, batches)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--use_norm_X", action="store_true", help="读取 *_X_norm.npy（建议打开）")
    p.add_argument("--split_train", default="train")
    p.add_argument("--split_val", default="val")

    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--segment_w", type=int, default=1)
    p.add_argument("--hidden_d", type=int, default=256)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--class_weight_scheme", choices=["inv", "inv_sqrt", "none"], default="inv_sqrt")

    p.add_argument("--seed", type=int, default=20251111)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt_dir", default="outputs/ckpts_memmap")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info_path = os.path.join(args.data_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise RuntimeError(f"找不到 {info_path}，请先运行 build_grid_memmap_dataset.py")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info["features"]
    L = int(info["lookback_steps"])
    H = int(info["horizon_steps"])
    C = len(features)

    counts = info.get("counts_written", {})
    if args.split_train not in counts or args.split_val not in counts:
        raise RuntimeError(f"dataset_info.json 的 counts_written 缺少 {args.split_train}/{args.split_val}：{counts}")

    Ntr = int(counts[args.split_train])
    Nva = int(counts[args.split_val])

    Xtr_path = os.path.join(args.data_dir, f"{args.split_train}_X_norm.npy" if args.use_norm_X else f"{args.split_train}_X.npy")
    Ytr_path = os.path.join(args.data_dir, f"{args.split_train}_Y.npy")
    Xva_path = os.path.join(args.data_dir, f"{args.split_val}_X_norm.npy" if args.use_norm_X else f"{args.split_val}_X.npy")
    Yva_path = os.path.join(args.data_dir, f"{args.split_val}_Y.npy")

    print("Device:", device)
    print("Seed:", args.seed)
    print("Train:", Xtr_path, Ytr_path, "N=", Ntr)
    print("Val:  ", Xva_path, Yva_path, "N=", Nva)
    print("grid_size:", args.grid_size, "segment_w:", args.segment_w, "batch:", args.batch, "hidden_d:", args.hidden_d)

    assert H % args.segment_w == 0, f"H={H} 必须能整除 segment_w={args.segment_w}"
    m = H // args.segment_w

    ds_tr = GridMemmapDataset(Xtr_path, Ytr_path, Ntr, L, C, H)
    ds_va = GridMemmapDataset(Xva_path, Yva_path, Nva, L, C, H)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=max(1, args.num_workers // 2))

    num_grids = args.grid_size * args.grid_size

    model = SegRNNGrid(
        segment_w=args.segment_w,
        hidden_d=args.hidden_d,
        num_grids=num_grids,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        max_m=m + 8,
        in_channels=C
    ).to(device)

    weight = None
    if args.class_weight and args.class_weight_scheme != "none":
        print("Computing class weights from memmap Y (streaming)...")
        weight = compute_class_weights_from_memmap(
            Y_path=Ytr_path, N=Ntr, H=H,
            num_classes=num_grids,
            scheme=args.class_weight_scheme,
            cap_min=0.1
        )
        print("weight stats min/median/max:",
              float(weight.min().item()),
              float(weight.median().item()),
              float(weight.max().item()))
        weight = weight.to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, loader_tr, optimizer, criterion, device, num_grids, args.segment_w)
        val_acc, val_top1, val_top5 = val_epoch(model, loader_va, device, num_grids, args.segment_w)
        print(f"Epoch {ep}/{args.epochs} train_loss={tr_loss:.4f} val_acc={val_acc:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}")

        ckpt = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args)
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, f"ckpt_ep{ep}.pth"))
        if val_acc > best:
            best = val_acc
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))

    print("Done. Best val_acc:", best)


if __name__ == "__main__":
    main()

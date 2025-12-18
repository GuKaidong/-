# src/train_improved.py
"""
python src/train_improved.py --train_npz data/1107/train_grid_filtered.npz --val_npz data/1107/val_grid.npz --grid_size 64 --segment_w 1 --hidden_d 128 --batch 128 --epochs 30 --class_weight --ckpt_dir outputs/ckpts_improved

python src/train_improved.py --train_npz data/1107/train_grid_filtered.npz --val_npz data/1107/val_grid.npz --grid_size 64 --segment_w 1 --hidden_d 128 --batch 128 --epochs 30 --class_weight --ckpt_dir outputs/ckpts_improved_1107

python src/train_improved.py --train_npz data/opensky_grid_1111/train_grid_filtered.npz --val_npz data/opensky_grid_1111/val_grid.npz --grid_size 64 --segment_w 1 --hidden_d 128 --batch 128 --epochs 30 --class_weight --ckpt_dir outputs/ckpts_improved_111115

python src/train_improved.py --train_npz data/opensky_grid_merged_111714/train_grid_filtered.npz --val_npz data/opensky_grid_merged_111714/val_grid.npz --grid_size 64 --segment_w 1 --hidden_d 128 --batch 128 --epochs 30 --class_weight --ckpt_dir outputs/ckpts_improved_111714

python src/train_improved.py --train_npz data/opensky_grid_merged_111714/remapped_grid32/train_grid_filtered_remapped_g32.npz --val_npz data/opensky_grid_merged_111714/remapped_grid32/val_grid_filtered_remapped_g32.npz --grid_size 32 --segment_w 8 --hidden_d 128 --batch 128 --epochs 30 --class_weight --ckpt_dir outputs/ckpts_remapped_g32


改进版训练脚本（含 segment_w > 1 时的 segment 标签聚合 mode）
- 使用说明: python src/train_improved.py --train_npz ... --val_npz ... (参见下方 CLI)
- 中文注释便于在你的项目中阅读和维护
"""
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(r'F:\研究生毕业论文\experiment')
from segrnn_model.segrnn_model_grid import SegRNNGrid


# ---------------- reproducibility ----------------
def set_seed(seed: int = 42):
    """设置全局随机种子，尽量使训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 更确定性的 cuDNN 行为（可能会变慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- focal loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        logpt = - nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = - ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------------- 稳健的 class weight 计算 ----------------
def compute_class_weights_from_npz(npz_path, grid_size, eps=1e-3, scheme='inv_sqrt', clip_min=0.1, clip_max=10.0):
    """
    从 npz 的 Y 统计并返回 torch tensor 权重
    - eps: 平滑项比例
    - scheme: 'inv' 或 'inv_sqrt' 或 'none'
    - clip_min/clip_max: 裁剪权重范围，避免极端值
    """
    d = np.load(npz_path, allow_pickle=True)
    if 'Y' not in d and 'y' not in d:
        raise RuntimeError(f"{npz_path} 中找不到 Y 或 y 键，无法计算 class weights")
    Y = d['Y'] if 'Y' in d else d['y']
    flat = Y.reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=grid_size*grid_size).astype(np.float64)
    counts = counts + eps * np.mean(counts + 1.0)
    if scheme == 'inv':
        w = 1.0 / counts
    elif scheme == 'inv_sqrt':
        w = 1.0 / np.sqrt(counts)
    else:
        w = np.ones_like(counts)
    w = w / np.mean(w)
    w = np.clip(w, clip_min, clip_max)
    return torch.from_numpy(w.astype(np.float32))

# ---------------- utility ----------------
def topk_accuracy_from_logits(logits, targets, k=5):
    # logits: [B,m,num_grids], targets: [B,m]
    topk = logits.topk(k, dim=-1).indices  # [B,m,k]
    targets_exp = targets.unsqueeze(-1).expand_as(topk)
    correct = (topk == targets_exp).any(dim=-1).float()  # [B,m]
    return correct.mean().item()

def mode_tensor_over_axis(tensor, axis):
    """
    对 tensor 在指定 axis 上取众数（mode）。
    优先使用 torch.mode（如果可用），否则回退到 numpy 实现。
    tensor: torch.Tensor
    axis: int
    返回：values （torch.Tensor）
    """
    try:
        # torch.mode 在大多数版本可用
        vals = torch.mode(tensor, dim=axis).values
        return vals
    except Exception:
        # 回退到 numpy（可能慢一些）
        arr = tensor.cpu().numpy()
        # axis 轴的多数票（对 tie 取第一个）
        # reshape到 (-1, K) 结构进行处理
        perm = list(range(arr.ndim))
        if axis != arr.ndim - 1:
            # move axis to last
            perm.pop(axis)
            perm.append(axis)
            arr = np.transpose(arr, perm)
        # arr shape (..., w) -> flatten leading dims
        lead = int(np.prod(arr.shape[:-1]))
        w = arr.shape[-1]
        arr2 = arr.reshape(lead, w)
        out = np.empty((lead,), dtype=arr.dtype)
        for i in range(lead):
            vals, counts = np.unique(arr2[i], return_counts=True)
            idx = np.argmax(counts)
            out[i] = vals[idx]
        out = out.reshape(arr.shape[:-1])
        return torch.from_numpy(out).to(tensor.device)

# ---------------- training / validation loops ----------------
def train_epoch(model, loader, optimizer, criterion, device, num_grids, args):
    model.train()
    running_loss = 0.0; iters = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)  # yb shape: [B, H]
        B, H = yb.shape
        w = args.segment_w
        assert H % w == 0, "H must be divisible by segment_w"
        m = H // w

        # reshape为 [B, m, w] 然后取 mode（多数票）作为该 segment 的标签
        yb_resh = yb.view(B, m, w)
        # 使用 torch.mode 或回退 numpy
        yb_seg = mode_tensor_over_axis(yb_resh, 2)  # [B, m]

        logits = model(xb, target_m=m)  # [B, m, num_grids]
        loss = criterion(logits.view(-1, num_grids), yb_seg.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running_loss += loss.item(); iters += 1
    return running_loss / max(1, iters)

def val_epoch(model, loader, device, num_grids, args):
    model.eval()
    correct = 0; total = 0
    top1 = 0.0; top5 = 0.0; batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            B, H = yb.shape
            w = args.segment_w
            assert H % w == 0
            m = H // w
            yb_resh = yb.view(B, m, w)
            yb_seg = mode_tensor_over_axis(yb_resh, 2)  # [B, m]

            logits = model(xb, target_m=m)  # [B,m,num_grids]
            preds = logits.argmax(dim=-1)  # [B, m]
            correct += (preds == yb_seg).sum().item()
            total += preds.numel()
            top1 += (preds == yb_seg).float().mean().item()
            top5 += topk_accuracy_from_logits(logits, yb_seg, k=5)
            batches += 1
    acc = correct / total if total>0 else 0.0
    return acc, top1 / max(1,batches), top5 / max(1,batches)

# ---------------- main ----------------
def main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Seed:", args.seed)
    print("Train npz:", args.train_npz)
    print("Val npz:", args.val_npz)
    print("grid_size:", args.grid_size, "segment_w:", args.segment_w, "batch:", args.batch, "hidden_d:", args.hidden_d)

    # load npz helper (兼容 X/Y 或 x/y)
    def load_npz_maybe_remap(path):
        d = np.load(path, allow_pickle=True)
        if 'X' in d and 'Y' in d:
            X = d['X'].astype(np.float32); Y = d['Y'].astype(np.int64)
            return X, Y, d
        elif 'x' in d and 'y' in d:
            X = d['x'].astype(np.float32); Y = d['y'].astype(np.int64)
            return X, Y, d
        else:
            raise RuntimeError(f"{path} 必须包含 X/Y 或 x/y 键")

    # load datasets
    X_tr, Y_tr, dtr = load_npz_maybe_remap(args.train_npz)
    X_va, Y_va, dva = load_npz_maybe_remap(args.val_npz)
    print("Train shape:", X_tr.shape, Y_tr.shape)
    print("Val shape:", X_va.shape, Y_va.shape)

    # 检查 Y 范围是否在 grid_size 范围内
    y_max = int(Y_tr.max()); y_min = int(Y_tr.min())
    if y_max >= args.grid_size * args.grid_size or y_min < 0:
        raise RuntimeError(f"训练集标签范围 [{y_min}, {y_max}] 与 grid_size={args.grid_size} (max idx={args.grid_size*args.grid_size-1}) 不匹配。请重新生成 npz。")
    yv_max = int(Y_va.max()); yv_min = int(Y_va.min())
    if yv_max >= args.grid_size * args.grid_size or yv_min < 0:
        raise RuntimeError(f"验证集标签范围 [{yv_min}, {yv_max}] 与 grid_size={args.grid_size} 不匹配。")

    # optionally compute class weights
    weight = None
    if args.class_weight:
        print("Computing class weights from train npz (stable)...")
        weight = compute_class_weights_from_npz(args.train_npz, args.grid_size, scheme=args.class_weight_scheme,
                                               clip_min=0.1, clip_max=10.0)
        print("Class weight tensor shape:", weight.shape)
        w_np = weight.cpu().numpy()
        print("Class weight stats: min/median/max:", float(w_np.min()), float(np.median(w_np)), float(w_np.max()))
    if args.use_gpu and weight is not None:
        weight = weight.to(device)

    # dataloaders
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))
    loader_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2)

    num_grids = args.grid_size * args.grid_size
    _, L, C = X_tr.shape
    H = Y_tr.shape[1]
    assert H % args.segment_w == 0
    m = H // args.segment_w

    # instantiate model
    model = SegRNNGrid(segment_w=args.segment_w, hidden_d=args.hidden_d, num_grids=num_grids,
                       gru_layers=args.gru_layers, dropout=args.dropout, max_m=m+8, in_channels=C)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=weight, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    os.makedirs(args.ckpt_dir, exist_ok=True)
    # 保存训练配置
    with open(os.path.join(args.ckpt_dir, "train_config.json"), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, loader_tr, optimizer, criterion, device, num_grids, args)
        val_acc, val_top1, val_top5 = val_epoch(model, loader_va, device, num_grids, args)
        print(f"Epoch {ep}/{args.epochs} train_loss={tr_loss:.4f} val_acc={val_acc:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}")
        if args.use_scheduler:
            scheduler.step()
        ckpt = {'epoch':ep, 'state_dict':model.state_dict(), 'opt':optimizer.state_dict(), 'args':vars(args)}
        torch.save(ckpt, os.path.join(args.ckpt_dir, f'ckpt_ep{ep}.pth'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(args.ckpt_dir, 'best.pth'))
    print("Done. Best val_acc:", best_acc)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_npz', required=True)
    p.add_argument('--val_npz', required=True)
    p.add_argument('--grid_size', type=int, default=128)
    p.add_argument('--segment_w', type=int, default=1)
    p.add_argument('--hidden_d', type=int, default=128)
    p.add_argument('--gru_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--ckpt_dir', default='./outputs/ckpts')
    p.add_argument('--class_weight', action='store_true', help='compute class weights from train npz')
    p.add_argument('--class_weight_scheme', choices=['inv','inv_sqrt','none'], default='inv_sqrt')
    p.add_argument('--focal', action='store_true', help='use focal loss instead of crossentropy')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--use_scheduler', action='store_true')
    p.add_argument('--meta_path', type=str, default='')
    p.add_argument('--seed', type=int, default=20251111)
    args = p.parse_args()
    main(args)

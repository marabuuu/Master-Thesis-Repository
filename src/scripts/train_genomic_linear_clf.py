#!/usr/bin/env python3
"""
Train a linear (or small-projection) classifier on per-sample genomic feature HDF5 files.

Usage examples:
  python scripts/train_genomic_linear_clf.py --train-dir data_gen/train --test-dir data_gen/test --out-dir models/gen_clf

The script supports two labelling modes:
 - directory-per-class: train/<class>/*.h5 and test/<class>/*.h5
 - labels CSV: pass --labels-csv labels.csv where CSV has columns `id,label` and filenames are `id.h5`

Each .h5 should contain a dataset named `feats` or `features` with shape (D,) or (1,D).
The script computes per-dimension mean/std on training features and saves them with the model
so you can reuse normalization in MoPaDi (keys: `conds_mean`, `conds_std`).
"""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GenomicFeatDataset(Dataset):
    def __init__(self, files, labels, feature_dim):
        self.files = files
        self.labels = labels
        self.feature_dim = feature_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, 'r') as f:
            if 'feats' in f:
                arr = np.array(f['feats'])
            elif 'features' in f:
                arr = np.array(f['features'])
            else:
                raise RuntimeError(f"No 'feats' or 'features' in {path}")
        # Preserve tile structure: arr may be (D,) or (N,D).
        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.feature_dim:
                raise RuntimeError(f"Feature dim mismatch for {path}: {arr.shape[0]} != {self.feature_dim}")
        elif arr.ndim == 2:
            if arr.shape[1] != self.feature_dim:
                raise RuntimeError(f"Feature dim mismatch for {path}: {arr.shape[1]} != {self.feature_dim}")
        else:
            raise RuntimeError(f"Unexpected feature array shape for {path}: {arr.shape}")
        label = self.labels[idx]
        return arr, label, os.path.basename(path)


class GenomicClassifier(nn.Module):
    """Accepts input of shape [B,N,D] or [B,D]. Returns logits [B,C]."""
    def __init__(self, in_dim, num_classes, proj_dim=0):
        super().__init__()
        if proj_dim and proj_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, num_classes)
            )
            self.out_dim = proj_dim
        else:
            self.net = nn.Linear(in_dim, num_classes)
            self.out_dim = in_dim

    def forward(self, x):
        # x may be [B,N,D] or [B,D]
        if x.dim() == 3:
            # pool over N (mean) -> [B,D]
            x = x.mean(dim=1)
        return self.net(x)


def collect_files_and_labels(root_dir, labels_csv=None, feature_dim=None, id_col='id', label_col='label', keep_classes=None):
    root = Path(root_dir)
    files = []
    labels = []
    classes = []

    if labels_csv is not None:
        import csv
        id2label = {}
        with open(labels_csv, 'r') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if id_col not in r or label_col not in r:
                    continue
                pid = r[id_col]
                lab = r[label_col]
                if lab is None or lab == '':
                    continue
                if keep_classes is not None and lab not in keep_classes:
                    continue
                id2label[pid] = lab

        for fname in sorted(os.listdir(root)):
            if not fname.endswith('.h5'):
                continue
            idname = os.path.splitext(fname)[0]
            if idname not in id2label:
                continue
            files.append(str(root / fname))
            labels.append(id2label[idname])
            classes.append(id2label[idname])
    else:
        # expect subdirs per class
        for cls in sorted(os.listdir(root)):
            cls_path = root / cls
            if not cls_path.is_dir():
                continue
            for fname in sorted(os.listdir(cls_path)):
                if not fname.endswith('.h5'):
                    continue
                files.append(str(cls_path / fname))
                labels.append(cls)
                classes.append(cls)

    if len(files) == 0:
        # Helpful diagnostic: if there are .h5 files directly in `root`, the user
        # likely provided a flat directory of samples but did not pass --labels-csv.
        try:
            entries = sorted(os.listdir(root))
            h5s = [e for e in entries if e.endswith('.h5')]
        except Exception:
            h5s = []
        if len(h5s) > 0:
            sample_list = h5s[:10]
            raise RuntimeError(
                f"No feature files were collected from {root_dir} using the expected layout.\n"
                f"I found {len(h5s)} .h5 files directly in the folder (examples: {sample_list}).\n"
                "This script expects either (a) subfolders per class (train/<class>/*.h5),"
                " or (b) a flat folder with a CSV mapping ids to labels passed via --labels-csv.\n"
                "Please either provide --labels-csv (columns: id,label) or reorganize your .h5 files into class subdirectories."
            )
        raise RuntimeError(f"No feature files found in {root_dir}")

    # map class names to indices
    class_names = sorted(list(set(classes)))
    class2idx = {n: i for i, n in enumerate(class_names)}
    labels_idx = [class2idx[l] for l in labels]

    # check feature dim if not specified
    if feature_dim is None:
        # read first file
        with h5py.File(files[0], 'r') as f:
            if 'feats' in f:
                arr = np.array(f['feats'])
            elif 'features' in f:
                arr = np.array(f['features'])
            else:
                raise RuntimeError(f"No 'feats' or 'features' in {files[0]}")
        # arr may be (D,) or (N,D). Determine per-vector dimensionality D.
        if arr.ndim == 1:
            feature_dim = int(arr.shape[0])
        elif arr.ndim == 2:
            feature_dim = int(arr.shape[1])
        else:
            raise RuntimeError(f"Unsupported feature shape in {files[0]}: {arr.shape}")

    return files, labels_idx, class2idx, feature_dim


def compute_mean_std(files, feature_dim, batch=256):
    # compute mean and std per-dimension (D) over all tile vectors across files
    s = np.zeros(feature_dim, dtype=np.float64)
    sq = np.zeros(feature_dim, dtype=np.float64)
    n = 0  # total number of vectors (rows)
    for p in tqdm(files, desc='computing mean/std'):
        with h5py.File(p, 'r') as f:
            if 'feats' in f:
                arr = np.array(f['feats'])
            elif 'features' in f:
                arr = np.array(f['features'])
            else:
                raise RuntimeError(f"No 'feats' or 'features' in {p}")
        arr = arr.astype(np.float64)
        if arr.ndim == 1:
            rows = arr.reshape(1, -1)
        elif arr.ndim == 2:
            rows = arr
        else:
            raise RuntimeError(f"Unsupported feature shape in {p}: {arr.shape}")
        s += rows.sum(axis=0)
        sq += (rows * rows).sum(axis=0)
        n += rows.shape[0]
    if n == 0:
        raise RuntimeError("No feature vectors found when computing mean/std")
    mean = (s / n).astype(np.float32)
    var = (sq / n - mean.astype(np.float64) ** 2).astype(np.float32)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, std


def train_loop(model, train_loader, val_loader, device, epochs, lr, out_dir):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} train')
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        # validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f'Epoch {epoch} val acc: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))

    print(f'Best val acc: {best_acc:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--feature-dim', type=int, default=None, help='Feature dimensionality (auto-detected if omitted)')
    parser.add_argument('--proj-dim', type=int, default=0, help='If >0, use small projection network before linear head')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--labels-csv', default=None, help='Optional CSV file mapping id -> label (columns: id,label)')
    parser.add_argument('--id-col', default='id', help='Column name in CSV that contains the sample id (filename without .h5)')
    parser.add_argument('--label-col', default='label', help='Column name in CSV that contains the class label')
    parser.add_argument('--keep-classes', default=None, help='Comma-separated list of classes to keep (e.g. "LumA,Basal")')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    keep = None
    if args.keep_classes is not None:
        keep = [c for c in args.keep_classes.split(',') if c]

    train_files, train_labels, class2idx, feature_dim = collect_files_and_labels(
        args.train_dir, args.labels_csv, args.feature_dim, id_col=args.id_col, label_col=args.label_col, keep_classes=keep
    )
    test_files, test_labels, _, _ = collect_files_and_labels(
        args.test_dir, args.labels_csv, feature_dim, id_col=args.id_col, label_col=args.label_col, keep_classes=keep
    )

    # Diagnostic summary: how many files were collected and class distribution
    from collections import Counter
    train_counts = Counter()
    for lab in train_labels:
        train_counts[lab] += 1
    test_counts = Counter()
    for lab in test_labels:
        test_counts[lab] += 1

    print(f'Found classes: {class2idx}')
    print(f'Collected {len(train_files)} train files, class counts: {dict(train_counts)}')
    print(f'Collected {len(test_files)} test files, class counts: {dict(test_counts)}')
    num_classes = len(class2idx)

    mean, std = compute_mean_std(train_files, feature_dim)
     # save normalization state as torch tensors (weights-only friendly)
    state = {
        'conds_mean': torch.from_numpy(mean).cpu(),
        'conds_std': torch.from_numpy(std).cpu(),
        'feature_dim': feature_dim,
        'class2idx': class2idx,
    }
    torch.save(state, os.path.join(args.out_dir, 'norm_state.pth'))

    # create datasets (normalize on the fly)
    def collate_fn(batch):
        # batch elements may be (D,) or (N,D). Handle both.
        samples = [b[0] for b in batch]
        ys = np.array([b[1] for b in batch], dtype=np.int64)

        # check dimensionality
        dims = [s.ndim for s in samples]
        if all(d == 1 for d in dims):
            # (B,D)
            xs = np.stack(samples, axis=0)  # (B,D)
            xs = (xs - mean[None, :]) / (std[None, :])
            xs = torch.from_numpy(xs).float()
        elif all(d == 2 for d in dims):
            Ns = [s.shape[0] for s in samples]
            if len(set(Ns)) == 1:
                # all have same N -> (B,N,D)
                xs = np.stack(samples, axis=0)  # (B,N,D)
                xs = (xs - mean[None, None, :]) / (std[None, None, :])
                xs = torch.from_numpy(xs).float()
            else:
                # variable N across samples: fall back to per-sample mean -> (B,D)
                stacked = np.stack([s.mean(axis=0) for s in samples], axis=0)
                xs = (stacked - mean[None, :]) / (std[None, :])
                xs = torch.from_numpy(xs).float()
        else:
            raise RuntimeError(f"Mixed or unsupported sample dimensions in batch: {dims}")

        ys = torch.from_numpy(ys)
        return xs, ys, [b[2] for b in batch]

    train_dataset = GenomicFeatDataset(train_files, train_labels, feature_dim)
    test_dataset = GenomicFeatDataset(test_files, test_labels, feature_dim)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GenomicClassifier(in_dim=feature_dim, num_classes=num_classes, proj_dim=args.proj_dim)

    train_loop(model, train_loader, test_loader, args.device, args.epochs, args.lr, args.out_dir)

    # save final model and metadata
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'final_model.pth'))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as fh:
        json.dump({'feature_dim': feature_dim, 'class2idx': class2idx}, fh, indent=2)

    print('Saved model and normalization state to', args.out_dir)


if __name__ == '__main__':
    main()

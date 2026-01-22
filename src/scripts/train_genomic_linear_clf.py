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

        arr = arr.astype(np.float32).reshape(-1)
        if arr.shape[0] != self.feature_dim:
            raise RuntimeError(f"Feature dim mismatch for {path}: {arr.shape[0]} != {self.feature_dim}")
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


def collect_files_and_labels(root_dir, labels_csv=None, feature_dim=None):
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
                id2label[r['id']] = r['label']

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
        feature_dim = int(arr.reshape(-1).shape[0])

    return files, labels_idx, class2idx, feature_dim


def compute_mean_std(files, feature_dim, batch=256):
    # compute mean and std over all training features
    s = np.zeros(feature_dim, dtype=np.float64)
    sq = np.zeros(feature_dim, dtype=np.float64)
    n = 0
    for p in tqdm(files, desc='computing mean/std'):
        with h5py.File(p, 'r') as f:
            if 'feats' in f:
                arr = np.array(f['feats'])
            elif 'features' in f:
                arr = np.array(f['features'])
            else:
                raise RuntimeError(f"No 'feats' or 'features' in {p}")
        arr = arr.reshape(-1).astype(np.float64)
        s += arr
        sq += arr * arr
        n += 1
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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    train_files, train_labels, class2idx, feature_dim = collect_files_and_labels(args.train_dir, args.labels_csv, args.feature_dim)
    test_files, test_labels, _, _ = collect_files_and_labels(args.test_dir, args.labels_csv, feature_dim)

    print(f'Found classes: {class2idx}')
    num_classes = len(class2idx)

    mean, std = compute_mean_std(train_files, feature_dim)
    # save normalization state
    state = {'conds_mean': mean, 'conds_std': std, 'feature_dim': feature_dim, 'class2idx': class2idx}
    torch.save(state, os.path.join(args.out_dir, 'norm_state.pth'))

    # create datasets (normalize on the fly)
    def collate_fn(batch):
        xs = np.stack([b[0] for b in batch], axis=0)
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        # normalize
        xs = (xs - mean[None, :]) / (std[None, :])
        xs = torch.from_numpy(xs).float()
        # expand to [B,1,D] so model can accept bag-shaped inputs
        xs = xs.unsqueeze(1)
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

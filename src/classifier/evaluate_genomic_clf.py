#!/usr/bin/env python3
"""
Evaluate a trained genomic classifier and perform dataset diagnostics.

Features:
- Check train/test overlap (by sample id filenames)
- Compute confusion matrix and per-class precision/recall/F1 on the test set
- Print examples of duplicated / overlapping sample ids

Usage:
  python src/scripts/evaluate_genomic_clf.py --train-dir path/to/train --test-dir path/to/test \
      --model-dir out/models/gen_clf --batch-size 64 --device cpu

The script expects `norm_state.pth` and `best_model.pth` in `--model-dir` (or pass explicit paths).
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
        if x.dim() == 3:
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
        try:
            entries = sorted(os.listdir(root))
            h5s = [e for e in entries if e.endswith('.h5')]
        except Exception:
            h5s = []
        if len(h5s) > 0:
            sample_list = h5s[:10]
            raise RuntimeError(
                f"No feature files were collected from {root_dir} using the expected layout. Found {len(h5s)} .h5 files directly in the folder (examples: {sample_list})."
            )
        raise RuntimeError(f"No feature files found in {root_dir}")

    class_names = sorted(list(set(classes)))
    class2idx = {n: i for i, n in enumerate(class_names)}
    labels_idx = [class2idx[l] for l in labels]

    if feature_dim is None:
        with h5py.File(files[0], 'r') as f:
            if 'feats' in f:
                arr = np.array(f['feats'])
            elif 'features' in f:
                arr = np.array(f['features'])
            else:
                raise RuntimeError(f"No 'feats' or 'features' in {files[0]}")
        if arr.ndim == 1:
            feature_dim = int(arr.shape[0])
        elif arr.ndim == 2:
            feature_dim = int(arr.shape[1])
        else:
            raise RuntimeError(f"Unsupported feature shape in {files[0]}: {arr.shape}")

    return files, labels_idx, class2idx, feature_dim


def collate_factory(mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)

    def collate_fn(batch):
        samples = [b[0] for b in batch]
        ys = np.array([b[1] for b in batch], dtype=np.int64)

        dims = [s.ndim for s in samples]
        if all(d == 1 for d in dims):
            xs = np.stack(samples, axis=0)
            xs = (xs - mean[None, :]) / (std[None, :])
            xs = torch.from_numpy(xs).float()
        elif all(d == 2 for d in dims):
            Ns = [s.shape[0] for s in samples]
            if len(set(Ns)) == 1:
                xs = np.stack(samples, axis=0)
                xs = (xs - mean[None, None, :]) / (std[None, None, :])
                xs = torch.from_numpy(xs).float()
            else:
                stacked = np.stack([s.mean(axis=0) for s in samples], axis=0)
                xs = (stacked - mean[None, :]) / (std[None, :])
                xs = torch.from_numpy(xs).float()
        else:
            raise RuntimeError(f"Mixed or unsupported sample dimensions in batch: {dims}")

        ys = torch.from_numpy(ys)
        return xs, ys, [b[2] for b in batch]

    return collate_fn


def compute_confusion_matrix(model, loader, device, num_classes):
    model.to(device)
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    examples = []
    with torch.no_grad():
        for x, y, names in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.numpy()
            for t, p, n in zip(y_np, preds, names):
                cm[t, p] += 1
                if t != p and len(examples) < 20:
                    examples.append((n, t, p))
    return cm, examples


def print_metrics_from_cm(cm, class_names):
    num_classes = cm.shape[0]
    precisions = []
    recalls = []
    f1s = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    print('\nPer-class metrics:')
    print('class\tprecision\trecall\tf1\tsupport')
    for i, name in enumerate(class_names):
        support = cm[i, :].sum()
        print(f"{name}\t{precisions[i]:.3f}\t{recalls[i]:.3f}\t{f1s[i]:.3f}\t{support}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--labels-csv', default=None)
    parser.add_argument('--id-col', default='id')
    parser.add_argument('--label-col', default='label')
    parser.add_argument('--keep-classes', default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--proj-dim', type=int, default=0)
    parser.add_argument('--allow-numpy-reconstruct', action='store_true',
                        help='If set, load model by allowlisting numpy._core.multiarray._reconstruct via torch.serialization.add_safe_globals (use for trusted checkpoints)')
    parser.add_argument('--check-overlap-only', action='store_true')
    args = parser.parse_args()

    keep = None
    if args.keep_classes is not None:
        keep = [c for c in args.keep_classes.split(',') if c]

    model_dir = Path(args.model_dir)
    norm_path = model_dir / 'norm_state.pth'
    model_path = model_dir / 'best_model.pth'
    if not norm_path.exists():
        raise RuntimeError(f"norm_state.pth not found in {model_dir}")
    if not model_path.exists() and not args.check_overlap_only:
        raise RuntimeError(f"best_model.pth not found in {model_dir}")

    # collect ids (filenames without .h5)
    train_ids = set(os.path.splitext(f)[0] for f in os.listdir(args.train_dir) if f.endswith('.h5'))
    test_ids = set(os.path.splitext(f)[0] for f in os.listdir(args.test_dir) if f.endswith('.h5'))
    inter = train_ids & test_ids
    print(f"Train count: {len(train_ids)}, Test count: {len(test_ids)}, Intersection: {len(inter)}")
    if len(inter) > 0:
        print('Examples of overlapping ids:', list(inter)[:20])

    if args.check_overlap_only:
        return

    # load norm state
    state = torch.load(str(norm_path), map_location='cpu')
    feature_dim = int(state.get('feature_dim'))
    mean = state['conds_mean']
    std = state['conds_std']
    # class2idx may be present in norm_state; if not, infer from test collection
    class2idx = state.get('class2idx', None)

    test_files, test_labels, inferred_class2idx, _ = collect_files_and_labels(
        args.test_dir, args.labels_csv, feature_dim, id_col=args.id_col, label_col=args.label_col, keep_classes=keep
    )

    if class2idx is None:
        class2idx = inferred_class2idx

    # sort class names by class2idx order
    class_names = [None] * len(class2idx)
    for k, v in class2idx.items():
        class_names[v] = k

    test_dataset = GenomicFeatDataset(test_files, test_labels, feature_dim)
    collate_fn = collate_factory(mean, std)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = len(class_names)
    model = GenomicClassifier(in_dim=feature_dim, num_classes=num_classes, proj_dim=args.proj_dim)

    # Try multiple loading strategies to handle checkpoints saved with older PyTorch/numpy:
    # 1) normal load (weights_only default)
    # 2) allowlist numpy._core.multiarray._reconstruct via torch.serialization.add_safe_globals
    # 3) last-resort: load with weights_only=False (may execute arbitrary code; use only for trusted files)
    sd = None
    if args.allow_numpy_reconstruct:
        # Explicit allowlist path requested by user
        try:
            import numpy as np
            from torch.serialization import add_safe_globals

            safe_objs = []
            for core_attr in ('_core', 'core'):
                core = getattr(np, core_attr, None)
                if core is None:
                    continue
                multi = getattr(core, 'multiarray', None)
                if multi is None:
                    continue
                rec = getattr(multi, '_reconstruct', None)
                if rec is not None:
                    safe_objs.append(rec)

            if len(safe_objs) == 0:
                raise RuntimeError('Could not find numpy._core.multiarray._reconstruct to allowlist')

            with add_safe_globals(safe_objs):
                sd = torch.load(str(model_path), map_location=args.device)
        except Exception as e:
            raise RuntimeError(f'Failed loading checkpoint with numpy reconstruct allowlist: {e}')
    else:
        # Try normal load; if it fails, try automatic allowlist; final fallback uses weights_only=False
        try:
            sd = torch.load(str(model_path), map_location=args.device)
        except Exception as e1:
            # try allowlist automatically
            try:
                import numpy as np
                from torch.serialization import add_safe_globals

                safe_objs = []
                for core_attr in ('_core', 'core'):
                    core = getattr(np, core_attr, None)
                    if core is None:
                        continue
                    multi = getattr(core, 'multiarray', None)
                    if multi is None:
                        continue
                    rec = getattr(multi, '_reconstruct', None)
                    if rec is not None:
                        safe_objs.append(rec)

                if len(safe_objs) > 0:
                    try:
                        with add_safe_globals(safe_objs):
                            sd = torch.load(str(model_path), map_location=args.device)
                    except Exception:
                        sd = None
            except Exception:
                sd = None

            if sd is None:
                # last resort: load with weights_only=False for trusted checkpoints
                try:
                    sd = torch.load(str(model_path), map_location=args.device, weights_only=False)
                except TypeError:
                    raise e1
                except Exception:
                    raise e1

    model.load_state_dict(sd)

    cm, examples = compute_confusion_matrix(model, test_loader, args.device, num_classes)

    print('\nConfusion matrix (rows=truth, cols=pred):')
    print(cm)
    print_metrics_from_cm(cm, class_names)

    if len(examples) > 0:
        print('\nSome misclassified examples (filename, true_idx, pred_idx):')
        for e in examples:
            print(e)


if __name__ == '__main__':
    main()

import mne
import numpy as np
import os
import re
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # ── Folders ─────────────────────────────────────────────
    stage1_folder = '/Users/ianchen/Desktop/eegdata/EEGproject/Data2/Analyze/stage1'
    stage2_folder = '/Users/ianchen/Desktop/eegdata/EEGproject/Data2/Analyze/stage2'

    # ── Subject split sizes ──────────────────────────────────
    n_train_subjects    = 100   # subjects used for pre-training
    n_finetune_subjects = 50    # subjects used for fine-tuning

    # ── Model hyper-parameters ───────────────────────────────
    F1      = 8
    F2      = 16
    D       = 2
    dropout = 0.5

    # ── Training (pre-training phase) ────────────────────────
    batch_size    = 32
    learning_rate = 0.001
    weight_decay  = 1e-4
    n_epochs      = 50
    random_seed   = 42

    # ── Fine-tuning phase ────────────────────────────────────
    finetune_lr      = 1e-4
    finetune_epochs  = 20

    # ── Output ───────────────────────────────────────────────
    output_dir = '/Users/ianchen/Desktop/eegdata/EEGproject/eegnet_results_subject_split'

    def __init__(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.output_dir = self.output_dir   # promote to instance attribute
            print(f"✓ Output directory: {self.output_dir}")
        except Exception:
            self.output_dir = os.path.join(os.getcwd(), 'eegnet_results_subject_split')
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✓ Fallback output directory: {self.output_dir}")


# ============================================================
# SUBJECT ID EXTRACTOR
# ============================================================
_RE_SUBJECT = re.compile(r'Function-(\d+)', re.IGNORECASE)

def extract_subject_id(filepath: str) -> str | None:
    """Return the numeric subject string (e.g. '041') or None."""
    m = _RE_SUBJECT.search(os.path.basename(filepath))
    return m.group(1) if m else None


# ============================================================
# STAGE DETECTOR  (folder-based labelling)
# ============================================================
class SegmentDetector:
    _RE_STAGE1 = re.compile(r'_trig(\d+)-(\d+)', re.IGNORECASE)
    _RE_STAGE2 = re.compile(r'_trigger(\d+)(?![\-\d])', re.IGNORECASE)

    _STAGE1_FOLDER_KEYWORDS = ['stage1', 'cropped_2sec']
    _STAGE2_FOLDER_KEYWORDS = ['epoched', 'stage2']

    @classmethod
    def detect_from_path(cls, filepath) -> int | None:
        folder   = os.path.dirname(filepath).lower()
        filename = os.path.basename(filepath)

        for kw in cls._STAGE1_FOLDER_KEYWORDS:
            if kw in folder:
                return 0
        for kw in cls._STAGE2_FOLDER_KEYWORDS:
            if kw in folder:
                return 1

        if cls._RE_STAGE2.search(filename):
            return 1

        m = cls._RE_STAGE1.search(filename)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            if s in [10, 11] and 61 <= e <= 120:
                return 0
            if 61 <= s <= 120 and e in [200, 201]:
                return 1

        return None


# ============================================================
# DATA LOADER  (with subject-level split)
# ============================================================
class SubjectSplitDataLoader:
    def __init__(self, config: Config):
        self.config   = config
        self.detector = SegmentDetector()

    # ── public API ──────────────────────────────────────────

    def load_and_split(self):
        """
        Returns
        -------
        splits : dict with keys 'train', 'finetune', 'test'
                 each value is (X, y) numpy arrays
        meta   : dict with n_channels, n_samples, subject_lists
        """
        print("\n" + "="*60)
        print("SUBJECT-LEVEL SPLIT DATA LOADER")
        print("="*60)

        # 1. Collect and label all files
        all_files   = self._collect_files()
        labelled    = self._label_files(all_files)   # list of (filepath, label, subject_id)

        if not labelled:
            raise RuntimeError("No labelled EEG files found. Check folder paths in Config.")

        # 2. Group by subject
        subject_files = defaultdict(list)   # subject_id -> [(fp, label), ...]
        for fp, lbl, sid in labelled:
            subject_files[sid].append((fp, lbl))

        all_subjects = sorted(subject_files.keys())
        n_total      = len(all_subjects)

        n_train = self.config.n_train_subjects
        n_ft    = self.config.n_finetune_subjects
        n_test  = n_total - n_train - n_ft

        if n_test <= 0:
            raise ValueError(
                f"Only {n_total} subjects found. "
                f"Need at least {n_train + n_ft + 1} "
                f"(100 train + 50 fine-tune + ≥1 test)."
            )

        train_subjects   = all_subjects[:n_train]
        ft_subjects      = all_subjects[n_train : n_train + n_ft]
        test_subjects    = all_subjects[n_train + n_ft :]

        self._print_split_summary(all_subjects, train_subjects, ft_subjects, test_subjects, subject_files)

        # 3. Load EEG data for each split
        train_pairs = self._gather_pairs(train_subjects,  subject_files)
        ft_pairs    = self._gather_pairs(ft_subjects,     subject_files)
        test_pairs  = self._gather_pairs(test_subjects,   subject_files)

        print("\n[1/3] Loading TRAIN segments …")
        X_tr, y_tr = self._load_pairs(train_pairs)

        print("\n[2/3] Loading FINE-TUNE segments …")
        X_ft, y_ft = self._load_pairs(ft_pairs)

        print("\n[3/3] Loading TEST segments …")
        X_ts, y_ts = self._load_pairs(test_pairs)

        # 4. Standardise dimensions across all data jointly
        all_X = X_tr + X_ft + X_ts
        all_X_arr, n_channels, n_samples = self._standardize(all_X)

        n_tr = len(X_tr)
        n_ft_ = len(X_ft)

        X_train_arr   = all_X_arr[:n_tr]
        X_ft_arr      = all_X_arr[n_tr : n_tr + n_ft_]
        X_test_arr    = all_X_arr[n_tr + n_ft_:]

        y_train = np.array(y_tr)
        y_ft    = np.array(y_ft)
        y_test  = np.array(y_ts)

        # 5. Balance within each split independently
        X_train_arr, y_train = self._balance(X_train_arr, y_train, "Train")
        X_ft_arr,    y_ft    = self._balance(X_ft_arr,    y_ft,    "Fine-tune")
        X_test_arr,  y_test  = self._balance(X_test_arr,  y_test,  "Test")

        splits = {
            'train'   : (X_train_arr, y_train),
            'finetune': (X_ft_arr,    y_ft),
            'test'    : (X_test_arr,  y_test),
        }
        meta = {
            'n_channels'      : n_channels,
            'n_samples'       : n_samples,
            'train_subjects'  : train_subjects,
            'finetune_subjects': ft_subjects,
            'test_subjects'   : test_subjects,
            'n_subjects_total': n_total,
        }
        return splits, meta

    # ── private helpers ─────────────────────────────────────

    def _collect_files(self):
        all_files = []
        for folder, tag in [(self.config.stage1_folder, 'Stage 1'),
                            (self.config.stage2_folder, 'Stage 2')]:
            if os.path.exists(folder):
                found = [os.path.join(folder, fn)
                         for fn in os.listdir(folder)
                         if fn.endswith('.set')
                         and not fn.startswith(('.', '._'))]
                print(f"  ✓ {tag} folder: {len(found)} .set files")
                all_files.extend(found)
            else:
                print(f"  ⚠️  {tag} folder NOT FOUND: {folder}")
        print(f"  Total .set files: {len(all_files)}")
        return all_files

    def _label_files(self, all_files):
        labelled, skipped_label, skipped_subject = [], 0, 0
        for fp in all_files:
            lbl = self.detector.detect_from_path(fp)
            sid = extract_subject_id(fp)
            if lbl is None:
                skipped_label += 1
                continue
            if sid is None:
                skipped_subject += 1
                continue
            labelled.append((fp, lbl, sid))

        print(f"\n  Labelled : {len(labelled)} files")
        if skipped_label:
            print(f"  Skipped (no stage label) : {skipped_label}")
        if skipped_subject:
            print(f"  Skipped (no subject ID)  : {skipped_subject}")
        return labelled

    @staticmethod
    def _gather_pairs(subjects, subject_files):
        pairs = []
        for s in subjects:
            pairs.extend(subject_files[s])
        return pairs

    @staticmethod
    def _load_pairs(pairs):
        X_list, y_list = [], []
        for i, (fp, lbl) in enumerate(pairs):
            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                print(f"    {i+1}/{len(pairs)}")
            try:
                raw = mne.io.read_raw_eeglab(fp, preload=True, verbose=False)
                X_list.append(raw.get_data())
                y_list.append(lbl)
            except Exception as e:
                print(f"    ⚠️  Skipping {os.path.basename(fp)}: {e}")
        print(f"    Loaded {len(X_list)}/{len(pairs)} segments")
        return X_list, y_list

    @staticmethod
    def _standardize(X_list):
        all_ch  = [x.shape[0] for x in X_list]
        all_len = [x.shape[1] for x in X_list]

        target_ch  = int(np.min(all_ch))
        target_len = int(np.median(all_len))

        print(f"\nStandardising across all splits:")
        print(f"  Channels : {target_ch}  (range {np.min(all_ch)}–{np.max(all_ch)})")
        print(f"  Samples  : {target_len}  (range {np.min(all_len)}–{np.max(all_len)})")

        def pad_or_crop(data):
            ch, samp = data.shape
            if ch > target_ch:
                data = data[:target_ch, :]
            elif ch < target_ch:
                data = np.pad(data, ((0, target_ch - ch), (0, 0)))
            _, samp = data.shape
            if samp > target_len:
                start = (samp - target_len) // 2
                data = data[:, start:start + target_len]
            elif samp < target_len:
                data = np.pad(data, ((0, 0), (0, target_len - samp)))
            return data

        X_arr = np.array([pad_or_crop(x) for x in X_list])
        return X_arr, target_ch, target_len

    def _balance(self, X, y, split_name):
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) <= 1:
            return X, y
        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        print(f"\n⚖️   Balancing [{split_name}] — Stage1={counts[0]}, Stage2={counts[1]}")
        if max_count > min_count * 1.1:
            rng = np.random.default_rng(self.config.random_seed)
            idx = np.concatenate([
                rng.choice(np.where(y == lbl)[0], size=min_count, replace=False)
                for lbl in unique
            ])
            rng.shuffle(idx)
            print(f"   After balance: {min_count} per class")
            return X[idx], y[idx]
        print("   Already balanced.")
        return X, y

    @staticmethod
    def _print_split_summary(all_subs, train_subs, ft_subs, test_subs, subject_files):
        def seg_count(subs):
            return sum(len(subject_files[s]) for s in subs)

        print(f"\n{'='*60}")
        print("SUBJECT SPLIT SUMMARY")
        print(f"{'='*60}")
        print(f"  Total subjects       : {len(all_subs)}")
        print(f"  ├─ Train     subjects: {len(train_subs):>4}  ({seg_count(train_subs)} segments)  IDs {train_subs[0]}…{train_subs[-1]}")
        print(f"  ├─ Fine-tune subjects: {len(ft_subs):>4}  ({seg_count(ft_subs)} segments)  IDs {ft_subs[0]}…{ft_subs[-1]}")
        print(f"  └─ Test      subjects: {len(test_subs):>4}  ({seg_count(test_subs)} segments)  IDs {test_subs[0]}…{test_subs[-1]}")
        print(f"{'='*60}")


# ============================================================
# EEGNET MODEL
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes=2,
                 F1=8, F2=16, D=2, dropout=0.5):
        super().__init__()
        self.conv1     = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.bn1       = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.elu1      = nn.ELU()
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropout)
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3       = nn.BatchNorm2d(F2)
        self.elu2      = nn.ELU()
        self.pool2     = nn.AvgPool2d((1, 8))
        self.drop2     = nn.Dropout(dropout)
        self.flatten   = nn.Flatten()
        self.fc        = nn.Linear(F2 * (n_samples // 32), n_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(self.elu1(self.bn2(self.depthwise(x)))))
        x = self.drop2(self.pool2(self.elu2(self.bn3(self.separable(x)))))
        return self.fc(self.flatten(x))


# ============================================================
# DATASET
# ============================================================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)   # (N, 1, C, T)
        self.y = torch.LongTensor(y)

    def __len__(self):           return len(self.X)
    def __getitem__(self, i):    return self.X[i], self.y[i]


# ============================================================
# TRAINER  (supports both pre-training and fine-tuning)
# ============================================================
class EEGNetTrainer:
    def __init__(self, config, model, device):
        self.config    = config
        self.model     = model.to(device)
        self.device    = device
        self.criterion = nn.CrossEntropyLoss()
        self.history   = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        self._reset_optimizer(config.learning_rate)

    def _reset_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

    # ── core training loop ──────────────────────────────────

    def train(self, train_loader, val_loader,
              n_epochs: int, ckpt_name: str, phase: str = "PRE-TRAINING"):
        best_acc, best_epoch = 0, 0
        ckpt = os.path.join(self.config.output_dir, ckpt_name)

        print(f"\n{'='*60}")
        print(f"  PHASE: {phase}  ({n_epochs} epochs)")
        print(f"{'='*60}")

        for epoch in range(n_epochs):
            self.model.train()
            t_loss, t_correct, t_total = 0, 0, 0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out  = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                t_loss  += loss.item()
                _, pred  = out.max(1)
                t_total   += yb.size(0)
                t_correct += pred.eq(yb).sum().item()

            train_acc = 100. * t_correct / t_total
            avg_loss  = t_loss / len(train_loader)
            val_acc   = self._evaluate(val_loader)
            self.scheduler.step(val_acc)

            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch + 1
                torch.save(self.model.state_dict(), ckpt)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Ep [{epoch+1:>3}/{n_epochs}]  "
                      f"Loss {avg_loss:.4f}  "
                      f"Train {train_acc:.1f}%  "
                      f"Val {val_acc:.1f}%  "
                      f"(Best {best_acc:.1f}% @ ep {best_epoch})")

        print(f"\n  ✓ Best Val Accuracy [{phase}]: {best_acc:.2f}%  (Epoch {best_epoch})")
        torch.save(self.model.state_dict(), ckpt)   # ensure latest is saved
        return best_acc

    def finetune(self, ft_loader, val_loader, ckpt_name: str):
        """Fine-tune with lower LR; resets optimiser."""
        self._reset_optimizer(self.config.finetune_lr)
        self.history = {'train_loss': [], 'train_acc': [], 'val_acc': []}   # fresh history
        return self.train(ft_loader, val_loader,
                          n_epochs=self.config.finetune_epochs,
                          ckpt_name=ckpt_name,
                          phase="FINE-TUNING")

    # ── evaluation helpers ──────────────────────────────────

    def _evaluate(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                _, pred = self.model(Xb).max(1)
                total   += yb.size(0)
                correct += pred.eq(yb).sum().item()
        return 100. * correct / total

    def get_predictions(self, loader):
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                _, pred = self.model(Xb).max(1)
                preds.extend(pred.cpu().numpy())
                truths.extend(yb.cpu().numpy())
        return np.array(truths), np.array(preds)


# ============================================================
# VISUALISATION
# ============================================================
def plot_history(history: dict, title: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], linewidth=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=f'{title} — Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'],   label='Val',   linewidth=2)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title=f'{title} — Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def plot_confusion(y_true, y_pred, title: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stage 1', 'Stage 2'],
                yticklabels=['Stage 1', 'Stage 2'],
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel('True Stage')
    plt.xlabel('Predicted Stage')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("EEGNet  —  Subject-Level Split Pipeline")
    print("  Train     : 100 subjects (pre-training)")
    print("  Fine-tune :  50 subjects (fine-tuning)")
    print("  Test      : remaining subjects (held-out)")
    print("="*60)

    cfg = Config()

    # ── Device ────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device('mps');  print("\n🚀 Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda'); print("\n🚀 NVIDIA CUDA")
    else:
        device = torch.device('cpu');  print("\n🚀 CPU")

    # ── Load & split ──────────────────────────────────────────
    loader = SubjectSplitDataLoader(cfg)
    splits, meta = loader.load_and_split()

    X_train, y_train = splits['train']
    X_ft,    y_ft    = splits['finetune']
    X_test,  y_test  = splits['test']
    n_channels = meta['n_channels']
    n_samples  = meta['n_samples']

    print(f"\n{'='*60}")
    print("DATASET SHAPES")
    print(f"{'='*60}")
    print(f"  Train    : {X_train.shape}  labels={np.bincount(y_train)}")
    print(f"  Fine-tune: {X_ft.shape}  labels={np.bincount(y_ft)}")
    print(f"  Test     : {X_test.shape}  labels={np.bincount(y_test)}")
    print(f"  Channels : {n_channels}")
    print(f"  Samples  : {n_samples}  (~{n_samples/250:.2f}s @ 250 Hz)")

    # ── Save test set for DeepLIFT analysis ──────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)   # ensure dir exists before writing
    test_npz_path = os.path.join(cfg.output_dir, 'test_data.npz')
    np.savez(test_npz_path, X=X_test, y=y_test)
    print(f"  ✓ Saved test data for DeepLIFT: {test_npz_path}")
    print(f"    shape X={X_test.shape}  labels={np.bincount(y_test)}")

    # ── DataLoaders ───────────────────────────────────────────
    def make_dl(X, y, shuffle=True):
        return DataLoader(EEGDataset(X, y),
                          batch_size=cfg.batch_size, shuffle=shuffle)

    train_dl = make_dl(X_train, y_train, shuffle=True)
    ft_dl    = make_dl(X_ft,    y_ft,    shuffle=True)
    test_dl  = make_dl(X_test,  y_test,  shuffle=False)

    # ── Build model ───────────────────────────────────────────
    model = EEGNet(n_channels, n_samples, n_classes=2,
                   F1=cfg.F1, F2=cfg.F2, D=cfg.D, dropout=cfg.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    trainer = EEGNetTrainer(cfg, model, device)

    # ── Phase 1 : Pre-training on 100 train subjects ──────────
    best_pretrain_acc = trainer.train(
        train_dl, test_dl,
        n_epochs   = cfg.n_epochs,
        ckpt_name  = 'eegnet_pretrain.pth',
        phase      = "PRE-TRAINING (100 subjects)"
    )
    plot_history(trainer.history,
                 title    = "Pre-Training",
                 out_path = os.path.join(cfg.output_dir, 'pretrain_history.png'))

    # ── Phase 2 : Fine-tuning on 50 fine-tune subjects ────────
    # Load best pre-trained weights before fine-tuning
    trainer.model.load_state_dict(
        torch.load(os.path.join(cfg.output_dir, 'eegnet_pretrain.pth'),
                   map_location=device)
    )

    best_ft_acc = trainer.finetune(
        ft_dl, test_dl,
        ckpt_name = 'eegnet_finetuned.pth'
    )
    plot_history(trainer.history,
                 title    = "Fine-Tuning",
                 out_path = os.path.join(cfg.output_dir, 'finetune_history.png'))

    # ── Phase 3 : Final evaluation on held-out test subjects ──
    # Load best fine-tuned weights
    trainer.model.load_state_dict(
        torch.load(os.path.join(cfg.output_dir, 'eegnet_finetuned.pth'),
                   map_location=device)
    )

    y_true, y_pred = trainer.get_predictions(test_dl)

    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION  (held-out subjects)")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                target_names=['Stage 1', 'Stage 2']))

    plot_confusion(y_true, y_pred,
                   title    = "Stage Classification — Test Confusion Matrix",
                   out_path = os.path.join(cfg.output_dir, 'test_confusion_matrix.png'))

    # ── Save summary ──────────────────────────────────────────
    summary = {
        'task'                    : 'Stage 1 vs Stage 2 — subject-level split',
        'n_subjects_total'        : meta['n_subjects_total'],
        'n_train_subjects'        : len(meta['train_subjects']),
        'n_finetune_subjects'     : len(meta['finetune_subjects']),
        'n_test_subjects'         : len(meta['test_subjects']),
        'train_subject_ids'       : meta['train_subjects'],
        'finetune_subject_ids'    : meta['finetune_subjects'],
        'test_subject_ids'        : meta['test_subjects'],
        'pretrain_best_val_acc'   : float(best_pretrain_acc),
        'finetune_best_val_acc'   : float(best_ft_acc),
        'n_channels'              : int(n_channels),
        'n_time_samples'          : int(n_samples),
        'train_segments'          : int(len(X_train)),
        'finetune_segments'       : int(len(X_ft)),
        'test_segments'           : int(len(X_test)),
    }
    summary_path = os.path.join(cfg.output_dir, 'subject_split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Summary: {summary_path}")

    # ── Phase 4 : DeepLIFT analysis on real test data ────────
    try:
        from eegnet_deeplift_analysis import run_deeplift_analysis
        print(f"\n{'='*60}")
        print("PHASE 4: DeepLIFT ANALYSIS  (real test data)")
        print(f"{'='*60}")
        run_deeplift_analysis(
            model      = trainer.model,
            X_test     = X_test,
            y_test     = y_test,
            output_dir = cfg.output_dir,
            device     = device,
            n_channels = n_channels,
            n_samples  = n_samples,
            sfreq      = 250.0,
            baseline   = 'zero',
            batch_size = 16,
            topk       = 4,
        )
    except ImportError:
        print("\n  ⚠️  eegnet_deeplift_analysis.py not found in the same folder.")
        print("     Place it next to EEGNet2.py to run DeepLIFT automatically.")
        print(f"     Or run it manually — test data saved at: {test_npz_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Output folder: {cfg.output_dir}")
    print(f"  eegnet_pretrain.pth               ← pre-trained weights")
    print(f"  eegnet_finetuned.pth              ← fine-tuned weights")
    print(f"  test_data.npz                     ← real test set for DeepLIFT")
    print(f"  pretrain_history.png")
    print(f"  finetune_history.png")
    print(f"  test_confusion_matrix.png")
    print(f"  subject_split_summary.json")
    print(f"  deeplift_*.png / .npz             ← attribution plots")
    print("="*60)


if __name__ == "__main__":
    main()
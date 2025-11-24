    import os
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    import timm
    import matplotlib.pyplot as plt
    import csv
    import random

    # ============ Custom Augmentations ============

    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=0.05, p=0.5):
            self.mean = mean
            self.std = std
            self.p = p

        def __call__(self, tensor):
            if random.random() < self.p:
                return tensor + torch.randn_like(tensor) * self.std + self.mean
            return tensor

    class AddGaussianBlur(object):
        def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
            self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            self.p = p

        def __call__(self, img):
            if random.random() < self.p:
                return self.blur(img)
            return img

    # ============ Mixup and CutMix ============

    def mixup_data(x, y, alpha=0.4, device='cuda'):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(x, y, alpha=1.0, device='cuda'):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size, C, H, W = x.size()
        index = torch.randperm(batch_size).to(device)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        x_new = x.clone()
        x_new[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        return x_new, y_a, y_b, lam

    def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ============ Set Seeds ============

    def set_seeds(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seeds()

    RESULTS_PARENT = "fatty_liver_results_aug_2"

    # ============ Dataset ============

    class ImageDataset(Dataset):
        def __init__(self, csv_file, data_root, transform=None):
            self.df = pd.read_csv(csv_file)
            self.image_paths = self.df['filename'].tolist()
            self.labels = self.df['label'].tolist()
            self.data_root = data_root
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = os.path.join(self.data_root, self.image_paths[idx])
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = int(self.labels[idx])
            return img, label

    # ============ Model Fetch ============

    def get_model(model_name, num_classes=2, pretrained=True, drop_rate=0.3):
        if model_name.lower() == "resnet50":
            try:
                model = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
            except:
                print("resnet50 not found in timm, using resnet18 instead.")
                model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif model_name.lower() == "resnet18":
            model = timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif model_name.lower() == "efficientnet_b4":
            model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif model_name.lower() == "swinv2":
            model = timm.create_model("swinv2_base_window12_192_22k", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif model_name.lower() == "vit":
            model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model

    # ============ Patient Split ============

    def extract_patient_id_from_path(image_path):
        return os.path.normpath(image_path).split(os.sep)[1]

    def create_folds(csv_path, out_dir, n_splits=5, random_state=42):
        df = pd.read_csv(csv_path)
        df['patient_id'] = df['filename'].apply(extract_patient_id_from_path)
        patient_labels = df.groupby('patient_id')['label'].agg(lambda x: x.iloc[0]).reset_index()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        patient_ids = patient_labels['patient_id'].values
        patient_y = patient_labels['label'].values
        for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_y)):
            train_patients = set(patient_ids[train_idx])
            val_patients = set(patient_ids[val_idx])
            train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
            val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
            train_csv = os.path.join(out_dir, f"train_fold{fold+1}.csv")
            val_csv = os.path.join(out_dir, f"val_fold{fold+1}.csv")
            train_df[['filename','label']].to_csv(train_csv, index=False)
            val_df[['filename','label']].to_csv(val_csv, index=False)
            print(f"Fold {fold+1}: Train images={len(train_df)}, Val images={len(val_df)} | Train patients={len(train_patients)}, Val patients={len(val_patients)}")

    # ============ Augmentations ============

    def get_transforms(model_name, train=True):
        if model_name.startswith("efficientnet"):
            size = 380
        elif model_name.startswith("vit"):
            size = 224
        elif model_name.startswith("swinv2"):
            size = 192
        else:
            size = 224
        normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(0.2),
                transforms.RandomVerticalFlip(0.2),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                AddGaussianBlur(kernel_size=3, sigma=(0.1, 1.0), p=0.3),
                transforms.ToTensor(),
                AddGaussianNoise(mean=0.0, std=0.01, p=0.3),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                normalize
            ])

    # ============ Parameter Freezing ============

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def freeze_all_but_last(model):
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            for m in reversed(list(model.modules())):
                if isinstance(m, torch.nn.Linear):
                    for param in m.parameters():
                        param.requires_grad = True
                    break

    def sequentially_unfreeze(model, unfrozen_blocks):
        if hasattr(model, 'blocks'):
            blocks = model.blocks
            if unfrozen_blocks < len(blocks):
                for param in blocks[unfrozen_blocks].parameters():
                    param.requires_grad = True
                return True
        elif hasattr(model, 'layer4'):
            layers = ['layer4', 'layer3', 'layer2', 'layer1']
            if unfrozen_blocks < len(layers):
                layer = getattr(model, layers[unfrozen_blocks])
                for param in layer.parameters():
                    param.requires_grad = True
                return True
        elif hasattr(model, 'blocks'):
            blocks = model.blocks
            if unfrozen_blocks < len(blocks):
                for param in blocks[unfrozen_blocks].parameters():
                    param.requires_grad = True
                return True
        return False

    def get_num_blocks(model):
        if hasattr(model, 'blocks'):
            return len(model.blocks)
        elif hasattr(model, 'layer4'):
            return 4
        return 0

    # ============ Train/Eval ============

    def train_one_epoch(
        model, loader, criterion, optimizer, device, 
        mixup=False, cutmix=False, mixup_alpha=0.4, cutmix_alpha=1.0, 
        scheduler=None
    ):
        model.train()
        losses, preds, targets = [], [], []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if mixup:
                x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
                logits = model(x_mix)
                loss = mixup_cutmix_criterion(criterion, logits, y_a, y_b, lam)
            elif cutmix:
                x_cut, y_a, y_b, lam = cutmix_data(x, y, alpha=cutmix_alpha, device=device)
                logits = model(x_cut)
                loss = mixup_cutmix_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            preds += torch.argmax(logits, 1).cpu().tolist()
            targets += y.cpu().tolist()
        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        return np.mean(losses), acc, bal_acc, f1

    def evaluate(
        model, loader, criterion, device, tta=1, tta_transforms=None
    ):
        model.eval()
        losses, preds, targets, probs = [], [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                if tta > 1 and tta_transforms is not None:
                    logits_sum = None
                    for tta_id in range(tta):
                        x_tta = torch.stack([tta_transforms(img.cpu()) for img in x]).to(device)
                        logits = model(x_tta)
                        logits_sum = logits if logits_sum is None else logits_sum + logits
                    logits = logits_sum / tta
                else:
                    logits = model(x)
                loss = criterion(logits, y)
                losses.append(loss.item())
                prob = torch.softmax(logits, 1)
                preds += torch.argmax(prob, 1).cpu().tolist()
                probs += prob[:,1].cpu().tolist()
                targets += y.cpu().tolist()
        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        try:
            auc = roc_auc_score(targets, probs)
        except:
            auc = 0.5
        cm = confusion_matrix(targets, preds)
        return np.mean(losses), acc, bal_acc, f1, auc, cm, preds, targets

    def ensemble_predict(
        model_paths, model_names, val_loader, criterion, device, 
        num_classes=2, tta=1, tta_transforms=None
    ):
        logits_all = []
        for model_path, model_name in zip(model_paths, model_names):
            model = get_model(model_name, num_classes=num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            logits_batches = []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    if tta > 1 and tta_transforms is not None:
                        logits_sum = None
                        for tta_id in range(tta):
                            x_tta = torch.stack([tta_transforms(img.cpu()) for img in x]).to(device)
                            logits = model(x_tta)
                            logits_sum = logits if logits_sum is None else logits_sum + logits
                        logits = logits_sum / tta
                    else:
                        logits = model(x)
                    logits_batches.append(logits.cpu())
            logits_all.append(torch.cat(logits_batches, dim=0))
        logits_ensemble = torch.stack(logits_all, dim=0).mean(0)
        preds = torch.argmax(logits_ensemble, 1).numpy().tolist()
        probs = torch.softmax(logits_ensemble, 1)[:,1].numpy().tolist()
        return preds, probs

    # ============ Logging/Plotting ============

    def plot_curves(log, save_dir, fold, model_name):
        metrics = ['loss', 'acc', 'bal_acc', 'f1', 'auc']
        plt.figure(figsize=(18,12))
        for i, m in enumerate(metrics):
            plt.subplot(2,3,i+1)
            plt.plot(log['train_'+m], label=f"Train {m}")
            plt.plot(log['val_'+m], label=f"Val {m}")
            plt.title(m)
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"fold{fold}_curves.png")
        plt.savefig(out_path)
        plt.close()

    def save_confusion_matrix(cm, save_dir, fold):
        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), va='center', ha='center')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fold{fold}_confusion_matrix.png"))
        plt.close()
        pd.DataFrame(cm).to_csv(os.path.join(save_dir, f"fold{fold}_confusion_matrix.csv"), index=False, header=False)

    def save_metrics_csv(log, save_dir, fold):
        df = pd.DataFrame(log)
        df.to_csv(os.path.join(save_dir, f"fold{fold}_log.csv"), index=False)

    def save_final_metrics(metrics, save_dir, fold):
        out_path = os.path.join(save_dir, f"fold{fold}_final_metrics.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    # ============ Main Training Loop ============

    def train_image_level(
        data_root, 
        folds_dir, 
        model_names,
        num_epochs=15,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        drop_rate=0.3,
        patience=8,
        class_weights=None,
        mixup=False,
        cutmix=False,
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        tta=1,
        ensemble=False,
        warmup_epochs=5,
        device=None
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(RESULTS_PARENT, exist_ok=True)
        for model_name in model_names:
            model_dir = os.path.join(RESULTS_PARENT, model_name)
            os.makedirs(model_dir, exist_ok=True)
            print(f"\n=============== Training {model_name} ===============\n")
            for fold in range(1,6):
                print(f"\n--- Fold {fold} ---")
                fold_dir = os.path.join(model_dir, f"fold{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                train_csv = os.path.join(folds_dir, f"train_fold{fold}.csv")
                val_csv = os.path.join(folds_dir, f"val_fold{fold}.csv")
                if class_weights is not None:
                    cw = torch.tensor(class_weights, dtype=torch.float32).to(device)
                else:
                    cw = None
                train_ds = ImageDataset(train_csv, data_root, transform=get_transforms(model_name, train=True))
                val_ds = ImageDataset(val_csv, data_root, transform=get_transforms(model_name, train=False))
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                model = get_model(model_name, num_classes=2, drop_rate=drop_rate).to(device)
                freeze_all_but_last(model)
                unfrozen_blocks = 0
                num_blocks = get_num_blocks(model)
                fully_unfrozen = False
                early_stop_counter = 0

                criterion = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.0)
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return float(epoch+1) / warmup_epochs
                    return 1.
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                best_bal_acc = -np.inf
                best_state = None
                log = {k:[] for k in [
                    'train_loss','train_acc','train_bal_acc','train_f1','train_auc',
                    'val_loss','val_acc','val_bal_acc','val_f1','val_auc',
                    'total_params', 'trainable_params'
                ]}
                patience_counter = 0

                total_params, trainable_params = count_parameters(model)
                print(f"Total parameters: {total_params}")
                print(f"Trainable parameters (initial): {trainable_params}")
                with open(os.path.join(fold_dir, f"fold{fold}_console.log"), 'a') as flog:
                    flog.write(f"Total parameters: {total_params}\n")
                    flog.write(f"Trainable parameters (initial): {trainable_params}\n")
                    if cw is not None:
                        flog.write(f"Class weights: {cw.detach().cpu().numpy().tolist()}\n")
                if tta > 1:
                    tta_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=20),
                        transforms.Lambda(lambda x: x)
                    ])
                else:
                    tta_transforms = None

                for epoch in range(1, num_epochs+1):
                    train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(
                        model, train_loader, criterion, optimizer, device,
                        mixup=mixup, cutmix=cutmix, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                        scheduler=scheduler
                    )
                    val_loss, val_acc, val_bal_acc, val_f1, val_auc, val_cm, val_preds, val_targets = evaluate(
                        model, val_loader, criterion, device, tta=tta, tta_transforms=tta_transforms
                    )
                    log['train_loss'].append(train_loss)
                    log['train_acc'].append(train_acc)
                    log['train_bal_acc'].append(train_bal_acc)
                    log['train_f1'].append(train_f1)
                    log['train_auc'].append(0)
                    log['val_loss'].append(val_loss)
                    log['val_acc'].append(val_acc)
                    log['val_bal_acc'].append(val_bal_acc)
                    log['val_f1'].append(val_f1)
                    log['val_auc'].append(val_auc)
                    total_params, trainable_params = count_parameters(model)
                    log['total_params'].append(total_params)
                    log['trainable_params'].append(trainable_params)
                    with open(os.path.join(fold_dir, f"fold{fold}_console.log"), 'a') as flog:
                        flog.write(f"Epoch {epoch:02d}: Train loss={train_loss:.4f} acc={train_acc:.3f} bal_acc={train_bal_acc:.3f} | Val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f} f1={val_f1:.3f} auc={val_auc:.3f}\n")
                        flog.write(f"Val confusion matrix:\n{val_cm}\n")
                        flog.write(f"Trainable params: {trainable_params}\n")
                    print(f"Epoch {epoch:02d}: Train loss={train_loss:.4f} acc={train_acc:.3f} bal_acc={train_bal_acc:.3f} | Val loss={val_loss:.4f} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f} f1={val_f1:.3f} auc={val_auc:.3f}")
                    print(f"Val confusion matrix:\n{val_cm}")
                    print(f"Trainable params: {trainable_params}")

                    improved = val_bal_acc > best_bal_acc

                    if improved:
                        best_bal_acc = val_bal_acc
                        best_state = {
                            "model": model.state_dict(),
                            "metrics": {
                                "val_loss": val_loss, "val_acc": val_acc, "val_bal_acc": val_bal_acc,
                                "val_f1": val_f1, "val_auc": val_auc
                            },
                            "confusion_matrix": val_cm,
                            "val_preds": val_preds,
                            "val_targets": val_targets
                        }
                        torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                        print(f"  [!] Best model saved to {os.path.join(fold_dir, 'best_model.pth')}")
                        patience_counter = 0
                        if fully_unfrozen:
                            early_stop_counter = 0
                    else:
                        patience_counter += 1
                        if fully_unfrozen:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                print(f"Early stopping triggered after {patience} epochs without improvement (fully unfrozen).")
                                with open(os.path.join(fold_dir, f"fold{fold}_console.log"), 'a') as flog:
                                    flog.write(f"Early stopping triggered after {patience} epochs without improvement (fully unfrozen).\n")
                                break
                        # If patience exceeded, unfreeze next block
                        if not fully_unfrozen and patience_counter >= patience:
                            unfrozen = sequentially_unfreeze(model, unfrozen_blocks)
                            if unfrozen:
                                unfrozen_blocks += 1
                                print(f"Unfroze block {unfrozen_blocks} for model {model_name} fold {fold}")
                                with open(os.path.join(fold_dir, f"fold{fold}_console.log"), 'a') as flog:
                                    flog.write(f"Unfroze block {unfrozen_blocks} for model {model_name} fold {fold}\n")
                                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
                                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                                patience_counter = 0
                                if unfrozen_blocks == num_blocks:
                                    fully_unfrozen = True
                                    print(f"All blocks are now unfrozen for model {model_name} fold {fold}. Activating early stopping logic.")
                                    with open(os.path.join(fold_dir, f"fold{fold}_console.log"), 'a') as flog:
                                        flog.write(f"All blocks are now unfrozen for model {model_name} fold {fold}. Activating early stopping logic.\n")
                            else:
                                fully_unfrozen = True  # Should only be possible if model has no blocks.

                # Save curves, logs, final metrics, confusion matrix
                plot_curves(log, fold_dir, fold, model_name)
                save_confusion_matrix(best_state["confusion_matrix"], fold_dir, fold)
                save_metrics_csv(log, fold_dir, fold)
                final_metrics = {
                    "val_loss": best_state["metrics"]["val_loss"],
                    "val_acc": best_state["metrics"]["val_acc"],
                    "val_bal_acc": best_state["metrics"]["val_bal_acc"],
                    "val_f1": best_state["metrics"]["val_f1"],
                    "val_auc": best_state["metrics"]["val_auc"],
                    "fold": fold,
                    "model": model_name
                }
                save_final_metrics(final_metrics, fold_dir, fold)
                y_df = pd.DataFrame({"y_true": best_state["val_targets"], "y_pred": best_state["val_preds"]})
                y_df.to_csv(os.path.join(fold_dir, f"fold{fold}_val_predictions.csv"), index=False)
                print(f"\nBest balanced accuracy for {model_name} fold {fold}: {best_bal_acc:.4f}")
                print(f"Results saved in {fold_dir}")

            # ========== ENSEMBLE =============
            if ensemble and len(model_names) > 1:
                print(f"\n--- Ensembling models for {model_name} ---")
                for fold in range(1, 6):
                    fold_dir = os.path.join(model_dir, f"fold{fold}")
                    val_csv = os.path.join(folds_dir, f"val_fold{fold}.csv")
                    val_ds = ImageDataset(val_csv, data_root, transform=get_transforms(model_name, train=False))
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                    model_paths = [os.path.join(RESULTS_PARENT, mname, f"fold{fold}", "best_model.pth") for mname in model_names]
                    preds, probs = ensemble_predict(
                        model_paths, model_names, val_loader, torch.nn.CrossEntropyLoss(), device,
                        num_classes=2, tta=tta, tta_transforms=tta_transforms
                    )
                    y_true = val_ds.labels
                    acc = accuracy_score(y_true, preds)
                    bal_acc = balanced_accuracy_score(y_true, preds)
                    f1 = f1_score(y_true, preds, average="macro")
                    try:
                        auc = roc_auc_score(y_true, probs)
                    except:
                        auc = 0.5
                    cm = confusion_matrix(y_true, preds)
                    print(f"Ensemble Fold {fold}: acc={acc:.3f}, bal_acc={bal_acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")
                    save_confusion_matrix(cm, fold_dir, f"{fold}_ensemble")
                    pd.DataFrame({"y_true": y_true, "y_pred": preds, "prob": probs}).to_csv(
                        os.path.join(fold_dir, f"fold{fold}_ensemble_val_predictions.csv"), index=False
                    )

    # ============ CLI ============

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Image-level classification with gradual unfreezing, manual weighted loss, patient-level split, advanced augmentation, dropout, weight decay, warmup, MixUp/CutMix, TTA, ensembling, and early stopping after full unfreezing.")
        parser.add_argument('--csv', type=str, required=False, help="(Unused) Input CSV with columns: filename,label (not required if folds already exist)")
        parser.add_argument('--data_root', type=str, required=True, help="Root folder containing images")
        parser.add_argument('--folds_dir', type=str, required=True, help="Directory with train_fold#.csv and val_fold#.csv files")
        parser.add_argument('--epochs', type=int, default=120)
        parser.add_argument('--batch_size', type=int, default=48)
        parser.add_argument('--learning_rate', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=3e-5)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--class_weights', type=float, nargs='*', default=None, help="Manual class weights as space-separated floats, e.g. --class_weights 1.0 3.0")
        parser.add_argument('--mixup', action='store_true', help="Enable MixUp augmentation")
        parser.add_argument('--cutmix', action='store_true', help="Enable CutMix augmentation")
        parser.add_argument('--mixup_alpha', type=float, default=0.1, help="Alpha value for MixUp (default 0.4)")
        parser.add_argument('--cutmix_alpha', type=float, default=1.0, help="Alpha value for CutMix (default 1.0)")
        parser.add_argument('--tta', type=int, default=1, help="Number of Test Time Augmentations (default 1, no TTA)")
        parser.add_argument('--ensemble', action='store_true', help="Enable ensembling across model_names")
        parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of warmup epochs (default 5)")
        # Remove --make_folds_only
        args = parser.parse_args()

        # DO NOT call create_folds.
        # Assume --folds_dir contains train_fold1.csv, val_fold1.csv, ..., train_fold5.csv, val_fold5.csv

        model_names = [
            # "efficientnet_b4",
            # "resnet18",
            # "resnet50",
            # "swinv2",
            "vit"
        ]
        train_image_level(
            data_root=args.data_root,
            folds_dir=args.folds_dir,
            model_names=model_names,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            drop_rate=args.drop_rate,
            patience=args.patience,
            class_weights=args.class_weights,
            mixup=args.mixup,
            cutmix=args.cutmix,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            tta=args.tta,
            ensemble=args.ensemble,
            warmup_epochs=args.warmup_epochs
        )

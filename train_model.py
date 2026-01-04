import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from log_results import log_epoch_result

def train_model(exp_name, model, dataloaders, criterion, optimizer, num_epochs, device="cuda"):
    since = time.time()
    model = model.to(device)
    best_f1 = 0.0
    best_val_acc = 0.0
    best_model_wts = None
    
    # [cite_start]建立結果儲存資料夾 [cite: 40]
    os.makedirs("results", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # --- 訓練階段 ---
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (running_corrects.double() / total).item()
        # --- 驗證階段 ---
        model.eval()
        val_corrects, val_total, val_preds, val_labels = 0, 0, [], []
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        # [cite_start]計算指標：Top-1 Acc 與 Macro-F1 [cite: 53]
        val_acc = (val_corrects.double() / val_total).item()
        macro_f1 = f1_score(val_labels, val_preds, average='macro')
        
        epoch_time = time.time() - epoch_start
        print(f"[{exp_name}] E{epoch} Loss:{train_loss:.4f} Acc:{train_acc:.4f} Val_Acc:{val_acc:.4f} F1:{macro_f1:.4f}")

        # [cite_start]紀錄至 CSV [cite: 40]
        log_epoch_result("training_results.csv", exp_name, epoch, train_loss, train_acc, val_acc, macro_f1)

        # [cite_start]儲存最佳模型 [cite: 39]
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy() 
            torch.save(best_model_wts, os.path.join("results", f"{exp_name}_best.pth"))

    # 訓練結束總結
    total_time = time.time() - since
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        
    # [cite_start]建立回傳字典供總表使用 [cite: 39, 40]
    summary_dict = {
        "best_macro_f1": round(best_f1, 4),
        "best_val_acc": round(best_val_acc, 4),
        "train_time_sec": round(total_time, 2)
    }

    return model, summary_dict
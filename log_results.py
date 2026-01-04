# 幫你設計一個 通用工具函數，讓你在每個 train_model() epoch 結束後呼叫它，就能自動把結果 append 到統一的 CSV 檔。


import os
import csv

def log_epoch_result(csv_path, exp_name, epoch, train_loss, train_acc, val_acc, macro_f1):
    """
    Append one epoch's training/validation result into a CSV file.
    
    Args:
        csv_path (str): CSV 檔路徑 (統一紀錄所有模型的結果)
        exp_name (str): 實驗名稱 (模型名)
        epoch (int): epoch 編號
        train_loss (float): 訓練損失
        train_acc (float): 訓練準確率
        val_acc (float): 驗證準確率
        macro_f1 (float): 驗證 F1 score (macro)
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        # 如果是新檔案，加上標題列
        if not file_exists:
            writer.writerow(["exp_name", "epoch", "train_loss", "train_acc", "val_acc", "macro_f1"])
        
        # 寫入數據
        writer.writerow([exp_name, epoch, train_loss, train_acc, val_acc, macro_f1])


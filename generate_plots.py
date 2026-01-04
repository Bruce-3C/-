# 獨立繪圖腳本.可以在不重新訓練模型的情況下，隨時讀取 training_results.csv 並產出圖表。 
# 確認你的資料夾裡有 training_results.csv，然後在終端機輸入 python generate_plots.py。
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_experiment_plots(csv_path="training_results.csv", output_dir="results"):
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取 CSV 資料
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到檔案 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 取得所有不同的實驗名稱
    experiments = df['exp_name'].unique()
    
    for exp in experiments:
        print(f"正在為實驗 {exp} 產生圖表...")
        exp_data = df[df['exp_name'] == exp]
        
        # 建立畫布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 繪製訓練損失 (Loss) 曲線
        ax1.plot(exp_data['epoch'], exp_data['train_loss'], color='tab:red', label='Train Loss', linewidth=2)
        ax1.set_title(f'{exp} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()

        # 2. 繪製驗證準確率 (Val Acc) 與 Macro-F1 曲線
        ax2.plot(exp_data['epoch'], exp_data['val_acc'], color='tab:blue', label='Val Acc', linewidth=2)
        ax2.plot(exp_data['epoch'], exp_data['macro_f1'], color='tab:green', label='Macro-F1', linestyle='--', linewidth=2)
        ax2.set_title(f'{exp} - Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1) # 分數通常在 0 到 1 之間
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        
        # 儲存圖表
        save_path = os.path.join(output_dir, f"{exp}_performance.png")
        plt.savefig(save_path, dpi=300) # 設定高解析度以便放進論文
        plt.close()
        print(f"圖表已儲存至：{save_path}")

if __name__ == "__main__":
    generate_experiment_plots()
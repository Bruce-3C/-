# 最終版train_experiments.py
# 「一鍵全模型訓練版本」的 train_experiments.py，可以單獨訓練某一個模型，也可以用 --run_all 自動依序訓練 AlexNet、GoogLeNet、ResNet-50、EfficientNet-B0、MobileNet-V3_Large，最後還會自動匯出一份比較表（CSV）。
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from train_model import train_model
import pandas as pd
import time
import os

# ------------------------
# 1. 資料加載函式
# ------------------------
def get_dataloaders(data_dir, batch_size=32):
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform["train"]),
        "val": datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform["val"])
    }

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    num_classes = len(image_datasets["train"].classes)
    print(f"📊 Detected {num_classes} classes: {image_datasets['train'].classes}")
    return dataloaders, num_classes


# ------------------------
# 2. 模型建立函式
# ------------------------
def build_model(model_name, num_classes):
    if model_name == "alexnet":
        model = models.alexnet(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
         model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
         model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"❌ Unsupported model: {model_name}")
    return model


# ------------------------
# 3. 訓練流程 (修正後的版本)
# ------------------------
def run_training(model_name, data_dir, batch_size, lr, epochs, device):
    dataloaders, num_classes = get_dataloaders(data_dir, batch_size)
    model = build_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    
    # 💥 修正：假設 train_model 現在回傳 (模型, 結果字典)
    # 如果你的 train_model 只回傳模型，這個程式碼會崩潰！
    # 你必須修改 train_model.py
    trained_model, result = train_model(model_name, model, dataloaders, criterion, optimizer,
                                        num_epochs=epochs, device=device)
    
    elapsed = (time.time() - start) / 60
    
    # 檢查是否為字典，如果 train_model 回傳錯誤，這裡會修正
    if not isinstance(result, dict):
        # 這是應急代碼：如果 train_model 只回傳了模型，我們需要手動建立字典，但無法取得最佳 F1
        result = {} 
        # 建議：確保 train_model 回傳最佳 F1
        
    result["train_time(min)"] = round(elapsed, 2)
    
    # 我們不再需要回傳模型，因為它已經被儲存到檔案
    return result

# ------------------------
# 4. 主程式入口
# ------------------------
def main(args):
    os.makedirs("results", exist_ok=True)
    summary = []

    # 單模型訓練
    if not args.run_all:
        res = run_training(args.model, args.data_dir, args.batch_size,
                           args.lr, args.epochs, args.device)
        summary.append({"Model": args.model, **res})
    else:
        models_to_run = ["alexnet", "googlenet", "resnet50", "efficientnet_b0", "mobilenet_v3_large"]
        print(f"🚀 開始依序訓練 {len(models_to_run)} 個模型...")
        for name in models_to_run:
            res = run_training(name, args.data_dir, args.batch_size,
                               args.lr, args.epochs, args.device)
            summary.append({"Model": name, **res})

    # 匯出結果
    df = pd.DataFrame(summary)
    csv_path = os.path.join("results", f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📄 所有結果已匯出到 {csv_path}")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset path (must contain train/ and val/)")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["alexnet", "googlenet", "resnet50", "efficientnet_b0", "mobilenet_v3_large"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_all", action="store_true", help="Train all five models sequentially")
    args = parser.parse_args()

    main(args)

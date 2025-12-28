import os
from PIL import Image
from torchvision import transforms

# 1. 設定路徑
input_folder = r"C:\DataSet\hard_samples" 
output_folder = r"C:\DataSet\FDM-3D-Printing-Defect-Dataset_6G\train\ok_336"
os.makedirs(output_folder, exist_ok=True)

# 2. 定義 TDA 增補策略 (包含自動縮放格式)
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 自動解決高解析度問題
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30), # 強制模型學習不同角度的幾何結構
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# 3. 執行批次增補
images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]

print(f"🚀 開始針對 {len(images)} 張困難樣本進行 TDA 增補...")

for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    
    # 每張圖產生 35 張變體，共約 100 張新樣本
    for i in range(35):
        aug_img = augment_transform(img)
        save_name = f"aug_hard_{i}_{img_name}"
        aug_img.save(os.path.join(output_folder, save_name))

print(f"✅ 增補完成！新樣本已存入: {output_folder}")
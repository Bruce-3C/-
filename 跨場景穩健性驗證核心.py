import os
import sys
import torch
import numpy as np
import cv2

# 強制導入當前同級目錄，確保載入模型
sys.path.insert(0, os.path.dirname(__file__))

def apply_illumination_stress(img, alpha=0.5, beta=-30):
    """
    光照強度變異壓力測試 (模擬現場極端曝光或採光不足)
    參數:
        alpha: 對比度調整係數 (0.5代表對比度減半，1.5代表高曝光)
        beta: 亮度調整偏差值 (-30代表環境昏暗，30代表強光干擾)
    """
    stressed_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return stressed_img

def apply_gaussian_noise_stress(img, mean=0, sigma=25):
    """
    高斯雜訊干擾壓力測試 (模擬感測器老化、訊號傳輸干擾或機台震動模糊)
    """
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    stressed_img = img.astype(np.float32) + noise
    stressed_img = np.clip(stressed_img, 0, 255).astype(np.uint8)
    return stressed_img

def evaluate_scenario_robustness(model, img_path, target_class_idx, device):
    """
    執行異質環境跨場景壓力測試與穩健性量化評估
    """
    model.eval()
    
    # 支援中文路徑讀取
    img_array = np.fromfile(img_path, np.uint8)
    src_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 定義四種異質環境測試場景
    scenarios = {
        "原始標準場景": src_img.copy(),
        "微弱光照壓力場景": apply_illumination_stress(src_img, alpha=0.6, beta=-40),
        "極端曝光壓力場景": apply_illumination_stress(src_img, alpha=1.4, beta=40),
        "感測器高斯雜訊場景": apply_gaussian_noise_stress(src_img, mean=0, sigma=30)
    }
    
    print(f"\n================ 跨場景穩健性壓力測試 ================")
    results_summary = {}
    
    for scenario_name, processed_img in scenarios.items():
        # 影像前處理至模型輸入尺寸
        img_resized = cv2.resize(processed_img, (224, 224))
        img_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
        
        # 推理預測
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            target_confidence = probabilities[target_class_idx].item()
            pred_class_idx = torch.argmax(probabilities).item()
            
        results_summary[scenario_name] = target_confidence
        print(f"[{scenario_name}] 目標類別置信度: {target_confidence:.4f} | 預測類別代碼: {pred_class_idx}")
        
        # 儲存異質環境模擬影像供論文圖表檢視
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, f"stress_{scenario_name}.png")
        _, img_encoded = cv2.imencode('.png', processed_img)
        img_encoded.tofile(save_path)
        
    print(f"======================================================\n")
    return results_summary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"當前運算裝置: {device}")
    
    # 1. 載入實體 FAC-NET 模型與本機架構
    try:
        from FACNet import FACNet
        model = FACNet(num_classes=6)
        weight_path = r"C:\DataSet\FAC-NET\舊電腦\FAC-NET\models\best_FACNet_版本管理_train_FACNet_v1_20260508_125300.pth"
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("【成功】已成功載入 FAC-NET 實體訓練權重進行穩健性測試。")
        model = model.to(device)
    except Exception as e:
        print(f"核心架構對接失敗: {e}")
        sys.exit(1)
        
    # 2. 指定測試樣本路徑 (此處以層位移樣本為例)
    image_path = r"C:\DataSet\FAC-NET\split_data\test\Layer_shifting\Image_20231128172036965.jpg"
    target_class = 3 # 層位移類別索引
    
    if not os.path.exists(image_path):
        print(f"找不到測試影像: {image_path}")
    else:
        # 3. 啟動異質環境評估
        evaluate_scenario_robustness(model, image_path, target_class, device)
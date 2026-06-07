import os
import sys
import torch
import numpy as np
import cv2

# 強制導入當前同級目錄
sys.path.insert(0, os.path.dirname(__file__))

def perform_occlusion_test(model, input_tensor, target_class_idx, window_size=32, stride=8):
    model.eval()
    _, channels, height, width = input_tensor.shape
    
    with torch.no_grad():
        original_output = torch.softmax(model(input_tensor), dim=1)
        original_confidence = original_output[0, target_class_idx].item()
    
    confidence_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    min_confidence = original_confidence
    
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            occluded_tensor = input_tensor.clone()
            occluded_tensor[0, :, y:y+window_size, x:x+window_size] = 0.0
            
            with torch.no_grad():
                output = torch.softmax(model(occluded_tensor), dim=1)
                current_confidence = output[0, target_class_idx].item()
            
            if current_confidence < min_confidence:
                min_confidence = current_confidence
            
            confidence_map[y:y+window_size, x:x+window_size] += current_confidence
            count_map[y:y+window_size, x:x+window_size] += 1.0

    count_map[count_map == 0] = 1.0
    heatmap = confidence_map / count_map
    max_cdr = ((original_confidence - min_confidence) / original_confidence) * 100.0
    
    print(f"原始置信度: {original_confidence:.4f}")
    print(f"遮擋後最低置信度: {min_confidence:.4f}")
    print(f"最大CDR: {max_cdr:.2f}%")
    
    return heatmap, max_cdr

def visualize_decision_area(heatmap, original_img_path, save_name="occlusion_heatmap_result.png"):
    # 支援中文路徑的讀取方式
    img_array = np.fromfile(original_img_path, np.uint8)
    bgr_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h, w, _ = bgr_img.shape
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_norm = cv2.normalize(heatmap_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_norm = 255 - heatmap_norm.astype(np.uint8) 
    
    color_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(bgr_img, 0.6, color_heatmap, 0.4, 0)
    
    # 核心修正：改用相對路徑並透過 imencode 寫入，完全繞過 Windows 絕對路徑與中文編碼限制
    current_dir = os.path.dirname(os.path.abspath(__file__))
    final_save_path = os.path.join(current_dir, save_name)
    
    _, img_encoded = cv2.imencode('.png', overlay_img)
    img_encoded.tofile(final_save_path)
    return final_save_path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"當前運算裝置: {device}")
    
    from FACNet import FACNet
    model = FACNet(num_classes=6)
    
    weight_path = r"C:\DataSet\FAC-NET\舊電腦\FAC-NET\models\best_FACNet_版本管理_train_FACNet_v1_20260508_125300.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print("【成功】已成功載入 FAC-NET 實體訓練權重！")
    model = model.to(device)
    
    # 測試影像路徑
    image_path = r"C:\DataSet\FAC-NET\split_data\test\Layer_shifting\Image_20231128172036965.jpg"
    
    if not os.path.exists(image_path):
        print(f"找不到測試影像: {image_path}")
    else:
        # 使用不懼怕中文路徑的解碼方式
        img_array = np.fromfile(image_path, np.uint8)
        src_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(src_img, (224, 224)) 
        
        img_tensor = img_resized.transpose((2, 0, 1)) / 255.0
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
        
        print("開始執行遮擋測試掃描...")
        heatmap, max_cdr = perform_occlusion_test(model, img_tensor, target_class_idx=3)
        
        # 儲存熱圖結果
        saved_path = visualize_decision_area(heatmap, image_path, "occlusion_heatmap_result.png")
        print(f"測試完成！結果圖表已確切儲存至: {saved_path}")
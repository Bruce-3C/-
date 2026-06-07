import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches

# 設定標楷體與微軟正黑體，確保中文正常顯示
plt.rcParams['font.sans-serif'] = ['DFKai-SB', 'BiauKai', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_3d_printing_object(ax, x, y, width, height, color='#80c0ff'):
    """繪製3D列印物件主體"""
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)

def add_layer_shifting(ax):
    """添加層位移瑕疵特徵"""
    upper = Rectangle((50, 50), 40, 25, facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(upper)
    ax.plot([30, 85], [50, 50], 'r--', linewidth=2)
    ax.text(57, 46, '位移界面', ha='center', color='red', fontsize=10, fontweight='bold')

def add_warping(ax):
    """添加翹曲瑕疵特徵"""
    warp = Polygon([[30, 20], [48, 23], [48, 60], [30, 60]], 
                   facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(warp)
    ax.annotate('', xy=(30, 65), xytext=(48, 65), arrowprops=dict(arrowstyle='->', color='red'))
    ax.text(39, 69, '翹曲', ha='center', color='red', fontsize=10, fontweight='bold')

def add_stringing(ax):
    """添加拉絲瑕疵特徵"""
    for (x_start, y_start, x_end, y_end) in [
        (75, 55, 92, 65), (78, 50, 95, 58), (72, 45, 90, 55),
        (35, 30, 22, 20), (38, 28, 20, 18)
    ]:
        ax.plot([x_start, x_end], [y_start, y_end], 'r-', linewidth=2, alpha=0.9)
    ax.text(87, 69, '拉絲', ha='center', color='red', fontsize=10, fontweight='bold')

def add_cracking(ax):
    """添加裂紋瑕疵特徵"""
    crack_points = [(40, 35), (52, 38), (58, 36), (65, 40), (70, 38)]
    xs, ys = zip(*crack_points)
    ax.plot(xs, ys, 'r-', linewidth=2.5)
    ax.plot([70, 76], [38, 43], 'r-', linewidth=2)
    ax.text(55, 47, '裂紋', ha='center', color='red', fontsize=10, fontweight='bold')

def add_off_platform(ax):
    """添加偏離平台瑕疵特徵"""
    obj = Rectangle((60, 25), 38, 30, facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(obj)
    orig = Rectangle((35, 25), 38, 30, facecolor='none', edgecolor='gray', linestyle='--', linewidth=1.5)
    ax.add_patch(orig)
    ax.annotate('', xy=(60, 40), xytext=(35, 40), arrowprops=dict(arrowstyle='->', color='red'))
    ax.text(47, 20, '脫離位移', ha='center', color='red', fontsize=10, fontweight='bold')

def add_environment_effects(ax):
    """添加環境參數效果（鏡面反射與陰影）"""
    highlight = patches.Ellipse((65, 70), 22, 14, facecolor='white', alpha=0.5)
    ax.add_patch(highlight)
    highlight2 = patches.Ellipse((40, 75), 12, 8, facecolor='white', alpha=0.4)
    ax.add_patch(highlight2)
    shadow = patches.Polygon([[15, 55], [40, 80], [75, 55], [75, 35], [40, 15], [15, 35]], 
                             facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)

def setup_single_canvas():
    """初始化單一正方形圖表畫布配置"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 100) # 為底部圖表子標題保留對齊空間
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax

def save_single_chart(fig, filename):
    """去除邊白並高解析度輸出圖片"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def export_twenty_tda_images(defect_type, defect_name):
    """為單一瑕疵類別導出abcd四張獨立影像"""
    
    # ------------------------------------------------------
    # (a) 獨立圖：原始正常樣本
    # ------------------------------------------------------
    fig, ax = setup_single_canvas()
    create_3d_printing_object(ax, 35, 25, 45, 50, color='#80c0ff')
    platform = Rectangle((25, 15), 65, 10, facecolor='#c0c0c0', edgecolor='black')
    ax.add_patch(platform)
    ax.text(50, -12, '(a) 原始正常樣本', ha='center', va='top', fontsize=12, fontweight='bold')
    save_single_chart(fig, f'TDA_{defect_type}_a.png')
    
    # ------------------------------------------------------
    # (b) 獨立圖：疊加瑕疵特徵
    # ------------------------------------------------------
    fig, ax = setup_single_canvas()
    create_3d_printing_object(ax, 35, 25, 45, 50, color='#ffcccc')
    ax.add_patch(Rectangle((25, 15), 65, 10, facecolor='#c0c0c0', edgecolor='black'))
    
    if defect_type == 'layer_shifting':
        add_layer_shifting(ax)
    elif defect_type == 'warping':
        add_warping(ax)
    elif defect_type == 'stringing':
        add_stringing(ax)
    elif defect_type == 'cracking':
        add_cracking(ax)
    elif defect_type == 'off_platform':
        add_off_platform(ax)
        
    ax.text(50, -12, '(b) 疊加瑕疵特徵\n(幾何參數$\\Gamma_{defect}$)', ha='center', va='top', fontsize=11, fontweight='bold')
    save_single_chart(fig, f'TDA_{defect_type}_b.png')
    
    # ------------------------------------------------------
    # (c) 獨立圖：加入環境變異
    # ------------------------------------------------------
    fig, ax = setup_single_canvas()
    create_3d_printing_object(ax, 35, 25, 45, 50, color='#ffcccc')
    ax.add_patch(Rectangle((25, 15), 65, 10, facecolor='#c0c0c0', edgecolor='black'))
    
    if defect_type == 'layer_shifting':
        add_layer_shifting(ax)
    elif defect_type == 'warping':
        add_warping(ax)
    elif defect_type == 'stringing':
        add_stringing(ax)
    elif defect_type == 'cracking':
        add_cracking(ax)
    elif defect_type == 'off_platform':
        add_off_platform(ax)
    
    add_environment_effects(ax)
    ax.text(50, -12, '(c) 加入環境變異\n(環境參數$\\Delta_{env}$：鏡面反射+陰影)', ha='center', va='top', fontsize=10, fontweight='bold')
    save_single_chart(fig, f'TDA_{defect_type}_c.png')
    
    # ------------------------------------------------------
    # (d) 獨立圖：TDA最終擴充樣本
    # ------------------------------------------------------
    fig, ax = setup_single_canvas()
    create_3d_printing_object(ax, 35, 25, 45, 50, color='#ff8888')
    ax.add_patch(Rectangle((25, 15), 65, 10, facecolor='#a0a0a0', edgecolor='black'))
    
    if defect_type == 'layer_shifting':
        add_layer_shifting(ax)
    elif defect_type == 'warping':
        add_warping(ax)
    elif defect_type == 'stringing':
        add_stringing(ax)
    elif defect_type == 'cracking':
        add_cracking(ax)
    elif defect_type == 'off_platform':
        add_off_platform(ax)
    
    add_environment_effects(ax)
    ax.text(50, -12, '(d) TDA最終擴充樣本\n(納入訓練資料集)', ha='center', va='top', fontsize=11, fontweight='bold')
    save_single_chart(fig, f'TDA_{defect_type}_d.png')

if __name__ == "__main__":
    defects = [
        ('layer_shifting', '層位移'),
        ('warping', '翹曲'),
        ('stringing', '拉絲'),
        ('cracking', '裂紋'),
        ('off_platform', '偏離平台')
    ]
    
    print("開始執行目標資料增補(TDA)多執行流圖表分拆程序...\n")
    
    total_images = 0
    for defect_type, defect_name in defects:
        export_twenty_tda_images(defect_type, defect_name)
        print(f"【成功導出】已完成 {defect_name} 瑕疵之 (a)(b)(c)(d) 共 4 張獨立正方形影像。")
        total_images += 4
        
    print(f"\n程序執行完畢！已成功產出全套共 {total_images} 張高解析度學術獨立圖表。")
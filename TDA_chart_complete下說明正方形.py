# TDA_chart_complete下說明正方形.py
# 產生5類瑕疵之TDA增補前後對照圖（繁體中文 + 標楷體 + 2x2 正方形矩陣編排）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches

# 設定標楷體
plt.rcParams['font.sans-serif'] = ['DFKai-SB', 'BiauKai', 'Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_3d_printing_object(ax, x, y, width, height, color='#80c0ff'):
    """繪製3D列印物件"""
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)

def add_layer_shifting(ax):
    """添加層位移瑕疵"""
    upper = Rectangle((50, 50), 40, 25, facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(upper)
    ax.plot([30, 85], [50, 50], 'r--', linewidth=2)
    ax.text(57, 46, '位移界面', ha='center', color='red', fontsize=10, fontweight='bold')

def add_warping(ax):
    """添加翹曲瑕疵"""
    warp = Polygon([[30, 20], [48, 23], [48, 60], [30, 60]], 
                   facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(warp)
    ax.annotate('', xy=(30, 65), xytext=(48, 65), arrowprops=dict(arrowstyle='->', color='red'))
    ax.text(39, 69, '翹曲', ha='center', color='red', fontsize=10, fontweight='bold')

def add_stringing(ax):
    """添加拉絲瑕疵"""
    for (x_start, y_start, x_end, y_end) in [
        (75, 55, 92, 65), (78, 50, 95, 58), (72, 45, 90, 55),
        (35, 30, 22, 20), (38, 28, 20, 18)
    ]:
        ax.plot([x_start, x_end], [y_start, y_end], 'r-', linewidth=2, alpha=0.9)
    ax.text(87, 69, '拉絲', ha='center', color='red', fontsize=10, fontweight='bold')

def add_cracking(ax):
    """添加裂紋瑕疵"""
    crack_points = [(40, 35), (52, 38), (58, 36), (65, 40), (70, 38)]
    xs, ys = zip(*crack_points)
    ax.plot(xs, ys, 'r-', linewidth=2.5)
    ax.plot([70, 76], [38, 43], 'r-', linewidth=2)
    ax.text(55, 47, '裂紋', ha='center', color='red', fontsize=10, fontweight='bold')

def add_off_platform(ax):
    """添加偏離平台瑕疵"""
    obj = Rectangle((60, 25), 38, 30, facecolor='#ff8888', edgecolor='black', linewidth=1.5)
    ax.add_patch(obj)
    orig = Rectangle((35, 25), 38, 30, facecolor='none', edgecolor='gray', linestyle='--', linewidth=1.5)
    ax.add_patch(orig)
    ax.annotate('', xy=(60, 40), xytext=(35, 40), arrowprops=dict(arrowstyle='->', color='red'))
    ax.text(47, 20, '脫離位移', ha='center', color='red', fontsize=10, fontweight='bold')

def add_environment_effects(ax):
    """添加環境參數效果（鏡面反射+陰影）"""
    highlight = patches.Ellipse((65, 70), 22, 14, facecolor='white', alpha=0.5)
    ax.add_patch(highlight)
    highlight2 = patches.Ellipse((40, 75), 12, 8, facecolor='white', alpha=0.4)
    ax.add_patch(highlight2)
    shadow = patches.Polygon([[15, 55], [40, 80], [75, 55], [75, 35], [40, 15], [15, 35]], 
                              facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)

def create_tda_comparison(defect_type, defect_name, fig_num):
    """建立單一瑕疵類別之TDA增補前後對照圖（2x2 排版，說明在各圖下方）"""
    # 調整畫布尺寸 (寬8, 高8.5) 讓整張大圖逼近正方形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8.5))
    axes = [ax1, ax2, ax3, ax4]
    
    # 統一設定子圖邊界與 1:1 比例
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(-25, 100)  # 下界調至 -25，為 2x2 緊湊排版預留足夠的文字空間
        ax.set_aspect('equal')
        ax.axis('off')

    # ----------------------------------------
    # (a) 左上：原始正常樣本
    # ----------------------------------------
    ax = ax1
    ax.set_facecolor('#f5f5f5')
    create_3d_printing_object(ax, 35, 25, 45, 50, color='#80c0ff')
    platform = Rectangle((25, 15), 65, 10, facecolor='#c0c0c0', edgecolor='black')
    ax.add_patch(platform)
    ax.text(50, -15, '(a) 原始正常樣本', ha='center', va='top', fontsize=11, fontweight='bold')
    
    # ----------------------------------------
    # (b) 右上：疊加瑕疵態樣
    # ----------------------------------------
    ax = ax2
    ax.set_facecolor('#f5f5f5')
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
        
    ax.text(50, -15, '(b) 疊加瑕疵特徵\n(幾何參數$\\Gamma_{defect}$)', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # ----------------------------------------
    # (c) 左下：加入環境變異
    # ----------------------------------------
    ax = ax3
    ax.set_facecolor('#f5f5f5')
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
    ax.text(50, -15, '(c) 加入環境變異\n(環境參數$\\Delta_{env}$：鏡面反射+陰影)', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # ----------------------------------------
    # (d) 右下：TDA最終擴充樣本
    # ----------------------------------------
    ax = ax4
    ax.set_facecolor('#e8e8e8')
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
    ax.text(50, -15, '(d) TDA最終擴充樣本\n(納入訓練資料集)', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # ----------------------------------------
    # 總標題與版面優化
    # ----------------------------------------
   # plt.suptitle(f'圖3.3-{fig_num} TDA增補前後對照圖\n（以{defect_name}瑕疵為例）', 
   #              fontsize=13, fontweight='bold', y=0.96, va='top')
    
    plt.tight_layout()
    # 稍微拉開上下兩組圖的垂直間距 (hspace)，避免上組圖的底字黏到下組圖
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.4, wspace=0.15) 
    
    plt.savefig(f'TDA_Comparison_{defect_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 瑕疵類別與對應圖號
defects = [
    ('layer_shifting', '層位移', '1'),
    ('warping', '翹曲', '2'),
    ('stringing', '拉絲', '3'),
    ('cracking', '裂紋', '4'),
    ('off_platform', '偏離平台', '5')
]

# 產生5類瑕疵之對照圖
for defect_type, defect_name, fig_num in defects:
    create_tda_comparison(defect_type, defect_name, fig_num)
    print(f'已產生：圖3.3-{fig_num} {defect_name}瑕疵對照圖 (格式：2x2 上下排列)')
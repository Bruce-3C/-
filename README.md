# -「一鍵全模型訓練版本」的 train_experiments.py，可以單獨訓練某一個模型，也可以用 --run_all 自動依序訓練 AlexNet、GoogLeNet、ResNet-50、EfficientNet-B0、MobileNetV3 large，最後還會自動匯出一份比較表（CSV）。
所以這個實驗的軟體架構是3個py所組成,分別是:
1.	train_experiments.py 實驗執行的主程式,可以單訓練一個模型或是用--run_all 自動依序訓練5個模型 
2.	train_model.py 訓練模型的程式
3.	log_results.py 產生一份比較表

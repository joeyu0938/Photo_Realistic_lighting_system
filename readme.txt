
目的: 將一張 360 的 LDR 圖片透過機器學習生成光源所在位置用來優化渲染並應用到 AR 中

input: 360 LDR 圖片
       -放在 Loader/images_LDR 下

執行: 在 LTL_learning 目錄下執行
      -執行 python LDR2HDRinference.py: 將 LDR 圖片轉換成 HDR
        ~ 結果存在 Loader/images_HDR
      -執行 python Loader/Filter_pytorch.py: 將 HDR 轉換成 regression
        ~ 結果存在 Loader/to_predict
      -執行 python inference.py: 完成所有結果

------------------------------------------------------------------------------

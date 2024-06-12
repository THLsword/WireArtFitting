通過GPU上的迭代，使用柏志的patch loss產生wire art

## 運行順序
- 先運行 render_utils 中的 run.sh
    - render -> alpha shape -> expand -> train
- 前面三部都做好的話直接運行 demo_deform.py(train)

## training 權重以及影響
- chamfer_loss
    - 1
- overlap_loss
    - 0.01 
- planar_loss
    - 2 
    - 試過小於1， 對結果有影響，而且結果會變差
- symmetry_loss
    - 0.1
- curvature_loss
    - 0.015
    - 1. 對於curve採樣點的權重分配
        - `(i-0.5)**8*64`
    - 2. 不同大小的loss weight影響
        - 1.0 會讓整個fitting出問題
- normal loss
    - 0.008: 腳還是不會分開
    - 0.08: 前腳會糾纏在一起
    - 0.1 * math.exp((i-(epoch_num*0.7))/100) 逐漸增大，在epoch num*0.7時權重為1
        - max(1, math.exp((i-(epoch_num*0.7))/100)) 不超過1 (沒區別)
- distance weight
    - 1 
    - ((d-m)*0.2) + 1
    - ((m-d)*0.2) + 1
        - 似乎兩腿之間收斂的比較快
        - 在ox上沒差，反而不如d-m
- mv weight
    - 看不出太大區別

## 動物模型
- cat：會在前100epoch無法分開前腿和後退
- dog：50epoch前腿和後退就完全分開了，效果非常好
- horse: 後腳完全沒有分離，並且扭曲重疊翻轉

# post-process
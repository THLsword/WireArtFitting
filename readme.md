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
        - 
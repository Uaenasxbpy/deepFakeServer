project_root/  
├── data/  
│   ├── coco/      # COCO数据集存放位置  
│   ├── processed/ # 处理后的数据集存放位置  
│   └── splits/    # 数据集划分文件存放位置  
├── models/  
│   ├── base_model.py  # 基础CNN模型定义  
│   └── forgery_detector.py  # 伪造检测模型定义  
├── utils/  
│   ├── data_loader.py  # 数据加载工具  
│   ├── metrics.py      # 评估指标计算工具  
│   └── visualization.py  # 可视化工具  
├── train.py         # 模型训练脚本  
├── evaluate.py      # 模型评估脚本  
├── optimize.py      # 模型优化脚本（可选）  
└── README.md        # 项目说明文档
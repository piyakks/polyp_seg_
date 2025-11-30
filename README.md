SAM과 Mamba구조를 이용한 PEFT(Parameter-Efficient Fine-Tuning)기반 Polyp Segmentation 모델 
## Model_Overview
<img width="4074" height="1547" alt="Polyp2" src="https://github.com/user-attachments/assets/9216536c-36c9-4192-b9da-bfabcf3cd296" />
## Environment
- **OS:** Ubuntu 20.04  
- **Python:** 3.10.18  
- **PyTorch:** 2.0.1  
- **CUDA:** 11.7  
- **GPU:** RTX 3090 24GB

---
## training
1. **기본 가중치 다운로드**  
[sam_vit_h_4b8939.pth]([https://github.com/hkchengrex/XMem](https://github.com/facebookresearch/segment-anything)) 를 다운로드 후 'checkpoint/` 폴더에 저장.

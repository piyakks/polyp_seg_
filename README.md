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
[sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything) 를 다운로드 후 `checkpoints/` 폴더에 저장.

```bash
checkpoints
 └─ sam_vit_h_4b8939.pth
```
2. 학습
```bash
python train_Extended_SAM.py
```
## inference

```bash
python tester.py
```
## Result
Total Param:745M

Learnable Param: 114M

<div style="display:flex; gap:8px; align-items:center; justify-content:center;">
  <img src="https://github.com/user-attachments/assets/6dd7d525-54e1-4ed7-8760-0f456b93f7e3" alt="이미지1" style="max-width:48%; height:auto;"/>
  <img src="https://github.com/user-attachments/assets/8cb0bd84-e9a2-423b-abdd-c73cbc7acf7d" alt="이미지2" style="max-width:48%; height:auto;"/>
</div>



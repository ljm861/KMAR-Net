# KMAR-Net
Application of an optimized Convolutional Neural Network (CNN) architecture to metal artifact reduction (MAR) in CT images

## Requirements
* Python (>3.5)
* Pytorch (>1.0)

## Run
* `Input = torch.randn(4, 1, 512, 512)` (Size of input images : 512 x 512)

```bash
python KMAR-Net.py
```

> output shape : torch.Size([4, 1, 512, 512])

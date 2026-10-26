| Metric                                |     ResNet-18 | MobileNetV3-Small |
| ------------------------------------- | ------------: | ----------------: |
| **Architecture**                      |  Residual CNN |   Lightweight CNN |
| **CIFAR-10 accuracy (Study 1)**       |    **96.52%** |            96.12% |
| **Parameters**                        | ~11.7 million |     ~2.54 million |
| **Weight precision**                  |          FP32 |              FP32 |
| **FP32 weight size**                  |      ~44.7 MB |           ~9.8 MB |
| **RTX 4060 training time**            |       ~90 min |     Not available |
| **RTX 4060 training epochs**          |           200 |     Not available |
| **RTX 4060 batch size**               |           128 |     Not available |
| **RTX 4060 training accuracy (test)** |        95.32% |     Not available |
| **Training time (Study 2)**           |    1.02 hours |        1.17 hours |
| **Inference latency (Study 2)**       |       16.6 ms |           28.0 ms |
| **CIFAR-10 accuracy (Study 2)**       |        88.61% |            74.18% |
| **RTX 4060 inference latency**        | Not available |     Not available |
| **RTX 4060 throughput**               | Not available |     Not available |


CNN1 EPOCH=10
GPU: NVIDIA RTX 4060 Laptop
Batch size: 64
Epochs: 10
Optimizer: Adam
Learning rate: 0.001

Training Results
Epoch	Training Time (s)	Test Accuracy
1	14.08	48.20%
2	13.84	54.00%
3	14.21	54.50%
4	13.90	56.82%
5	16.75	57.48%
6	18.21	59.69%
7	31.41	59.57%
8	13.52	60.71%
9	13.35	62.16%
10	13.30	61.22%
Overall Benchmark
Metric	Result
Total training time	162.57 s
Average epoch time	16.26 s
Best test accuracy	62.16%
Final test accuracy	61.22%
Parameters	11,642
Model weight size	0.04 MiB
Total inference time	0.347 s
Inference time per image (batched)	0.035 ms
Throughput	28,810 images/sec

Note: No cropping or data augmentation. Standard CNN with 2 convolutional layers, trained on CIFAR-10.
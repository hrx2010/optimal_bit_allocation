# optimal_bit_allocation

We provide the scripts and pre-trained models to reproduce the results reported in paper "Optimizing the Bit Allocation for Compression of Weights and Activations of Deep Neural Networks" published in ICIP 2019.

Please extract folders "bit_allocation_resnet50", "bit_allocation_vgg16", "codebooks_resnet50", "codebooks_vgg16" and "libs" before running the experiments.

The detailed information of each component in this library is listed below:

1. bit_allocation_resnet50: bit allocation files of resnet50
2. bit_allocation_vgg16: bit allocation files of vgg16
3. codebooks_resnet50: quantization codebooks of resnet50
4. codebooks_vgg16: quantization codebooks of vgg16
5. libs: third-party libraries.
6. models: pre-trained models

 - run "bash run_vgg.sh" to generate results of vgg-16 (results in figure 2)
 - run "bash run_resnet.sh" to generate results of resnet-50 (results in figure 3)

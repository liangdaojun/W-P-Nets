# WPNets and PWNets: from The Perspective of Channel Fusion

The performance and parameters of neural networks have a positive correlation, and there are a lot of parameters redundancy in the existing neural network architectures. By exploring the channels relationship of the whole and part of the neural network, the architectures of the convolution network with the tradeoff between the parameters and the performance are obtained. Two network architectures are implemented by dividing the convolution kernels of one layer into multiple groups, thus ensuring that the network has more connections and fewer parameters. In these two network architectures, one with information flowing from whole to part, which is called whole-to-part connected networks (WPNets), and the other one with information flowing from part to whole, which is called part-to-whole connected networks (PWNets). WPNets use the whole channel information to enhance partial channel information, and the PWNets use partial channel information to generate or enhance the whole channel information. 

<img src="https://github.com/liangdaojun/W-P-Nets/blob/master/images/WPNets.jpg" width="480">
<img src="https://github.com/liangdaojun/W-P-Nets/blob/master/images/PWNets.jpg" width="480">

## Training
.. code::
    python run.py --model_type=WPNet(PWNet) --dataset=C10(C10+,C100,C100+,SVHN,ImageNet) --initial_channel=50(72,96,100,120,150,180)

- Model was tested with Python 3.5.2 with CUDA.
- Model should work as expected with TensorFlow >= 0.10. Tensorflow 1.0 support was recently included.

## Test
-----
Test results on various datasets. 

====================== ====== ====== =========== =========== ============== ============== ==============
Model type             Depth  Params C10          C10+       C100           C100+          SVHN
====================== ====== ====== =========== =========== ============== ============== ==============
WPNet(C=50; G=10)      64     1.4M   5.86        4.37        24.58          21.96          1.71
WPNet(C=100; G=25)     154    4M     5.45        4.10        21.78          20.03          1.70
====================== ====== ====== =========== =========== ============== ============== ==============
PWNet(C=50; G=10)      64     0.3M   6.88        5.67        27.89          25.33          1.76
PWNet(C=200; G=25)     154    8M     5.41        4.03        21.31          19.75          1.56
====================== ====== ====== =========== =========== ============== ============== ==============
## Acknowledgement
This reimplementation are adapted from the [vision_networks](https://github.com/ikhlestov/vision_networks) repository by [ikhlestov] (https://github.com/ikhlestov).

====================== ====== =========== =========== ============== ==============
Model type             Depth  C10          C10+       C100           C100+
====================== ====== =========== =========== ============== ==============
DenseNet(*k* = 12)     40     6.67(7.00)  5.44(5.24)  27.44(27.55)   25.87(24.42)
DenseNet-BC(*k* = 12)  100    5.54(5.92)  4.87(4.51)  24.88(24.15)   22.85(22.27)
====================== ====== =========== =========== ============== ==============

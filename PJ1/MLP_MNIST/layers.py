import numpy as np
from scipy.signal import convolve2d,correlate2d

# InputLayer
class InputLayer:
    def forward(self, inputs, training=True):
    #     # Automatically flatten the input if it is not already flattened
    #     if len(inputs.shape) > 2:  # If input is multi-dimensional, like images
    #         self.output = inputs.reshape(inputs.shape[0], -1)  # Flatten to (batch_size, features)
    #     else:
    #         self.output = inputs
        """
        智能输入层:自动处理MLP的展平输入和CNN的图像输入
        - MLP输入: (batch, features) → 直接传递
        - CNN图像输入: 
          - 若输入是(batch, features): 自动reshape为(batch, 1, 28, 28) [MNIST]
          - 若输入是(batch, channels, height, width): 直接传递
        """
        # 情况1：已经是4D输入（CNN直接使用）
        if len(inputs.shape) == 4:
            self.output = inputs
            
        # 情况2：2D输入（需判断是MLP数据还是需要reshape的CNN数据）
        elif len(inputs.shape) == 2:
            if hasattr(self, 'force_reshape_for_cnn') and self.force_reshape_for_cnn:
                # 显式指定了要转为CNN格式（适用于知道是图像数据但被展平的情况）
                self.output = inputs.reshape(-1, 1, 28, 28)  # MNIST默认reshape
            else:
                # 默认作为MLP处理
                self.output = inputs
                
        # 情况3：3D输入（如灰度图缺少通道维）
        elif len(inputs.shape) == 3:
            self.output = np.expand_dims(inputs, axis=1)  # 添加通道维
            
        else:
            raise ValueError(f"can not process: {inputs.shape}")
            
        # print(f"[InputLayer] {inputs.shape} change into {self.output.shape}")  # 调试用
        return self.output
    # 添加空方法
    def backward(self, dvalues):
        pass  # InputLayer不需要反向传播

# FullyConnectedLayer
class FullyConnectedLayer:
    def __init__(self, n_inputs, n_neurons, l2_reg_weight=0, l2_reg_bias=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l2_reg_weight = l2_reg_weight
        self.l2_reg_bias = l2_reg_bias
    
    # Forward pass
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        # L2 regularization gradients
        if self.l2_reg_weight > 0:
            self.dweights += 2 * self.l2_reg_weight * self.weights
        if self.l2_reg_bias > 0:
            self.dbiases += 2 * self.l2_reg_bias * self.biases
    
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set weights and biases
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

#######################################################################
#                             ReLU                                    #
#######################################################################
class ActivationReLU:
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs

#######################################################################
#                            Sigmoid                                  #
#######################################################################
class ActivationSigmoid:
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5).astype(int)

#######################################################################
#                             Tanh                                    #
#######################################################################
class ActivationTanh:
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)
    
    def predictions(self, outputs):
        return outputs  
    
#######################################################################
#                           Softmax                                   #
#######################################################################
class ActivationSoftmax:
    def forward(self, inputs, training=True):
        # For numerical stability
        exps = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.output = probs
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)  # Create an empty array
        for i, (out, dvalue) in enumerate(zip(self.output, dvalues)):
            out = out.reshape(-1, 1)  # Column vector
            jacobian = np.diagflat(out) - np.dot(out, out.T)
            self.dinputs[i] = np.dot(jacobian, dvalue)
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


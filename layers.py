import numpy as np


def im2col(x, kernel_height, kernel_width):
    [batch, in_channels, h, w] = x.shape
    h_out = h - kernel_height + 1
    w_out = w - kernel_width + 1
    col = np.zeros((kernel_height *
                    kernel_width *
                    in_channels, int(((in_channels *
                                       kernel_height *
                                       kernel_width *
                                       w_out *
                                       h_out -
                                       1) %
                                      (h_out *
                                       w_out)) *
                                     batch +
                                     batch)))
    for b in range(batch):
        for k in range(
                in_channels *
                kernel_height *
                kernel_width *
                h_out *
                w_out):
            pp = int(k / (h_out * w_out))
            qq = int(k % (h_out * w_out))
            d0 = int(pp / kernel_height / kernel_width)
            i0 = int(qq / w_out) + int((pp / kernel_width) % kernel_height)
            j0 = int(qq % w_out) + int(pp % kernel_width)
#             print (b, qq)
            col[pp, batch * qq + b] = x[b, d0, i0, j0]
    return col


def col2im(col, x_shape, kernel_height, kernel_width):
    [batch, in_channels, h, w] = x_shape
    h_out = h - kernel_height + 1
    w_out = w - kernel_width + 1
    x = np.zeros(x_shape)
    for b in range(batch):
        for k in range(
                in_channels *
                kernel_height *
                kernel_width *
                h_out *
                w_out):
            pp = int(k / (h_out * w_out))
            qq = int(k % (h_out * w_out))
            d0 = int(pp / kernel_height / kernel_width)
            i0 = int(qq / w_out) + int((pp / kernel_width) % kernel_height)
            j0 = int(qq % w_out) + int(pp % kernel_width)
#             print (b, qq)
            x[b, d0, i0, j0] += col[pp, batch * qq + b].T
    return x


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """

        raise NotImplementedError("Not implemented in interface")

    def forward(self, input):
        """
        Takes input data of shape [batch, ...], returns output data [batch, ...]
        """

        raise NotImplementedError("Not implemented in interface")

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input. Updates layer parameters and returns         gradient for next layer
        Let x be layer weights, output – output of the layer on the given input and grad_output – gradient of layer with respect         to output

        To compute loss gradients w.r.t parameters, you need to apply chain rule (backprop):
        (d loss / d x)  = (d loss / d output) * (d output / d x)
        Luckily, you already receive (d loss / d output) as grad_output, so you only need to multiply it by (d output / d x)
        If your layer has parameters (e.g. dense layer), you need to update them here using d loss / d x. The resulting update is         a sum of updates in a batch.

        returns (d loss / d input) = (d loss / d output) * (d output / d input)
        """

        raise NotImplementedError("Not implemented in interface")


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        This layer does not have any parameters.
        """

    def forward(self, input):
        """
        Perform ReLU transformation
        input shape: [batch, input_units]
        output shape: [batch, input_units]
        """
#         self.num_inputs = input.shape
        output = np.zeros(input.shape)
        if (input.dtype == np.float):
            output = np.clip(input, 0, np.finfo(input.dtype).max)
        else:
            output = np.clip(input, 0, np.iinfo(input.dtype).max)
        return output

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """
        dx, x = None, input
        dx = np.array(grad_output, copy=True)
        dx[x <= 0] = 0
        return dx


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = Wx + b

        W: matrix of shape [num_inputs, num_outputs]
        b: vector of shape [num_outputs]
        """

        self.learning_rate = learning_rate

        # initialize weights with small random numbers from normal distribution

        self.weights = np.random.normal(
            0, 1 / input_units, size=(input_units, output_units))
        self.biases = np.random.normal(0, 1 / input_units, size=output_units)
#         raise NotImplementedError("Implement me plz ;(")

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        num_batches = input.shape[0]
        x_size = int(np.size(input) / num_batches)
        x = np.reshape(input, (num_batches, x_size))
        output = np.dot(x, self.weights) + self.biases
        return output

    def backward(self, input, grad_output):
        """
        input shape: [batch, input_units]
        grad_output: [batch, output units]

        Returns: grad_input, gradient of output w.r.t input
        """
        num_batches = input.shape[0]
        x_size = int(np.size(input) / num_batches)
        x = np.reshape(input, (num_batches, x_size))
        dx = np.dot(grad_output, self.weights.T)
        dw = np.dot(grad_output.T, x)
        db = np.dot(grad_output.T, np.ones(grad_output.shape[0]))
        update = -self.learning_rate * dw.T
        self.weights += update
        update = -self.learning_rate * db
        self.biases += update
        return dx


class Conv2d(Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            learning_rate=0.1):
        """
        A convolutional layer with out_channels kernels of kernel_size.

        in_channels — number of input channels
        out_channels — number of convolutional filters
        kernel_size — tuple of two numbers: k_1 and k_2

        Initialize required weights.
        """
        self.kernel_size = kernel_size
#         self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.normal(
            0,
            1 / in_channels,
            size=(
                in_channels,
                out_channels,
                kernel_size[0],
                kernel_size[1]))
        self.learning_rate = learning_rate

    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        [batch, in_channels, h, w] = input.shape
        h_out = h - self.kernel_size[0] + 1
        w_out = w - self.kernel_size[1] + 1
        output = np.zeros((batch, self.out_channels, h_out, w_out))
#         for b in range(batch):
#             for ch in range(self.out_channels):
#                 for x in range(w_out):
#                     for y in range(h_out):
#                         output[b, ch, y, x] = (self.weights[:, ch, :, :] *
#                                          input[b, :, y:y + self.kernel_size[0],
#                                          x:x + self.kernel_size[1]]).sum()

        x_cols = im2col(input, self.kernel_size[0], self.kernel_size[1])
#         print(x_cols.shape)
        output = self.weights.reshape(self.out_channels, -1).dot(x_cols)
        output = output.reshape(self.out_channels, h_out, w_out, batch)
        output = output.transpose(3, 0, 1, 2)
        return output

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights
        """
#         [batch, in_channels, h, w] = input.shape
#         dx = np.zeros_like(input)
#         dw = np.zeros_like(self.weights)
        x_cols = im2col(input, self.kernel_size[0], self.kernel_size[1])
        grad_output_col = grad_output.transpose(
            1, 2, 3, 0).reshape(
            self.out_channels, -1)
        dw = grad_output_col.dot(x_cols.T).reshape(self.weights.shape)
        dx_cols = self.weights.reshape(
            self.out_channels, -1).T.dot(grad_output_col)
        dx = col2im(
            dx_cols,
            input.shape,
            self.kernel_size[0],
            self.kernel_size[1])
#         for b in range(batch):
#             for ch in range(self.out_channels):
#                 for x in range(w_out):
#                     for y in range(h_out):
#                         dx[b, :, y:y + self.kernel_size[0], x:x + kernel_size[1]] +=
#                         self.weights[:, ch, :, :] * grad_output[b, ch, y, x]
#                         dw[:, ch, :, :] += input[:, y:y + self.kernel_size[0], x:x + kernel_size[1]]
#                         * grad_output[b, ch, y, x]

        update = - dw * self.learning_rate
        self.weights += update
        return dx


class Maxpool2d(Layer):
    def __init__(self, kernel_size):
        """
        A maxpooling layer with kernel of kernel_size.
        This layer donwsamples [kernel_size, kernel_size] to
        1 number which represents maximum.

        Stride description is identical to the convolution
        layer. But default value we use is kernel_size to
        reduce dim by kernel_size times.

        This layer does not have any learnable parameters.
        """

        self.stride = kernel_size
        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Perform maxpooling transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """
        [batch, in_channels, h, w] = input.shape
        if h % self.kernel_size == 0 and w % self.kernel_size == 0:
            x_reshaped = input.reshape(batch,
                                       in_channels,
                                       int(h / self.kernel_size),
                                       self.kernel_size,
                                       int(w / self.kernel_size),
                                       self.kernel_size)
            return x_reshaped.max(axis=3).max(axis=4)
        else:
            h_out = int((h - self.kernel_size) / self.stride + 1)
            w_out = int((w - self.kernel_size) / self.stride + 1)
            output = np.zeros((batch, in_channels, h_out, w_out))
            for b in range(batch):
                for x in range(w_out):
                    for y in range(h_out):
                        y1 = y * self.stride
                        y2 = y * self.stride + self.kernel_size
                        x1 = x * self.stride
                        x2 = x * self.stride + self.kernel_size
                        window = input[b, :, x1:x2, y1:y2]
                        output[b, :, x, y] = np.max(window.reshape(
                            (in_channels, self.kernel_size**2)), axis=1)
#             print("OK")
            return output

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Maxpool2d input
        """
        [batch, in_channels, h, w] = input.shape
        h_out = int((h - self.kernel_size) / self.stride + 1)
        w_out = int((w - self.kernel_size) / self.stride + 1)
        dx = np.zeros_like(input)
        for b in range(batch):
            for ch in range(in_channels):
                for x in range(w_out):
                    for y in range(h_out):
                        y1 = y * self.stride
                        y2 = y * self.stride + self.kernel_size
                        x1 = x * self.stride
                        x2 = x * self.stride + self.kernel_size
                        window = input[b, ch, x1:x2, y1:y2].reshape(
                            (self.kernel_size**2))
                        window = (window == window.max())
                        dx[b, ch, x1:x2, y1:y2] = window.reshape(
                            (self.kernel_size, self.kernel_size)) * grad_output[b, ch, x, y]
        return dx


class Flatten(Layer):
    def __init__(self):
        """
        This layer does not have any parameters
        """

    def forward(self, input):
        """
        input shape: [batch_size, channels, feature_nums_h, feature_nums_w]
        output shape: [batch_size, channels * feature_nums_h * feature_nums_w]
        """
        return input.reshape(input.shape[0], -1)

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Flatten input
        """
        return grad_output.reshape(input.shape)


def softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    output is a number
    """
    # softmax
    logits_exp = np.exp(logits)
    logits_exp_sum = np.sum(logits_exp, axis=1, keepdims=True)
    probabilities = logits_exp / logits_exp_sum
    # cross-entropy
    log_probabilities = np.log(probabilities + 1e-9)
    N = probabilities.shape[0]
    loss = -np.sum(log_probabilities[range(N), y_true]) / N
    return loss


def grad_softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy gradient from logits and ids of correct answers
    Output should be divided by batch_size, so any layer update can be simply computed as sum of object updates.
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    """
    logits_exp = np.exp(logits)
    logits_exp_sum = np.sum(logits_exp, axis=1, keepdims=True)
    probabilities = logits_exp / logits_exp_sum
    dloss = probabilities
    batch_size = y_true.shape[0]
    dloss[range(batch_size), y_true] -= 1
    dloss /= batch_size
    return dloss

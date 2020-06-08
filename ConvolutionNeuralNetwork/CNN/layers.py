import numpy as np 
from f_utils import *

class Layers:
    def __init__(self):
        self.parameters = dict()
        self.grads = dict()
        self.net = dict()
        self.name = " "        
    

class Convolution(Layers):
    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, name, num):
        super().__init__()
        ll = num
        self.out_c = num_filters
        self.f = kernel_size
        self.in_c = inputs_channel      
        self.lr = 0.01        
        self.name = name     
        self.s = stride 
        self.padding = padding
        
        weights  = np.random.randn(self.out_c,self.f, self.f, self.in_c) / np.sqrt(self.out_c * self.f * self.f * self.in_c)
        #for i in range(0, self.out_c):
        #    weights[i,:,:,:] = 
            
        self.parameters['W%s' %  str(ll)] = weights
        self.parameters['b%s' %  str(ll)] = np.zeros((self.out_c, 1))
    
    def convolve(self,input_image, mask,stride = 1,bp = False):
    
        i_row, i_col,i_ch = input_image.shape
        k_row, k_col,k_ch = mask.shape
        output = np.zeros((input_image.shape[0],input_image.shape[1]))
        pad_height = ((i_row*stride)-i_row+k_row - stride) // 2
        pad_width = ((i_col*stride)-i_col+k_col - stride) // 2
     
        padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width),i_ch))
        mask_without_ch = np.zeros((k_row,k_col))
        for ch in range(i_ch):
            padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width,ch] = input_image[:,:,ch]
        for ch in range(i_ch):
            for row in range(i_row):
                for col in range(i_col):
                    mask_without_ch = mask[:,:,ch]
                    if(bp):
                        mask_without_ch = np.flip(mask_without_ch)
                    output[row, col] = output[row, col] + np.sum(mask_without_ch * padded_image[row:row + k_row, col:col + k_col,ch])
        return output
    def validconvolve(self,input_image, Kernal,stride = 1):
        i_row, i_col = input_image.shape
        k_row, k_col = Kernal.shape
        pad_height = self.padding
        pad_width = self.padding
        padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = input_image
        
        i_row, i_col = padded_image.shape
        new_row = ((i_row - k_row)//stride) + 1
        new_col = ((i_col - k_col)//stride) + 1
        output = np.zeros((new_row,new_col))
        for row in range(new_row):
            for col in range(new_row):
                output[row,col] = np.sum(padded_image[row:row + k_row, col:col + k_col] * Kernal)
        return output
    
    
    def forward(self, inputs, ll):
        output = np.zeros((inputs.shape[0],inputs.shape[1],self.out_c))
        for i in range(self.out_c):
            output[:,:,i] = self.convolve(inputs, self.parameters['W%s' %  str(ll)][i,:,:,:]) + self.parameters['b%s' %  str(ll)][i]
        
        self.net['A'] = inputs
        return output
    
    
    def convolution_back(self,image, kernel):
        i_row, i_col = image.shape
        k_row, k_col = kernel.shape
        output = np.zeros(image.shape)
        pad_height = (k_row - 1) // 2
        pad_width = (k_col - 1) // 2
     
        padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width)))
     
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
        for row in range(i_row):
            for col in range(i_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + k_row, col:col + k_col])
        return output            
    

    def backward(self, gradients, ll):
        
        inputs = self.net['A']
        Doutput = np.zeros(inputs.shape)
        #print(self.parameters['W%s' %  str(ll)][0,:,:,:].shape)
        #print(gradients.shape)
        for j in range(self.in_c):
            for i in range(self.out_c):
                f_map = gradients[:,:,i]
                kernal = self.parameters['W%s' %  str(ll)][i,:,:,j]
                kernal = np.flip(kernal)
                Doutput[:,:,j] += self.convolution_back(f_map, kernal)
                
        dw = np.zeros((self.out_c,self.f, self.f, self.in_c))
        for i in range(inputs.shape[2]):
            for j in range(gradients.shape[2]):
               dw[j,:,:,i] = self.validconvolve(inputs[:,:,i],gradients[:,:,j])
                 
                
        self.grads['dW%s' %  str(ll)] = dw
        self.grads['db%s' %  str(ll)] = np.sum(gradients, axis=(0,1), keepdims=True)
        self.grads['db%s' %  str(ll)] = self.grads['db%s' %  str(ll)].reshape(self.parameters['b%s' % ll].shape)
        
        return Doutput
    def update_parameters(self,ll):
        self.parameters["W%s" %ll] = self.parameters["W%s" % ll] - (self.lr * self.grads["dW%s"%ll])
        self.parameters['b%s' %ll] = self.parameters['b%s' % ll] - (self.lr * self.grads['db%s'%ll])
        


class Maxpooling(Layers):
    def __init__(self, pool_size, stride, num):
        super().__init__()
        ll = num
        self.size = pool_size
        self.stride = stride
        
        
    def convolve(self,input_image, size = 2,stride = 2):
    
        i_row, i_col,i_ch = input_image.shape
        k_row, k_col = size,size
        new_row = ((i_row - size)//stride) + 1
        new_col = ((i_col - size)//stride) + 1
        output = np.zeros((new_row,new_col,i_ch))
        #doutput_index = np.zeros((new_row,new_col,i_ch,2))
        #pad_height = ((i_row*stride)-i_row+k_row - stride) // 2
        #pad_width = ((i_col*stride)-i_col+k_col - stride) // 2
        #padded_image = np.zeros((i_row + (2 * pad_height), i_col + (2 * pad_width),i_ch))
        #for ch in range(i_ch):
        #    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width,ch] = input_image[:,:,ch]
        for ch in range(i_ch):
            for row in range(new_row):
                for col in range(new_row):
                    #doutput_index[row,col,ch] = np.unravel_index(input_image[(row*stride):(row*stride) + k_row, (col * stride):(col * stride) + k_col,ch].argmax(), input_image[(row*stride):(row*stride) + k_row, (col * stride):(col * stride) + k_col,ch].shape)
                    output[row,col,ch] = np.max(input_image[(row*stride):(row*stride) + k_row, (col * stride):(col * stride) + k_col,ch])
        #self.net['index'] = doutput_index
        return output
    def forward(self, inputs, ll):
        self.net['A'] = inputs
        return self.convolve(inputs, self.size,self.stride)
    
    def create_mask(self, x):
        return x == np.max(x)
    def backward(self, gradients, ll):
        dout = np.zeros(self.net['A'].shape)
        n_H, n_W, n_C = gradients.shape
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h * self.stride
                vert_end = vert_start + self.size
                horiz_start = w * self.stride
                horiz_end = horiz_start + self.size
                for c in range(n_C):
                    a_slice = self.net['A'][vert_start: vert_end, horiz_start: horiz_end, c]
                    #mask = self.create_mask(a_slice)
                    #dout[vert_start: vert_end, horiz_start: horiz_end, c] += \
                        #gradients[h, w, c] * mask
                    idx_max = np.argmax(a_slice)
                    (idx_x, idx_y) = np.unravel_index(idx_max, (self.size, self.size))
                    dout[h+idx_x, w+idx_y, c] = gradients[h//self.size, w//self.size, c]
        return dout

    
class FullyConnected(Layers):
    def __init__(self, num_inputs, num_outputs, name, num):
        super().__init__()
        l = num
        self.inp = num_inputs
        self.output = num_outputs      
        self.lr = 0.01        
        self.name = name   
        self.parameters['W%s' % l] = np.random.randn(self.output, self.inp) /np.sqrt(self.inp/2.)
        self.parameters['b%s' % l] = np.zeros((self.output, 1))

    def forward(self, inputs, ll): 
        Z = np.dot(self.parameters['W%s' % ll], inputs) + self.parameters['b%s' % ll]
        self.net['A'] = inputs
        return Z
    def backward(self, gradients, ll):
        self.grads['dW%s' % ll] = np.dot(gradients, self.net['A'].T)
        self.grads['db%s' % ll] = np.sum(gradients, axis=1, keepdims=True)
        return np.dot(self.parameters['W%s' % ll].T, gradients)
    def update_parameters(self,ll):
        self.parameters["W%s" %ll] = self.parameters["W%s" % ll] - (self.lr * self.grads["dW%s"%ll])
        self.parameters['b%s' %ll] = self.parameters['b%s' % ll ] - (self.lr * self.grads['db%s'%ll])
    
class Flatten(Layers):
    def __init__(self, num):
        super().__init__()
        
    
    def forward(self, inputs, ll):
        self.net['A'] = inputs
        Z = inputs.reshape(inputs.shape[0]*inputs.shape[1]*inputs.shape[2], 1)
        return Z
        
    def backward(self, dy, ll):
        shape = self.net['A'].shape
        return dy.reshape(shape)


class Softmax(Layers):
    def __init__(self, num):
        super().__init__()
    
    def forward(self, inputs, ll):
        
        inputs_ = inputs - inputs.max()
        e = np.exp(inputs_)
        self.net['A'] = e / np.sum(e, axis=0, keepdims=True)
        return self.net['A']
    
    def backward(self, gradients, ll):
        output =  self.net['A']
        gradients = gradients.reshape(10,1)
        return  output - gradients 


class ReLu(Layers):
    def __init__(self, num):
        super().__init__()
        self.name = "relu"
        
        
    def forward(self, inputs, ll):
        self.net['A'] = inputs
        return np.where(inputs >= 0, inputs, 0)
    
    def backward(self, gradients, ll):
        Z = self.net['A']
        return gradients * (np.where(Z >= 0, 1, 0))
    

class Sigmoid(Layers):
    def __init__(self, num):
        super().__init__()
        
    def forward(self, inputs, ll):
        self.net['A'] = inputs
        return 1/(1 + np.exp(-inputs))
    
    def backward(self, gradients, ll):
        a =  self.net['A']
        return gradients * np.exp(-a) / ((1 + np.exp(-a)) ** 2)


class Tanh(Layers):
    def __init__(self, num):
        super().__init__()
        
    def forward(self, inputs, ll):
        self.net['A'] = inputs
        t=(np.exp(inputs)-np.exp(-inputs))/(np.exp(inputs)+np.exp(-inputs))
        return t 

   
    def backward(self, gradients, ll):
        a = self.net['A']
        t=(np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
        dt=1-t**2
        return dt * gradients

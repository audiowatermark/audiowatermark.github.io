# Untargeted version.

import torch
import torch.nn as nn

from models.crop import centre_crop
from models.resample import Resample1d
from models.conv import ConvLayer
import torch.nn.functional as F

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        # print("upsampled:", list(upsampled.size()))

        for i, conv in enumerate(self.pre_shortcut_convs):
            upsampled = conv(upsampled)
            # print(f"conv {i}:", list(upsampled.size()))

        # Prepare shortcut connection

        # print("shortcut:", list(shortcut.size()))
        combined = centre_crop(shortcut, upsampled)
        # print("combined:", list(combined.size()))

        # Combine high- and low-level features
        for i, conv in enumerate(self.post_shortcut_convs):
            # print(f"cat {i}", list(torch.cat([combined, centre_crop(upsampled, combined)], dim=1).size()))
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
            # print(f"conv {i}", list(combined.size()))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # conv before shortcut
        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])
        # print (self.pre_shortcut_convs)
        # conv after shortcut
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

        self.fc = nn.Linear(256, 1) # style-embedding dimension as input, output unknown 

    def forward(self, x, style_emb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # PREPARING SHORTCUT FEATURES
        # print ("before 1d conv shape {}".format(x.shape))
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)
        # print ("after 1d conv shape {}".format(out.shape))
        # DOWNSAMPLING
        out = self.downconv(out)
        # print ("after downsample shape {}".format(out.shape))
        #######################################
        # Oct.24 style token FC
        #######################################
        # style_out = out.shape[1]*out.shape[2]
        # self.fc = nn.Linear(256, style_out).to(device)  # convert style embedding shape to out shape
        # style_emb = self.fc(style_emb)
        # style_emb = style_emb.view(1,  out.shape[1], out.shape[2])
        # # print ("Style_emb shape : {}".format(style_emb.shape))
        # out = out + style_emb  # insert style embedding to ds block
        #######################################
        # Oct.24 style token dup
        #######################################
        style_emb_repeated = torch.tile(style_emb, (out.shape[1], out.shape[2]//256))  # [1, 256] => [64, 22441]
        pad_right = out.shape[2] - style_emb_repeated.shape[2]  # pad to match the out shape
        # print ("style_emb : {}, repeated : {}".format(style_emb.shape, style_emb_repeated.shape))
        padded_style_emb = F.pad(style_emb_repeated, pad=[0, pad_right, 0, 0, 0, 0]) 
        # print ("padded_style_emb : {}".format(padded_style_emb.shape))
        out = out + padded_style_emb
        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size, target_output_size, conv_type, res, separate=False, depth=1, strides=2, debug=False):
        super(Waveunet, self).__init__()

        # what is different between num_levels and depth?
        # :param depth: Number of convs per block
        # :param num_levels: Number of DS/US blocks

        self.num_levels = len(num_channels)

        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments  # list: ["bass", "drums", "other", "vocals"]
        self.separate = separate
        self.debug = debug

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()
            module.throttle = nn.Linear(in_features = 1, out_features = self.num_levels-1)

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

                # print (module.downsampling_blocks)
                # exit (0)
            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        # The difference between input and output have to be even,
        # so that output can be put in the middle of the input. 
        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            # bottleneck++ until (output_size >= target_output_size)
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)

            if out is not False:
                # return curr_size (the input size), output_size
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):

        # This is equivalent to list(self.waveunets.keys())[0]
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:

            # get the output size of the last UP-SAMPLING layer
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # get the input size of the first DOWN-SAMPLING layer
            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, style_emb, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        
        throttled input x

        :param x: (original input mix, eps)

        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''

        shortcuts = []
        out, eps = x
        thro = module.throttle(eps).unsqueeze(-1).unsqueeze(-1)

        # DOWNSAMPLING BLOCKS
        for i, block in enumerate(module.downsampling_blocks):
            out, short = block(out, style_emb)
            shortcuts.append(short)

            if self.debug:
                print(f"DS Layer {i}: out ({list(out.size())}), short ({list(short.size())})")
        # BOTTLENECK CONVOLUTION
        for i, conv in enumerate(module.bottlenecks):
            out = conv(out)

            if self.debug:
                print(f"Bottlenecks Layer {i}: out ({list(out.size())})")

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1-idx]*thro[:, -1-idx])

            if self.debug:
                print(f"UP Layer {idx}: out ({list(out.size())}), short ({-1 - idx}:{list(shortcuts[-1 - idx].size())})")

        # OUTPUT CONV
        out = module.output_conv(out)

        if self.debug:
            print(f"Output: out ({list(out.size())})")

        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, style_emb, inst=None):
        curr_input_size = x[0].shape[-1]
        # print (curr_input_size, self.input_size)
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, style_emb, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, style_emb, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict

def init_stylewaveunet():
    instruments = ["styled"]  # record the name of the output, non-sense
    features = 32  # 1D conv feature 
    levels = 6     # How many downsampling and upsampling  
    depth = 1      # bottlenecks depth
    sr = 16000
    channels = 1
    kernel_size = 5
    output_size = 5
    strides = 4
    conv_type = "gn"
    res = "fixed"
    separate = 0
    feature_growth = "double"

    num_features = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
                    [features*2**i for i in range(0, levels)]
    # [32, 64, 128, 256, 512, 1024]
    target_outputs = int(output_size * sr)
    # print ("target_outputs : {}".format(target_outputs))
    mywaveunet = Waveunet(
                            num_inputs=channels, 
                            num_channels=num_features, 
                            num_outputs=channels, 
                            instruments=instruments, 
                            kernel_size=kernel_size,
                            target_output_size=target_outputs, 
                            depth=depth, 
                            strides=strides,
                            conv_type=conv_type, 
                            res=res, 
                            separate=separate
                        )
    return mywaveunet

# waveunet = init_waveunet()

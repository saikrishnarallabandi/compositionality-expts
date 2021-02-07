from torch.autograd import Variable
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

input_size = 2048
output_size = 16000
mcb = CompactBilinearPooling(input_size, input_size, output_size).cuda()
x = torch.rand(4,input_size).cuda()
y = torch.rand(4,input_size).cuda()

z = mcb(x,y)

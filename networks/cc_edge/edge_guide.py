import os
import torch
from torch import nn
from torch.autograd import Function, gradcheck
from torch.utils.cpp_extension import load
import time

DEBUG = False


def timer(func):
    if DEBUG:
        def _task(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(func.__name__ + ' consume %.4f s' % (time.time() - start))
            return res
    else:
        _task = func
    return _task


this_file = os.path.dirname(os.path.realpath(__file__))

weight_cuda = load(
    name="cc_edge", sources=[os.path.join(
        this_file, "src/cc_edge.cu")], verbose=DEBUG, build_directory=os.path.join(this_file, 'build'),
    extra_cuda_cflags=["--expt-extended-lambda"]
)


class ComputeMapping(Function):
    @staticmethod
    @timer
    def forward(ctx, h_interal, w_interal):
        weight = weight_cuda.compute_weight_forward(h_interal, w_interal)
        return weight

    @staticmethod
    @timer
    def backward(ctx, grad):
        grad = grad.contiguous()
        # print(grad.size(),grad.dtype)
        grad_h, grad_w = weight_cuda.compute_weight_backward(grad)
#        print(grad_h.max().item(),grad_h.min().item(),grad_w.min().item(),grad_w.max().item())
        return grad_h, grad_w, None


class MapWithWeight(Function):
    @staticmethod
    @timer
    def forward(ctx, weight, feature):
        weight = weight.contiguous()
        ctx.save_for_backward(weight, feature)

        out = weight_cuda.aggregate_forward(weight, feature)
        return out

    @staticmethod
    @timer
    def backward(ctx, grad_out):
        weight, feature = ctx.saved_tensors
        # print(grad_out.size(),weight.size(),feature.size())
        # print(grad_out[0,-1])
        grad_out = grad_out.contiguous()
        grad_weight, grad_feature = weight_cuda.aggregate_backward(grad_out, weight, feature)
        return grad_weight, grad_feature


compute_mapping = ComputeMapping.apply
aggregate = MapWithWeight.apply

class RotateEdgeGuide(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,mask,edge):
        pred= torch.cat([mask,edge],dim=1)
        #rotate
        r_pred = []
        r_weight = []

#
#
# compute the connect cost in a block
class GaussianEdgeGuide(nn.Module):
    def __init__(self, theta=40, kernel_size=5):
        super().__init__()
        self.theta = theta
        #self.register_buffer('spatial_kernel',torch.Tensor([1,1,1,1,0,1,1,1,1]).view(1,1,9,1,1))
        self.kernel_size = kernel_size

    def forward(self, mask, edge, iter=1):
        max_edge = edge.max()
        n, c, h, w = edge.size()
        # local_edge = nn.functional.pad(edge, pad=[1, 1, 1, 1], value=10 * max_edge)
        # print(edge.size())
        local_edge = nn.functional.unfold(edge, kernel_size=3, padding=1, stride=1).view(n, c, 9, h, w)
        local_edge = edge.unsqueeze(2) +local_edge
        spatial_kernel=torch.Tensor([1, 1, 1, 1, 0, 1, 1, 1, 1]).view(1, 1, 9, 1, 1).to(edge.device)
        local_edge = local_edge#*spatial_kernel
        local_edge = nn.functional.softmax(-1 * self.theta * (local_edge - max_edge), dim=2)
        # print(mask.size())
        for _ in range(iter):
            mask = nn.functional.unfold(mask, kernel_size=3, padding=1, stride=1).view(n, mask.size(1), 9, h, w)
            mask = mask * local_edge
            mask=mask.sum(dim=2)
        return mask

class ASPPGaussianEdgeGuide(nn.Module):
    def __init__(self,theta=40,input_chanel=512):
        super().__init__()

    def forward(self,mask,edge,iter=1):
        pass

class GaussianEdgeGuideV2(nn.Module):
    def __init__(self, theta=10, kernel_size=5):
        super().__init__()
        self.theta = theta
        self.kernel_size = kernel_size

    def forward(self, mask, edge):
        # max_edge = edge.max()
        n, c, h, w = mask.size()
        edge = nn.functional.interpolate(edge, size=(h, w), mode='bilinear', align_corners=True)
        # local_edge = nn.functional.pad(edge, pad=[1, 1, 1, 1], value=10 * max_edge)
        # print(edge.size())
        local_edge = nn.functional.unfold(edge, kernel_size=3, padding=1, stride=1).view(n, edge.size(1), 9, h, w)
        local_edge = edge.unsqueeze(2) - local_edge
        local_edge = nn.functional.softmax(-1 * self.theta * (local_edge ** 2), dim=2)
        # print(mask.size())
        mask = nn.functional.unfold(mask, kernel_size=3, padding=1, stride=1).view(n, mask.size(1), 9, h, w)
        mask = mask * local_edge
        return mask.sum(dim=2)


class CCEdgeGuide(nn.Module):
    def __init__(self, theta=40, kernel_length=-1):
        super().__init__()
        self.kernel_size = kernel_length
        self.theta = theta
        # self.gamma =
    def forward(self, mask, edge, iter=3):
        edge = self.prepare(edge)
        max_edge = edge.max()
        # edge = torch.exp(edge - max_edge)
        h_cumsum = torch.cumsum(edge, dim=2)
        w_cumsum = torch.cumsum(edge, dim=3)  # n,c,h,w
        weight = compute_mapping(h_cumsum, w_cumsum)  # n,1,h,w,k
        weight = weight.squeeze(1).permute(0, 3, 1, 2).contiguous()

        weight = nn.functional.softmax(-1 * self.theta * (weight - max_edge), dim=1)
        # print(weight.size())
        # print(mask.size())

        for _ in range(iter):
            # start=time.time()
            mask = aggregate(weight, mask)
            # torch.cuda.synchronize()
            # print(time.time()-start)
        return mask

    def prepare(self, edge):
        return edge.relu()


@timer
def py_compute_mapping(edge_h, edge_w):
    n, c, h, w = edge_h.size()
    out = edge_w.new_zeros(n, c, h, w, h + w - 1)
    for i in range(n):
        for j in range(c):
            for k in range(h):
                for l in range(w):
                    for m in range(h + w - 1):
                        if m < h:
                            if m < k:
                                out[i, j, k, l, m] = edge_h[i, j, k, l] - edge_h[i, j, m, l]
                            else:
                                out[i, j, k, l, m] = edge_h[i, j, m, l] - edge_h[i, j, k, l]
                        else:
                            if m - h < l:
                                out[i, j, k, l, m] = edge_w[i, j, k, l] - edge_w[i, j, k, m - h]
                            else:
                                out[i, j, k, l, m] = edge_w[i, j, k, m + 1 - h] - edge_w[i, j, k, l]
    return out


@timer
def py_mapping(weight, feature):
    """
    :param weight: [n,c,h,w,k]
    :param feature: [n,c,h,w]
    :return:
    """
    n, c, h, w = feature.size()

    out = feature.new_zeros(n, c, h, w)
    for i in range(n):
        for j in range(c):
            for k in range(h):
                for l in range(w):

                    for m in range(h + w - 1):

                        if m < h:

                            out[i, j, k, l] += weight[i, j, k, l, m] * feature[i, j, m, l]
                        else:
                            if m - h < l:
                                out[i, j, k, l] += weight[i, j, k, l, m] * feature[i, j, k, m - h]
                            else:
                                out[i, j, k, l] += weight[i, j, k, l, m] * feature[i, j, k, m - h + 1]
                        # print(i,j,k,l,m)
    return out


def check_status(func):
    def wrapper(*args, **kwargs):
        print("=" * 10 + func.__name__ + " Begin" + "=" * 10)
        func(*args, **kwargs)
        print("=" * 9 + "%s Success" % func.__name__ + "=" * 9)

    return wrapper


@check_status
@timer
def test_compute_weight(device_id=0):
    # check
    for i in range(1, 10):
        for j in range(1, 10):
            edge_h = torch.abs(torch.randn(i, j, 10, 10)).cuda().double()
            edge_h.requires_grad = True
            edge_w = torch.abs(torch.randn(i, j, 10, 10)).cuda().double()
            edge_w.requires_grad = True
            out = compute_mapping(edge_h, edge_w)
            torch.sum(out).backward()
            gradh = edge_h.grad.clone()
            gradw = edge_w.grad.clone()
            edge_h.grad.zero_()
            edge_w.grad.zero_()
            out2 = py_compute_mapping(edge_h, edge_w)
            torch.sum(out2).backward()
            gradh2 = edge_h.grad.clone()
            gradw2 = edge_w.grad.clone()
            assert torch.abs(gradh2 - gradh).max() == 0
            assert torch.abs(gradw2 - gradw).max() == 0
            assert torch.abs((out - out2)).max() == 0
            edge_h.grad.zero_()
            edge_w.grad.zero_()
    gradcheck(py_compute_mapping, [edge_h, edge_w])
    edge_h.grad.zero_()
    edge_w.grad.zero_()
    gradcheck(compute_mapping, [edge_h, edge_w])


@check_status
@timer
def test_mapping():
    for i in range(1, 3):
        for j in range(19, 20):
            h = 10
            w = 10
            print('=' * 10)
            weight = torch.randn(i, h + w - 1, h, w).cuda().double()
            weight.requires_grad = True
            # weight = weight.expand(i, j, h, w, h + w - 1)

            feature = torch.randn(i, j, h, w).cuda().double()
            feature.requires_grad = True
            # out1 = py_mapping(weight.expand(i, j, h, w, h + w - 1), feature)
            # torch.sum(out1).backward()
            # grad_w1 = weight.grad.clone()
            # grad_f1 = feature.grad.clone()
            # weight.grad.zero_()
            # feature.grad.zero_()
            # # print(weight,feature)
            # out2 = aggregate(weight, feature)
            # torch.sum(out2).backward()
            # grad_w2 = weight.grad.clone()
            # grad_f2 = feature.grad.clone()
            # torch.cuda.synchronize()
            # if torch.abs(out1 - out2).max() != 0:
            #     print(out1)
            #     print(out2)
            #     raise Exception('Wrong')
            # if torch.abs(grad_w1 - grad_w2).max() != 0:
            #     print(i, j)
            #     # print(weight, feature)
            #     print(grad_w1)
            #     print(grad_w2)
            #     raise Exception("Weight gradient wrong")
            #
            # if torch.abs(grad_f1 - grad_f2).max() != 0:
            #     print(i, j)
            #     print(grad_f1)
            #     print(grad_f2)
            #     raise Exception("Feature gradient wrong")
    gradcheck(aggregate, (weight, feature))


if __name__ == '__main__':
    from multiprocessing.pool import Pool

    # _pool = Pool(2)
    # _pool.apply_async(test_compute_weight, args=(0,))
    # torch.cuda.set_device(0)
    test_mapping()
    # out = CCEdgeGuide()(torch.randn(1, 19, 128, 256).cuda(), torch.randn(1, 1, 128, 256).cuda())
    # torch.cuda.synchronize()
    # test_compute_weight()
    # _pool.close()
    #
    # _pool.join()

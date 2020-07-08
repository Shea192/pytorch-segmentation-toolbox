#include<torch/torch.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef AT_CHECK
#define AT_CHECK AT_ASSERT
#endif
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

extern THCState *state;
/*
@h_interal [n,c,h,w]
@w_interal [n,c,h,w]
@out [n,c,h,w,k]
*/
template <typename T>
__global__ void compute_weight_forward_kernel(const T* h_interal, const T* w_interal, T* out,const int batch,const int channel,const int height,const int width)
{
   //
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int sp =height*width;
   int k = blockIdx.z;
   if(x<width && y<height && k<(height+width-1))
   {
        for(int b=0;b<batch;b++)
        {

            for(int ch=0;ch<channel;ch++)
            {

                T diff=0;
                if(k<height)
                {
                    if(k<y)
                        diff = h_interal[((b*channel+ch)*sp+y*width+x)]- h_interal[((b*channel+ch)*sp+k*width+x)];
                    else
                        diff = h_interal[((b*channel+ch)*sp+k*width+x)]-h_interal[((b*channel+ch)*sp+y*width+x)];
                }
                else
                {
                    if(k-height<x)
                        diff = w_interal[((b*channel+ch)*sp+y*width+x)] - w_interal[((b*channel+ch)*sp+y*width+k-height)];
                    // one step to left
                    else
                        diff = w_interal[((b*channel+ch)*sp+y*width+k+1-height)]-w_interal[((b*channel+ch)*sp+y*width+x)];
                }
                out[((b*channel+ch)*sp+y*width+x)*(height+width-1)+k]=diff;

            }
        }
        // printf("%d %d %d\n",y,x,k);
    }
}

/*
@grad_out [n,c,h,w,h+w-1]
@grad_h [n,c,h,w]
@grad_w [n,c,h,w]
*/
template <typename T>
__global__ void compute_weight_backward_kernel(const T* grad_out,T* grad_h,T* grad_w,const int batch,const int channel,const int height,const int width)
{

   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int sp =height*width;
   int plane = blockIdx.z;
   if(x<width && y<height && plane<channel)
   {
        for(int b=0;b<batch;b++)
        {
           //dh
          for(int k=0;k<height;k++)
          {
             T dout_self = grad_out[((b*channel+plane)*sp+y*width+x)*(height+width-1)+k];
             T dout_other = grad_out[((b*channel+plane)*sp+k*width+x)*(height+width-1)+y];
             T dout = dout_self + dout_other;
             if(k<y)
                grad_h[(b*channel+plane)*sp+y*width+x]+=dout;
             else if(k>y)
                grad_h[(b*channel+plane)*sp+y*width+x]-=dout;
          }
          //dw
          for(int k=0;k<width;k++)
          {
            if(k==x)
                continue;
            if(k<x)
            {

                // self : g[height+0],...,g[height+x-1],
                // self : w[0],...,w[x-1]

                T dout_self = grad_out[((b*channel+plane)*sp+y*width+x)*(height+width-1)+height+k];
                T dout_other = grad_out[((b*channel+plane)*sp+y*width+k)*(height+width-1)+height+x-1];
                grad_w[(b*channel+plane)*sp+y*width+x]+=(dout_self+dout_other);
            }
             else
             {
                 // self : g[height+x],...,g[height+width-2],
                // self : w[x+1],...,w[width-1]
                T dout_self = grad_out[((b*channel+plane)*sp+y*width+x)*(height+width-1)+height+k-1];
                T dout_other = grad_out[((b*channel+plane)*sp+y*width+k)*(height+width-1)+height+x];
                grad_w[(b*channel+plane)*sp+y*width+x]-=(dout_self+dout_other);
             }

           }
        }
   }

}

/*
@weight : [n,c,h,w,k]
@feature: [n,c,h,w]
@out: [n,c,h,w]
*/
template <typename T>
__global__ void mapping_with_weight_forward_kernel(const T* weight, const T* feature, T* out,const int batch,const int channel,const int height,const int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int len= height+width-1;
    int sp =height*width;
    int ch = blockIdx.z;

    if(x<width && y<height && ch<channel)
    {
         //printf("%d %d %d %d %d\n",y,x,k,batch,channel);
         for(int b=0;b<batch;b++)
         {
 
             for(int k=0;k<height;k++)
             {
                T point_weight = weight[(b*len+k)*sp+y*width+x];
                T point_feature = feature[(b*channel+ch)*sp+k*width+x];
                out[(b*channel+ch)*sp+y*width+x]+=point_weight*point_feature;
             }
             for(int k=0;k<width;k++)
             {
                if(k==x) continue;
                int j=k<x?k:k-1;
                T point_feature = feature[(b*channel+ch)*sp+y*width+k];
                T point_weight = weight[(b*len+height+j)*sp+y*width+x];
                out[(b*channel+ch)*sp+y*width+x]+=point_weight*point_feature;
             }
             }
        }
}
/*
@grad_out [n,c,h,w]
@weight : [n,c,h,w,k]
@feature: [n,c,h,w]
@grad_feature [n,c,h,w]
@grad_weight  [n,c,h,w,k]
*/
template <typename T>
__global__ void mapping_with_weight_backward_weight_kernel(const T* grad_out, const T* feature, T* grad_weight,const int batch,const int channel,const int height,const int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp =height*width;
    int k = blockIdx.z;
    int len=height+width-1;
    if(x<width && y<height && k<len)
    {

        T point_feature=0,dout=0;
        int cord=0;
        for(int b=0;b<batch;b++)
        {
            for(int ch=0;ch<channel;ch++)
            {
                dout = grad_out[(b*channel+ch)*sp+y*width+x];
                if(k<height)
                    point_feature=feature[(b*channel+ch)*sp+k*width+x];
                else
                {
                    cord=k-height;
                    cord = cord<x? cord : cord+1;
                    point_feature = feature[(b*channel+ch)*sp+y*width+cord];
                }
                grad_weight[(b*sp+y*width+x)*(height+width-1)+k] += point_feature*dout;
            }
        }
    }

}

template <typename T>
__global__ void mapping_with_weight_backward_feature_kernel(const T* grad_out, const T* weight,T* grad_feature,const int batch,const int channel,const int height,const int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sp =height*width;
    int ch = blockIdx.z;

    if(x<width && y<height && ch<channel)
    {
         //printf("%d %d %d %d %d\n",y,x,k,batch,channel);
         for(int b=0;b<batch;b++)
         {
             for(int k=0;k<height+width-1;k++)
             {
                if(k<height)
                    grad_feature[(b*channel+ch)*sp+y*width+x] += weight[(b*sp+k*width+x)*(height+width-1)+y]*grad_out[(b*channel+ch)*sp+k*width+x];
                else
                {
                    int cord = k -height;
                    if(cord<x)
                        grad_feature[(b*channel+ch)*sp+y*width+x] += weight[(b*sp+y*width+cord)*(height+width-1)+x+height-1]*grad_out[(b*channel+ch)*sp+y*width+cord];
                    else
                        grad_feature[(b*channel+ch)*sp+y*width+x] += weight[(b*sp+y*width+cord+1)*(height+width-1)+x+height]*grad_out[(b*channel+ch)*sp+y*width+cord+1];
                }

             }
        }
    }
}
/*
@h_interal Tensor [n,c,h,w]
@w_interal Tensor [n,c,h,w]
*/
at::Tensor compute_mapping_full_forward(const at::Tensor h_interal,const at::Tensor w_interal)
 {
     CHECK_INPUT(h_interal);
     CHECK_INPUT(w_interal);
     AT_CHECK(h_interal.ndimension() == 4,"h_interal should be BXCXHXW");
     AT_CHECK(w_interal.ndimension() == 4,"w_interal should be BXCXHXW");
     cudaStream_t stream = THCState_getCurrentStream(state);
     int batch = h_interal.size(0), channel = h_interal.size(1), height=h_interal.size(2),width=h_interal.size(3);

     auto out= h_interal.new_zeros({batch,channel,height,width,height+width-1});

     dim3 threads(32, 32); // 1024 thread
     const int d1 = (width+threads.x-1)/threads.x; // get how many blocks needed for x
     const int d2 = (height+threads.y-1)/threads.y;
     const int d3=height+width-1;
     dim3 blocks(d1, d2, d3); // each block counts for a position and one neighbor.
    AT_DISPATCH_FLOATING_TYPES(h_interal.type(), "compute_mapping_full_forward", ([&]{
    compute_weight_forward_kernel<scalar_t><<<blocks,threads,0,stream>>>(
    h_interal.data<scalar_t>(),
    w_interal.data<scalar_t>(),
    out.data<scalar_t>(),
    batch,
    channel,
    height,
    width);
    }));
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return out;
}

 std::vector<at::Tensor> compute_mapping_full_backward(const at::Tensor grad_out)
{
     CHECK_INPUT(grad_out);
     AT_CHECK(grad_out.ndimension() == 5,"w_interal should be BXCXHXW");
     int batch = grad_out.size(0), channel = grad_out.size(1), height=grad_out.size(2),width=grad_out.size(3);
     auto grad_h= grad_out.new_zeros({batch,channel,height,width});
     auto grad_w= grad_out.new_zeros({batch,channel,height,width});
     dim3 threads(32, 32); // 1024 thread
     const int d1 = (width+threads.x-1)/threads.x; // get how many blocks needed for x
     const int d2 = (height+threads.y-1)/threads.y;
     const int d3=channel;
     dim3 blocks(d1, d2, d3); // each block counts for a position and one neighbor.

     AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weight_forwardward_cuda", ([&] {
     compute_weight_backward_kernel<scalar_t><<<blocks,threads>>>(
     grad_out.data<scalar_t>(),
     grad_h.data<scalar_t>(),
     grad_w.data<scalar_t>(),
     batch,
     channel,
     height,
     width);
     }));
    return {grad_h,grad_w};
}

at::Tensor mapping_with_weight_forward(const at::Tensor weight, const at::Tensor feature)
{
    CHECK_INPUT(weight);
    CHECK_INPUT(feature);
    AT_CHECK(weight.ndimension() == 4,"weight should be BXKXHXW");
    AT_CHECK(feature.ndimension() == 4,"feature should be BXCXHXW");
    cudaStream_t stream = THCState_getCurrentStream(state);
    int batch = feature.size(0), channel = feature.size(1), height=feature.size(2),width=feature.size(3);
    auto out = at::zeros_like(feature);
    dim3 threads(32, 32); // 1024 thread
    const int d1 = (width+threads.x-1)/threads.x; // get how many blocks needed for x
    const int d2 = (height+threads.y-1)/threads.y;
    const int d3=channel;
    dim3 blocks(d1, d2, d3); // each block counts for a position and one neighbor.
    
    const float* weight_data = weight.data<float>();
    const float* feature_data = feature.data<float>();
    float* out_data = out.data<float>();
    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,NULL);
    std::cout<<weight.size(0)<<" "<<weight.size(1)<<" "<<weight.size(2)<<" "<<weight.size(3)<<std::endl;
    std::cout<<feature.size(0)<<" "<<feature.size(1)<<" "<<feature.size(2)<<" "<<feature.size(3)<<std::endl;
    mapping_with_weight_forward_kernel<float><<<blocks,threads,0,stream>>>(weight_data,feature_data,out_data,batch,channel,height,width);
    cudaEventRecord(stop1,NULL);
    cudaEventSynchronize(stop1);
    float msecTotal1=0.0f;
    cudaEventElapsedTime(&msecTotal1,start1,stop1);
    std::cout<<msecTotal1<<std::endl;
    
    // AT_DISPATCH_FLOATING_TYPES(out.type(),"mapping forward cuda",([&]{
    //     mapping_with_weight_forward_kernel<scalar_t><<<blocks,threads,0,stream>>>(
    //         weight.data<scalar_t>(),
    //         feature.data<scalar_t>(),
    //         out.data<scalar_t>(),
    //         batch,
    //         channel,
    //         height,
    //         width   
    //     );
    // }));
    
    return out;
}
std::vector<at::Tensor> mapping_with_weight_backward(const at::Tensor grad_out,const at::Tensor weight, const at::Tensor feature)
{


    CHECK_INPUT(grad_out);
    CHECK_INPUT(weight);
    CHECK_INPUT(feature);
    AT_CHECK(weight.ndimension() == 5,"weight should be BXCXHXWXK");
    AT_CHECK(feature.ndimension() == 4,"feature should be BXCXHXW");  
    cudaStream_t stream = THCState_getCurrentStream(state);

    int batch = feature.size(0), channel = feature.size(1), height=feature.size(2),width=feature.size(3);
    auto grad_weight = at::zeros_like(weight);
    auto grad_feature = at::zeros_like(feature);

    dim3 threads(32, 32); // 1024 thread
     const int d1 = (width+threads.x-1)/threads.x; // get how many blocks needed for x
     const int d2 = (height+threads.y-1)/threads.y;
     const int d3=height+width-1;
     dim3 blocks(d1, d2, d3); // each block counts for a position and one neighbor.
    //std::cout << grad_out<< std::endl;
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(),"mapping backward weight cuda",([&]{
        mapping_with_weight_backward_weight_kernel<scalar_t><<<blocks,threads,0,stream>>>(
            grad_out.data<scalar_t>(),
            feature.data<scalar_t>(),
            grad_weight.data<scalar_t>(),
            batch,
            channel,
            height,
            width   
        );
    }));
    const int f_d3=channel;
    dim3 blocks_f(d1, d2, f_d3); // each block counts for a position and one neighbor.
    AT_DISPATCH_FLOATING_TYPES(grad_out.type(),"mapping backward feature cuda",([&]{
        mapping_with_weight_backward_feature_kernel<scalar_t><<<blocks_f,threads>>>(
            grad_out.data<scalar_t>(),
            weight.data<scalar_t>(),
            grad_feature.data<scalar_t>(),
            batch,
            channel,
            height,
            width
        );
    }));
    return {grad_weight,grad_feature};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("compute_weight_forward", &compute_mapping_full_forward, "weight forward (CUDA)");
     m.def("compute_weight_backward", &compute_mapping_full_backward, "weight backward (CUDA)");
     m.def("aggregate_forward",&mapping_with_weight_forward,"aggregate with weight forward(CUDA)");
     m.def("aggregate_backward",&mapping_with_weight_backward,"aggregate with weight backward(CUDA)");
   }
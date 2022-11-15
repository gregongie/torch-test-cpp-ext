#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// computes one projetion view
__global__ void projection_view_kernel(
                    const torch::PackedTensorAccessor32<float,2> image,
                    torch::PackedTensorAccessor32<float,2> sinogram,
                    const float dx,
                    const float dy,
                    const float x0,
                    const float y0,
                    const float fanangle2,
                    const float detectorlength,
                    const float u0,
                    const float du,
                    const float ds,
                    const float radius,
                    const float source_to_detector,
                    const int nbins){

  const int nx = image.size(0);
  const int ny = image.size(1);

  // get view index "sindex" from block/thread
  // const int n = blockIdx.y;
  // const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int sindex = threadIdx.x;

  auto s = sindex*ds;

  // location of the source
  auto xsource = radius*cos(s);
  auto ysource = radius*sin(s);

  // detector center
  auto xDetCenter = (radius - source_to_detector)*cos(s);
  auto yDetCenter = (radius - source_to_detector)*sin(s);

  // unit vector in the direction of the detector line
  auto eux = -sin(s);
  auto euy =  cos(s);

  //loop over detector views
  for (int uindex = 0; uindex < nbins; uindex++){
    auto u = u0 + (uindex+0.5)*du;
    auto xbin = xDetCenter + eux*u;
    auto ybin = yDetCenter + euy*u;

    auto xl = x0;
    auto yl = y0;

    auto xdiff = xbin-xsource;
    auto ydiff = ybin-ysource;
    auto xad = abs(xdiff)*dy;
    auto yad = abs(ydiff)*dx;

    float raysum = 0.0; // acculumator variable

    if (xad > yad){  // loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
      auto slope = ydiff/xdiff;
      auto travPixlen = dx*sqrt(1.0+slope*slope);
      auto yIntOld = ysource+slope*(xl-xsource);
      int iyOld = static_cast<int>(floor((yIntOld-y0)/dy));
      // loop over x-layers
      for (int ix = 0; ix < nx; ix++){
         auto x=xl+dx*(ix + 1.0);
         auto yIntercept=ysource+slope*(x-xsource);
         int iy = static_cast<int>(floor((yIntercept-y0)/dy));
         if (iy == iyOld){ // if true, ray stays in the same pixel for this x-layer
            if ((iy >= 0) && (iy < ny)) {
               raysum += travPixlen*image[ix][iy];
            }
         } else {    // else case is if ray hits two pixels for this x-layer
            auto yMid=dy*std::max(iy,iyOld)+yl;
            auto ydist1=abs(yMid-yIntOld);
            auto ydist2=abs(yIntercept-yMid);
            auto frac1=ydist1/(ydist1+ydist2);
            auto frac2=1.0-frac1;
            if ((iyOld >= 0) && (iyOld < ny)){
               raysum += frac1*travPixlen*image[ix][iyOld];
             }
            if ((iy>=0) && (iy<ny)){
               raysum += frac2*travPixlen*image[ix][iy];
             }
         }
         iyOld=iy;
         yIntOld=yIntercept;
       }

    } else {// through y-layers of image if xad<=yad
      auto slopeinv=xdiff/ydiff;
      auto travPixlen=dy*sqrt(1.0+slopeinv*slopeinv);
      auto xIntOld=xsource+slopeinv*(yl-ysource);
      int ixOld= static_cast<int>(floor((xIntOld-x0)/dx));
      // loop over y-layers
      for (int iy = 0; iy < ny; iy++){
         auto y=yl+dy*(iy + 1.0);
         auto xIntercept=xsource+slopeinv*(y-ysource);
         int ix = static_cast<int>(floor((xIntercept-x0)/dx));
         if (ix == ixOld){// if true, ray stays in the same pixel for this y-layer
            if ((ix >= 0) && (ix < nx)){
               raysum += travPixlen*image[ix][iy];
             }
         } else {  // else case is if ray hits two pixels for this y-layer
            auto xMid=dx*std::max(ix,ixOld)+xl;
            auto xdist1=abs(xMid-xIntOld);
            auto xdist2=abs(xIntercept-xMid);
            auto frac1=xdist1/(xdist1+xdist2);
            auto frac2=1.0-frac1;
            if ((ixOld >= 0) && (ixOld < nx)){
               raysum += frac1*travPixlen*image[ixOld][iy];
            }
            if ((ix>=0) && (ix<nx)){
               raysum += frac2*travPixlen*image[ix][iy];
            }
         }
         ixOld = ix;
         xIntOld = xIntercept;
       }
    }
    sinogram[sindex][uindex]=raysum;
 }
}

torch::Tensor circularFanbeamProjection_cuda(const torch::Tensor image, float ximageside, float yimageside,
                              float radius, float source_to_detector,
                              int nviews, float slen, int nbins) {
    const float dx = ximageside/image.size(0);
    const float dy = yimageside/image.size(1);
    const float x0 = -ximageside/2.0;
    const float y0 = -yimageside/2.0;

    // compute length of detector so that it views the inscribed FOV of the image array
    const float fanangle2 = std::asin((ximageside/2.0)/radius);  //This only works for ximageside = yimageside
    const float detectorlength = 2.0*std::tan(fanangle2)*source_to_detector;
    const float u0 = -detectorlength/2.0;

    const float du = detectorlength/nbins;
    const float ds = slen/nviews;

    const auto image_a = image.packed_accessor32<float,2>();

    const auto options = torch::TensorOptions().device(torch::kCUDA);
    torch::Tensor sinogram = torch::zeros({nviews, nbins}, options);
    auto sinogram_a = sinogram.packed_accessor32<float,2>();

    const int threads = 512; //one per view?
    // const dim3 blocks((512 + threads - 1) / threads, 1);
    const int blocks = 1; //match to batch size in future?

    projection_view_kernel<<<blocks, threads>>>(image_a,
                                                sinogram_a,
                                                dx,
                                                dy,
                                                x0,
                                                y0,
                                                fanangle2,
                                                detectorlength,
                                                u0,
                                                du,
                                                ds,
                                                radius,
                                                source_to_detector,
                                                nbins);

    return sinogram;
}

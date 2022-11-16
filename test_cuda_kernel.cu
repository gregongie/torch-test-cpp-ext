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

// computes backprojetion over one view
__global__ void backprojection_view_kernel(
                    torch::PackedTensorAccessor32<float,2> image,
                    const torch::PackedTensorAccessor32<float,2> sinogram,
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
                    const int nbins,
                    const float fov_radius,
                    const float pi){

                    const int nx = image.size(0);
                    const int ny = image.size(1);

                    const int sindex = threadIdx.x;
                    float s = sindex*ds;

                    // location of the source
                    float xsource = radius*cos(s);
                    float ysource = radius*sin(s);

                    // detector center
                    float xDetCenter = (radius - source_to_detector)*cos(s);
                    float yDetCenter = (radius - source_to_detector)*sin(s);

                    // unit vector in the direction of the detector line
                    float eux = -sin(s);
                    float euy =  cos(s);

                    //Unit vector in the direction perpendicular to the detector line
                    float ewx = cos(s);
                    float ewy = sin(s);

                    for (int iy = 0; iy < ny; iy++){
                       float pix_y = y0 + dy*(iy+0.5);
                       for (int ix = 0; ix < nx; ix++){
                          float pix_x = x0 + dx*(ix+0.5);

                          float frad = sqrt(pix_x*pix_x + pix_y*pix_y);
                          float fphi = atan2(pix_y,pix_x);
                          if (frad<=fov_radius){
                             float bigu = (radius+frad*sin(s-fphi-pi/2.0))/radius;
                             float bpweight = 1.0/(bigu*bigu);

                             float ew_dot_source_pix = (pix_x-xsource)*ewx + (pix_y-ysource)*ewy;
                             float rayratio = -source_to_detector/ew_dot_source_pix;

                             float det_int_x = xsource+rayratio*(pix_x-xsource);
                             float det_int_y = ysource+rayratio*(pix_y-ysource);

                             float upos = ((det_int_x-xDetCenter)*eux +(det_int_y-yDetCenter)*euy);
                             float det_value;

                             if ((upos-u0 >= du/2.0) && (upos-u0 < detectorlength-du/2.0)){
                                float bin_loc = (upos-u0)/du + 0.5;
                                int nbin1 = static_cast<int>(bin_loc)-1;
                                int nbin2 = nbin1+1;
                                float frac= bin_loc - static_cast<int>(bin_loc);
                                det_value = frac*sinogram[sindex][nbin2]+(1.0-frac)*sinogram[sindex][nbin1];
                                atomicAdd(image[ix][iy],bpweight*det_value*ds);
                              }
                             // } else {
                             //    det_value = 0.0;
                             // }
                             // image[ix][iy] += bpweight*det_value*ds;
                         }
                      }
                   }

}

torch::Tensor circularFanbeamProjection_cuda(const torch::Tensor image, const int nx, const int ny, const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
    const float dx = ximageside/nx;
    const float dy = yimageside/ny;
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

torch::Tensor circularFanbeamBackProjection_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
   const float dx = ximageside/nx;
   const float dy = yimageside/ny;
   const float x0 = -ximageside/2.0;
   const float y0 = -yimageside/2.0;

   // compute length of detector so that it views the inscribed FOV of the image array
   const float fanangle2 = asin((ximageside/2.0)/radius);  //This only works for ximageside = yimageside
   const float detectorlength = 2.0*tan(fanangle2)*source_to_detector;
   const float u0 = -detectorlength/2.0;

   const float du = detectorlength/nbins;
   const float ds = slen/nviews;

   const float fov_radius = ximageside/2.0;

   const auto options = torch::TensorOptions().device(torch::kCUDA);
   torch::Tensor image = torch::zeros({nx, ny}, options); //initialize image
   auto image_a = image.packed_accessor32<float,2>(); //accessor for updating values of image

   const auto sinogram_a = sinogram.packed_accessor32<float,2>(); //accessor for accessing values of sinogram

   const float pi = 4*atan(1);

   const int threads = 512; //one per view?
   // const dim3 blocks((512 + threads - 1) / threads, 1);
   const int blocks = 1; //match to batch size in future?

   backprojection_view_kernel<<<blocks, threads>>>(image_a,
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
                                                   nbins,
                                                   fov_radius,
                                                   pi);
   return image;
}

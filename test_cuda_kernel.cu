#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

inline int roundup_div(const int x, const int y) {
    return x / y + (x % y != 0);
}

// computes one projetion view
__global__ void projection_view_kernel(
                    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> image,
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> sinogram,
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
                    const int nviews,
                    const int nx,
                    const int ny){

  // const int nx = image.size(1);
  // const int ny = image.size(2);

  // const int ib = blockIdx.x;
  // const int sindex = threadIdx.x;

  const int uindex = blockIdx.x * blockDim.x + threadIdx.x;
  const int sindex = blockIdx.y * blockDim.y + threadIdx.y;
  const int ib = blockIdx.z;

  //loop over detector views
  // for (int uindex = 0; uindex < nbins; uindex++){
  if ((uindex < nbins) && (sindex < nviews)) {
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
               raysum += travPixlen*image[ib][ix][iy];
            }
         } else {    // else case is if ray hits two pixels for this x-layer
            auto yMid=dy*max(iy,iyOld)+yl;
            auto ydist1=abs(yMid-yIntOld);
            auto ydist2=abs(yIntercept-yMid);
            auto frac1=ydist1/(ydist1+ydist2);
            auto frac2=1.0-frac1;
            if ((iyOld >= 0) && (iyOld < ny)){
               raysum += frac1*travPixlen*image[ib][ix][iyOld];
             }
            if ((iy>=0) && (iy<ny)){
               raysum += frac2*travPixlen*image[ib][ix][iy];
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
               raysum += travPixlen*image[ib][ix][iy];
             }
         } else {  // else case is if ray hits two pixels for this y-layer
            auto xMid=dx*max(ix,ixOld)+xl;
            auto xdist1=abs(xMid-xIntOld);
            auto xdist2=abs(xIntercept-xMid);
            auto frac1=xdist1/(xdist1+xdist2);
            auto frac2=1.0-frac1;
            if ((ixOld >= 0) && (ixOld < nx)){
               raysum += frac1*travPixlen*image[ib][ixOld][iy];
            }
            if ((ix>=0) && (ix<nx)){
               raysum += frac2*travPixlen*image[ib][ix][iy];
            }
         }
         ixOld = ix;
         xIntOld = xIntercept;
       }
    }
    sinogram[ib][sindex][uindex]=raysum;
 }
}

// computes one backprojection view
__global__ void backprojection_view_kernel(
                    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> image,
                    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> sinogram,
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
                    const float fov_radius,
                    const int nbins,
                    const int nviews,
                    const int nx,
                    const int ny){

  const int uindex = blockIdx.x * blockDim.x + threadIdx.x; //detector index
  const int sindex = blockIdx.y * blockDim.y + threadIdx.y; //view index
  const int ib = blockIdx.z; //batch index
  // const int ib = blockIdx.x;
  // const int sindex = threadIdx.x;

  //compute backprojection for a single ray
  if ((uindex < nbins) && (sindex < nviews)) {
    const auto sinoval = sinogram[ib][sindex][uindex];

    const float s = sindex*ds;

    // location of the source
    const float xsource = radius*cos(s);
    const float ysource = radius*sin(s);

    // detector center
    const float xDetCenter = (radius - source_to_detector)*cos(s);
    const float yDetCenter = (radius - source_to_detector)*sin(s);

    // unit vector in the direction of the detector line
    const float eux = -sin(s);
    const float euy =  cos(s);

    const float fov_radius2 = fov_radius*fov_radius; //used to set image mask

    float u = u0+(uindex+0.5)*du;
    float xbin = xDetCenter + eux*u;
    float ybin = yDetCenter + euy*u;

    float xl=x0;
    float yl=y0;

    float xdiff=xbin-xsource;
    float ydiff=ybin-ysource;
    float xad=abs(xdiff)*dy;
    float yad=abs(ydiff)*dx;

    if (xad>yad){   // loop through x-layers of image if xad>yad. This ensures ray hits only one or two pixels per layer
       float slope=ydiff/xdiff;
       float travPixlen=dx*sqrt(1.0+slope*slope);
       float yIntOld=ysource + slope*(xl-xsource);
       int iyOld = static_cast<int>(floor((yIntOld-y0)/dy));
       for (int ix = 0; ix < nx; ix++){
          float x = xl + dx*(ix + 1.0);
          float yIntercept=ysource+slope*(x-xsource);
          int iy = static_cast<int>(floor((yIntercept-y0)/dy));
          float pix_x = x0 + dx*(ix+0.5); //used to set mask
          float pix_y = y0 + dy*(iy+0.5); //used to set mask
          float pix_y_old = y0 + dy*(iyOld+0.5); // used to set mask
            if (iy == iyOld){ // if true, ray stays in the same pixel for this x-layer
             if ((pix_x*pix_x + pix_y*pix_y <= fov_radius2) && (iy >= 0) && (iy < ny)){
                atomicAdd(&image[ib][ix][iy],sinoval*travPixlen);
                // image[ib][ix][iy] += sinoval*travPixlen;
              }
          } else {    // else case is if ray hits two pixels for this x-layer
             float yMid = dy*max(iy,iyOld)+yl;
             float ydist1 = abs(yMid-yIntOld);
             float ydist2 = abs(yIntercept-yMid);
             float frac1 = ydist1/(ydist1+ydist2);
             float frac2 = 1.0-frac1;
             if ((iyOld >= 0) && (iyOld < ny) && (pix_x*pix_x + pix_y_old*pix_y_old <= fov_radius2)){
                atomicAdd(&image[ib][ix][iyOld],frac1*sinoval*travPixlen);
                // image[ib][ix][iyOld] += frac1*sinoval*travPixlen;
              }
             if ((iy >= 0) && (iy < ny) && (pix_x*pix_x + pix_y*pix_y <= fov_radius2)) {
                atomicAdd(&image[ib][ix][iy],frac2*sinoval*travPixlen);
                // image[ib][ix][iy] += frac2*sinoval*travPixlen;
              }
          }
          iyOld=iy;
          yIntOld=yIntercept;
        }
    } else { //loop through y-layers of image if xad<=yad
       float slopeinv=xdiff/ydiff;
       float travPixlen=dy*sqrt(1.0+slopeinv*slopeinv);
       float xIntOld=xsource+slopeinv*(yl-ysource);
       int ixOld = static_cast<int>(floor((xIntOld-x0)/dx));
       for (int iy = 0; iy < ny; iy++){
          float y = yl + dy*(iy + 1.0);
          float xIntercept = xsource+slopeinv*(y-ysource);
          int ix = static_cast<int>(floor((xIntercept-x0)/dx));
          float pix_x = x0 + dx*(ix+0.5);
          float pix_y = y0 + dy*(iy+0.5);
          float pix_x_old = x0 + dx*(ixOld+0.5); // used to set mask
          if (ix == ixOld){ // if true, ray stays in the same pixel for this y-layer
             if ((ix >= 0) && (ix < nx) && (pix_x*pix_x + pix_y*pix_y <= fov_radius2)) {
                atomicAdd(&image[ib][ix][iy],sinoval*travPixlen);
                // image[ib][ix][iy] += sinoval*travPixlen;
              }
          } else { // else case is if ray hits two pixels for this y-layer
             float xMid = dx*max(ix,ixOld)+xl;
             float xdist1 = abs(xMid-xIntOld);
             float xdist2 = abs(xIntercept-xMid);
             float frac1 = xdist1/(xdist1+xdist2);
             float frac2=1.0-frac1;
             if ((ixOld >= 0) && (ixOld < nx) && (pix_x_old*pix_x_old + pix_y*pix_y <= fov_radius2)){
                atomicAdd(&image[ib][ixOld][iy],frac1*sinoval*travPixlen);
                // image[ib][ixOld][iy] += frac1*sinoval*travPixlen;
              }
             if ((ix >= 0) && (ix < nx) && (pix_x*pix_x + pix_y*pix_y <= fov_radius2)){
                atomicAdd(&image[ib][ix][iy],frac2*sinoval*travPixlen);
                // image[ib][ix][iy] += frac2*sinoval*travPixlen;
              }
          }
          ixOld = ix;
          xIntOld = xIntercept;
       }
     }
   } // end uindex for loop

}

// computes pixel-driven backprojetion over one view
__global__ void backprojection_pix_view_kernel(
                    torch::PackedTensorAccessor32<float,3> image,
                    const torch::PackedTensorAccessor32<float,3> sinogram,
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

                    const int nx = image.size(1);
                    const int ny = image.size(2);

                    const int sindex = threadIdx.x;
                    const int ib = blockIdx.x;

                    const float s = sindex*ds;

                    // location of the source
                    const float xsource = radius*cos(s);
                    const float ysource = radius*sin(s);

                    // detector center
                    const float xDetCenter = (radius - source_to_detector)*cos(s);
                    const float yDetCenter = (radius - source_to_detector)*sin(s);

                    // unit vector in the direction of the detector line
                    const float eux = -sin(s);
                    const float euy =  cos(s);

                    //Unit vector in the direction perpendicular to the detector line
                    const float ewx = cos(s);
                    const float ewy = sin(s);

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
                                det_value = frac*sinogram[ib][sindex][nbin2]+(1.0-frac)*sinogram[ib][sindex][nbin1];
                                atomicAdd(&image[ib][ix][iy],bpweight*det_value*ds);
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

    const auto image_a = image.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    const int batch_size = image_a.size(0); //batch_size

    const auto options = torch::TensorOptions().dtype(image.dtype()).device(image.device());
    auto sinogram = torch::zeros({batch_size, nviews, nbins}, options);
    auto sinogram_a = sinogram.packed_accessor32<float,3,torch::RestrictPtrTraits>();

    // parallize over rays, batches
    dim3 block_dim(16, 16);
    dim3 grid_dim(roundup_div(nbins, 16), roundup_div(nviews, 16), batch_size );

    projection_view_kernel<<<grid_dim, block_dim>>>(image_a,
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
                                                nviews,
                                                nx,
                                                ny);

    return sinogram;
}

// exact matrix transpose of circularFanbeamProjection
torch::Tensor circularFanbeamBackProjection_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
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

   const float fov_radius = ximageside/2.0;

   const auto sinogram_a = sinogram.packed_accessor32<float,3,torch::RestrictPtrTraits>();
   const int batch_size = sinogram_a.size(0); //batch_size

   const auto options = torch::TensorOptions().dtype(sinogram.dtype()).device(sinogram.device());
   auto image = torch::zeros({batch_size, nx, ny}, options);
   auto image_a = image.packed_accessor32<float,3,torch::RestrictPtrTraits>();

   // const int threads = nviews; //one per view, max 1024 -- todo: add input validation
   // const int blocks = batch_size; //match to batch size

   // parallize over rays, batches
   dim3 block_dim(16, 16);
   dim3 grid_dim(roundup_div(nbins, 16), roundup_div(nviews, 16), batch_size );

   backprojection_view_kernel<<<grid_dim, block_dim>>>(image_a,
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
                                               fov_radius,
                                               nbins,
                                               nviews,
                                               nx,
                                               ny);

    return image;
}


torch::Tensor circularFanbeamBackProjectionPixelDriven_cuda(const torch::Tensor sinogram, const int nx, const int ny,
                              const float ximageside, const float yimageside,
                              const float radius, const float source_to_detector,
                              const int nviews, const float slen, const int nbins) {
   const int batch_size = sinogram.size(0);
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
   torch::Tensor image = torch::zeros({batch_size, nx, ny}, options); //initialize image
   auto image_a = image.packed_accessor32<float,3>(); //accessor for updating values of image

   const auto sinogram_a = sinogram.packed_accessor32<float,3>(); //accessor for accessing values of sinogram

   const float pi = 4*atan(1);

   const int threads = nviews; //one per view, max 1024
   const int blocks = batch_size; //match to batch size

   backprojection_pix_view_kernel<<<blocks, threads>>>(image_a,
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

#include <torch/extension.h>
#include <vector>
#include <cmath>

torch::Tensor circularFanbeamProjection(torch::Tensor image, float ximageside, float yimageside,
                              float radius, float source_to_detector,
                              int nviews, float slen, int nbins) {
  const auto aimage = image.accessor<float,2>();
  const auto nx = aimage.size(0);
  const auto ny = aimage.size(1);

  const auto dx = ximageside/nx;
  const auto dy = yimageside/ny;
  const auto x0 = -ximageside/2;
  const auto y0 = -yimageside/2;

  // compute length of detector so that it views the inscribed FOV of the image array
  const auto fanangle2 = asin((ximageside/2)/radius);  //This only works for ximageside = yimageside
  const auto detectorlength = 2*tan(fanangle2)*source_to_detector;
  const auto u0 = -detectorlength/2;

  const auto du = detectorlength/nbins;
  const auto ds = slen/nviews;

  torch::Tensor sinogram = torch::zeros({nviews,nbins});
  auto asinogram = sinogram.accessor<float, 2>(); //accessor for updating values of sinogram

  //loop over views -- parallelize over this loop!
  for (int sindex = 0; sindex < nviews; sindex++){
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

    //Unit vector in the direction perpendicular to the detector line
    // auto ewx = cos(s);
    // auto ewy = sin(s);

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
        auto travPixlen = dx*sqrt(1+slope*slope);
        auto yIntOld = ysource+slope*(xl-xsource);
        int iyOld = static_cast<int>(floor((yIntOld-y0)/dy));
        // loop over x-layers
        for (int ix = 0; ix < nx; ix++){
           auto x=xl+dx*(ix + 1.0);
           auto yIntercept=ysource+slope*(x-xsource);
           int iy = static_cast<int>(floor((yIntercept-y0)/dy));
           if (iy == iyOld){ // if true, ray stays in the same pixel for this x-layer
              if ((iy >= 0) && (iy < ny)) {
                 raysum=raysum+travPixlen*aimage[ix][iy];
              }
           } else {    // else case is if ray hits two pixels for this x-layer
              auto yMid=dy*std::max(iy,iyOld)+yl;
              auto ydist1=abs(yMid-yIntOld);
              auto ydist2=abs(yIntercept-yMid);
              auto frac1=ydist1/(ydist1+ydist2);
              auto frac2=1.0-frac1;
              if ((iyOld >= 0) && (iyOld < ny)){
                 raysum += frac1*travPixlen*aimage[ix][iyOld];
               }
              if ((iy>=0) && (iy<ny)){
                 raysum += frac2*travPixlen*aimage[ix][iy];
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
                 raysum += travPixlen*aimage[ix][iy];
               }
           } else {  // else case is if ray hits two pixels for this y-layer
              auto xMid=dx*std::max(ix,ixOld)+xl;
              auto xdist1=abs(xMid-xIntOld);
              auto xdist2=abs(xIntercept-xMid);
              auto frac1=xdist1/(xdist1+xdist2);
              auto frac2=1.0-frac1;
              if ((ixOld >= 0) && (ixOld < nx)){
                 raysum += frac1*travPixlen*aimage[ixOld][iy];
              }
              if ((ix>=0) && (ix<nx)){
                 raysum += frac2*travPixlen*aimage[ix][iy];
              }
           ixOld = ix;
           xIntOld = xIntercept;
          }
         }
      }
    asinogram[sindex][uindex]=raysum;
   }
 }
 return sinogram;
}

torch::Tensor test_backward(torch::Tensor x) {
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &circularFanbeamProjection, "Test forward");
  m.def("backward", &test_backward, "Test backward");
}

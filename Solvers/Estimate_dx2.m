function uu_dx2 = Estimate_dx2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
  uu_l = interp2(xx - h,yy,uu_LUT,xx_LUT,yy_LUT,'spline');
  uu_r = interp2(xx + h,yy,uu_LUT,xx_LUT,yy_LUT,'spline');
  uu_c = interp2(xx,yy,uu_LUT,xx_LUT,yy_LUT,'spline');
  uu_dx2 = (uu_l+uu_r-2*uu_c)./(h.*h)
endfunction

function uu_dy2 = Estimate_dy2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
  uu_l = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy-h,'spline')
  uu_r = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy+h,'spline')
  uu_c = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'spline')
  uu_dy2 = (uu_l+uu_r-2*uu_c)./(h.*h)
endfunction
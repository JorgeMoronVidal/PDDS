#! octave-qf
 function [Dx,Dy,x,y] = cheb(N,VEC_x, VEC_y) 
   if N==0 
     Dx =0, Dy = 0, x =1, y = 1; 
     return
   end
   x = (VEC_x(N)+0.5*(1+cos(pi*(0:N)/N))*(VEC_x(1)-VEC_x(N))).';
   y = (VEC_y(N)+0.5*(1+cos(pi*(0:N)/N))*(VEC_y(1)-VEC_y(N))).'; 
   c = [2; ones(N-1,1); 2] .* (-1) .^ (0:N)' ;
   X = repmat(x,1,N+1) ; 
   dX = X - X.' ;
   % The off-diagnoal entries 
   Dx = (c* (1 ./ c)') ./ (dX + eye(N+1)) ; 
   % diagnoal entries 
   Dx = Dx - diag(sum(Dx.')) ;
   Y = repmat(y,1,N+1) ; 
   dY = Y - Y.' ;
   % The off-diagnoal entries 
   Dy = (c* (1 ./ c)') ./ (dY + eye(N+1)) ; 
   % diagnoal entries 
   Dy = Dy - diag(sum(Dy.')) ;
 end
%function uu_dx2 = Estimate_dx2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
%  uu_l = interp2(xx_LUT,yy_LUT,uu_LUT,xx - h,yy,'spline');
%  uu_r = interp2(xx_LUT,yy_LUT,uu_LUT,xx + h,yy,'spline');
%  uu_c = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'spline');
%  uu_dx2 = (uu_l+uu_r-2*uu_c)./(h.*h);
%endfunction
%function uu_dy2 = Estimate_dy2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
%  uu_l = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy-h,'spline');
%  uu_r = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy+h,'spline');
%  uu_c = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'spline');
%  uu_dy2 = (uu_l+uu_r-2*uu_c)./(h.*h);
%endfunction
%p36.m (Modified by Jorge Moron) - Poisson eq. on [-1,1]x[-1,1] with nonzero BC's
%Set up grid and 2D Laplacian, boundary points included:
args = argv();
id = args{1};
file = sprintf("Input/Interfaces/North_%s.txt", id);
table = csvread(file);
x_north = table(:,1);
y_north = table(:,2);
sol_north = table(:,3);
%sol_noisy_north = table(:,4);
clear table;
file = sprintf("Input/Interfaces/South_%s.txt", id);
table = csvread(file);
x_south = table(:,1);
y_south = table(:,2);
sol_south = table(:,3);
%sol_noisy_south = table(:,4);
clear table;
file = sprintf("Input/Interfaces/East_%s.txt", id);
table = csvread(file);
x_east = table(:,1);
y_east = table(:,2);
sol_east = table(:,3);
%sol_noisy_east = table(:,4);
clear table;
file = sprintf("Input/Interfaces/West_%s.txt", id);
table = csvread(file);
x_west = table(:,1);
y_west = table(:,2);
sol_west = table(:,3);
%sol_noisy_west = table(:,4);
clear table;
file = sprintf("Output/Subdomains/X_%s_%s.txt", args{2},args{3});
table = csvread(file);
x_LUT = table(:,1);
file = sprintf("Output/Subdomains/Y_%s_%s.txt", args{2},args{3});
table = csvread(file);
y_LUT = table(:,1);
file = sprintf("Output/Subdomains/Sol_%s_%s.txt", args{2},args{3});
table = csvread(file);
u_LUT = table(:,1);
file = sprintf("Output/Subdomains/Correction_%s_%s.txt", args{2},args{3});
table = csvread(file);
v_LUT = table(:,1);
[xx_LUT,yy_LUT]=meshgrid(x_LUT,y_LUT);
uu_LUT = reshape(u_LUT,size(yy_LUT))';
vv_LUT = reshape(v_LUT,size(yy_LUT))';
N = size(x_north)(1); [Dx,Dy,x,y] = cheb(N,x_north,y_west);
[xx,yy] = meshgrid(x,y);
uu_LUT = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'spline');
vv_LUT = interp2(xx_LUT,yy_LUT,vv_LUT,xx,yy,'spline');
xx_LUT = xx;
yy_LUT = yy;
rhs = (interp2(xx_LUT,yy_LUT,vv_LUT,xx,yy,'spline').^2).*(3*(interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'spline'))-2*interp2(xx_LUT,yy_LUT,vv_LUT,xx,yy,'spline'));
rhs = rhs(:);
xx = xx(:); yy = yy(:);
D2x = Dx^2; D2y = Dy^2; I = eye(N+1); L = kron(I,D2x) + kron(D2y,I)-3*diag((interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'cubic')).^2);
%Impose boundary conditions and -f function by replacing appropriate rows of L:
b = find(xx==x_west(1) | xx == x_east(1) | yy==y_north(1) | yy == y_south(1));
% boundary pts
L(b,:) = zeros(4*N,(N+1)^2); 
L(b,b) = eye(4*N);
%rhs(b) = sin(omegax*pi*xx(b) + omegay*pi*yy(b)) + cos(omegapx*pi*xx(b) + omegapy*pi*yy(b));
b_west = find(xx==x_west(1));
rhs(b_west) = interp1(y_west,sol_west, yy(b_west),'spline');
b_east = find(xx==x_east(1));
rhs(b_east) = interp1(y_east,sol_east, yy(b_east),'spline');
b_north = find(yy==y_north(1));
rhs(b_north) = interp1(x_north,sol_north, xx(b_north),'spline');
b_south = find(yy==y_south(1));
rhs(b_south) = interp1(x_south,sol_south, xx(b_south),'spline');
% Solve Poisson equation, reshape to 2D, and plot:
v = L\rhs; vv = reshape(v,N+1,N+1);
file = sprintf("Output/Subdomains/X_%s_%s.txt", args{2},args{3});
save("-ascii",file,"x")
file = sprintf("Output/Subdomains/Y_%s_%s.txt", args{2},args{3});
save("-ascii",file,"y")
file = sprintf("Output/Subdomains/Correction_%s_%s.txt", args{2},args{3});
v = (vv')(:);
save("-ascii",file,"v");
file = sprintf("Output/Subdomains/Sol_%s_%s.txt", args{2},args{3});
u = (vv'+uu_LUT')(:);
save("-ascii",file,"u");
%rhs(b_west) = interp1(y_west,sol_noisy_west, yy(b_west),'spline');
%rhs(b_east) = interp1(y_east,sol_noisy_east, yy(b_east),'spline');
%rhs(b_north) = interp1(x_north,sol_noisy_north, xx(b_north),'spline');
%rhs(b_south) = interp1(x_south,sol_noisy_south, xx(b_south),'spline');
%u = L\rhs; uu = reshape(u,N+1,N+1);
%file = sprintf("Output/Subdomains/Sol_noisy_%s_%s.txt", args{2},args{3});
%u = (uu')(:);
%save("-ascii",file,"u");
%[xx,yy] = meshgrid(x,y);
%uu = reshape(u,size(xx))';
%[xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
%uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
%clf, subplot(1,3,1),
%mesh(xx,yy,uu), colormap([0 0 0])
%view(-20,45),
%title('Pseudopspectral solution')
%u_a = sin(pi*xx).*sin(pi*yy);
%uu_a = reshape(u_a,N+1,N+1);
%uuu_a = interp2(xx,yy,uu_a,xxx,yyy,'spline');
%subplot(1,3,2),
%mesh(xx,yy,uu_a), colormap([0 0 0])
%view(-20,45),
%title('Analytical solution')
%error = abs(uu_a - uu);
%subplot(1,3,3),
%mesh(xx,yy,error), colormap([0 0 0]),
%title('Absolute error')
%waitfor(gcf)

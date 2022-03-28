#! octave-qf
function u = Monegros(x,y,op)
k_1 = 1.0; k_2 = 0.01; k_3 = 0.02; k_4 = 0.12; k_5 = 0.05;
k_6 = 0.05; k_7 = -0.12; scaling_factor = 0.333;
switch op
    case 'u'
        u = 3 + scaling_factor*(sin(sqrt(k_1+k_2*x.^2 +k_3*y.^2)) + tanh(sin(k_4*x + k_5*y)+sin(k_6*x + k_7*y)));
    case 'dudx'
        u = k_2*x.*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./sqrt(k_2*x.^2 + k_3*y.^2 + k_1) -...
        (k_4*cos(k_4*x + k_5*y) + k_6*cos(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1);
        u = u*scaling_factor;
    case 'dudy'
        u = k_3*y.*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./sqrt(k_2*x.^2 + k_3*y.^2 + k_1) -...
        (k_5*cos(k_4*x + k_5*y) + k_7*cos(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1);
        u = u*scaling_factor;
    case 'd2udx2'
        u = -k_2^2*x.^2.*sin(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1) + ...
        2*(k_4*cos(k_4*x + k_5*y) + k_6*cos(k_6*x + k_7*y)).^2.*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1).*tanh(sin(k_4*x + k_5*y) + sin(k_6*x + k_7*y)) -...
        k_2^2*x.^2.*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1).^(3/2) +...
        (k_4^2*sin(k_4*x + k_5*y) + k_6^2*sin(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1) + k_2*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./sqrt(k_2*x.^2 +...
        k_3*y.^2 + k_1);
        u = u*scaling_factor;
    case 'd2udy2'
        u = -k_3^2*y.^2.*sin(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1) + ...
        2*(k_5*cos(k_4*x + k_5*y) + k_7*cos(k_6*x + k_7*y)).^2.*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1).*tanh(sin(k_4*x + k_5*y) + sin(k_6*x + k_7*y)) -...
        k_3^2*y.^2.*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1).^(3/2) +... 
        (k_5^2*sin(k_4*x + k_5*y) + k_7^2*sin(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) +...
        sin(k_6*x + k_7*y)).^2 - 1) + k_3*cos(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./sqrt(k_2*x.^2 +...
        k_3*y.^2 + k_1);
        u = u*scaling_factor;
    case 'd2udxdy'
        u = -k_2*k_3*x.*y.*sin(sqrt(k_2*x.^2 + k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1) + ...
        2*(k_4*cos(k_4*x + k_5*y) + k_6*cos(k_6*x + k_7*y)).*(k_5*cos(k_4*x + k_5*y) +...
        k_7*cos(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) + sin(k_6*x + k_7*y)).^2 - 1)...
        .*tanh(sin(k_4*x + k_5*y) + sin(k_6*x + k_7*y)) - k_2*k_3*x.*y.*cos(sqrt(k_2*x.^2 + ...
        k_3*y.^2 + k_1))./(k_2*x.^2 + k_3*y.^2 + k_1).^(3/2) + (k_4*k_5*sin(k_4*x + k_5*y) +...
        k_6*k_7*sin(k_6*x + k_7*y)).*(tanh(sin(k_4*x + k_5*y) + sin(k_6*x + k_7*y)).^2 - 1);
        u = u*scaling_factor;
    otherwise
        disp('Error')
end
    
end

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
 function uu_dx2 = Estimate_dx2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
  uu_l = interp2(xx_LUT,yy_LUT,uu_LUT,xx - h,yy,'cubic');
  uu_r = interp2(xx_LUT,yy_LUT,uu_LUT,xx + h,yy,'cubic');
  uu_c = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'cubic');
  uu_dx2 = (uu_l+uu_r-2*uu_c)./(h.*h);
endfunction
function uu_dy2 = Estimate_dy2(h,xx,yy,uu_LUT,xx_LUT,yy_LUT)
  uu_l = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy-h,'cubic');
  uu_r = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy+h,'cubic');
  uu_c = interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'cubic');
  uu_dy2 = (uu_l+uu_r-2*uu_c)./(h.*h);
endfunction
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
[xx_LUT,yy_LUT]=meshgrid(x_LUT,y_LUT);
uu_LUT = reshape(u_LUT,size(yy_LUT));
uu_LUT = uu_LUT';
N = size(x_north)(1); [Dx,Dy,x,y] = cheb(N,x_north,y_west);
[xx,yy] = meshgrid(x,y); xx = xx(:); yy = yy(:);
D2x = Dx^2; D2y = Dy^2; I = eye(N+1); L = kron(I,D2x) + kron(D2y,I)-3*diag((interp2(xx_LUT,yy_LUT,uu_LUT,xx,yy,'cubic')).^2);
%Impose boundary conditions and -f function by replacing appropriate rows of L:
b = find(xx==x_west(1) | xx == x_east(1) | yy==y_north(1) | yy == y_south(1));
% boundary pts
L(b,:) = zeros(4*N,(N+1)^2); 
L(b,b) = eye(4*N);
d2udx2_LUT=Estimate_dx2(0.005*ones(size(xx)),xx,yy,uu_LUT,xx_LUT,yy_LUT);
d2udy2_LUT=Estimate_dy2(0.005*ones(size(yy)),xx,yy,uu_LUT,xx_LUT,yy_LUT);
rhs = Monegros(xx,yy,'d2udx2') +Monegros(xx,yy,'d2udy2')- Monegros(xx,yy,'u').^3 - d2udx2_LUT -d2udy2_LUT + uu_LUT(:).^3;
%rhs(b) = sin(omegax*pi*xx(b) + omegay*pi*yy(b)) + cos(omegapx*pi*xx(b) + omegapy*pi*yy(b));
b_west = find(xx==x_west(1));
rhs(b_west) = interp1(y_west,sol_west, yy(b_west),'spline');
b_east = find(xx==x_east(1));
rhs(b_east) = interp1(y_east,sol_east, yy(b_east),'spline');
b_north = find(yy==y_north(1));
rhs(b_north) = interp1(x_north,sol_north, xx(b_north),'spline');
b_south = find(yy==y_south(1));
rhs(b_south) = interp1(x_south,sol_south, xx(b_south),'spline');
%rhs(b) = Monegros(xx(b),yy(b),'u');
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
u = (vv'+'uu_LUT')(:);
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
%[xxx,yyy] = meshgrid(-1:.04:1,-1:.04:1);
%uuu = interp2(xx,yy,uu,xxx,yyy,'spline');
%clf, subplot(1,3,1),
%mesh(xx,yy,uu), colormap([0 0 0])
%view(-20,45),
%title('Pseudopspectral solution')
%u_a = Monegros(xx,yy,'u');
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

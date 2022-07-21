function u = tau_square(x,y,L)
u = zeros(size(x));
for k = 0:50
    for l = 0:50
        u = u + (-1)^(k+l)*cos((2*k+1)*pi*x/L).*cos((2*k+1)*pi*y/L)/((2*k+1)*(2*l+1)*(((2*k+1)^2)+((2*l+1)^2)));
    end
end
u = u*(L^2)*32/pi^4;
end
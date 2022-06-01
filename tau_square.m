function u = tau_square(x,y,L)
L_half = 0.5*L;
u = zeros(size(x));
for k = 0:50
    for l = 0:50
        u = u + (-1)^(k+l)*cos(0.5*(2*k+1)*pi*x/L_half).*cos(0.5*(2*k+1)*pi*y/L_half)/((2*k+1)*(2*l+1)*(((2*k+1)^2)+((2*l+1)^2)));
    end
end
u = u*(L_half^2)*128/pi^4;
end
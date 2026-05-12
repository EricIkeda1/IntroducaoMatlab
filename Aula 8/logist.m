function g = logist(u)
% Funcao logistica
% g(u) = 1/(1+exp(-beta.u))
beta = 1;
g = 1./(1+exp(-beta*u));
end


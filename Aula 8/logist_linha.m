function g = logist_linha(u)
% Funcao logistica - derivada
% g'(u) = beta*g(u)(1-g(u))
beta = 1;
g = beta*logist(u).*(1-logist(u));
end

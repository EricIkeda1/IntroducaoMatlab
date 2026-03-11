%Funções de Ativação


%%Funções degrau
clear; clc;

u = (-2:0.01:2); %Exemplo saída combinador linear
for i =1:size(u,2)
    if u(1,i)< 0;
       y(1,i)= 0;
    else 
       y(1,i)= 1;
    end
end

y1=hardlim(u); %necessario Toolbox Deep Learning
figure(1)
plot (y,'ko','LineWidth',[3]);
hold on;
plot(y1, 'g*');
grid;
ylim([-0.5 1.5]);
title ('Função de Ativação - Degrau');
% Exemplo laço for
clear; %Limpa workspace 
close all; %fecha janelas
clc; %Limpa command window
%for i=1:10
%    if(i>=3 && i<7)
%    fprintf('Contagem: %d*\n',i)
%    else
%    fprintf('Contagem: %d\n',i)
%    end
%    pause(1);
%end 

i=1;
while (1)                           %i<=10
    if(i>=3 && i<7)
        fprintf('Contagem: %d*\n',i)
    else
        fprintf('Contagem: %d\n',i)
    end
    pause(1);
    i=i+1;
    if(i>10)
        i=1;
        break;
    end
end 
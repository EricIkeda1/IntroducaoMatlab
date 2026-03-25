%Algoritmo para Perceptron Aula 3 - Ex.2
clear; clc; close all;

% -----dados de treinamento --------------

renda=[2800;1300;1400;500;1100;1800;...

    2400;1950;450;2750;850;1300;2100;...

    900;2700;1600];

divida=[550;500;80;200;270;450;650;...

    600;70;730;90;200;750;300;250;500];

d=[1;0;1;0;0;1;1;1;0;1;0;0;1;0;1;0]; % Item <2>

data=[renda,divida,d]; %Item <1>



%------dados de validacao----------------

X=[1900 150 -1;2500 800 -1; 1600 700 -1;...

    2300 500 -1; 2100 250 -1]; % X1, X2 amostra 117-121

D=[1;1;0;1;1]; % saida deseja amostra 117-121


%------Analise Grafica----------------
for i=1:16

    if data(i, 3)==1

        plot(data(i, 1), data(i,2),'r+')
        hold on;
    end

    if data(i, 3)==0

        plot(data(i, 1), data(i,2),'bo')
        hold on;
    end
end

%% Inicialização e ajustes iniciais

data(:,3)=-1; %inserir - 1 na coluna 3 (bias)
norm=max(max(data)); %maior valor da matriz
data=data/norm; %normalização dos dados da matriz
n_ent=2; %Numero de entradas
fprintf("Pesos iniciais: ")
w=rand(1,(n_ent+1)) %Item <3>
n=0.0001; % Taxa de aprendizagem Item <4>
epoca=0; %Item<5>
errof=1000;
u=0;
pause(2);
%% Treinamento de Perceptron
fprintf("Iniciando o Treinamento de Perceptron \n")
pause(2);
[L C]=size(data); %verifica o tamanho da matriz L(Linhas) C(colunas)
tic; %inicializa contador de tempo tic-toc
while(errof>0.01) %Item <6.1>
    eroo=0;
    for i=i:L %Item <6.2>
        u(i)=w*data(i,:)'; %Item <6.2.1>

        %--------funcao de ativacao Item <6.2.2>--------------
        if u(i)>=0
            y(1)=1;
        else
            y(1)=0;
        end
        %--------Fim de ativacao-----------------
    end
end


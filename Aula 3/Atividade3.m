% Algoritmo para Perceptron Aula 3 - Ex.2
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
% ------Análise gráfica--------------------
for i=1:16
    if data(i,3)==1
        plot(data(i,1),data(i,2),'r+')
        hold on;
    end
    if data(i,3)==0
        plot(data(i,1),data(i,2),'bo')
        hold on;
    end
end
%% Inicialização e ajustes iniciais
data(:,3)=-1; %inserir -1 na coluna 3 (bias)
norm=max(max(data)); %maior valor da matriz
data=data/norm; %normalizacao dos dados da matriz
n_ent=2; %Número de entradas
fprintf("Pesos iniciais: ")
w=rand(1,(n_ent+1)) %Item <3>
n=0.0001; % Taxa de aprendizagem Item<4>
epoca=0; %Item<5>
erro=1000;
errof=1000;
u=0;
pause(2);
%% Treinamento da Perceptron
fprintf("Iniciando o Treinamento da Perceptron \n")
pause(2);
[L C ]=size(data); %verifica o tamanho da matriz L(linhas) C(col)
tic; %inicializa contador de tempo tic-toc
while(errof>0.01) %Item <6.1>
    erro=0;
    for i=1:L %Item <6.2>
        u(i)=w*data(i,:)'; %Item<6.2.1>
        %--------------funcao de ativacao-----Item <6.2.2>---
        if u(i)>=0
            y(i)=1;
        else
            y(i)=0;
        end
        %-----------fim funcao de ativacao--------------
        d1(i)=d(i);
        if y(i)~=d1(i) %Item<6.2.3>
            dif=d1(i)-y(i);
            w=w+(n*dif*data(i,:)) %Item<6.2.3.1>
        else
            dif=0;
        end
        erro=erro+abs(dif);
    end
    errof=erro;
    epoca=epoca+1; %incrementa o contador epocas
    %condição de parada
    if epoca>50000
        fprintf("Não converge!\n")
        break
    end
end
t=toc; %armazena tempo de treinamento
fprintf("Finalizando o Treinamento da Perceptron\n")
pause(2);
fprintf("Tempo de treinamento: %0.3f seg\n",t)
pause(2);
fprintf("Número de épocas: %d\n",epoca)
pause(2);
fprintf("Pesos finais:")
w
%% Validação da Perceptron
fprintf("Iniciando a Validação \n")
pause(2);
% Item <1> foi realizado inicio do codigo
wfinal=w; % Item<2>
[L1 C1]=size(X);
for i=1:L1
    U(i)=wfinal*X(i,:)'; %Item<3.1>
end
Y=hardlim(U); %Item<3.2> hardlim -> degrau
fprintf("A saída esperada é:\n")
D=D'
fprintf("A saída da rede é: \n")
Y

% Projeto Pr�tico Rede PMC Classificador
% Supervisionado backpropagation
% Renato Kazuo Miyamoto
%-------------------------------------------------------------------------%
%% In�cio
clear all; clc;
%-------------------------------------------------------------------------%
%% Ajustes iniciais
fprintf('Rede Neural <strong>Arquitetura Perceptron M�ltiplas Camadas</strong>\n');
fprintf('Treinamento <strong>Backpropagation com Momentum</strong>\n');
pause(2);
importfile('dados_projeto.txt') 
%-------------------------------------------------------------------------%
[R C]=size(data);
d = [data(:,5), data(:,6), data(:,7)];%Sa�das desejadas -> coluna 5,6,7
dados_orig=data; % (apenas pq na validacao o arquivo data ser� substituido.)
data=[-1*ones(R,1),data(:,1),data(:,2),data(:,3), data(:,4)]; %bias limiar na coluna 1
n_ent=C-3; %numero de entradas
%-------------------------------------------------------------------------%
%% Inicializa��o dos vetores W
% Duas camadas neurais
%-------------------------------------------------------------------------%
Neuronios_entrada = 15;%20 25
Neuronios_saida = 3;

% MESMOS PESOS DO BP CONVENCIONAL
W1 = load('w1_inicial.txt');
W2 = load('w2_inicial.txt');

fprintf('------------------------------------------------------\n');
pause(2);
%-------------------------------------------------------------------------%
n=0.1; %taxa de aprendizagem
e = 1e-6; % precis�o
Momentum = 0.9; % fator de momentum
%-------------------------------------------------------------------------%
%inicializa��o do contador de epocas e variaveis
Soma_Eqm = 0;
epocas = 0;
Em_at = 1;
Em_ant =0;

VariacaoW1 = zeros(Neuronios_entrada,n_ent+1);
VariacaoW2 = zeros(Neuronios_saida,Neuronios_entrada+1);

%% Treinamento e Valida��o da Rede
fprintf('Iniciando Treinamento da Perceptron M�ltlipas Camadas\n');
pause (2);
tic;

while abs(Em_at-Em_ant)>e
    Em_ant = Em_at;
    Soma_Eqm = 0;

    for i=1:R

    %------Passo foward-------------------------------------------%
     Y_1=W1*data(i,:)'; % saida primeira camada -passo forward
        
     Y1=[-1 (logsig(Y_1))']'; % funcao sigmoide
        
     Y_2= W2*Y1; %saida segunda camada
        
     Y2=logsig(Y_2); % funcao sigmoide

   %------Passo backward-------------------------------------------%        

        for j=1:Neuronios_saida %3
            Delta2(j)=(d(i,j)-Y2(j))*(Y2(j)*(1-Y2(j)));
        end

%-------------Momentum ------------------------        

        W2a = W2;

        for j=1: Neuronios_saida
        for z=1:Neuronios_entrada+1
            W2(j,z)=W2(j,z)+n*Delta2(j)*Y1(z)' + (Momentum*VariacaoW2(j,z));
        end
        end

        for j=1:Neuronios_saida
        for z=1:Neuronios_entrada+1
            VariacaoW2(j,z)=W2(j,z)-W2a(j,z);
        end
        end

%-------------Momentum ------------------------     

    for j=1:Neuronios_entrada
        Delta_aux=0;
        for z=1:Neuronios_saida
            Delta_aux=Delta_aux+Delta2(z)*W2(z,j);
        end
        Delta1(j)=Delta_aux*(Y1(j+1)'*(1-Y1(j+1)));
    end
  
%-------------Momentum ------------------------        

        W1a = W1;

        for j=1:n_ent+1
        for z=1:Neuronios_entrada
            W1(z,j)=W1(z,j)+n*Delta1(z)*data(i,j) + (Momentum*VariacaoW1(z,j));
        end
        end

        for j=1:n_ent+1
        for z=1:Neuronios_entrada
            VariacaoW1(z,j)=W1(z,j)-W1a(z,j);
        end
        end

%-------------Momentum ------------------------     

        erro = d(i,:)' - Y2;
        Soma_Eqm = Soma_Eqm + sum(erro.^2);

    end

    Em_at = Soma_Eqm/R;

    epocas = epocas+1;
    Epoca_plot(epocas)= epocas;
    Erro = abs(Em_at-Em_ant);
    erro_plot(epocas)=Erro;
    
    if epocas > 2500
        break
    end                    
end

fprintf('------------------------------------------------------\n');
fprintf('Finalizando Treinamento da Perceptron M�ltiplas Camadas \n');
t=toc;
fprintf('Tempo de treinamento: %0.3f secs\n',t);

figure(1)
subplot(2,1,1)
plot (Epoca_plot, erro_plot,'b');
xlabel('�pocas');
ylabel('Erro');
title ('Comportamento do erro por �pocas');
grid on;

fprintf ('N�meros de �pocas:%d\n',epocas);

%-------------------------------------------------------------------------%
%% Valida��o da PMC
fprintf('Iniciando Valida��o Perceptron M�ltiplas Camadas \n');
pause (2);  

importfile('dados_validacao.txt');
[Li Co ]=size(data);
D=[data(:,5), data(:,6), data(:,7)];
data=[-1*ones(Li,1),data(:,1),data(:,2),data(:,3),data(:,4)];

d1=0; d2=0; d3=0;
 
for i=1:Li    
   Y_1=W1*data(i,:)';
   Y1=[-1 (logsig(Y_1))']';
   Y_2=W2*Y1;
   Y2=logsig(Y_2);
    
   d1(i)=Y2(1)>=0.5;
   d2(i)=Y2(2)>=0.5;
   d3(i)=Y2(3)>=0.5;
end

Saida_rede_validada(:,1) = d1';
Saida_rede_validada(:,2) = d2';
Saida_rede_validada(:,3) = d3';

acerto=0; erro=0;

for i=1:Li 
    if isequal(Saida_rede_validada(i,:),D(i,:))
        acerto=acerto+1;
    else
        erro=erro+1;
    end
end  

fprintf('Porcentagem de acerto %.2f %%\n',(acerto/Li)*100);
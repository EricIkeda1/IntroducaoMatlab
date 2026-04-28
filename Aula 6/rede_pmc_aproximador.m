% Projeto Pr�tico Rede PMC Aproximador de Fun��es
% Prof. Renato Kazuo Miyamoto
%-------------------------------------------------------------------------%
%% In�cio 
clear all; close all; clc;
%% Ajustes iniciais
fprintf('Rede Neural <strong>Arquitetura Perceptron M�ltiplas Camadas</strong>\n');
pause(2);
importfile('dados_treinamento.txt') %3 entradas e uma sa�da
%-------------------------------------------------------------------------%
[L C]=size(data);
d = data(:,C);%Sa�das desejadas -> coluna 4
dados_orig=data; % (apenas pq na validacao o arquivo data ser� substituido.)
data1=[-1*ones(L,1),data(:,1),data(:,2),data(:,3)]; %bias limiar na coluna 1
n_ent=C-1; %numero de entradas
%-------------------------------------------------------------------------%
%% Inicializa��o dos vetores W - gerando matriz aleat�ria valores 0 e 1
% Duas camadas neurais
Neuronios_entrada = 10;
Neuronios_saida = 1;
W1 = rand(Neuronios_entrada,n_ent+1); % --> 10 neuronios (10,4) rand(1,(n_ent+1));
fprintf('Pesos iniciais primeira camada neural: [');
fprintf('%g, ', W1(1:end-1));
fprintf('%g]\n', W1(end));

W2=rand(Neuronios_saida,Neuronios_entrada+1);
% um neuronio camada dois recebe sinal de 10 neuronios cam. 1 mais o limiar (1,11)
fprintf('Pesos iniciais segunda camada neural: [');
fprintf('%g, ', W2(1:end-1));
fprintf('%g]\n', W2(end));
fprintf('------------------------------------------------------\n');
pause(2);
%-------------------------------------------------------------------------%
n=0.1; %taxa de aprendizagem
e = 1e-6; % precis�o
%-------------------------------------------------------------------------%
%inicializa��o do contador de epocas e vari�veis
epoca = 0; 
Em_at = 1; % Erro Quadratico M�dio Atual
Em_ant =0; % Erro Quadratico M�dio Anterior
Soma_Eqm = 0;
beta = 1;
%erro_plot=0;
%-------------------------------------------------------------------------%
%% Treinamento
fprintf('Iniciando Treinamento da Perceptron M�ltlipas Camadas\n');
pause (2);
tic;
while abs(Em_at-Em_ant)>e
    Em_ant = Em_at;
  for i=1:L %200
  %------Passo foward-------------------------------------------%
   I_1=W1*data1(i,:)'; %saida primeira camada -passo forward 5.1
    
   g=(1./(1+(exp(-beta*I_1'))))';
   Y1=[-1; g];
   % Y1=[-1 (logsig(Y_1))']'; %funcao sigmoide e armazenarmento em vetor transposto 5.4 
   %Y1=[-1 (tanh(Y_1))']';
     

   I_2= W2*Y1; %saida segunda camada - passo forward
   Y2=(1./(1+(exp(-beta*I_2'))))';
   %Y2=logsig(Y_2); % funcao sigmoide
   %Y2=tanh(Y_2);
  %------Passo backward-------------------------------------------%  
    Delta_2=(d(i)-Y2)*(Y2*(1-Y2));
    %Delta_2=(d(i)-Y2)*(1-(Y2'*Y2));
    % calculo gradiente delta conforme eq (5.15)--> delta=(d-yj).g'(i)- passo
    % g'(i)=derivada funcao sigmoide/logistica --> g(y)=y(1-y)
    % backward da ultima camada voltando a primeira
        
    W2=W2+n*Delta_2*Y1' 
    % calculo peso w2 conforme eq(5.17) wji=wji+n*delta*Y
        
    Delta_1=-Delta_2*W2*(Y1'*(1-Y1));
    %Delta_1=-Delta_2*W2*(1-(Y1'*Y1));
    %calculo delta 1 5.26
        
    for j=1:Neuronios_entrada % 1: Neuronios_entrada
    W1(j,:)=W1(j,:)+n*Delta_1(j)*data(i,:);
      %calculo peso 1
    end
        
    Soma_Eqm = Soma_Eqm + 0.5*((d(i) - Y2)^2);% Eqm conforme eq(5.7) 
        
  end
 %---------------------------------------------------------------------%
 % atualiza Erro quadratico atual    
 Em_at = (Soma_Eqm/L);
 Soma_Eqm = 0;
 %---------------------------------------------------------------------%
 % Contador de epocas e erro 
 epoca = epoca+1;
 Epoca_plot(epoca)= epoca;
 Erro = abs(Em_at-Em_ant); %E
 erro_plot(epoca)=Erro;
        
% condicao de parada    
 if epoca > 10000
    break
 end
        
end
fprintf('------------------------------------------------------\n');
fprintf('Finalizando Treinamento da Perceptron M�ltiplas Camadas \n');
pause (2);
t=toc;
fprintf('Tempo de treinamento: %0.3f secs\n',t);
pause (2);

figure(1)
plot (Epoca_plot, erro_plot,'b');
norm=size(Epoca_plot);
norm=norm(:,2);
ylim([-0.05 1]);
xlim([-1 norm]);
xlabel('�pocas');
ylabel('Erro');
title 'Comportamento do erro por �pocas de Treinamento'
grid on;

fprintf ('N�meros de �pocas:%d\n',epoca);
fprintf('Pesos Finais W1: [');
fprintf('%g, ', W1(1:end-1));
fprintf('%g]\n', W1(end));
fprintf('Pesos Finais W2: [');
fprintf('%g, ', W2(1:end-1));
fprintf('%g]\n', W2(end));
pause(4);
fprintf('------------------------------------------------------\n');

%% Valida��o da PMC
fprintf('Iniciando Valida��o Perceptron M�ltiplas Camadas \n');
pause (4);  
%----------------------------------------------------------------------%
importfile('dados_validacao.txt');
[Li Co ]=size(data);
erm=0;
variancia=0;
%saida=0;
y2=0;
D = data(:,4); % saida desejada
data2=[-1*ones(Li,1),data(:,1),data(:,2),data(:,3)]; %matriz de bias
%----------------------------------------------------------------------%
%calculo do erm
for i=1:Li   
    u1=W1*data2(i,:)';
    g1=(1./(1+(exp(-beta*u1'))))';
    y1=[-1; g1];
    %y1=[-1 (logsig(u1))']';
    %y1=[-1 (tanh(u1))']';
    u2=W2*y1;
    y2(i)=(1./(1+(exp(-beta*u2'))))';
    %y2(i)=logsig(u2);
    %y2(i)=tanh(u2);
    er(i)=(D(i)-y2(i))/D(i); %calculo do erro relativo
    
    erm=erm+er(i);%armazena erro relativo medio
end
erm=erm/Li; %calculo erro rel medio

%----------------------------------------------------------------------%
%calculo variancia
for i=1:Li
    var_(i)=(erm-y2(i))^2; %armazena variancia
    variancia=variancia+var_(i);
end
variancia=variancia/Li; %calculo variancia

fprintf('A sa�da esperada �:  [');
fprintf('%g, ', D(1:end-1));
fprintf('%g]\n',D(end));
pause(2);
fprintf('A sa�da da Rede � :  [');
fprintf('%g, ', y2(1:end-1));
fprintf('%g]\n',y2(end));
pause(1);
fprintf('Erro Relativo M�dio = %d\n',erm);
fprintf ('Variancia = %d\n',variancia);

z=1:Li
figure(2)
plot(z,y2','-*',z,D,'-x')
xlabel('Posi��o');
ylabel('Amostras');
title ('Valida��o Perceptron M�ltiplas Camadas')
legend('Sa�da Rede','Sa�da Esperada')
grid on;
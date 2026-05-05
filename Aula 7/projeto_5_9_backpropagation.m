% Projeto Pr�tico Rede PMC Classificador
% Supervisionado backpropagation
% Renato Kazuo Miyamoto
%-------------------------------------------------------------------------%
%% In�cio
clear all; close all; clc;
%-------------------------------------------------------------------------%
%% Ajustes iniciais
fprintf('Rede Neural <strong>Arquitetura Perceptron M�ltiplas Camadas</strong>\n');
fprintf('Treinamento <strong>Backpropagation convencional</strong>\n');
pause(2);
importfile('dados_projeto.txt') 
%-------------------------------------------------------------------------%
[R C]=size(data);
d = [data(:,5), data(:,6), data(:,7)];%Sa�das desejadas -> coluna 5,6,7
dados_orig=data; % (apenas pq na validacao o arquivo data ser� substituido.)
data=[-1*ones(R,1),data(:,1),data(:,2),data(:,3), data(:,4)]; %bias limiar na coluna 1
n_ent=C-3; %numero de entradas
%-------------------------------------------------------------------------%
%% Inicializa��o dos vetores W - gerando matriz aleat�ria valores 0 e 1
% Duas camadas neurais
%-------------------------------------------------------------------------%
 %inicializa��o dos vetores radomicos de pesos W1 e W2
Neuronios_entrada = 15;%20 25
Neuronios_saida = 3;
W1 = rand(Neuronios_entrada,n_ent+1);
fprintf('Pesos iniciais primeira camada neural: [');
fprintf('%g, ', W1(1:end-1));
fprintf('%g]\n', W1(end));
% %salvar pesos para utilizar na rede momentum
save('w1_inicial.txt','W1','-ascii');

W2 = rand(Neuronios_saida,Neuronios_entrada+1);
fprintf('Pesos iniciais segunda camada neural: [');
fprintf('%g, ', W2(1:end-1));
fprintf('%g]\n', W2(end));
% %salvar pesos para utilizar na rede momentum
save('w2_inicial.txt','W2','-ascii');

% Descomentar quando for implementar BP com momentum
% MESMA MATRIZ DE PESOS UTILIZADOS NO TREINAMENTO BP CONVENCIONAL 
%  load ('w1_1.txt');
%  W1=w1_1;
%  
%  load ('w2_1.txt');
%  W2=w2_1;
 
fprintf('------------------------------------------------------\n');
pause(2);
%-------------------------------------------------------------------------%
n=0.1; %taxa de aprendizagem
e = 1e-6; % precis�o
%-------------------------------------------------------------------------%
%inicializa��o do contador de epocas e variaveis
Soma_Eqm = 0;
epocas = 0;
Em_at = 1;
Em_ant =0;
VariacaoW1 = 0;
% Momentum = 0;
VariacaoW1 = zeros(Neuronios_entrada,n_ent+1);
VariacaoW2 = zeros(Neuronios_saida,Neuronios_entrada+1);

%% Treinamento e Valida��o da Rede
fprintf('Iniciando Treinamento da Perceptron M�ltlipas Camadas\n');
pause (2);
tic;
while abs(Em_at-Em_ant)>e
    Em_ant = Em_at;
    for i=1:R
    %------Passo foward-------------------------------------------%
     Y_1=W1*data(i,:)'; % saida primeira camada -passo forward
        
     Y1=[-1 (logsig(Y_1))']'; % funcao sigmoide e armazenarmento em vetor transposto 
        
     Y_2= W2*Y1; %saida segunda camada - passo forward
        
     Y2=logsig(Y_2); % funcao sigmoide
   %------Passo backward-------------------------------------------%        
        for j=1:Neuronios_saida %3
            Delta2(j)=(d(i,j)-Y2(j))*(Y2(j)*(1-Y2(j)));
        end
       %Delta2=(d(i)-Y2)*(Y2'*(1-Y2));

        for j=1: Neuronios_saida
        for z=1:Neuronios_entrada+1
           %W2a(j,z)= W2(j,z);
           W2(j,z)=W2(j,z)+n*Delta2(j)*Y1(z)'; %+(Momentum*VariacaoW2(j,z))
        end
        end
%-------------Momentum ------------------------        




%-------------Momentum ------------------------     

    for j=1:Neuronios_entrada
        Delta_aux=0;
    for z=1:Neuronios_saida
        Delta_aux=Delta_aux+Delta2(z)*W2(z,j);%unico valor
    end
        Delta1(j)=Delta_aux*(Y1(j)'*(1-Y1(j)));
    end
    %Delta1=Delta2*W2*(Y1'*(1-Y1)); %1x16 preciso 1x15
  
    for j=1:n_ent+1
    for z=1:Neuronios_entrada
        %W1a(z,j)= W1(z,j);
        W1(z,j)=W1(z,j)+n*Delta1(z)*data(i,j); %+(Momentum*VariacaoW1(z,j))
    end
    end
%-------------Momentum ------------------------        




%-------------Momentum ------------------------   

    for j=1:Neuronios_saida
        Soma_Eqm = Soma_Eqm + 0.5*((d(i,j) - Y2(j))^2);
    end
    end
                    
  %---------------------------------------------------------------------%
  %Atualiza��o de Eqm atual    
  Em_at = (Soma_Eqm/130);
  Soma_Eqm = 0;
  %---------------------------------------------------------------------%
  % Contador de epocas e erro 
  epocas = epocas+1;
  Epoca_plot(epocas)= epocas;
  Erro = abs(Em_at-Em_ant); %E
  erro_plot(epocas)=Erro;
    
 % condicao de parada    
   if epocas > 2500
   break
   end                    
end
fprintf('------------------------------------------------------\n');
fprintf('Finalizando Treinamento da Perceptron M�ltiplas Camadas \n');
pause (1);
t=toc;
fprintf('Tempo de treinamento: %0.3f secs\n',t);
pause (1);

figure(1)
subplot(2,1,1)
plot (Epoca_plot, erro_plot,'b');
norm=size(Epoca_plot);
norm=norm(:,2);
ylim([-0.05 1]);
xlim([-1 norm]);
xlabel('�pocas');
ylabel('Erro');
legend('BP convencional');
title ('Comportamento do erro por �pocas de Treinamento');
grid on;

fprintf ('N�meros de �pocas:%d\n',epocas);
fprintf('Pesos Finais W1: [');
fprintf('%g, ', W1(1:end-1));
fprintf('%g]\n', W1(end));
fprintf('Pesos Finais W2: [');
fprintf('%g, ', W2(1:end-1));
fprintf('%g]\n', W2(end));
pause(3);
fprintf('------------------------------------------------------\n');

%% Valida��o da PMC
fprintf('Iniciando Valida��o Perceptron M�ltiplas Camadas \n');
pause (4);  
%----------------------------------------------------------------------%
importfile('dados_validacao.txt');
[Li Co ]=size(data);
D=[data(:,5), data(:,6), data(:,7)];
data=[-1*ones(Li,1),data(:,1),data(:,2),data(:,3),data(:,4)];
% inicializacao saidas
d1=0;
d2=0;
d3=0;
 
  for i=1:Li    
   Y_1=W1*data(i,:)';
   Y1=[-1 (logsig(Y_1))']';
   Y_2=W2*Y1;
   Y2=logsig(Y_2);
    
   d1(i)=Y2(1);
   if d1(i)>=0.5
      d1(i)=1;
   else
      d1(i)=0;
   end
   %if Neuronios_saida>1
   d2(i)=Y2(2);
   if d2(i)>=0.5
      d2(i)=1;
   else
      d2(i)=0;
   end
   %end
   %if Neuronios_saida>2
   d3(i)=Y2(3);
   if d3(i)>=0.5
      d3(i)=1;
    else
      d3(i)=0;
    end
    %end
end
Saida_rede_validada(:,1) = d1';
Saida_rede_validada(:,2) = d2';
Saida_rede_validada(:,3) = d3';
%%
% c�lculo taxa de acertos
acerto=0;erro=0;
 for i=1:Li 
     z=isequal (Saida_rede_validada(i,:),D(i,:));
 if z==1
     acerto=acerto+1;
     fprintf('Amostra %d: Y2[',i);
     fprintf('%g,', Saida_rede_validada(i,:));
     fprintf(']');
     fprintf('D[');
     fprintf('%g,', D(i,:));
     fprintf(']');
     fprintf(' --> Acertou\n');
 end
 if z==0
     erro=erro+1;
     fprintf('Amostra %d: Y2[',i);
     fprintf('%g,', Saida_rede_validada(i,:));
     fprintf(']');
     fprintf('D[');
     fprintf('%g,', D(i,:));
     fprintf(']');
     fprintf('--> Errou\n');
 end
 end  
fprintf('Porcentagem de acerto %.2f %%\n',(acerto/Li)*100);

%%
if(((acerto/Li)*100)==100)
figure(1)
subplot(2,1,1)
hold on;
dimen = [.45 .3 .5 .5];
text = 'Acerto de 100%';
a = annotation('textbox',dimen,'String',text,'FitBoxToText','on');
end
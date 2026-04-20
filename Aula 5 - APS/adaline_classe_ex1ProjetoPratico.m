% Implementação Adaline Ex1 Aula 5
%%-------------------------------------------------------------------------%
%% Início
clear all; close all; clc;

%% Manipulacao de dados iniciais
fprintf('Rede Neural <strong>Arquitetura Adaline</strong>\n');
importfile('C:\Users\Eric Yuji Ikeda\Documents\VScode\IntroducaoMatlab\Aula 4 - APS\Tabela_dados.txt') %Item <1>
d= data (:,4); %Saídas desejadas %Item <2>  % Corrigidos aqui
%%
%Ajustes iniciais
data(:,5)= -1; %bias
%data=data/max(max(data)); %Ja estão normalizados
n_ent=4; %numero de entradas
[Linhas Colunas]=size(data); %L numero de linhas e C numero colunas

%% Inicializacao 
%inicialização do vetor W - gerando matriz aleatória valores 0 e 1
W = rand(1,(n_ent+1)); %Item <3>

fprintf('Pesos iniciais: [');
fprintf('%g, ', W(1:end-1));
fprintf('%g]\n', W(end));

n=0.01; %taxa de aprendizagem %Item <4>

%errof =1000;

%--------novo da Adaline-------------------
Eqm_at = 1; % Erro Quadratico Médio Atual
Eqm_ant =0; % Erro Quadratico Médio Anterior
Soma_Eqm = 0;
erro_plot=0;
e = 1*10^-6; % precisão %Item <4>
%------------------------------------------

epoca = 0; %Item <5>
erro=0;
u=0;

pause(2);

%% Treinamento da Adaline
fprintf('Iniciando Treinamento da Adaline\n');
pause (2);
tic;

while abs(Eqm_at - Eqm_ant)>=e  %Item <6>
     
     Eqm_ant = Eqm_at;   %Item <6.1>

%---------Item <6.2>---------------------
for i=1:Linhas
    
    u = 0;

    for j=1:(n_ent+1) % Corrigidos aqui também
       u = u + (data(i,j)* W(1,j));  %Item <6.2.1>
    end
    
    A(i,1)=u;
    
    for j=1:(n_ent+1)
     % w=w+n(d-u).x treinamento Delta;
     W(1,j) = (W(1,j) + ((n*(d(i) - A(i,1)))*data(i,j))) %Item <6.2.2>
    end

    %soma Eqm(W) Eqm=Eqm+(d-u)^2;
    Soma_Eqm = Soma_Eqm +((d(i) - A(i,1))^2); 
end

%condicao pra while;
Eqm_at = Soma_Eqm/Linhas;
Erro = abs(Eqm_at - Eqm_ant);

u=0;  

epoca = (epoca+1); %Item <6.3>

if epoca > 5000  %condicao de parada de treinamento
    disp('Não converge!')
   break
end

Eqm_plot(epoca) = Eqm_at;
Epoca_plot(epoca)= epoca;
erro_plot(epoca)=Erro;

Soma_Eqm = 0;

end

t=toc;

fprintf('Finalizando Treinamento da Adaline \n');
pause (2);
fprintf('Tempo de treinamento: %0.3f secs\n',t);
pause (2);
fprintf('A Adaline convergiu em %d épocas\n',epoca);
pause (2);

fprintf('Os Pesos Finais: [');
fprintf('%g, ', W(1:end-1));
fprintf('%g]\n', W(end));

pause(4);

fprintf('--------------------------------------------------\n');

figure()
plot (Epoca_plot, Eqm_plot,'b');
ylim([-1 20]);
xlim([-50 1200]);
xlabel('Épocas');
ylabel('Eqm (w)');
title ('Comportamento do Eqm por Épocas de Treinamento');
grid on;

figure()
plot (Epoca_plot, erro_plot,'b');
ylim([-1 20]);
xlim([-50 1200]);
xlabel('Épocas');
ylabel('Erro');
title ('Comportamento do erro por Épocas de Treinamento');
grid on;

%% Validação da Adaline
fprintf('Iniciando Validação da Adaline \n');

pause (4);

Wfinal=W;

dados_val = data;
D= d;

[Li Co ]=size(dados_val);

X= dados_val;

U=0;f=1;g=1;U=0;

for g = 1:Li
    
    U = 0;
    
    for j = 1:(n_ent+1)
        U = U + (X(g,j)* Wfinal(1,j));
    end
    
   U = hardlims(U);
   Y(g,f) = U;
   f = f+1;

   U=0;
   f=1;
end

Y=Y(:,1)';

fprintf('A saída Esperada é:\n '); 
D=D'

fprintf('A saída da Rede é:\n '); 
Y

%% cálculo taxa de acertos
acerto=0;erro=0;

for i=1:Li 
    
     z=isequal (Y(:,i),D(:,i));
     
     if z==1
         acerto=acerto+1;
         fprintf('Amostra %d: Y[',i);
         fprintf('%g,', Y(:,i));
         fprintf(']');
         fprintf('D[');
         fprintf('%g,', D(:,i));
         fprintf(']');
         fprintf(' --> Acertou\n');
     end
     
     if z==0
         erro=erro+1;
         fprintf('Amostra %d: Y[',i);
         fprintf('%g,', Y(:,i));
         fprintf(']');
         fprintf('D[');
         fprintf('%g,', D(:,i));
         fprintf(']');
         fprintf('--> Errou\n');
     end
end  

fprintf('Porcentagem de acerto %.2f %%\n',(acerto/Li)*100);

figure()
div=zeros(1,Li);
plot(D,'*r')
hold on
plot(Y,'bo')
hold on
plot(div,'black')
grid on
title('Amostras de validação')
axis([1 30 -1.5 1.5])
text(15,0.5,'CLASSE B')
text(15,-0.5,'CLASSE A')
legend('D - Saída Desejada','Y - Saída Rede')
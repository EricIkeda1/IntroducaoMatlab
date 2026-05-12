%% Projeto Prático Rede PMC Sistemas variantes no tempo TDNN Vs2 2024
% Supervisionado backpropagation MOMENTUM
% Renato Kazuo Miyamoto 
% TDNN1 -> 5 entradas (np=5) e 10 neuronios int. (n=10)
% TDNN2 -> 10 entradas (np=10) e 15 neuronios int. (n=15)
% TDNN3 -> 15 entradas (np=15) e 25 neuronios int. (n=25)
%% Início
clear all; clc; close all;
%-------------------------------------------------------------------------%
%% Ajustes iniciais
fprintf('<strong>PMC Sistemas variantes no tempo - TDNN </strong>\n');
fprintf('Treinamento Backpropagation Momentum\n');

pause(1);

% Dados de entrada treinamento
dados = xlsread('Tabela_treinamento.xls');
% Dados de entrada validacao
dados_v = xlsread('Tabela_validacao.xls');
[R C]=size(dados);

% Parametros do treinamento
n = 0.1;                        % taxa de aprendizado
e = 0.5e-6;                     % precisao
Momentum = 0.8;                 % fator de momentum
tr = 3;                         % treinamentos

% Inicializacao de variaveis das topologias
epoca_p5 = zeros(tr,1);
Eqm_p5 = zeros(20000,tr);
epoca_p10 = zeros(tr,1);
Eqm_p10 = zeros(20000,tr);
epoca_p15 = zeros(tr,1);
Eqm_p15 = zeros(20000,tr);

%% Topologia TDNN1 - inicializacao
fprintf('TOPOLOGIA TDNN 1 - p=5, n1=10\n');
fprintf('Configuração TDNN 1\n');
p=5;   %entradas
n1=10; % neuronios camada escondida
n2=1; % neuronios saida
%X = zeros(size(dados,1),p+1);
%d = zeros(size(dados,1),1);
for t = (p+1):size(dados,1)
    X(t,1) = -1;
    for col = 2:(p+1)
        X(t,col) = dados(t-(col-1));
    end
    d(t) = dados(t);
end
X1=X; % apenas para análise do for
X = X(p+1:end,:)';
d = d(p+1:end);
[M,k] = size(X);                    % atributos M e amostras k

rng('shuffle');
wi1 = rand(n1,M,tr);                % peso inicial camada 1
wi2 = rand(n2,n1+1,tr);             % peso inicial camada 2
wf1 = zeros(n1,M,tr);               % peso final camada 1
wf2 = zeros(n2,n1+1,tr);            % peso final camada 2

%% Topologia 1 - Treinamento
fprintf('Treinamento TDNN 1\n');

for i = 1:tr

    w1 = zeros(n1,M,k+1);
    w1(:,:,1) = wi1(:,:,i);
    w2 = zeros(n2,n1+1,k+1);
    w2(:,:,1) = wi2(:,:,i);

    Eqm_atual = 0;
    Eqm_anterior = 1;
    ek = zeros(n2,k);
    aux = 0;

    while abs(Eqm_atual - Eqm_anterior) > e
        Eqm_anterior = Eqm_atual;
        for j = 1:k
            % passo forward
            I1 = w1(:,:,j)*X(:,j);
            Y1 = logist(I1);
            Y1b = [-1; Y1];
            I2 = w2(:,:,j)*Y1b;
            Y2 = logist(I2);
            % passo backward
            ek(:,j) =d(:,j)'-Y2; %d(j,:)'-Y2;
            delta2 = ek(:,j).*logist_linha(I2);
            w2(:,:,j+1) = w2(:,:,j) + n*delta2*Y1b';
            % momentum
            if j > 1
                w2(:,:,j+1) = w2(:,:,j+1) + Momentum*(w2(:,:,j)-w2(:,:,j-1));
            end
            delta1 = (delta2'*w2(:,2:end,j+1))'.*logist_linha(I1);
            w1(:,:,j+1) = w1(:,:,j) + n*delta1*X(:,j)';
            % momentum
            if j > 1
                w1(:,:,j+1) = w1(:,:,j+1) + Momentum*(w1(:,:,j)-w1(:,:,j-1));
            end
        end
        w1(:,:,1) = w1(:,:,j+1);
        w2(:,:,1) = w2(:,:,j+1);
        Eqm_atual = mean(sum(ek.^2))/2;
        aux = aux + 1;
        Eqm_p5(aux,i) = Eqm_atual;
        epoca_p5(i) = epoca_p5(i) + 1;
        %fprintf('\nEpoca %d %f MSE \n\n',aux,Eqm_atual)
    end
    wf1(:,:,i) = w1(:,:,j+1);
    wf2(:,:,i) = w2(:,:,j+1);
    fprintf('\nTreinamento %d encerrado \n',i)
end


%% Topologia 1 - Validacao
fprintf('Validação TDNN 1\n');

Xv = zeros(size(dados_v,1),p+1);
dados_temp = [dados(size(dados,1)-(p-1):end);dados_v];

for t = (p+1):size(dados_temp,1)
    Xv(t,1) = -1;
    for col = 2:(p+1)
        Xv(t,col) = dados_temp(t-(col-1));
    end
end

Xv = Xv(p+1:end,:)';
[~,kv] = size(Xv);

yv_p5 = zeros(kv,n2,tr);
ERM_p5 = zeros(tr,1);
VARI_p5 = zeros(tr,1);

for i = 1:tr

    w1 =  wf1(:,:,i);
    w2 =  wf2(:,:,i);
    Erro_rel = zeros(n2,kv);

    for j = 1:kv
        I1 = w1*Xv(:,j);
        Y1 = logist(I1);
        Y1b = [-1; Y1];
        I2 = w2*Y1b;
        Y2 = logist(I2);
        Erro_rel(j) = abs(Y2 - dados_v(j))/abs(dados_v(j));
        yv_p5(j,:,i) = Y2';
    end
    ERM_p5(i) = mean(Erro_rel)*100;
    VARI_p5(i) = var(yv_p5(:,:,i));
end

[~,I_p5] = min(ERM_p5);


%% Topologia 2 - inicializacao

%% Topologia 2 - Treinamento

%% Topologia 2 - Validacao

%% Topologia 3 - inicializacao

%% Topologia 3 - Treinamento

%% Topologia 3 - Validacao

%% Resultados

disp('TOPOLOGIA 1 - Resultados')

disp('Epocas dos 3 treinamentos - p=5')
disp('         E1          E2          E3')
disp(epoca_p5')
disp('Erro quadratico medio dos 3 treinamentos - p=5')
disp('     Eqm1      Eqm2      Eqm3')
disp([Eqm_p5(epoca_p5(1),1) Eqm_p5(epoca_p5(2),2) Eqm_p5(epoca_p5(3),3)])


disp('Saidas de validacao e desejado - p=5')
disp('      dv     yv1      yv2      yv3')
disp([dados_v yv_p5(:,:,1) yv_p5(:,:,2) yv_p5(:,:,3)])

disp('Erro relativo medio - p=5')
disp('     ERM1      ERM2      ERM3')
disp(ERM_p5')

disp('Variancia - p=5')
disp('     VAR1      VAR2      VAR3')
disp((VARI_p5*100)')


% disp('TOPOLOGIA 2 - Resultados')

%disp('TOPOLOGIA 3 - Resultados')



figure
plot(Eqm_p5(1:epoca_p5(I_p5),I_p5),'b')
hold on
%plot(Eqm_p10(1:epoca_p10(I_p10),I_p10),'r')
%hold on
% plot(Eqm_p15(1:epoca_p15(I_p15),I_p15),'g')
xlim([0 10000])
ylim([0 1])
title('Erro de treinamento Eqm')
xlabel('Epoca')
ylabel('Erro Eqm')
legend('Topologia 1','Topologia 2','Topologia 3')


figure
plot(dados_v,'k--','LineWidth',2.5)
hold on
plot(yv_p5(:,:,I_p5),'-r')
%hold on
%plot(yv_p10(:,:,I_p10),'-g')
% hold on
% plot(yv_p15(:,:,I_p15),'-b')
title('Comparação entre saída e desejado')
xlabel('Amostras')
ylabel('Valores')
legend('Desejado','Saída top. 1','Saída top. 2','Saída top. 3')
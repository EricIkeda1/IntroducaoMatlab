%% Projeto Prático Rede PMC Sistemas variantes no tempo TDNN Vs2 2024
% Supervisionado backpropagation MOMENTUM
% Eric Yuji Ikeda
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
n = 0.1;
e = 0.5e-6;
Momentum = 0.8;
tr = 3;

% Inicializacao de variaveis das topologias
epoca_p5 = zeros(tr,1);
Eqm_p5 = zeros(20000,tr);

epoca_p10 = zeros(tr,1);
Eqm_p10 = zeros(20000,tr);

epoca_p15 = zeros(tr,1);
Eqm_p15 = zeros(20000,tr);

%% Topologia TDNN1 - inicializacao %comentar após o
fprintf('TOPOLOGIA TDNN 1 - p=5, n1=10\n');
fprintf('Configuração TDNN 1\n');

p=5;
n1=10;
n2=1;

clear X d

for t = (p+1):size(dados,1)

    X(t,1) = -1;

    for col = 2:(p+1)
        X(t,col) = dados(t-(col-1));
    end

    d(t) = dados(t);

end

X = X(p+1:end,:)';
d = d(p+1:end);

[M,k] = size(X);

rng('shuffle');

wi1 = rand(n1,M,tr);
wi2 = rand(n2,n1+1,tr);

wf1 = zeros(n1,M,tr);
wf2 = zeros(n2,n1+1,tr);

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

            I1 = w1(:,:,j)*X(:,j);
            Y1 = logist(I1);

            Y1b = [-1;Y1];

            I2 = w2(:,:,j)*Y1b;
            Y2 = logist(I2);

            ek(:,j) = d(:,j)' - Y2;

            delta2 = ek(:,j).*logist_linha(I2);

            w2(:,:,j+1) = w2(:,:,j) + n*delta2*Y1b';

            if j > 1
                w2(:,:,j+1) = w2(:,:,j+1) + Momentum*(w2(:,:,j)-w2(:,:,j-1));
            end

            delta1 = (delta2'*w2(:,2:end,j+1))'.*logist_linha(I1);

            w1(:,:,j+1) = w1(:,:,j) + n*delta1*X(:,j)';

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

    w1 = wf1(:,:,i);
    w2 = wf2(:,:,i);

    Erro_rel = zeros(n2,kv);

    for j = 1:kv

        I1 = w1*Xv(:,j);
        Y1 = logist(I1);

        Y1b = [-1;Y1];

        I2 = w2*Y1b;
        Y2 = logist(I2);

        Erro_rel(j) = abs(Y2 - dados_v(j))/abs(dados_v(j));

        yv_p5(j,:,i) = Y2';

    end

    ERM_p5(i) = mean(Erro_rel)*100;
    VARI_p5(i) = var(yv_p5(:,:,i));

end

[~,I_p5] = min(ERM_p5);

%% Topologia TDNN2 - inicializacao
fprintf('TOPOLOGIA TDNN 2 - p=10, n1=15\n');
fprintf('Configuração TDNN 2\n');

p=10;
n1=15;
n2=1;

clear X d

for t = (p+1):size(dados,1)

    X(t,1) = -1;

    for col = 2:(p+1)
        X(t,col) = dados(t-(col-1));
    end

    d(t) = dados(t);

end

X = X(p+1:end,:)';
d = d(p+1:end);

[M,k] = size(X);

rng('shuffle');

wi1 = rand(n1,M,tr);
wi2 = rand(n2,n1+1,tr);

wf1_p10 = zeros(n1,M,tr);
wf2_p10 = zeros(n2,n1+1,tr);

%% Topologia 2 - Treinamento
fprintf('Treinamento TDNN 2\n');

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

            I1 = w1(:,:,j)*X(:,j);
            Y1 = logist(I1);

            Y1b = [-1;Y1];

            I2 = w2(:,:,j)*Y1b;
            Y2 = logist(I2);

            ek(:,j) = d(:,j)' - Y2;

            delta2 = ek(:,j).*logist_linha(I2);

            w2(:,:,j+1) = w2(:,:,j) + n*delta2*Y1b';

            if j > 1
                w2(:,:,j+1) = w2(:,:,j+1) + Momentum*(w2(:,:,j)-w2(:,:,j-1));
            end

            delta1 = (delta2'*w2(:,2:end,j+1))'.*logist_linha(I1);

            w1(:,:,j+1) = w1(:,:,j) + n*delta1*X(:,j)';

            if j > 1
                w1(:,:,j+1) = w1(:,:,j+1) + Momentum*(w1(:,:,j)-w1(:,:,j-1));
            end

        end

        w1(:,:,1) = w1(:,:,j+1);
        w2(:,:,1) = w2(:,:,j+1);

        Eqm_atual = mean(sum(ek.^2))/2;

        aux = aux + 1;

        Eqm_p10(aux,i) = Eqm_atual;

        epoca_p10(i) = epoca_p10(i) + 1;

    end

    wf1_p10(:,:,i) = w1(:,:,j+1);
    wf2_p10(:,:,i) = w2(:,:,j+1);

    fprintf('\nTreinamento %d encerrado \n',i)

end

%% Topologia 2 - Validacao
fprintf('Validação TDNN 2\n');

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

yv_p10 = zeros(kv,n2,tr);

ERM_p10 = zeros(tr,1);
VARI_p10 = zeros(tr,1);

for i = 1:tr

    w1 = wf1_p10(:,:,i);
    w2 = wf2_p10(:,:,i);

    Erro_rel = zeros(n2,kv);

    for j = 1:kv

        I1 = w1*Xv(:,j);
        Y1 = logist(I1);

        Y1b = [-1;Y1];

        I2 = w2*Y1b;
        Y2 = logist(I2);

        Erro_rel(j) = abs(Y2 - dados_v(j))/abs(dados_v(j));

        yv_p10(j,:,i) = Y2';

    end

    ERM_p10(i) = mean(Erro_rel)*100;
    VARI_p10(i) = var(yv_p10(:,:,i));

end

[~,I_p10] = min(ERM_p10);

%% Topologia TDNN3 - inicializacao
fprintf('TOPOLOGIA TDNN 3 - p=15, n1=25\n');
fprintf('Configuração TDNN 3\n');

p=15;
n1=25;
n2=1;

clear X d

for t = (p+1):size(dados,1)

    X(t,1) = -1;

    for col = 2:(p+1)
        X(t,col) = dados(t-(col-1));
    end

    d(t) = dados(t);

end

X = X(p+1:end,:)';
d = d(p+1:end);

[M,k] = size(X);

rng('shuffle');

wi1 = rand(n1,M,tr);
wi2 = rand(n2,n1+1,tr);

wf1_p15 = zeros(n1,M,tr);
wf2_p15 = zeros(n2,n1+1,tr);

%% Topologia 3 - Treinamento
fprintf('Treinamento TDNN 3\n');

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

            I1 = w1(:,:,j)*X(:,j);
            Y1 = logist(I1);

            Y1b = [-1;Y1];

            I2 = w2(:,:,j)*Y1b;
            Y2 = logist(I2);

            ek(:,j) = d(:,j)' - Y2;

            delta2 = ek(:,j).*logist_linha(I2);

            w2(:,:,j+1) = w2(:,:,j) + n*delta2*Y1b';

            if j > 1
                w2(:,:,j+1) = w2(:,:,j+1) + Momentum*(w2(:,:,j)-w2(:,:,j-1));
            end

            delta1 = (delta2'*w2(:,2:end,j+1))'.*logist_linha(I1);

            w1(:,:,j+1) = w1(:,:,j) + n*delta1*X(:,j)';

            if j > 1
                w1(:,:,j+1) = w1(:,:,j+1) + Momentum*(w1(:,:,j)-w1(:,:,j-1));
            end

        end

        w1(:,:,1) = w1(:,:,j+1);
        w2(:,:,1) = w2(:,:,j+1);

        Eqm_atual = mean(sum(ek.^2))/2;

        aux = aux + 1;

        Eqm_p15(aux,i) = Eqm_atual;

        epoca_p15(i) = epoca_p15(i) + 1;

    end

    wf1_p15(:,:,i) = w1(:,:,j+1);
    wf2_p15(:,:,i) = w2(:,:,j+1);

    fprintf('\nTreinamento %d encerrado \n',i)

end

%% Topologia 3 - Validacao
fprintf('Validação TDNN 3\n');

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

yv_p15 = zeros(kv,n2,tr);

ERM_p15 = zeros(tr,1);
VARI_p15 = zeros(tr,1);

for i = 1:tr

    w1 = wf1_p15(:,:,i);
    w2 = wf2_p15(:,:,i);

    Erro_rel = zeros(n2,kv);

    for j = 1:kv

        I1 = w1*Xv(:,j);
        Y1 = logist(I1);

        Y1b = [-1;Y1];

        I2 = w2*Y1b;
        Y2 = logist(I2);

        Erro_rel(j) = abs(Y2 - dados_v(j))/abs(dados_v(j));

        yv_p15(j,:,i) = Y2';

    end

    ERM_p15(i) = mean(Erro_rel)*100;
    VARI_p15(i) = var(yv_p15(:,:,i));

end

[~,I_p15] = min(ERM_p15);

%% Resultados

fprintf('\n');
fprintf('Tabela de Treinamentos\n\n');

fprintf('--------------------------------------------------------------------------\n');
fprintf('Treinamento     TDNN1(Eqm)   TDNN1(Ep)   TDNN2(Eqm)   TDNN2(Ep)   TDNN3(Eqm)   TDNN3(Ep)\n');
fprintf('--------------------------------------------------------------------------\n');

fprintf('1(T1)%18.4f%12d%14.4f%12d%14.4f%12d\n',...
Eqm_p5(epoca_p5(1),1),epoca_p5(1),...
Eqm_p10(epoca_p10(1),1),epoca_p10(1),...
Eqm_p15(epoca_p15(1),1),epoca_p15(1));

fprintf('2(T1)%18.4f%12d%14.4f%12d%14.4f%12d\n',...
Eqm_p5(epoca_p5(2),2),epoca_p5(2),...
Eqm_p10(epoca_p10(2),2),epoca_p10(2),...
Eqm_p15(epoca_p15(2),2),epoca_p15(2));

fprintf('3(T1)%18.4f%12d%14.4f%12d%14.4f%12d\n',...
Eqm_p5(epoca_p5(3),3),epoca_p5(3),...
Eqm_p10(epoca_p10(3),3),epoca_p10(3),...
Eqm_p15(epoca_p15(3),3),epoca_p15(3));

fprintf('--------------------------------------------------------------------------\n');

fprintf('\n\n');
fprintf('Tabela de Valores\n\n');

fprintf('-------------------------------------------------------------------------------------------------------------\n');
fprintf('Amostras      x(t)      TDNN1(T1)  TDNN1(T2)  TDNN1(T3)  TDNN2(T1)  TDNN2(T2)  TDNN2(T3)  TDNN3(T1)  TDNN3(T2)  TDNN3(T3)\n');
fprintf('-------------------------------------------------------------------------------------------------------------\n');

for i = 1:length(dados_v)

    fprintf('t=%3d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n',...
    100+i,...
    dados_v(i),...
    yv_p5(i,:,1),...
    yv_p5(i,:,2),...
    yv_p5(i,:,3),...
    yv_p10(i,:,1),...
    yv_p10(i,:,2),...
    yv_p10(i,:,3),...
    yv_p15(i,:,1),...
    yv_p15(i,:,2),...
    yv_p15(i,:,3));

end

fprintf('-------------------------------------------------------------------------------------------------------------\n');

fprintf('Erro medio(%%)%12s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n',...
'',...
ERM_p5(1),ERM_p5(2),ERM_p5(3),...
ERM_p10(1),ERM_p10(2),ERM_p10(3),...
ERM_p15(1),ERM_p15(2),ERM_p15(3));

fprintf('Variancia(%%)%13s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n',...
'',...
VARI_p5(1),VARI_p5(2),VARI_p5(3),...
VARI_p10(1),VARI_p10(2),VARI_p10(3),...
VARI_p15(1),VARI_p15(2),VARI_p15(3));

fprintf('-------------------------------------------------------------------------------------------------------------\n');

figure

plot(Eqm_p5(1:epoca_p5(I_p5),I_p5),'b')
hold on

plot(Eqm_p10(1:epoca_p10(I_p10),I_p10),'r')
hold on

plot(Eqm_p15(1:epoca_p15(I_p15),I_p15),'g')

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
hold on

plot(yv_p10(:,:,I_p10),'-g')
hold on

plot(yv_p15(:,:,I_p15),'-b')

title('Comparação entre saída e desejado')
xlabel('Amostras')
ylabel('Valores')

legend('Desejado','Saída top. 1','Saída top. 2','Saída top. 3')
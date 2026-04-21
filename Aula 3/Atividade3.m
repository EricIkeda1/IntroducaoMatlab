% Algoritmo para Perceptron Aula 3 - Ex.2
clear; clc; close all;

% Importar dados do arquivo (gerado pelo Import Tool)
importfile('C:\Users\Eric Yuji Ikeda\Documents\VScode\IntroducaoMatlab\Aula 4 - APS\Tabela_dados.txt') %Item <1>

% Os dados importados estão nas variáveis: x1, x2, x3, d
% Vamos usar x1, x2 como entradas e d como saída desejada
renda = x1;  % x1 será tratado como 'renda'
divida = x2; % x2 será tratado como 'divida'
d = d;       % saída desejada

% Montar matriz de dados [renda, divida, d]
data = [renda, divida, d];

% Converter saídas -1 para 0 (para compatibilidade com o algoritmo)
for i = 1:length(data)
    if data(i,3) == -1
        data(i,3) = 0;
    end
end

% ------dados de validacao----------------
X = [1900 150 -1; 2500 800 -1; 1600 700 -1;...
     2300 500 -1; 2100 250 -1]; % X1, X2 amostra 117-121
D = [1; 1; 0; 1; 1]; % saida deseja amostra 117-121

% ------Análise gráfica--------------------
figure;
for i = 1:length(data)
    if data(i,3) == 1
        plot(data(i,1), data(i,2), 'r+')
        hold on;
    end
    if data(i,3) == 0
        plot(data(i,1), data(i,2), 'bo')
        hold on;
    end
end
xlabel('x1 (renda)');
ylabel('x2 (divida)');
title('Dados de Treinamento');
grid on;

%% Inicialização e ajustes iniciais
data(:,3) = -1; % inserir -1 na coluna 3 (bias)
norm = max(max(data)); % maior valor da matriz
data = data / norm; % normalizacao dos dados da matriz
n_ent = 2; % Número de entradas
fprintf("Pesos iniciais: ")
w = rand(1, (n_ent + 1)) % Item <3>
n = 0.0001; % Taxa de aprendizagem Item<4>
epoca = 0; % Item<5>
erro = 1000;
errof = 1000;
u = 0;
pause(2);

%% Treinamento da Perceptron
fprintf("Iniciando o Treinamento da Perceptron \n")
pause(2);

[L, C] = size(data); % verifica o tamanho da matriz L(linhas) C(col)
tic; % inicializa contador de tempo tic-toc

while(errof > 0.01 && epoca < 1000) % Item <6.1> - limite de 1000 épocas
    erro = 0;
    for i = 1:L % Item <6.2>
        u(i) = w * data(i,:)'; % Item<6.2.1>
        %--------------funcao de ativacao-----Item <6.2.2>---
        if u(i) >= 0
            y(i) = 1;
        else
            y(i) = 0;
        end
        %-----------fim funcao de ativacao--------------
        d1(i) = d(i);
        if y(i) ~= d1(i) % Item<6.2.3>
            dif = d1(i) - y(i);
            w = w + (n * dif * data(i,:)); % Item<6.2.3.1>
        else
            dif = 0;
        end
        erro = erro + abs(dif);
    end
    errof = erro;
    epoca = epoca + 1; % incrementa o contador epocas
    %condição de parada
    if epoca > 50000
        fprintf("Não converge!\n")
        break
    end
end

t = toc; % armazena tempo de treinamento

fprintf("Finalizando o Treinamento da Perceptron\n")
pause(2);
fprintf("Tempo de treinamento: %0.3f seg\n", t)
pause(2);
fprintf("Número de épocas: %d\n", epoca)
pause(2);
fprintf("Pesos finais:")
w

%% Validação da Perceptron
fprintf("Iniciando a Validação \n")
pause(2);

% Item <1> foi realizado inicio do codigo
wfinal = w; % Item<2>
[L1, C1] = size(X);

for i = 1:L1
    U(i) = wfinal * X(i,:)'; % Item<3.1>
end

Y = hardlim(U); % Item<3.2> hardlim -> degrau

fprintf("A saída esperada é:\n")
D = D'
fprintf("A saída da rede é: \n")
Y

%% Calculo o acerto
acerto = 0;
for i = 1:L1 
    z = isequal(Y(:,i), D(:,1));
    if z == 1
        acerto = acerto + 1;
    end
end

fprintf("Porcentagem de acerto e: %0.2f %%\n", (acerto/L1)*100)

% Função importfile (adicione no final do script)
function importfile(fileToRead)
    % Inicializar variáveis
    delimiter = ' ';
    startRow = 2;
    
    % Formato da coluna
    formatSpec = '%f%f%f%f%[^\n\r]';
    
    % Abrir arquivo
    fileID = fopen(fileToRead,'r');
    
    % Ler colunas de dados
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, ...
        'MultipleDelimsAsOne', true, 'HeaderLines', startRow-1, ...
        'ReturnOnError', false);
    
    % Fechar arquivo
    fclose(fileID);
    
    % Atribuir variáveis
    x1 = dataArray{:, 1};
    x2 = dataArray{:, 2};
    x3 = dataArray{:, 3};
    d = dataArray{:, 4};
    
    % Salvar no workspace base
    assignin('base', 'x1', x1);
    assignin('base', 'x2', x2);
    assignin('base', 'x3', x3);
    assignin('base', 'd', d);
end
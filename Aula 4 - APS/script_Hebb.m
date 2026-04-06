% script_Hebb.m
clear; clc;

fid = fopen('Tabela_dados.txt', 'r');
if fid == -1
    error('Não foi possível abrir o arquivo');
end

% Pular a primeira linha (cabeçalho)
fgetl(fid);

% Ler os dados numéricos
dados = fscanf(fid, '%f %f %f %f', [4, Inf]);
dados = dados';
fclose(fid);

% Separar
X = dados(:,1:3);
d = dados(:,4);

% Adicionar bias
X = [ones(size(X,1),1), X];

% Normalizar
for j = 2:4
    X(:,j) = X(:,j) / max(abs(X(:,j)));
end

% Treinar
eta = 0.1;
w = [0,0,0,0];
melhor_taxa = 0;
melhor_w = w;

for epoca = 1:200
    erros = 0;
    for i = 1:length(d)
        u = w * X(i,:)';
        y = 1*(u>=0) + (-1)*(u<0);
        if y ~= d(i)
            w = w + eta * (d(i)-y) * X(i,:);
            erros = erros + 1;
        end
    end
    
    % Testar
    acertos = 0;
    for i = 1:length(d)
        u = w * X(i,:)';
        y = 1*(u>=0) + (-1)*(u<0);
        acertos = acertos + (y == d(i));
    end
    taxa = acertos/length(d)*100;
    
    if taxa > melhor_taxa
        melhor_taxa = taxa;
        melhor_w = w;
    end
    
    if mod(epoca,20)==0
        fprintf('Época %3d: acerto = %.1f%%\n', epoca, taxa);
    end
    
    if erros == 0
        fprintf('Convergiu na época %d!\n', epoca);
        break;
    end
end

% Resultado final
w = melhor_w;
acertos = 0;
for i = 1:length(d)
    u = w * X(i,:)';
    y = 1*(u>=0) + (-1)*(u<0);
    acertos = acertos + (y == d(i));
end

fprintf('\n=== RESULTADO ===\n');
fprintf('Acertos: %d/%d\n', acertos, length(d));
fprintf('Taxa de acerto: %.2f%%\n', acertos/length(d)*100);
% script_Hebb.m
clear; clc; close all;

fid = fopen('Tabela_dados.txt', 'r');
if fid == -1
    error('Não foi possível abrir o arquivo');
end

fgetl(fid);

dados = fscanf(fid, '%f %f %f %f', [4, Inf]);
dados = dados';
fclose(fid);

X = dados(:,1:3);
d = dados(:,4);

X = [ones(size(X,1),1), X];

for j = 2:4
    X(:,j) = X(:,j) / max(abs(X(:,j)));
end

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

%% ================= PLOT 3D =================

figure; hold on; grid on;

% Dados de treino
for i = 1:size(X,1)
    if d(i) == 1
        plot3(X(i,2), X(i,3), X(i,4), 'r+', 'MarkerSize', 8)
    else
        plot3(X(i,2), X(i,3), X(i,4), 'bo', 'MarkerSize', 8)
    end
end

%% Dados de validação
load('dados_val.mat')

X_val = dados_val(:,1:3);
d_val = dados_val(:,4);

X_val = [ones(size(X_val,1),1), X_val];

for j = 2:4
    X_val(:,j) = X_val(:,j) / max(abs(X(:,j)));
end

for i = 1:size(X_val,1)
    if d_val(i) == 1
        plot3(X_val(i,2), X_val(i,3), X_val(i,4), 'ks', 'MarkerSize', 8)
    else
        plot3(X_val(i,2), X_val(i,3), X_val(i,4), 'kd', 'MarkerSize', 8)
    end
end

%% Plano de decisão

[x1, x2] = meshgrid(...
    linspace(min(X(:,2)), max(X(:,2)), 10), ...
    linspace(min(X(:,3)), max(X(:,3)), 10));

x3 = -(w(1) + w(2)*x1 + w(3)*x2) / w(4);

surf(x1, x2, x3, 'FaceAlpha', 0.3)

xlabel('X1')
ylabel('X2')
zlabel('X3')

legend('Classe +1 treino','Classe -1 treino', ...
       'Classe +1 validação','Classe -1 validação', ...
       'Plano de decisão')

title('Hebb 3D (Cubo + Plano)')
view(3)
%% Função degrau bipolar
% Funções de Ativação
clear; clc;

u = (-2:0.01:2);

for i = 1:size(u,2)
    if u(1,i) < 0
        y(1,i) = -1;
    elseif u(1,i) == 0
        y(1,i) = 0;
    else 
        y(1,i) = 1;
    end
end

y1 = sign(u); 

figure(1)
plot(u, y, 'ko', 'LineWidth', 3);
hold on;
plot(u, y1, 'g*');

grid;
ylim([-1.5 1.5]);

legend('por código', 'por sign')
title('Função de Ativação - Degrau Bipolar');
xlabel('u');
ylabel('g(u)');

%% Função Sigmoidal (Refatorado)

clear; clc;

% Vetor de entrada
u = -2:0.01:2;

% Intervalo de beta
beta_min = 1;
beta_max = 6;

% Cores para cada curva
cores = ['b', 'r', 'g', 'k', 'y', 'm'];

figure(2);
hold on;
grid on;

% Loop para cada valor de beta
for beta = beta_min:beta_max
    
    % Função sigmoidal (vetorizada - sem for interno)
    b = 1 ./ (1 + exp(-beta .* u));
    
    % Plot
    plot(u, b, cores(beta), 'LineWidth', 1.5);
end

% Ajustes do gráfico
ylim([-0.5 1.5]);
title('Função de Ativação - Sigmoidal/logística');
xlabel('u');
ylabel('f(u)');

% Legenda automática
legend_strings = arrayfun(@(x) sprintf('beta=%d', x), beta_min:beta_max, 'UniformOutput', false);
legend(legend_strings);

%% Função Degrau Bipolar 
clear; clc;

% Vetor de entrada
u = -2:0.01:2;

g = zeros(size(u));

g(u > 0) = 1;
g(u < 0) = -1;

figure(3)
plot(u, g, 'k', 'LineWidth', 1.5);
grid on;

ylim([-1.5 1.5]);
title('Função de Ativação - Degrau Bipolar');
xlabel('u');
ylabel('g(u)');

%% Função Tangente Hiperbólica
clear; clc;

% Vetor de entrada
u = -2:0.01:2;

% Intervalo de beta
beta_min = 1;
beta_max = 6;

% Cores para cada curva
cores = ['b', 'r', 'g', 'k', 'y', 'm'];

figure(4);
hold on;
grid on;

% Loop para cada valor de beta
for beta = beta_min:beta_max
    
    % Aplicando a função tanh com beta
    g = (1 - exp(-beta .* u)) ./ (1 + exp(-beta .* u));
    
    % Plot
    plot(u, g, cores(beta), 'LineWidth', 1.5);
end

% Ajustes do gráfico
ylim([-1.5 1.5]);
title('Função de Ativação - Tangente Hiperbólica');
xlabel('u');
ylabel('g(u)');

% Legenda automática
legend_strings = arrayfun(@(x) sprintf('beta=%d', x), beta_min:beta_max, 'UniformOutput', false);
legend(legend_strings);

%% Função Linear
clear; clc;

% Vetor de entrada
u = -2:0.01:2;

% Função linear
g = u;

% Plot
figure(5);
plot(u, g, 'b', 'LineWidth', 1.5);

% Ajustes do gráfico
grid on;
title('Função de Ativação - Linear');
xlabel('u');
ylabel('g(u)');
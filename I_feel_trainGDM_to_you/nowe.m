clear;
format compact;
load prima.mat;

% Przygotowanie zmiennych
P = Pns;
T = Ts;
% P = Pns(1:500);
% T = Ts(1:500);

% Wyczyszczenie zbednych zmiennych
clearvars -except P T;

% Wektor neuronow warstwy 1 i 2
S1_vec=20:5:70;
S2_vec=5:5:50;

% Wektory inc/dec learning ratio i b³êdu
lr_inc_vec = 1.03:0.01:1.07;
lr_dec_vec = .6:.05:.8;
er_vec = 1.02:0.01:1.06;

% work_count = length(S1_vec)*length(S2_vec)*length(lr_inc_vec)*length(lr_dec_vec)*length(er_vec);


% Zapis nag³ówka
header = 'S1\tS2\t\tlr_inc\tlr_dec\tmax_err\tacc[%%]\t\tlr\t\t\tMSE\t\tepoch\twork_progress[%%]\n';
file_var = fopen('kryha.txt', 'wt');
fprintf(file_var, header);
formating = '%2g \t %2g \t %1.6g \t %1.6g \t %1.6g \t %3.4g \t %1.6g \t %4.6g \t %3.6g \t %3.6g\n';
fclose(file_var);


% Testowanie
% S1_vec=60;
% S2_vec=30;
lr_inc_vec = 1.05;
lr_dec_vec = 0.7;
er_vec = 1.04;

work_count = length(S1_vec)*length(S2_vec)*length(lr_inc_vec)*length(lr_dec_vec)*length(er_vec);
PK_v5=zeros(length(S1_vec),length(S2_vec),length(lr_inc_vec),length(lr_dec_vec),length(er_vec));
MSE_v5=PK_v5;
work_counter = 0;

for ind_S1=1:length(S1_vec)
    for ind_S2=1:length(S2_vec)
        for ind_lr_inc=1:length(lr_inc_vec)
            for ind_lr_dec=1:length(lr_dec_vec)
                for ind_er=1:length(er_vec)
                    
                    % Zwiekszenie licznika
                    work_counter=work_counter+1;
                    work_progress = work_counter/work_count *100;
                    
                    % Utworzenie sieci
                    net = feedforwardnet([S1_vec(ind_S1) S2_vec(ind_S2)],'traingda');
                    
                    % Ustawienie parametrow sieci
                    net.trainParam.epochs = 500; % default = 1000
                    net.trainParam.lr = 0.001; % default = 0.01
                    net.trainParam.lr_inc = lr_inc_vec(ind_lr_inc); % default = 1.05
                    net.trainParam.lr_dec = lr_dec_vec(ind_lr_dec); % default = 0.7
                    net.trainParam.max_perf_inc = er_vec(ind_er); % default = 1.04
                    net.trainParam.showCommandLine = true;
                    %net.trainParam.showWindow=false;
                    
%                   net.trainParam.max_fail = net.trainParam.epochs;
                    net.trainParam.max_fail = 50;
                    net.trainParam.goal =0.25/length(T);
                    net.trainParam.show = 100;
                    %                     net.performFcn = 'sse';
                    %                     net.trainParam.goal =0.25
                    
                    % Trenowanie
                    [net, tr] = train(net, P, T);
                    % netOutput = net(P);
                    % perf = perform(net,netOutput,T)
                    
                    % Symulacja
                    A3=net(P);
                    
                    %                     SSE = sumsqr((A3-T)')
                    MSE = immse(A3,T) % u¿ywaæ tego czy best_perf?
                    
                    accuracy = (1-sum((abs(A3-T)>=.5))/length(T))*100 % TODO: przerobiæ na tylko z testu
                    
                    % Zapis do konsoli
                    % fprintf("\nS1 = %d S2 = %d lr_inc = %d lr_dec = %d er = %d\n\n", S1_vec(ind_S1), S2_vec(ind_S2), lr_inc_vec(ind_lr_inc), lr_dec_vec(ind_lr_dec), er_vec(ind_er));
                    
                    % Zapis do macierzy
                    PK_v5(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec, ind_er) = accuracy;
                    MSE_v5(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec, ind_er) = MSE;
                    
                    % Zapis do pliku
                    file_var = fopen('kryha.txt', 'at');
                    fprintf(file_var, formating, S1_vec(ind_S1), S2_vec(ind_S2), lr_inc_vec(ind_lr_inc), lr_dec_vec(ind_lr_dec), er_vec(ind_er), accuracy, tr.lr(tr.best_epoch),tr.best_perf,tr.best_epoch, work_progress);
                    fclose(file_var);
                    
                    
surf(S1_vec,S2_vec,PK_v5')
xlabel('S1');
ylabel('S2');
zlabel('PK [%]');
                end
            end
        end
    end
end

clear; %czysci pamiêc i zmienne z workspace'a
format compact; %czyœci wyjscia(brak pustych przestrzeni)
load prima.mat; %³adowanie danych znormalizowanych

% Przygotowanie zmiennych
P=Pns; % wejœcie
T=Ts; % po¿¹dane wyjœcie

% Wyczyszczenie zbednych zmiennych
clearvars -except P T;
%Testowanie
%Pn = Pns(1:200);
%T = Ts(1:200);


% Wektor neuronow warstwy 1 i 2
%S1_vec = 20:10:100;
%S2_vec = S1_vec;
S1_vec=20:10:100;%wiecej neuronow to trzeba zmniejszyc lr //poczytac w dokumentacji traingd
S2_vec=10:10:50;
% Wektory learning ratio i momentum constans
lr_vec = 0.001 ; % wysoki lr(bliski 1) zle wplywa na PK tak samo niski, najlepszy to chyba 0.01 lub ten teraz 
mc_vec =  0.8; %dla mc=0.95 i 0.99 PK~ 65% a dla ma³ych sie jebie
%Poprawnosc klasyfikacji 4wymiarowa(dla 4 zmiennych) i jej b³¹d
PK_v4 = zeros(length(S1_vec),length(S2_vec),length(lr_vec),length(mc_vec));
MSE_v4 = PK_v4;
% licznik który bedzie pokazywa³ ile zosta³o do koñca nauki
licznikWC= 0;
work_count = length(S1_vec)*length(S2_vec)*length(lr_vec)*length(mc_vec);

% Zapis nag³ówka
header = 'S1 \t\t S2 \t\t lr \t\t mc \t\t PK[%%] \t\t MSE \t\t epoch \t\t work_progress[%%]\n';
file_var = fopen('danetest1.txt', 'wt');
fprintf(file_var, header);
formating = '%2g \t\t %2g \t\t %1.5g \t\t %1.3g \t\t %1.4g \t\t %3.4g \t\t %g \t\t %3.2g\n';
fclose(file_var);


for ind_S1=1:length(S1_vec)
    for ind_S2=1:length(S2_vec)    %:length(S2_vec) ??? :ind_S1
        for ind_lr=1:length(lr_vec)
            for ind_mc=1:length(mc_vec)
               %for ex=1:10 
                % Zwiekszenie licznika
                licznikWC=licznikWC+1;
                % Procent wykonania 
                work_progress = licznikWC/work_count *100;
                % Utworzenie sieci
                net = feedforwardnet([S1_vec(ind_S1) S2_vec(ind_S2)],'traingdm');
                % Ustawienie parametrow sieci
                net.trainParam.lr = lr_vec(ind_lr); 
                net.trainParam.epochs = 1000;
                net.trainParam.goal = 0.25/length(T); %b³¹d docelowy roznica miedzy klasami to 1 wiec /2 i do kwadratu
                net.trainParam.mc=mc_vec(ind_mc);
                net.trainParam.max_fail=50;
                % Trenowanie
                [net,tr] = train(net,P,T);
                %Symulacja
                y = net(P);    %wyjscie sieci
                %Poprawnoœæ Klasyfikacji(precyzja) w %
                PK = (1-sum(abs(T-y)>=.5)/length(T))*100; %sum sumuje wartosci w macierzy
                MSE = immse(y,T); %immse liczy blad pomiedzy macierzami o takich samych rozmiarach i klasach
                % Zapis do macierzy 4D- PK i B³¹d œrednio kwadratowy
                PK_v4 (ind_S1, ind_S2, ind_lr, ind_mc) = PK; %dolaczenie PK do ... tablicy PK
                %MSE_v4(ind_S1, ind_S2, ind_lr, ind_mc) = tr.best_perf; tr 
                MSE_v4(ind_S1, ind_S2, ind_lr, ind_mc) = MSE;%dolaczenie pojedynczego bledu do ... tablicy bledow
                % Zapis do konsoli
                fprintf("\nS1 = %d S2 = %d lr = %d mc = %1.2d PK = %d MSE= %d Epoka = %d WP = %d \n",  S1_vec(ind_S1), S2_vec(ind_S2), lr_vec(ind_lr),mc_vec(ind_mc), PK,tr.best_perf,tr.num_epochs,work_progress);
                
                % Zapis do pliku
                file_var = fopen('danetest1.txt', 'at');
                %fprintf(file_var, formating, S1_vec(ind_S1), S2_vec(ind_S2), lr_vec(ind_lr), mc_vec(ind_mc), tr.best_perf, tr.num_epochs, work_progress, PK);
                fprintf(file_var, formating, S1_vec(ind_S1), S2_vec(ind_S2), lr_vec(ind_lr), mc_vec(ind_mc), PK, MSE, tr.best_epoch, work_progress);
                fclose(file_var);
                     
                %Prezentacja danych %jeszcze lr, mc i PK oraz S1 i S2 od
                %koncowego b³êdu oraz lr,mc od koncowego b³êdu
                surf(S1_vec,S2_vec,PK_v4')
                xlabel('S1');
                ylabel('S2');
                zlabel('PK [%]');
               %end
            end
        end
    end
end

%figure, plotfit(net,x,t)

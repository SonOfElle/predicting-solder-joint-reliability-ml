%***
% This function has been prepared for the paper entitled
% "Correlation-Driven Machine Learning for Accelerated Reliability 
% Assessment of Solder Joints in Electronics" submitted to the Scientific
% Reports Journal.
%***
% Note: the concepts and notations that we used here are taken from the
% above-mentioned paper written by: 
% Vahid Samavatian, Mahmud Fotuhi-Firuzabad, Majid Samavatian,
% Payman Dehghanian and Frede Blaabjerg 
%***
% trainConv.m: this source code allows us to train a multilayer neural
% network based on the proposed correlation driven algorithm using delta
% rule method.
    %%%%    Author:        Vahid SAMAVATIAN
    %%%%    UNIVERSITY:     Sharif University Technology
    %%%%    EMAIL:          v.samavatian@gmail.com
    %%%%    Updated: 06/07/2020

function [WC, W1, W2, W3, Wo] = trainConv(X,D,hidden_layers, CC, epoch, N1, N2)

alpha = 0.01; 
beta  = 0.95;

[i,j,~]=size(X);
input=i*j+((i-CC(1)+1)*(j-CC(2)+1))*CC(3)/(2*2);
output=size(D,1);
network_architecture=[input hidden_layers output];

rng(1);
WC = 1e-2*randn(CC);
W1 = (2 * rand(network_architecture(2),network_architecture(1)) - 1) * sqrt(6) / sqrt(360 + 100);
W2 = (2 * rand(network_architecture(3),network_architecture(2)) - 1) * sqrt(6) / sqrt(360 + 100);
W3 = (2 * rand(network_architecture(4),network_architecture(3)) - 1) * sqrt(6) / sqrt(360 + 100);
Wo = (2 * rand(network_architecture(5),network_architecture(4)) - 1) * sqrt(6) / sqrt( 10 +  10);

momentumC = zeros(size(WC));
momentum1 = zeros(size(W1));
momentum2 = zeros(size(W2));
momentum3 = zeros(size(W3));
momentumo = zeros(size(Wo));


% waite bar
h = waitbar(0,'... CBNN Training process  ...');

for kk=1:epoch
    waitbar(kk/epoch);% wait bar function
    N = length(D);
    bsize = 5;
    blist = 1:bsize:(N-bsize+1);
    for batch = 1:length(blist)
   
        dWC = zeros(size(WC));
        dW1 = zeros(size(W1));
        dW2 = zeros(size(W2));
        dW3 = zeros(size(W3));
        dWo = zeros(size(Wo));
        
        begin = blist(batch);
        
        for k = begin:begin+bsize-1
            %% Data Correlating
            x    = X(:, :, k);
            yC1  = Conv(x, WC);
            yC2  = ReLU(yC1);
            yC   = Pool(yC2);
            %% Data Flattening
            yC_f         = reshape(yC, [], 1);
            x_f          = reshape(x, [], 1);
            x_flattened  = [yC_f;x_f];
            
            %% Conventional neural network with 3 hidden layers
            h1 = W1*x_flattened;
            y1 = ReLU(h1);
            
            h2 = W2*y1;
            y2 = ReLU(h2);
            
            h3 = W3*y2;
            y3 = ReLU(h3);            
            
            ho = Wo*y3;
            y  = Sigmoid(ho);
            %% Error Calculation
            
             eo      = D(k) - y;
            deltao  = eo * Sigmoid(ho)*(1-Sigmoid(ho));   % e * derivative of sigmoid
            
                        
            e3     = Wo' * deltao;
            delta3 = (y3 > 0) .* e3;
            
            e2     = W3' * delta3;
            delta2 = (y2 > 0) .* e2;
            
            
            e1     = W2' * delta2;
            delta1 = (y1 > 0) .* e1;
            
            
            e_flattened  = W1' * delta1;
            e_yC_f       = e_flattened(1:length(yC_f),1);
            
            eC     = reshape(e_yC_f, size(yC));
            
            [~,~,Q] = size(WC);                % filter WC
            eC2     = zeros(size(yC2));
            W4      = ones(size(yC2)) / (2*2);
            for c=1:Q
                eC2(:, :, c) = kron(eC(:, :, c), ones ([2 2])) .* W4(:, :, c);
            end
            deltaC2 = (yC2 > 0) .* eC2;
            
            
            delta_x = zeros (size(WC));
            for c=1:Q
                delta_x(:, :, c) = conv2(x(:, :), rot90(deltaC2(:, :, c), 2), 'valid');
            end
            
            dWC = dWC + delta_x;
            dW1 = dW1 + delta1 * x_flattened';
            dW2 = dW2 + delta2 * y1';
            dW3 = dW3 + delta3 * y2';
            dWo = dWo + deltao * y3';
        end
        
        dWC = dWC / bsize;
        dW1 = dW1 / bsize;
        dW2 = dW2 / bsize;
        dW3 = dW3 / bsize;
        dWo = dWo / bsize;
        
        momentumC = alpha*dWC + beta*momentumC;
        WC        = WC + momentumC;
        
        momentum1 = alpha*dW1 + beta*momentum1;
        W1        = W1 + momentum1;
        
        momentum2 = alpha*dW2 + beta*momentum2;
        W2        = W2 + momentum2;
        
        momentum3 = alpha*dW3 + beta*momentum3;
        W3        = W3 + momentum3;  
        
        momentumo = alpha*dWo + beta*momentumo;
        Wo        = Wo + momentumo ;
        
        
    end
end
close (h)


N = length(D);


for k = 1:N
    
    x =X(:, :, k);
    
    yC1=Conv(x, WC);
    yC2=ReLU(yC1);
    yC=Pool(yC2);
    
    yC_f=reshape(yC, [], 1);
    x_f=reshape(x, [], 1);
    x_flattened=[yC_f;x_f];
    
    h1=W1*x_flattened;
    y1=ReLU(h1);
    
    h2=W2*y1;
    y2=ReLU(h2);
    
    h3=W3*y2;
    y3=ReLU(h3);
    
    ho=Wo*y3;
    y =Sigmoid(ho);
    
    Y(k)=y;
end

figure (1)
axes('fontsize',36,'fontweight','Bold')
title('Training data')
hold on

Y_den=normalize(Y, 'range', [N1 N2]);
D_den=normalize(D, 'range', [N1 N2]);

scatter(D_den(1,:)/1000,Y_den(1,:)/1000,100,'filled')

xlabel('Measured UL(h)','fontsize',36)
ylabel('Predicted UL(h)','fontsize',36)

xlim([23 29]);
ylim([23 29]);

ax = gca;
ax.LineWidth=1;
ax.XTick = 23:29;

ay = gca;
ay.LineWidth=1;
ay.YTick =23:29;

set(gcf,'color','w')

axis square
box on
axis 
end

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
% predict.m: this source code allows us to predict the useful lifetime
% via the proposed neural network and the results compare with the measured 
% lifetime and reports RMSE, error and r coefficients of determination. 
    %%%%    Author:        Vahid SAMAVATIAN
    %%%%    UNIVERSITY:     Sharif University Technology
    %%%%    EMAIL:          v.samavatian@gmail.com
    %%%%    Updated: 06/07/2020
    
function [RMSE, Error, r]=predict(X,D,WC,W1,W2,W3,Wo)

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

Error=sum((Y-D)/N);
RMSE=sqrt(sum((Y-D).^2)/N); 
rr=corrcoef(Y,D);
r=rr(1,2);
end
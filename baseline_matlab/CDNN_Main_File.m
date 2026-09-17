%***
% This supplementary source code has been prepared for the paper entitled
% "Correlation-Driven Machine Learning for Accelerated Reliability 
% Assessment of Solder Joints in Electronics" submitted to the Scientific
% Reports Journal.
%***
% Note: the concepts and notations that we used here are taken from the
% above-mentioned paper written by: 
% Vahid Samavatian, Mahmud Fotuhi-Firuzabad, Majid Samavatian,
% Payman Dehghanian and Frede Blaabjerg 
%***
% CDNN_Main_File: this source code allows us to train a multilayer neural
% network based on the proposed correlation driven algorithm. It by default
% is capable to consider three hidden layers with optional number of perceptrons by 
% defining "hidden_layers" vector variable. Cross correlation architecture
% was defined by "CC" vector variable. The training repeats for finite
% number of iterations as defined by "epoch".
% plotting : predicted and measured useful lifetime in kh is shwon by end
% of training.
% CDNN.mat: contains the most important caracteristics of the network
% weights.mat: contains weights of the network 
    %%%%    Author:        Vahid SAMAVATIAN
    %%%%    UNIVERSITY:     Sharif University Technology
    %%%%    EMAIL:          v.samavatian@gmail.com
    %%%%    Updated: 06/07/2020
%%
clear all;
close all;
clc;

%% Data extracting
Main_Directory=pwd;	% Save the main directory of source code

% Change the name of the Data folder in the following line, between Data
% (for original samples) and Synth_Data for synthetic dataset
cd(fullfile(Main_Directory, 'Data'));	% Change the directory of the saved Data


files = dir('*.txt');	% Save all .txt files in the data folder
numfiles = length(files);
Data_Import=zeros(7,4,numfiles);	% Define a variable for saving all data in the .txt files
for k = 1:numfiles                  
  Data_Import(:,:,k) = importdata(files(k).name);	% Save it for all data file
end

cd (Main_Directory);        % Return to the main directory of source code
Data=Data_Import(1:6,:,:);	% Data beaking (Training data)
UL(1,:)=Data_Import(7,1,:);	% Data breaking (lifetime data)

%% Data prepration
X  = Scale(Data,0.2,0.8);                    % Scale the training data to the range of 0.2 to 0.8
D  = normalize(UL, 'range', [0.2 0.8]);	     % Scale the lifetime data to the range of 0.2 to 0.8
N1 = min(UL);                                % Save min for denormalization
N2 = max(UL);                                % Save max for denormalization

%% Initialize parameters
CC=[3 3 80];                     % Cross correlation architecture 3*3*80
hidden_layers=[640 580 500];    % 3 hidden layers for FC network
epoch=10;                       % Repeatation of neural network training

%% Network Training
tic
[WC, W1, W2, W3, Wo]=trainConv(X,D,hidden_layers, CC, epoch,N1,N2); % Call the training function in order to calculate the weights for the CDNN
toc
save('CDNN.mat','X','D','WC', 'W1', 'W2', 'W3', 'Wo', 'N1', 'N2');  % Save the results in CDNN.mat
save('weights.mat','WC', 'W1', 'W2', 'W3','Wo');                    % Save the weights in weights.mat

%% Predict
[RMSE, Error, r]=predict(X,D,WC,W1,W2,W3,Wo)

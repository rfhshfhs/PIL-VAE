clear;
clc;
% Load data set
X = importdata('./Dataset/fashion_mnistX.mat');%d*N
Label = importdata('./Dataset/fashion_mnistnumY.mat');
idx = find(Label == 2);
X1 = X(:,idx);
num_trainingSamples = 2000;
ridx =  randi([1, size(X1,2)], [1,num_trainingSamples]);
X_train = X1(:,ridx);

% Hyperparameters
input_dim = size(X_train, 1);   % dim of input data
hidden_dim = size(X_train, 2);  % Gn-PIL dim of H
latent_dim = 8;% dim of Z   8
MAX_EN_LAYER =1;
mean_value = 0;
sig = 1e-2;
para = 1;
actFun = 'prelu';% prelu gau
lambda = 1e-3;%1e-6

vae = {};
HiddenO = {};
l = 1;


X_train = mapminmax(X_train',0,1)';


InputLayer = X_train;
tic;
while(l<=MAX_EN_LAYER)
%%%%%%%%%% Encoder %%%%%%%%%%
%%%%%% 1st layer: PIL0
InputWeight=randn(hidden_dim,input_dim);
if hidden_dim >= input_dim
    InputWeight = orth(InputWeight);
else
    InputWeight = orth(InputWeight')';
end
tempH = InputWeight*InputLayer;
H1 = ActivationFunc(tempH,actFun,para);
vae{l}.WI = InputWeight;
HiddenO{l}=H1;
l = l + 1;

InputLayer = H1;
hidden_dim = size(InputLayer, 2);
input_dim = size(InputLayer, 1);
end
%%%%%% 2nd layer: Gn-PIL
% % 1 using pseudoinverse(Gn-PIL)
% 1---------------------------------------
InputLayer_pinv=pinv(InputLayer);
;

% Adding nosie from standard normalization distribution
InputLayer_pinv = InputLayer_pinv + normrnd(mean_value, sig,size(InputLayer_pinv));
vae{l}.WI = InputLayer_pinv;
tempH = vae{l}.WI * InputLayer;
H2 = ActivationFunc(tempH,actFun,para);
HiddenO{l}=H2;
l = l + 1;

Y = H2';


%[coeff,score,pcvariance,mu,v,S] = ppca(Y,latent_dim,'Options',opt);
[ mu,w,sigma,L ] = ppcamle(Y,latent_dim);
Z = (w'*w + sigma*eye(latent_dim))\w'*(Y-repmat(mu',1,num_trainingSamples));
%Z = w'*(InputLayer - repmat(mu',1,num_trainingSamples));
vae{l}.WI = w;
l = l + 1;

%%%%%%%%%% Decoder %%%%%%%%%%
%%%%%%% 1st layer: Using Z to reconstruct H2
OutputWeight = H2*Z'/(Z*Z'+lambda*eye(latent_dim));
vae{l}.WO = OutputWeight;
tempH = OutputWeight*Z;
%rec_H2 = ActivationFunc(tempH,actFun,para);
rec_H2 = tempH;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%rec_H2 = w * Z + repmat(mu',1,num_trainingSamples);
l = l + 1;

%%%%%%% 2nd layer: Using H2 to reconstruct H1
dl = l-4;
while(dl>0)
OutputWeight = HiddenO{dl}*tempH'/(tempH*tempH'+lambda*eye(hidden_dim));
vae{l}.WO = OutputWeight;
tempH = OutputWeight*tempH;
rec_H1 = tempH;
l = l + 1;
dl=dl-1;
end
%%%%%%% 3rd layer:  Using H1 to reconstruct X
OutputWeight = X_train*rec_H1'/(rec_H1*rec_H1'+lambda*eye(hidden_dim));
vae{l}.WO = OutputWeight;
tempX = OutputWeight*rec_H1;
rec_X = tempX;
toc;
%%%%%%% reconstructing new samples
num_samples_test = 1000;
ridx_test =  randi([60001, 70000], [1,num_samples_test]);
X_test = X(:,ridx_test);
X_test = mapminmax(X_test',0,1)';
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
InputLayer = X_test;
l = 1;
while(l<=MAX_EN_LAYER)
tempH_test1 = vae{l}.WI * InputLayer;
H_test1 = ActivationFunc(tempH_test1,actFun,para);
InputLayer = H_test1;
l = l + 1;
end
tempH_test2 = vae{l}.WI * InputLayer;
H_test2 = ActivationFunc(tempH_test2,actFun,para);
InputLayer = H_test2;
l = l + 1;

Z_test = (vae{l}.WI'*vae{l}.WI + sigma*eye(latent_dim))\vae{l}.WI'*(InputLayer-repmat(mu',1,num_samples_test));
l = l + 1;

H_rec_test2 = w * Z_test + repmat(mu',1,num_samples_test);
%H_rec_test2 = vae{l}.WO*Z_test;
l = l + 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dl = l-4;
while(dl>0)
H_rec_test1 = vae{l}.WO*H_rec_test2;
l = l + 1;
dl = dl-1;
end
X_rec_test = vae{l}.WO*H_rec_test1;

%%%%%%% Generating new sample
num_samples = 1000;

% Sample Z from standard normalization distribution
latent_samples = randn(latent_dim, num_samples, 'double'); %CLASSNAME can be 'double' or 'single'.

% Generate the covariance matrix
covariance = sigma * eye(hidden_dim);

% Generate noise from the Gaussian distribution
ns = mvnrnd(zeros(num_samples, hidden_dim), covariance, num_samples);

% Generate new samples
hidden_samples1 = w * latent_samples + repmat(mu',1,num_samples) + ns';

l = 4+MAX_EN_LAYER;
generated_samples2 = vae{l}.WO * hidden_samples1;
temprecH=generated_samples2;
while(l<4+MAX_EN_LAYER*2)
l = l + 1;
temprecH = vae{l}.WO * temprecH;


end
generated_X=temprecH;
displaySamplesGeneral(generated_X,10,'MNIST',28,28);
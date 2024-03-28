function [ mu,w,sigma,L ] = ppcamle(data,q)%ppca极大似然估计
%	myppcaMle:			perform the ML estimation for ppca
%   data:               the input data
%	q:					the dimension of latent space
%	mu:        			the mean of data
%	w: 					the factor loading matrix
%	sigma:				the noise variance
%	L:					the log likelihood 
[N,d] = size(data);
mu = mean(data,1);
T = data-repmat(mu,N,1); %Tn-mu   N x d matrix
S = T'*T/N; % sample covirance matrix
[V,D] = eig(S); % Eigenvalue decomposition
[D,ind] = sort(diag(D),'descend');
V = V(:,ind);
sigma = sum(D((q+1):d))/(d-q);
Uq = V(:,1:q);
lambda = D(1:q);
w = Uq*sqrt(diag(lambda)-sigma*eye(q));
C = w*w'+sigma*eye(d);
L = -N*(d*log(2*pi)+log(det(C))+trace(C\S))/2;
end


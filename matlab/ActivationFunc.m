function H = ActivationFunc( tempH, ActivationFunction,p)
%ACTIVATIONFUNC Summary of this function goes here
%   Detailed explanation goes here
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-p.*tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
    case {'gau'}
        H = 1./sqrt(2*pi)*exp(-1./2*(tempH.^2));
        %%%%%%%% ReLU
    case {'relu'}
        idx = find(tempH(:)<0);
        tempH(idx)=0;
        H = tempH;
    case {'srelu'}
        idx = find(tempH(:)<p);
        tempH(idx)=0;
        H = tempH;
    case {'tan'}
        H = tanh(p.*tempH);
    case {'prelu'}
        alpha = 0.02;
        idx = find(tempH(:)<0);
        tempH(idx)=alpha.*tempH(idx);
        H = tempH;
    case {'mor'}
        H = cos(0.4.*tempH).*exp(-1./2*(tempH.^2));
    case {'gelu'}
        H = tempH .* 1 ./ (1 + exp(-p.*tempH.*1.702));
end
end


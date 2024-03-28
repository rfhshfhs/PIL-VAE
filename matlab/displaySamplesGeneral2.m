function displaySamplesGeneral2(X_raw,X_new,num,xpix,ypix)
%DISPLAYSAMPLES 此处显示有关此函数的摘要
%   此处显示详细说明
figure;
idx =  randi([1, size(X_raw,2)], [1,num]);
X_raw = X_raw(:,idx);
X_new = X_new(:,idx);
for i = 1:num
    subplot(2, num, i);
    imshow(reshape(X_raw(:,i), xpix, ypix));
    subplot(2, num, i+num);
    imshow(reshape(X_new(:,i), xpix, ypix));
end
end
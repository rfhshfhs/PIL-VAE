function displaySamplesGeneral(X,num,storeName,xpix,ypix)
%DISPLAYSAMPLES 此处显示有关此函数的摘要
%   此处显示详细说明
figure;
idx =  randi([1, size(X,2)], [1,num]);
X = X(:,idx);
for i = 1:num
    subplot(1, num, i);
    
    imshow(reshape(X(:,i), xpix, ypix),[0,0.8]);
%     imwrite(reshape(X(:,i), xpix, ypix),'image/'+string(storeName)+string(i)+'.png');
end
end


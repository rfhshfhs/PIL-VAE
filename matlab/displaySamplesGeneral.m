function displaySamplesGeneral(X,num,storeName,xpix,ypix)
%DISPLAYSAMPLES �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
figure;
idx =  randi([1, size(X,2)], [1,num]);
X = X(:,idx);
for i = 1:num
    subplot(1, num, i);
    
    imshow(reshape(X(:,i), xpix, ypix),[0,0.8]);
%     imwrite(reshape(X(:,i), xpix, ypix),'image/'+string(storeName)+string(i)+'.png');
end
end


%复现成功
close all;clear;clc
% sparse representation—based classification
tic;

% 1 - matlab_lasso
% 2 - fista
% 3 - pinv
% 4 - fista + pinv
method = 2;

load('nums_person.mat');
load('nums_image');
load('face.mat');  % load face allface dataset



person_label = 4;   %1-30，
test_path =['resource\occlusion_img\yaleB29_1.pgm'];
testImg = imread(test_path);

%% 30%
%testImg  = testImg + uint8(100 * randn(size(testImg )));
%%
figure,imshow(testImg),title('Test');
testImg = imresize(testImg,[12,10],'lanczos3');
testImg = double(testImg(:));



%% L1 regression by Lasso;
switch method
    case 1
        disp("-----------matlab_lasso-----------");
        x_spare = myLASSO(dataset,testImg,nums_person*nums_image);
    case 2
        disp("-------------fista---------------");
        x0=double(zeros(nums_person*nums_image,1));
        x_spare = fista_lasso(dataset,testImg,x0);
    case 3   
        disp("-------------Unconstrained---------------");
        x_spare = pinv(dataset+10)*testImg;
    case 4
        %加入最小二乘校正
        disp("-------------fista_2---------------");
        x0=double(zeros(nums_person*nums_image,1));
        x_spare = fista_lasso(dataset,testImg,x0);
        x_spare = adjust(dataset,x_spare,testImg,nums_person*nums_image);
end
        
figure,plot(x_spare); 


%% 误差
binError = zeros(1,nums_person);
step = 30;
for i =1:nums_person
    temp_vector = zeros(nums_person*nums_image,1);
    for j = 1:nums_image
        k_person = j + step*(i-1);
        temp_vector(k_person,:) = x_spare(k_person ,:);
        binError(:,i) = norm(testImg - dataset * temp_vector);
    end
end
figure,
bar(binError,0.5);
title("reconstruct error");
[value,whoInd] = min(binError)

if(whoInd == person_label)
   disp('is true')
else
    disp('is false')
end
whoImg_path = ['resource\yaleB',num2str(whoInd),'\1.pgm'];
whoImage = imread(whoImg_path);

figure,imshow(whoImage),title('Origin'); %show the original person
%%
toc;

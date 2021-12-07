close all;clear;clc
% spare representation—based classification
tic;

% 1 - matlab_lasso
% 2 - fista
% 3 - pinv
% 4 - fista + pinv
method = 2 ;


load('nums_person.mat');
load('nums_image');
load('face.mat');  % load face allface dataset


%initial
right = 0;

for k = 1:nums_person
    for j = 1:nums_image
        %close all;
        str=['person ',num2str(k),'--->','photo ',num2str(j)];
        disp(str)
        test_path =['resource\yaleB',num2str(k),'\',num2str(2*j),'.pgm'];
        testImg = imread(test_path);
        %figure,imshow(testImg);
        testImg = imresize(testImg,[12,10],'lanczos3');
        testImg = double(testImg(:));

    %% L1 regression by Lasso;
    switch method
        case 1
            %disp("-----------matlab_lasso-----------");
            x_spare = myLASSO(dataset,testImg,nums_person*nums_image);
        case 2
            %disp("-------------fista---------------");
            x0=double(zeros(nums_person*nums_image,1));
            x_spare = fista_lasso(dataset,testImg,x0);
        case 3   
            %disp("-------------Unconstrained---------------");
            x_spare = pinv(dataset)*testImg;
        case 4
            %加入最小二乘校正
            %disp("-------------fista_2---------------");
            x0=double(zeros(nums_person*nums_image,1));
            x_spare = fista_lasso(dataset,testImg,x0);
            x_spare = adjust(dataset,x_spare,testImg,nums_person*nums_image);
    end

    % figure,plot(x_spare); 
    %% 误差
        binError = zeros(1,nums_person);
        step = 30;
        for i =1:nums_person
            temp_vector = zeros(nums_person*nums_image,1);
            for m = 1:nums_image
                k_person = m + step*(i-1);
                temp_vector(k_person,:) = x_spare(k_person ,:);
                binError(:,i) = norm(testImg - dataset * temp_vector);
            end
        end
%         figure,
%         bar(binError,0.5);
%         title("reconstruct error");
        [value,whoInd] = min(binError);

        if(whoInd == k)
            right=right+1;
            %disp('is true')
        else
            str_temp = [num2str(k),',',num2str(2*j),'---->',num2str(whoInd)];
            disp(str_temp);
            %disp('is false')
        end
    end
end

%Calculation accuracy
right_rate = right/(nums_image*nums_person)

toc;

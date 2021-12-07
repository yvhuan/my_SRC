close all;clear;clc
nums_person =30;
nums_image =30;
path_father = 'resource\yaleB';
dataset = zeros(120,nums_person*nums_image);
k = 1;

for i = 1:nums_person
    for j = 1:nums_image
        path = [path_father,num2str(i),'\',num2str(2*j-1),'.pgm'];
        temp_img = imread(path);
        img = imresize(temp_img,[12 10],'lanczos3' );
        img = img(:);    %flatten to vector
        dataset(:,k) = img;
        k = k+1;
    end
end

% normalize columns of dataset
for k=1: nums_person*nums_image
     dataset(:,k) = dataset(:,k)/norm(dataset(:,k));
end


save('nums_person.mat','nums_person');
save('nums_image.mat','nums_image');
save('face.mat','dataset');



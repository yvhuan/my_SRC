function x = adjust(A,x_sparse,B,nums)
      %A is dataset
      %B is testImage
      %nums is column of dataset
     x_temp = pinv(A(:,abs(x_sparse)>0))*B;
     j = 1;
     for i = 1:nums
         if(x_sparse(i,:) ~=0)
             x_sparse(i,:) = x_temp(j);
             j = j+1;
         end
     end
     x = x_sparse;
end
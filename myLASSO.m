%nums is column 
function [x] = myLASSO(A,B,nums)
     [XL1, FitInfo] = lasso(A,B,'CV',10);
     x_lasso= XL1(:,FitInfo.Index1SE);
     %去点负值
%      for i = 1:nums
%          if(x_lasso(i,:) <0)
%              x_lasso(i,:) = 0;
%          end
%      end
     
%      x_temp = pinv(A(:,abs(x_lasso)>0))*B;
%      j = 1;
%      for i = 1:nums
%          if(x_lasso(i,:) ~=0)
%              x_lasso(i,:) = x_temp(j);
%              j = j+1;
%          end
%      end
%      
%      for i = 1:nums
%          if(x_lasso(i,:) <0)
%              x_lasso(i,:) =0;
%          end
%      end
     x = x_lasso;
     
end


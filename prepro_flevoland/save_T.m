%% save T as T_R and T_I
dim = size(T);
newT = zeros([dim,9]);
for i = 1:dim(1)
    for j = 1:dim(2)
       ele = T(i,j,:);
       ele = ele{1};
       for n = 1:9
          newT(i,j,n) = ele(n); 
       end
    end
end  
for n = 1:9
    Ts = newT(:,:,n);
    xlswrite('T_R.xlsx',real(Ts),n);
    xlswrite('T_I.xlsx',imag(Ts),n); 
end


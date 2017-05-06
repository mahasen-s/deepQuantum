function [V,D,H] = exact_TFI(N,h_drive,h_inter,h_detune,method)
sx  = sparse([[0,1];[1,0]]);
sz  = sparse([[1,0];[0,-1]]);
I   = sparse(eye(2));
Z   = sparse(zeros(2));


% Blank H
H   = sparse(1);
for i = 1:N
    H = kron(H,Z);
end

% Build H
for i = 1:N
    
    sum_sig_x   = sparse(1);
    sum_sig_z   = sparse(1);
    sum_int_sig_z2 = sparse(1);
    for j = 1:N
        if i==j
            sum_sig_x = kron(sum_sig_x,sx);
            sum_sig_z = kron(sum_sig_z,sz);
        else
            sum_sig_x = kron(sum_sig_x,I);
            sum_sig_z = kron(sum_sig_z,I);
        end
        if j==i
            sum_int_sig_z2 = kron(sum_int_sig_z2,sz);
        elseif j==mod(i ,N)+1
            sum_int_sig_z2 = kron(sum_int_sig_z2,sz);
        else
            sum_int_sig_z2 = kron(sum_int_sig_z2,I);
        end
    end
    H       = H + h_drive*sum_sig_x + h_detune*sum_sig_z + h_inter*sum_int_sig_z2;
end

[V,D]   = eigs(H,1,method);

end



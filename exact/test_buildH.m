function test_buildH
N               = 3;
h_drive         = 0;
h_inter         = 1;
h_detune        = 0;

states          = [1,1,1;-1,-1,-1;-1,1,1;1,-1,1;1,1,-1]';
basis           = [1,0;0,1]';

n_states        = size(states,2);
states_full     = zeros(2^N,n_states);
for i=1:n_states
    new     = 1;
    for j=1:N
        if states(j,i) == 1
            ind     = 1;
        elseif states(j,i) == -1
            ind     = 2;
        end
        new = kron(new,basis(:,ind));
    end
    states_full(:,i) = new;
end
states
states_full
[V,D,H] = exact_TFI(N,h_drive,h_inter,h_detune,'sa');
H_full  = full(H)

vals    = zeros(1,n_states);
for i=1:n_states
    vals(i)     = states_full(:,i)'*H_full*states_full(:,i);
end
vals
    




end
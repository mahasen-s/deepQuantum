function test_hamiltonian
N               = 3;

states          = [1,1,1;-1,-1,-1;1,-1,-1;-1,1,-1;-1,-1,1]';
down            = [1,0];
up              = [0,1];
basis           = [down;up]';

n_states        = size(states,2);
states_full     = zeros(2^N,n_states);
for i=1:n_states
    new     = 1;
    for j=1:N
        if states(j,i) == 1 % if up 
            ind     = 2;
        elseif states(j,i) == -1 % if down
            ind     = 1;
        end
        new = kron(new,basis(:,ind));
    end
    states_full(:,i) = new;
end

% Disp states, full states, hamiltonian
states
states_full


%% Test sig_z*sig_z
h_drive = 0;
h_inter = 1;
h_detune= 0;

% Get Hamiltonian
[~,~,H] = exact_TFI(N,h_drive,h_inter,h_detune,'sa');
H       = full(H);

disp('sig_z*sig_z eigenvalues')
disp(diag(states_full'*H*states_full)')

%% Test sig_x term
h_drive = 1;
h_inter = 0;
h_detune= 0;

% Get Hamiltonian
[~,~,H] = exact_TFI(N,h_drive,h_inter,h_detune,'sa');
H       = full(H);

% Apply Hamiltonian, convert states to shorthand
H1  = H*states_full
states  = cell(1,n_states);
for i=1:n_states
    states{i}  = convertState(H1(:,i));
end
x = cat(3,states{:});
disp('sig_x terms')
permute(x,[1,3,2])



end

function T = convertState(S)
% convert 2^N element full state to NxM shorthand.
N       = length(S);
Nsites  = log(N)/log(2);
Nvecs   = sum(S);

T       = -ones(Nsites,Nvecs);
inds    = find(S);

for i=(Nsites-1):-1:0
    inds1             = inds>2^i;
    if isempty(inds1) == false
        inds(inds1)       = inds(inds1) - 2^i;
        T(i+1,find(inds1))  = 1; %up
    end
end

end

function [ matrix_octavius_mat ] = F_OCTAV_read1500( filename, varargin)
% Reading the mcc file from OCTAVIUS 729

if nargin == 1
    tmp_intv = 1;
    oct_read_intv = 3;
else
    tmp_intv = 0.5;
    oct_read_intv = 2;
end

% tmp_intv = 0.5;
% oct_read_intv = 3;

% tmp_x = [-13:tmp_intv:13];
% N_tmp_x = numel(tmp_x);

st_idx_j = [];
end_idx_j = [];
temp_mcc = textread(filename, '%s');

for j = 1:length(temp_mcc)
    if ismember(temp_mcc(j),'BEGIN_DATA')
        st_idx_j = [st_idx_j; j];
    end
    
    if ismember(temp_mcc(j),'END_DATA')
        end_idx_j = [end_idx_j; j];
    end
end

delimt_ind(:,1) = st_idx_j;
delimt_ind(:,2) = end_idx_j;
delimt_ind(:,3) = delimt_ind(:,2) -delimt_ind(:,1)-1;

N_tmp_x = length(st_idx_j);

matrix_octavius_temp = cell(N_tmp_x, max(delimt_ind(:,3)));
% matrix_octavius_cell = cell(N_tmp_x, N_tmp_x);
% Measurement data storage point
matrix_octavius_mat = zeros(N_tmp_x, N_tmp_x) - 1;
matrix_octavius_mat_tmp = zeros(N_tmp_x, N_tmp_x) - 1;

for j = 1:N_tmp_x
    for k = 1:delimt_ind(j,3)
        matrix_octavius_temp(j,k) = temp_mcc( delimt_ind(j,1) + k );
        %         matrix_octavius_temp(j,k) = temp_mcc( delimt_ind(j,1) +1: delimt_ind(j,2) -1 );
    end
end

% real number of measurement point in x direction
% (alternating in y direciton)

x_lngt = length(matrix_octavius_temp(j,:))/oct_read_intv;
if nargin == 1       
    for j = 1:2:N_tmp_x % start from -130
        for k = 1 : x_lngt
            matrix_octavius_mat_tmp(j, 2*k-1) = str2num( matrix_octavius_temp{ j, 2 + oct_read_intv*(k-1) } );
        end
    end    
    for j = 2:2:N_tmp_x % start from -125
        for k = 1 : x_lngt -1
            matrix_octavius_mat_tmp(j, 2*k)    = str2num( matrix_octavius_temp{ j, 2 + oct_read_intv*(k-1) } );
        end
    end
else
    for j = 1:N_tmp_x % merged 1500
        for k = 1:x_lngt
            matrix_octavius_mat_tmp(j, k)    = str2num( matrix_octavius_temp{ j, k*oct_read_intv} );
        end
    end
end

% start from -130
matrix_octavius_mat = flipud(matrix_octavius_mat_tmp);

end



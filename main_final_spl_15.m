warning('off');
close all; clc
addpath './tool';
addpath './metric';
addpath './data';
addpath './ulli';
noise = [0.03, 0.05, 0.07, 0.1];
shape_files = [600, 1200, 1200, 1500, 0];
%% todo
for shape = 1 % soldier
    switch shape
        case 1
            shapename = 'soldier_vox10_0';
        case 2
            shapename = 'longdress';
        case 3
            shapename = 'loot';
        case 4
            shapename = 'redandblack';
        case 5
            shapename = 'Frame_0141';
    end
    for noisetype = 4:4
        for file_e = 1:6
            file = file_e + shape_files(shape);
            %shapenam e = 'gargoyle';%dc,gargoyle,anchor,daratech,lordquas
            gt_filename = ['./data/', shapename , num2str(file), 'ds.ply'];
            % noisename = [shapename,'_gaussian_noise_',num2str(0.03)];%0.03,0.04
            n_filename = ['./data/', shapename , num2str(file), 'ds_gaussian_noise_', num2str(noise(noisetype)), '.ply'];
            n_filename_ne = ['./data/', shapename , num2str(file), 'ds_gaussian_noise_', num2str(noise(noisetype))];
            if file_e == 1
                np_filename = gt_filename; %empty file
            else
                np_filename = ['./data/', shapename , num2str(file-1), 'ds_gaussian_noise_', num2str(noise(noisetype)), '_dn15.ply'];% pre frame
            end
            
            %% read in file for ground-truth and noisy model
            pt_gt = pcread(gt_filename);
            X_gt = pt_gt.Location;
            pt_X= pcread(n_filename);
            X = pt_X.Location;
            pt_p = pcread(np_filename);
            X_p = pt_p.Location;
            pt_p = pointCloud(X_p);

            temp = meandistance(X_gt, X);
            disp(temp);
            fid = fopen('result16.txt', 'a');
            fprintf(fid, 'initial mse %s %d %.2f %.4f\r\n', shapename, file, noise(noisetype), temp);
            fclose(fid);
            
            %% parameter settings
            max_itr = 4; change_tolerance = 0.0001; % stop criteria
            % choose fps for uniform sampling
            %% change from 0.5 to 0.05
            flag_sample = 1; SAMPLING_SET = ceil(0.5*size(X,1));
            % if not, use grid sample
            gridStep = 0.7;
            % patch size, searching window size,
            pk = 30; wink = 16;
            % graph construction parameter: epsilon, patch similarity threshold
            eps1 = 0.45*ones(max_itr,1); eps1(1:2)=[1.05, 0.75]; eps1 = eps1.^2;
            threshold_d = 20;
            if noisetype ~= 2 % noise variance = 0.05
                threshold_d = 20;
            end
            % optimization parameter
            gamma=0.5; lambda1 = 30*ones(max_itr,1); lambda2=zeros(max_itr, 1);
            if (noisetype == 1)
                lambda1 = [7, 40, 70, 300];
                if file_e ~= 1
                    lambda2(1:2) = [0.4, 0];
                end
            end
            if (noisetype == 2)
                lambda1 = [1.1, 10, 17, 30];
                if file_e ~= 1
                    lambda2 = [28, 0.1 , 0, 0];
                end
            end
            if (noisetype == 3)
                lambda1 = [1, 4, 12, 50];
                if file_e ~= 1
                    lambda2 = [270, 0, 0, 0];
                end
            end
            if (noisetype == 4) % change parameter
                % lambda1 = [10, 2.1, 6, 12];
                lambda1 = [10 30 500 40]; % 10 30 300-1000-500 40-100-10
                if file_e ~= 1
                    lambda2 = [30, 0, 0, 0]; % 3000-30
                end
            end
            % lambda2 = [0 0 0 0]; % baseline
            %% todo
            winf = 10; % window to find most similar patch in adjcent frames
            %% todo end
            
            %% pixel graph (no prefiltering now)
            X_m = X;
            
            %% patch center, patch construction with size=pk, and patch graph with k=wink
            pt_pre = pointCloud(X);
            % patch center
            if flag_sample
                srf = struct('X',X(:,1),'Y',X(:,2),'Z',X(:,3));
                ifps = fps_euc(srf,SAMPLING_SET);
                pt_C = pointCloud(X(ifps,:));
            else
                pt_C = pcdownsample(pt_pre,'gridAverage',gridStep);
            end
            % patch construction
            pn = pt_C.Count; % patch number
            P = zeros(pn,pk); % patch center and the node indices in the patch
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_pre,pt_C.Location(i,:),pk);
                P(i,:) = indices;
            end
            Nf = pn*pk; % node graph size
            % patch graph
            P_win = zeros(pn, wink);
            P_winf = zeros(pn, winf);
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_C,pt_C.Location(i,:),wink);
                P_win(i,:) = indices;
            end
            N_p = pt_p.Count; % pre frame point number

            %% compute pre frame normal
            % inter_normal = compute_normal(pt_p, pk);
            inter_normal = pcnormals(pt_p);

            P_p_win = zeros(N_p, pk);
            for i = 1:N_p
                [indices,dists] = findNearestNeighbors(pt_p,pt_p.Location(i,:),pk);
                P_p_win(i,:) = indices;
            end
            P_win = P_win(:,2:end);
            
            %% Denoising
            dM = zeros(max_itr,1); dH = dM;
            X_pre = X;
            % set center
            X_c = zeros(pn,3);
            for c = 1:3
                u = X_pre(:,c);
                X_c(:,c) = mean(u(P),2);
            end
            for i = 1:pn
                [indices,dists] = findNearestNeighbors(pt_p,X_c(i,:),winf);
                P_winf(i,:) = indices;
            end
            
            for itr = 1:max_itr
                %% prepare reference frame for each patch: so as to compute height field and projection
                
%                 fid = fopen('time15.txt', 'a');
%                 t1 = clock;
                
                % shift the point wrt the center
                f = zeros(pn,pk,3); fo = f;
                for c = 1:3
                    u = X_m(:,c);
                    f(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                    u = X_pre(:,c);
                    fo(:,:,c) = u(P)-repmat(X_c(:,c),1,pk);
                end
                % normal at each center
                fn = zeros(pn,3); ptmp = zeros(pk,3);
                for i = 1:pn
                    ptmp(:,:) = f(i,:,:);
                    cov = ptmp'*ptmp;
                    [Vtmp,Dtmp] = eig(cov);
                    fn(i,:) = Vtmp(:,1); % reference plane with normal vertor stored in fn
                end
                %% todo
                P_p = zeros(pk*pn, 3);
                P_n = zeros(pk*pn, 3);
                %% end todo
                
                %% patch similarity and connection
                D_i = zeros(Nf,wink-1);D_w = zeros(Nf,wink-1); tmp_p1 = (1:pk)';
                D_wp = zeros(Nf, 1);
                % temp variable for comparing fixed and moving patches
                fixed = zeros(pk,3); fixed_n = zeros(3,1); moving = fixed;

                inter_W = zeros(pk*pn, 1);

                % intra_normal = compute_normal(pointCloud(X_pre), pk);
                intra_normal = pcnormals(pointCloud(X_pre));

                inter_order = zeros(pn,1);

                W_t = zeros(pn*pk,1);
                % W_c = zeros(length(X_pre));

                inter_F = [X_p inter_normal]; % n*6
                intra_F = [X_pre intra_normal]; % n*6

                for i = 1:pn
                    fixed(:,:) = f(i,:,:); fixed_n(:) = fn(i,:);
                    % compute height field, and projection on reference plane
                    hf = fixed*fixed_n; fixed_proj = fixed-hf*fixed_n';
                    % search the neighbors of fixed
                    wedge = P_win(i,:);

                    %% find most similar patch in pre frame
                    pt_p = pointCloud(X_p);
                    indices = P_winf(i,:);
                    Dist_min = 9999999999;

                    similar_order = 0;
                    for j = 1:length(indices)

                        % order of pre frame patch
                        indices_p = P_p_win(indices(j),:);

                        % center of pre frame patch (mean coordinates)
                        center = zeros(1,3);
                        for c=1:3
                            center(c) = mean(X_p(indices_p,c));
                        end

                        % metric for similar inter-frame patch
                        moving = X_p(indices_p,:) - repmat(center, pk, 1);
                        hm = moving*fixed_n; moving_proj = moving-hm*fixed_n';
                        xx = sum((fixed_proj.^2),2);
                        yy = sum((moving_proj.^2),2);
                        xy = fixed_proj*moving_proj';
                        DistMat = repmat(xx,1,length(yy))-2*xy+repmat(yy',length(xx),1);
                        [dmin, idx] = min(DistMat,[],2);
                        preserve_id = dmin<=threshold_d;
                        discard_id = dmin>threshold_d;
                        Dsum = 0;
                        Dist = (sum((hm(idx(preserve_id))-hf(preserve_id)).^2)+Dsum)/pk;
                        
                        if Dist < Dist_min
                            similar_order = j;
                            Dist_min = Dist;
                        end
                    end

                    % order of the inter_patch
                    inter_patch = P_p_win(indices(similar_order),:); % 1*pk

                        %inter_order(i) = indices(similar_order); % order of the similar patch center of patch i

                    %% find corresponding point
                        % patch_c = X_pre(P(i,:),:);
                    intra_patch = zeros((wink-1), pk);
                    for v = 1:(wink-1)
                        [indices,dists] = findNearestNeighbors(pt_pre,pt_C.Location(P_win(i,v),:),pk);
                        intra_patch(v,:) = indices;
                    end
                    % return order
                    [new_inter_patch, new_intra_patch] = findCorrespondingPoint_eu(P(i,:), inter_patch, intra_patch, pt_p, pt_pre);

                    %% graph learning

                    % inter patch
                    patch_s = intra_F(P(i,:),:);
                    patch_t = inter_F(new_inter_patch,:);
                    % [M,~] = proximal_gradient_descent(patch_s, patch_t, true); % use normal
                    
                    for j = 1:length(wedge)
                    	patch_t = [patch_t; intra_F(new_intra_patch(j,:),:)]; % inter + intra
                    end

%                     ftid = fopen('iter15.txt', 'a');
%                     fprintf(ftid, 'total_itr: %d      i: %d\r\n', itr, i);
%                     fclose(ftid);
                    
                    patch_s = repmat(patch_s, (length(wedge)+1), 1);
                    [R,weights, M] = proximal_gradient_descent3(patch_s, patch_t, true); % use normal
                    % [R,weights, M] = proximal_gradient_descent_compare(patch_s, patch_t, true); % compare

                    for x = 1:pk

                        % W_t(pk*(i-1)+x) = exp(-diff' * M * diff);
                        W_t(pk*(i-1)+x) = weights(x);
                    end

                    % intra patch
                    for j = 1:length(wedge) % wedge = P_win(i,:);
                        % patch_r = intra_F(new_intra_patch(j,:),:);
                        % [M,~] = proximal_gradient_descent(patch_s, patch_r, true); % use normal
                        for x = 1:pk

                            % Dist = exp(-diff' * M * diff);
%                             Dist = exp(-sum((diff * M) .* diff, 2));
                            D_w(pk*(i-1)+x,j) = weights(j*pk+x);
                            D_i(pk*(i-1)+x,j) = pk*(wedge(j)-1)+x; % new_intra_patch is in the order of patch_c
                        end
                    end
                end

                %% weight construction
                               
                % compute L
                tmp = repmat((1:Nf)',1,wink-1);
                D_p = sparse(tmp(:),D_i(:),D_w(:),Nf,Nf);
                A = D_p;
                if noisetype == 1
                    A(A>5) = 0;
                else
                    if noisetype == 3
                        A(A>10) = 0;
                    elseif noisetype == 4
                        A(A>30) = 0;
                    end
                end
                mask1 = logical(tril(A));
                A = A-mask1'.*A;
                A = A+A';
                A(A>0)=exp(-A(A>0)./(2*eps1(itr)));
                weight_rec=diag(sum(A).^(-gamma));
                A=weight_rec*A*weight_rec;
                Dn=sum(A);
                Dn=diag(Dn);
                L=Dn-A;                

                % W = exp(-(fi-fj)' * M * (fi-fj));
                % compute L
                % D = diag(sum(W_c,2));
                % L = D - W_c;

                % compute W_t
                    %W_t = diag(W_t);
                    % D_pp = W_t' * W_t;
                W_t = sparse((1:Nf)',(1:Nf),W_t,Nf,Nf);
                D_pp = W_t;

                %% optimization
                % U - S_t
                % D_pp - W' * W
                X_rec = zeros(length(X_pre),3);
                tmp = P';
                U = sparse(1:Nf,tmp(:),ones(Nf,1),Nf,length(X_pre)); 
                % noisetype=4, lambda1(1)=15, mset=2.5661
                for c = 1:3
                    t = repmat(X_c(:,c),1,pk);
                    t = t'; t = t(:);
                    u0 = X_pre(:,c);
                    C = U'*L*t+lambda1(itr)*u0+lambda2(itr)*U'*D_pp*(t+P_p(:,c));
                    B = U'*L*U+lambda1(itr)*speye(length(X_pre))+lambda2(itr)*U'*D_pp*U;
                    X_rec(:,c) = lsqr(B,double(C),1e-06,10000);
                end

                %% result
                mset = meandistance(X_gt, X_rec);
                ftid = fopen('result16.txt', 'a');
                fprintf(ftid, '%s %d %.2f %d %.4f\r\n', shapename, file, noise(noisetype), itr, mset);
                fclose(ftid);
                X_pre = X_rec;
                X_m = X_pre;
                
%                 t2 = clock;
%                 fprintf(fid, '%.4f\r\n', etime(t2,t1));
%                 fclose(fid);
                write_ply_only_points(X_rec,[n_filename_ne, '_itr', num2str(itr),  '_dn15.ply']);

            end
            mdc = meandistance(X_gt, X_rec);
            write_ply_only_points(X_rec,[n_filename_ne '_dn15.ply']);
            disp(['meandistance=',num2str(mdc)]);
            fid = fopen('result16.txt', 'a');
            fprintf(fid, 'final mse %s %d %.2f %.4f\r\n', shapename, file, noise(noisetype), mdc);
            fclose(fid);
        end
    end
end

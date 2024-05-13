%% Main
% Hajer J
% Document topic discovery via k-means algorithm.

close all;  clear;  clc;

load("wikipedia_m.mat");

k = 8;  % number of clusters/groups
X = tdmatrix; % Data points matrix
N = size(X, 2); % number of points
n = size(X, 1); % dimension of points
run_time = 13; % run times
it_max=zeros(20,1);


% init history variables
J_history = zeros(1, run_time);
iterations_history = zeros(1, run_time);
group_history = zeros(run_time, N);

for r = 1:run_time
    % create an initial group assignment
    group = randi(k, 1, N); % vector of N entries!

    J_clust = zeros(1, k); % init objective function for each cluster
    Jprev = 0; % init "previous" objective function

    iterations = 0; % count for reference
    stop = 0; % init stopping criterion

    dist_xi = zeros(1, k); % init distance to each centroid
    Pj = zeros(1, k); % extra / ignore
    j_it=[0];

    while stop == 0
        % init k centroids
        z = zeros(n, k); % n-by-k matrix collection of all the centroids

        % Update centroids and objective function for each cluster
        for j = 1:k

        % indices of the points assigned to group j
        Gj = find(group == j); %vector, size pj
        pj = length(Gj); %how many points in group j;
        
        Pj(j) = pj; % extra / ignore

        %init distance between centroid and each point in the cluster
        dist_Xj = zeros(1, pj); %vector size pj
        
        % all points assigned to group j
        Xj = X(:, Gj) ; %submatrix n-by-pj
        
        %------ update centroid!!-----------%
        zj = mean(Xj, 2); % mean of all points in the group
        z(:, j) = zj; %n-vector

        % Matrix of the distances between each of the points in 
        % cluster j and their representative
        A = Xj - zj; %matrix size n-by-pj
        for p = 1:pj
        %this vector contains pj distances 
        dist_Xj(p) = sum(A(:, p).^2); %squared norm of each col      
        end
        J_clust(j)  = 1/N*sum(dist_Xj); % update objective for cluster j
        end

    %---------stopping criterion=----------% 
    % evaluate the quality of the clustering
        J = sum(J_clust);
        if abs((J - Jprev) / J) < 1e-8
            stop = 1;
        end
        Jprev = J; % update objective
        j_it=[j_it Jprev];

    % -------------update partition------%
    for i = 1:N  % for all data points 
        for j = 1:k % for all centroids
            xi = X(:, i);
            zj = z(:, j);
            dist_xi(j) = norm(xi - zj)^2; % distance between xi & zj
        end        
        [~, j] = min(dist_xi);  % index of closest centroid 
        group(i) = j; % assign point i to cluster j
    end
    iterations = iterations + 1;
    end

    % Store results for the current run
    J_history(r) = J;
    iterations_history(r) = iterations;
    group_history(r, :) = group;
end

%% Reporting Final Results!!
%-------------------------------------------------------------------------%


% Find the best run
[~, best_r] = min(J_history);
for j = 1:k
    % Extract cluster representatives
    zj = mean(X(:, group_history(best_r, :) == j), 2);
    % Sort indices of cluster representatives
    [~, I_top_words] = sort(zj, 'descend');
    Top_Words = dictionary(I_top_words(1:5));
    % Find closest articles
    Gj = find(group_history(best_r, :) == j);
    [~, I_closest_points] = sort(sum((X(:, Gj) - zj).^2, 1), 'ascend');
    Top_Articles = articles(I_closest_points(1:5));
    % Print results
    fprintf('Cluster %d: \n', j);
    fprintf('Five top terms\n');
    disp(Top_Words);
    fprintf('Five top articles:\n');
    disp(Top_Articles);
    fprintf('---------------- \n');
    fprintf(' \n');
end

% Print results of the best run
fprintf('Results of the best run:\n');
fprintf('Objective function: %d\n', J_history(best_r));
fprintf('Number of iterations: %d\n', iterations_history(best_r));


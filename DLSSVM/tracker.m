% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: implement the dlssvm tracker                             %
% parameters:                                                        %
%      input: path of image sequences                                %
%      ext:extension name of file, for example, '.jpg'               %
%      show_img:                                                     %
%      init_rect: initial position of the target                     %
%      start_frame:                                                  %
%      end_frame:                                                    %
%      s_frames: the number of frames                                %
%                                                                    %
% ********************************************************************
%     you need configure the opencv for run this program.            %
%     The program is successfully run under opencv 2.4.8             %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = tracker(input, ext, show_img, init_rect, start_frame, end_frame, s_frames)

show_img = 1;
% Display runtime result

addpath(genpath('.'));
D = dir(fullfile(input, ['*.', ext]));
file_list = { D.name };
% Load images in the path to file_list

if nargin < 4
    init_rect = -ones(1, 4);
    % The default starting rectangle is [x=-1, y=-1, w=-1, h=-1]
end
if nargin < 5
    start_frame = 1;
    % Default starting frame
end
if nargin < 6
    end_frame = numel(file_list);
    % Default ending frame
end

global sampler
global tracker
global config
global finish

config.display = true;
% Whether to display runtime result.
% This is redundant because in "makeConfig.m", it will be set to show_img

sampler = createSampler();
% THIS IS JUST:
% sampler.radius = 1

finish = 0;
timer = 0;
result.res = nan(end_frame - start_frame + 1, 4);
% "nan()" is like "ones()", NaNs instead of 1's
% e.g., nan(1,1) is [NaN] and nan(2) is [Nan NaN; NaN NaN]
result.len = end_frame - start_frame + 1;
result.startFrame = start_frame;
result.type = 'rect';

if show_img
    figure(1); set(1,'KeyPressFcn', @handleKey);
end

output = zeros(1,4);

patterns = cell(1, 1);
% Don't know what is "cell"? See: https://www.mathworks.com/help/matlab/ref/cell.html

params = makeParams();
% THIS IS JUST:
% params.lambda = 100
% params.nBudget = 100

k = 1; % Pattern index

for frame_id = start_frame:end_frame
    if finish == 1
        break;
    end

    if ~config.display
        clc
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end

    if nargin == 7
        I_orig = imread(s_frames{frame_id-start_frame+1});
    else
        I_orig = imread(fullfile(input,file_list{frame_id}));
    end

    if frame_id == start_frame  % For the first frame
        init_rect = round(init_rect); % [x y width height]
        % Using round in case the indexes are not integers

        % Set config parameters according to the region size of the first frame
        config = makeConfig(I_orig, init_rect, true, false, true, show_img);
        % PARAMS:
        % I_orig - The first frame
        % init_rect - The selected rectangle
        % true - Whether to use color
        % false - Whether to use Experts
        % true - Whether to use IIF (Illumination Invariant Feature)
        % show_img - Whether to display runtime result

        tracker.output = init_rect * config.image_scale;
        % [x y width height] * image_scale, because the image is rescaled
        tracker.output(1:2) = tracker.output(1:2) + config.padding; % [x+padding y+padding width height]
        tracker.output_exp = tracker.output;
        output = tracker.output;
        % output = tracker.output = tracker.output_exp
    end

    % I_orig is the raw frame image
    [I_scale]= getFrame2Compute(I_orig);
    % THIS IS JUST:
    % 1. resizing I_orig with config.image_scale
    % 2. padding top, bottom, left, right of I_orig with the width of config.padding

    sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    % rsz_rt IS JUST RESIZING RECTANGLE,
    % note that output = [x y width height].
    % 1. let r = sqrt(width * height)
    % 2. x0 = x - 0.5 * config.search_roi * r
    %    y0 = y - 0.5 * config.search_roi * r
    %    x1 = x + width + 0.5 * config.search_roi * r
    %    y1 = y + height + 0.5 * config.search_roi * r
    % 3. let rect = [x0 y0 x1 y1]
    % 4. Shift rect inside if it goes outside I_scale // because "rect" is bigger then "output"
    % 5. sampler.roi = rect

    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    % crop a rectangle from I_scale by the [x0 y0 x1 y1] above

    [BC, F] = getFeatureRep(I_crop, config.hist_nbin);
    % These are the features used in MEEM, you shoud see the other paper called
    % "MEEM: Robust Tracking via Multiple Experts using Entropy Minimization"
    % I guess, it transforms RGB into Lab(1) and LRT(2) channel:
    % (1) Lab color space: Lightness, green-red and blue-yellow (color-opponents),
    %                      and it is designed to approximate human vision.
    % (2) Local Rank Transform: I don't know, you can Google it yourself.
    % I noticed that only BC will be used in the rest of the code

    tic

    if frame_id == start_frame
        initSampler(tracker.output, BC, F, config.use_color);
        % Here you should refer to "initSampler.m", it is very important.
        % Although F is passed in the function, it is not used at all, just ignore it.

        patterns{1}.X = sampler.patterns_dt;
        % "sampler.patterns_dt" is calculated in "resample.m"
        %% patterns{1}.X is denoted by phi_i(y) = phi(x_i,y_i) - phi(x_i,y) in the next line

        patterns{1}.X = repmat(patterns{1}.X(1, :), size(patterns{1}.X, 1), 1) - patterns{1}.X;
        % patterns{1}.X(1, :) is tracker.output_feat
        % patterns{1}.X = [
        %     tracker.output_feat - tracker.output_feat  ;
        %     [ tracker.output_feat - im2colstep(1, ...) ; // this is delta_Psi_i(y_1)
        %       tracker.output_feat - im2colstep(2, ...) ; // this is delta_Psi_i(y_2)
        %       ...                                        ] // #candidates
        % ]

        patterns{1}.Y = sampler.state_dt;
        % patterns{1}.Y = [
        %     x y width height ; // tracker.output
        %     [
        %       x_1 y_1 width height ;
        %       x_2 y_2 width height ;
        %       ...
        %     ] // #candidates
        % ]

        patterns{1}.lossY = sampler.costs;    % loss function: L(y_i,y), computed in resample
        patterns{1}.supportVectorNum = [];    % save structured output index whose alpha is not zero
        patterns{1}.supportVectorAlpha = [];  % save dual variable
        patterns{1}.supportVectorWeight = []; % save weight related to dual variable

        w0 = zeros(1, size(patterns{1}.X, 2));  % initilize the classifer w0

        %% Training classifier w0 by the proposed dlssvm optimization method
        [w0, patterns] = dlssvmOptimization(patterns, params, w0);
        % Read paper yourself
        % w0' * Psi(x, y) will give a score that how much x is to y

        if config.display
            figure(1);
            imshow(I_orig);
            res = tracker.output;
            res(1:2) = res(1:2) - config.padding;
            res = res / config.image_scale;
            rectangle('position',res,'LineWidth',2,'EdgeColor','b')
            % Show tracker.output on the frame
        end
    else
        if config.display
            figure(1)
            imshow(I_orig);
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
            % Recall: sampler.roi [x0 y0 x1 y1]
            % is the shifted resized padded tracker.output
            roi_reg(1:2) = roi_reg(1:2) - config.padding;
            rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
            % Show sampler.roi on the image
        end

        feature_map = imresize(BC, config.ratio,'nearest');
        % Get the feature map of candiadtes
        % Recall: BC is the MEEM features computed from sampler.roi cropped from the frame

        ratio_x = size(BC, 2) / size(feature_map, 2);
        ratio_y = size(BC, 1) / size(feature_map, 1);

        detX = im2colstep(feature_map, ...
                          [sampler.template_size(1:2), size(BC,3)], ...
                          [1, 1, size(BC,3)] );
        % detX: candidates in the search region 

        x_sz = size(feature_map, 2) - sampler.template_size(2) + 1;
        y_sz = size(feature_map, 1) - sampler.template_size(1) + 1;
        [X, Y] = meshgrid(1:x_sz,1:y_sz);
        detY = repmat(tracker.output, [numel(X), 1]);
        detY(:,1) = (X(:)-1) * ratio_x + sampler.roi(1);
        detY(:,2) = (Y(:)-1) * ratio_y + sampler.roi(2);

        % detect the object
        % detX is feature(Lab+LIF+Explicit feature map), w0 is linear classifer
        % because we use linear w0, we can evaluate the candidate region by simple dot product
        score = w0 * detX; % w0' * Psi(x) is the score that x fits the last output,
                           % because w0 is trained by the last dataset: delta_Psi_i(candidates)
        [~, maxInd] = max(score);

        % detect the target position by maximal response
        output = detY(maxInd, :);  % find the best y
        % end detecting the object

        if config.display
            figure(1)
            res = output;
            res(1:2) = res(1:2) - config.padding;
            res = res/config.image_scale;
            % display the object
            rectangle('position',res,'LineWidth',2,'EdgeColor','b');
        end

        step = round(sqrt((y_sz*x_sz)/120));
        mask_temp = zeros(y_sz,x_sz);
        mask_temp(1:step:end, 1:step:end) = 1;
        mask_temp = mask_temp > 0;
        mask_temp(maxInd) = 0;
        k = k + 1;

        % construct the training set from the current tracking results.
        % detX(:, maxInd) is the tracking result, and its loss is 0.
        patterns{k}.X = [detX(:, maxInd)'; detX(:, mask_temp(:))'];
        patterns{k}.X = repmat(patterns{k}.X(1, :), size(patterns{k}.X, 1), 1) - patterns{k}.X;
        patterns{k}.Y = [ detY(maxInd, :); detY(mask_temp(:), :) ];
        patterns{k}.lossY = 1 - getIOU(patterns{k}.Y, output);       % loss function for all Y
        patterns{k}.supportVectorNum = [];
        patterns{k}.supportVectorAlpha = [];
        patterns{k}.supportVectorWeight = [];
        [w0, patterns] = dlssvmOptimization(patterns, params, w0);   % train w0 with previous frames
        k = size(patterns, 2);
    end

    timer = timer + toc;
    res = output;
    res(1:2) = res(1:2) - config.padding;
    result.res(frame_id - start_frame+1,:) = res / config.image_scale;
end

result.fps = result.len / timer;

clearvars -global sampler tracker config finish

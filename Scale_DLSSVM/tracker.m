function result = tracker(input, ext, show_img, init_rect, start_frame, end_frame, s_frames)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function: implement the dlssvm tracker (Scale-DLSSVM)              %
% parameters:                                                        %
%      input: path of image sequences                                %
%      ext:extension name of file, for example, '.jpg'               %
%      show_img:                                                     %
%      init_rect: initial position of the target                     %
%      start_frame:                                                  %
%      end_frame:                                                    %
%      s_frames: the number of frames                                %
%  Note: According to our test,there may be slightly different       %
%  on different MATLAB environment due to using fft transformation.  % 
%                                                                    %
% ********************************************************************
%     you need configure the opencv for run this program.            %
%     The program is successfully run under opencv 2.4.8             %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('.'));

D = dir(fullfile(input,['*.', ext]));
file_list={D.name};

if nargin < 4
    init_rect = -ones(1,4);
end
if nargin < 5
    start_frame = 1;
end
if nargin < 6
    end_frame = numel(file_list);
end

global sampler
global tracker
global config
global finish

config.display = true;
sampler = createSampler();
finish = 0;

timer = 0;
result.res = nan(end_frame-start_frame+1,4);
result.len = end_frame-start_frame+1;
result.startFrame = start_frame;
result.type = 'rect';

if show_img
    figure(1); set(1,'KeyPressFcn', @handleKey); 
end

output = zeros(1,4);

patterns = cell(1, 1);
params = makeParams();  
k = 1;

if init_rect(3)<=25 || init_rect(4)<=25
    search_size=1;
else   
    search_size = [1 0.995 1.005];
end
res = init_rect;
for frame_id = start_frame:end_frame
    if finish == 1
        break;
    end

    if ~config.display
        clc
        display('Scale_DLSSVM');
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end
    
    if nargin == 7
        I_orig=imread(s_frames{frame_id-start_frame+1});
    else
        I_orig=imread(fullfile(input,file_list{frame_id}));
    end
    
    if frame_id==start_frame
        init_rect = round(init_rect);  
        %% makeConfig is slightly different with fixed dlssvm in the parameter of config.padding
        config = makeConfig(I_orig,init_rect,true,false,true,show_img);  
        tracker.output = init_rect*config.image_scale;
        tracker.output(1:2) = tracker.output(1:2) + config.padding;
        tracker.output_exp = tracker.output;        
        output = tracker.output;
    end
    
    [I_scale]= getFrame2Compute(I_orig);
    sampler.roi = rsz_rt(output,size(I_scale),config.search_roi,true);
    
    tic
    window_sz = [sampler.roi(4) - sampler.roi(2)+1, sampler.roi(3) - sampler.roi(1)+1];
    if (frame_id==start_frame) 
        window_sz_old =window_sz; 
    end
    
    % Three parameters for multi-scale estimations
    detX = [];
    detY = [];
    score = [];   

    for i=1:size(search_size,2) 
        tmp_sz = window_sz*search_size(i);
        pos = [(sampler.roi(1)+sampler.roi(3))/2,(sampler.roi(2)+sampler.roi(4))/2];
        param0 = [pos(1), pos(2), tmp_sz(2)/window_sz_old(2), 0,...
            tmp_sz(1)/window_sz_old(2)/(window_sz_old(1)/window_sz_old(2)),0];
        param0 = affparam2mat(param0);
        I_crop = uint8(warpimg(double(I_scale), param0, window_sz_old));
        
        % we employ the the feature used by MEEM (Jianming, Zhang et al, ECCV2014)to represent the object
        [BC, F] = getFeatureRep(I_crop,config.hist_nbin);
        
        % for the first frame, don't need to estimate scale
        if frame_id==start_frame && i == 1
            initSampler(tracker.output,BC,F,config.use_color);
            patterns{1}.X = sampler.patterns_dt;
            patterns{1}.X = repmat(patterns{1}.X(1, :), size(patterns{1}.X, 1),1) - patterns{1}.X;
            patterns{1}.Y = sampler.state_dt;
            patterns{1}.lossY = sampler.costs;
            patterns{1}.supportVectorNum=[];
            patterns{1}.supportVectorAlpha=[];
            patterns{1}.supportVectorWeight=[];
            w0 = zeros(1, size(patterns{1}.X, 2));
            [w0, patterns]=dlssvmOptimization(patterns,params, w0);
            
            if config.display
                figure(1);
                imshow(I_orig);
                res = tracker.output;
                res(1:2) = res(1:2) - config.padding;
                res = res/config.image_scale;
                rectangle('position',res,'LineWidth',2,'EdgeColor','b')
                title([num2str(frame_id-start_frame+1),'/',num2str(end_frame-start_frame+1)]);
            end
            break;
        else  % for the other frame, we need to estimate scale
            if config.display
                figure(1)
                imshow(I_orig);
                roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2)+1;
                roi_reg(1:2) = roi_reg(1:2) - config.padding;
                rectangle('position',roi_reg/config.image_scale,'LineWidth',1,'EdgeColor','r');
                title([num2str(frame_id-start_frame+1),'/',num2str(end_frame-start_frame+1)]);
            end
            
            % calculate the feature under the different scales
            feature_map(:, :, :, i) = imresize(BC,config.ratio,'nearest');
            
            % we use fast Fourier transform to speed up to estimate scale
            [h1,w1,~]=size(feature_map(:, :, :, i));
            % reshape w to matrix to use fft
            w = reshape(w0, sampler.template_size(1), sampler.template_size(2), sampler.template_size(3));
            [h2,w2, ~]=size(w);
            fa = fft2(feature_map(:, :, :, i),h1+h2-1,w1+w2-1);
            fb = fft2(w,h1+h2-1,w1+w2-1);
            yf=conj(fb).*fa;
            y=ifft2(yf);
            y=y(1:end-(2*h2)+2,1:end-(2*w2)+2,:);
            score=real(sum(y,3));
            
            % maximal score under the current scale
            maxScore(i) = max(score(:));
            rows = [];
            cols = [];
            [rows, cols] = find(score == maxScore(i));
            row(i) = rows(1);
            col(i) = cols(1);
            
            target_sz = output(3:4) * search_size(i);
            topLeft = [pos(1) - tmp_sz(2)/2, pos(2) - tmp_sz(1)/2];
            ratio_x = size(BC,2)/size(feature_map(:, :, :, i),2);
            ratio_y = size(BC,1)/size(feature_map(:, :, :, i),1);
            outputs(i, :) = [(col(i)-1)*ratio_x*(tmp_sz(2)/window_sz_old(2)) + topLeft(1), ...
                (row(i)-1)*ratio_y*(tmp_sz(1)/window_sz_old(1)) + topLeft(2), ...
                target_sz];
            
            if i == size(search_size,2)      % estimate true scale
                [~, layer] = max(maxScore);  % the layer corresponding to maxScore
                output = outputs(layer, :);  % the estimated position in the layer

                if config.display
                    figure(1)
                    res = output;
                    res(1:2) = res(1:2) - config.padding;
                    res = res/config.image_scale;
                    rectangle('position',res,'LineWidth',2,'EdgeColor','b')
                end
                % extract the training sample from the estimated scale
                feature_map = feature_map(:, :, :, layer);
                ratio_x = size(BC,2)/size(feature_map,2);
                ratio_y = size(BC,1)/size(feature_map,1);
                x_sz = size(feature_map,2)-sampler.template_size(2)+1;
                y_sz = size(feature_map,1)-sampler.template_size(1)+1;
                [X, Y] = meshgrid(1:x_sz,1:y_sz);
                step = round(sqrt((y_sz*x_sz)/120));
                detX = im2colstep(feature_map,[sampler.template_size(1:2), size(BC,3)],[step, step, size(BC,3)]);
                mask_temp = zeros(y_sz,x_sz);
                mask_temp(1:step:end,1:step:end) = 1;
                mask_temp = mask_temp > 0;
                X = X(mask_temp);
                Y = Y(mask_temp);
                detY = repmat(output,[numel(X),1]);
                tmp_sz = window_sz*search_size(layer);
                topLeft = [pos(1) - tmp_sz(2)/2, pos(2) - tmp_sz(1)/2];
                detY(:,1) = (X(:)-1)*ratio_x*(tmp_sz(2)/window_sz_old(2)) + topLeft(1);
                detY(:,2) = (Y(:)-1)*ratio_y*(tmp_sz(1)/window_sz_old(1)) + topLeft(2);
                k = k+1;
                target_feature = feature_map (round(row(layer):row(layer)+sampler.template_size(1)-1),...
                    round(col(layer):col(layer)+sampler.template_size(2)-1),:);
                patterns{k}.X = [target_feature(:)'; detX'];
                patterns{k}.X = repmat(patterns{k}.X(1, :), size(patterns{k}.X, 1),1) - patterns{k}.X;
                patterns{k}.Y = [output; detY];
                patterns{k}.lossY = 1 - getIOU(patterns{k}.Y,output);
                patterns{k}.supportVectorNum=[];
                patterns{k}.supportVectorAlpha=[];
                patterns{k}.supportVectorWeight=[];
                [w0,patterns]=dlssvmOptimization(patterns,params, w0);
                k=size(patterns,2);
                
                timer = timer + toc;
                res = output;
                res(1:2) = res(1:2) - config.padding;
                res = res/config.image_scale;
            end
        end
    end
    result.res(frame_id-start_frame+1,:) = res;
end

result.fps = result.len/timer;
clearvars -global sampler tracker config finish 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Implemetation of the tracker described in paper
%	"MEEM: Robust Tracking via Multiple Experts using Entropy Minimization",
%   Jianming Zhang, Shugao Ma, Stan Sclaroff, ECCV, 2014
%
%	Copyright (C) 2014 Jianming Zhang
%
%	This program is free software: you can redistribute it and/or modify
%	it under the terms of the GNU General Public License as published by
%	the Free Software Foundation, either version 3 of the License, or
%	(at your option) any later version.
%
%	This program is distributed in the hope that it will be useful,
%	but WITHOUT ANY WARRANTY; without even the implied warranty of
%	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%	GNU General Public License for more details.
%
%	You should have received a copy of the GNU General Public License
%	along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%	If you have problems about this software, please contact: jmzhang@bu.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function resample(I_vf, step_size)
% "initSampler.m" doesn't pass step_size to resample()
% I_vf - BC (MEEM's features), region cropped from I_scale by sampler.roi

global sampler;
global tracker;
global config;

rect = tracker.output;

upleft = round([rect(1)-sampler.roi(1)+1,rect(2)-sampler.roi(2)+1]);
% The up-left boundary of tracker.output in BC

if ~((upleft(1)<1) || (upleft(2)<1) || (round(upleft(1)+rect(3)-1)>size(I_vf,2)) || (round(upleft(2)+rect(4)-1)>size(I_vf,1)))
    % If window is inside BC (MEEM's features)

    sub_win = I_vf(round(upleft(2):(upleft(2)+rect(4)-1)),round(upleft(1): (upleft(1)+rect(3)-1)),:);
    % Crop the region of "tracker.output" at the posotion of "upleft" on BC

    output_feat = imresize(sub_win, config.template_sz);
    tracker.output_feat = output_feat(:)'; % 1 x N
    % Resize and vectorize that region as tracker.output_feat

else
    warning('tracking window outside of frame');
    keyboard
end

step_size = max(round(min(sampler.template_size(1:2))/4),1);
step_size = step_size([1 1]); % step_size = [step_size(1) step_size(1)]
% The "max(round(min(.)/4), 1)" here is in case round(min/4) = 0, I guess

feature_map = imresize(I_vf, config.ratio, 'nearest');
% Resize BC (MEEM's features) using "Nearest-Neighbor"
% config.ratio = sqrt(template_max_numel/win_area); at "makeConfig.m:86"

sampler.patterns_dt = [tracker.output_feat; ...
    im2colstep( feature_map, sampler.template_size, [step_size, size(I_vf, 3)] )'];
% sampler.patterns_dt = [
%     tracker.output_feat;  // vectorized resized region cropped from BC
%     im2colstep(.)'        // vectorized candidates cropped from resized BC, but why resizing?
% ] // (1 + #candidates) x N
% This will be the X

temp = repmat(rect, [size(sampler.patterns_dt, 1), 1]);
% temp = [
%     x y width height ;
%     [ x y width height ; 
%       ...                ] // #candidates'
% ]

[X, Y] = meshgrid(1 : step_size(2) : size(feature_map,2) - sampler.template_size(2) + 1, ...
    1 : step_size(1) : size(feature_map,1) - sampler.template_size(1) + 1);

temp(2:end, 1) = (X(:)-1) / config.ratio + sampler.roi(1);
temp(2:end, 2) = (Y(:)-1) / config.ratio + sampler.roi(2);

%% compute cost table
left = max( round(temp(:,1)), round(rect(1)) );
top = max( round(temp(:,2)), round(rect(2)) );
right = min( round(temp(:,1)+temp(:, 3)), round(rect(1)+rect(3)) );
bottom = min( round(temp(:,2)+temp(:, 4)), round(rect(2)+rect(4)) );
ovlp = max(right - left,0).*max(bottom - top, 0); % ovlp is overlap

sampler.costs = 1 - ovlp ./ (2*rect(3)*rect(4) - ovlp);
% This is the loss function L(y, yi) in the paper
% sampler.costs = [
%     L(tracker.output, tracker.output) ; // 0
%     [ 
%       L(tracker.output, y_1) ;
%       L(tracker.output, y_2) ;
%       ...
%     ] // #candidates
% ]

sampler.state_dt = temp;
% sampler.state_dt = [
%     x y width height ;
%     [
%       x_1 y_1 width height ;
%       x_2 y_2 width height ;
%       ...
%     ] // #candidates
% ]

end

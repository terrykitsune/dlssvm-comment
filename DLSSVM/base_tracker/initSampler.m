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

function initSampler(init_rect, I_vf, I, use_color)
% init_rect - tracker.output ([x y width height])
% I_vf      - BC (MEEM's features), region cropped from I_scale by sampler.roi
% I         - This is redundant, just ignore it
% use_color - Whether to use RGB to compute features or just gray scale

global config
global sampler;

init_rect_roi = init_rect;

init_rect_roi(1:2) = init_rect(1:2) - sampler.roi(1:2)+1;
% sampler.roi is set in "tracker.m" by rsz_rt(),
% which is the resized and shifted tracker.output, [x0 y0 x1 y1].
% init_rect_roi will be [x-x0+1, y-y0+1, width, height]

template = I_vf( round( init_rect_roi(2) : init_rect_roi(2)+init_rect_roi(4)-1 ), ...
    round(init_rect_roi(1):init_rect_roi(1)+init_rect_roi(3)-1),:);
% Crop the region of init_rect_roi from BC (MEEM's features) as template
% NOTE: This is the region of tracker.output on BC
% "template" is the region of tracker.output cropped from BC!

sampler.template = imresize(template, config.template_sz);
sampler.template_size = size(sampler.template);
sampler.template = sampler.template(:)'; % "sampler.template" is vectorized resized "template"
sampler.template_width = init_rect(3);   % width of "tracker.output"
sampler.template_height = init_rect(4);  % height of "tracker.output"

if use_color
    sampler.feature_num = 4;
else
    sampler.feature_num = 2;
end


%% for collecting initial training data
resample(I_vf);

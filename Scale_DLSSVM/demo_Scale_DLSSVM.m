% ********************************************************************
% you need configure the opencv for run this program.                %
% The program is successfully run under opencv 2.4.8                 %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
videoDir='G:\tracker_benchmark_v1.0\';
res = tracker('G:\tracker_benchmark_v1.0\data\basketball\img','jpg',true,[198,214,34,81]);
res = tracker([videoDir,'data\Singer1\img'],'jpg',true,[51 53 87 290]);
res = tracker([videoDir,'data\car4\img'],'jpg',true,[70 51 107 87]);
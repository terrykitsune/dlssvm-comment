function results=run_Scale_DLSSVM(seq, res_path, bSaveImage)

results=tracker(seq.path, seq.ext, false, seq.init_rect, seq.startFrame, seq.endFrame, seq.s_frames);

disp(['fps: ' num2str(results.fps)])

results.type = 'rect';

end
function [trans_img,trans_facial5point]=face_align_512(impath,facial5point,savedir)
% align the faces by similarity transformation.
% using 5 facial landmarks: 2 eyes, nose, 2 mouth corners.
%   impath: path to image
%   facial5point: 5x2 size, 5 facial landmark positions, detected by MTCNN
%   savedir: savedir for cropped image and transformed facial landmarks

%% alignment settings
imgSize = [512,512];
coord5point = [180,230;
    300,230;
    240,301;
    186,365.6;
    294,365.6];%480x480
coord5point = (coord5point-240)/560 * 512 + 256;

%% face alignment

% load and align, resize image to imgSize
img      = imread(impath);
facial5point = double(facial5point);
transf   = cp2tform(facial5point, coord5point, 'similarity');
trans_img  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                    'YData', [1 imgSize(1)],...
                                    'Size', imgSize,...
                                    'FillValues', [255;255;255]);
trans_facial5point = round(tformfwd(transf,facial5point));


%% save results
if ~exist(savedir,'dir')
    mkdir(savedir)
end
[~,name,~] = fileparts(impath);
% save trans_img
imwrite(trans_img, fullfile(savedir,[name,'_aligned.png']));
fprintf('write aligned image to %s\n',fullfile(savedir,[name,'_aligned.png']));
% save trans_facial5point
write_5pt(fullfile(savedir, [name, '_aligned.txt']), trans_facial5point);
fprintf('write transformed facial landmark to %s\n',fullfile(savedir,[name,'_aligned.txt']));

%% show results
imshow(trans_img); hold on;
plot(trans_facial5point(:,1),trans_facial5point(:,2),'b');
plot(trans_facial5point(:,1),trans_facial5point(:,2),'r+');

end

function [] = write_5pt(fn, trans_pt)
fid = fopen(fn, 'w');
for i = 1:5
    fprintf(fid, '%d %d\n', trans_pt(i,1), trans_pt(i,2));%will be read as np.int32
end
fclose(fid);
end
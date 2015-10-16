% ---------------------------------------  
%  Setup array of numFrames images
% ---------------------------------------
% implay uses 20fps by default
FPS = 60;
% time for one loop to complete
speed = 1;
% determine statically the number of frames
% for constant speed and constant fps
numFrames = ceil(FPS*speed);
canvas = 400;
images = uint8(ones(canvas,canvas,3,numFrames)); 
A = imread('pudding.png', 'png','BackgroundColor', [1 1 1]);

% ---------------------------------------
%  Transformation
% ---------------------------------------
for i = 1:numFrames
    shy = 0;
    % Finish one rotation in these frames
    shx = 0.2*cos(2*pi * i/numFrames);
    xform=[ 1   shy 0
           shx 1   0 
           0   0   1 ];
    af = affine2d(xform);
    [B, RB] = imwarp(A, af, 'FillValues', [255]);% white bg
    sizeX = size(B,2);
    start = canvas/2 - 128;

    % shear to the right! we need to move the starting point!
    if RB.XWorldLimits(2) > 257
        start = start - RB.XWorldLimits(2) + 257;
    end
    start = ceil(start);

    % old implementation
    %images(:,:,:,i) = images(:,:,:,i)*255; % fill rest of bg with white
    %images(canvas/2-128:canvas/2+127,start:sizeX+start-1,:,i) = B;
    
    % new implementation with use of padarray
    I = padarray(B,[canvas/2-128 start 0], 255, 'pre');
    I = padarray(I,[canvas-size(I,1) canvas-size(I,2) 0], 255, 'post');
    images(:,:,:,i) = I;
end

% ---------------------------------------
%  Displaying
% ---------------------------------------
implay(images, FPS);
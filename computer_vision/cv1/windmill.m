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
A = imread('windmill.png', 'png','BackgroundColor', [1 1 1]);
maskA = imread('windmill_mask.png', 'png','BackgroundColor', [1 1 1]);
bg = imread('windmill_back.jpeg', 'jpeg');
images = uint8(zeros(size(bg,1),size(bg,2),3,numFrames));
interpol = 'cubic'; % nearest/linear

% ---------------------------------------
%  Transformation
% ---------------------------------------
A = A-maskA;
for i = 1:numFrames
    % ------ Scale the windmill
    sx = 1;
    sy = 1;
    sform=[  sx  0 0
             0  sy 0 
             0   0 1 ];
    sf = affine2d(sform);
    [B, RB] = imwarp(A, sf, 'FillValues', 0, 'Interp', interpol);    % black bg
    maskB = imwarp(maskA, sf, 'FillValues', 255, 'Interp', interpol);% white bg
      
    % ------ Finish one rotation in these frames
    %q = pi/4 * (i-2);
    q = -2*pi * (i/numFrames);
    xform=[  cos(q)  sin(q) 0
            -sin(q)  cos(q) 0 
               0      0     1 ];
    af = affine2d(xform);
    [C, RC] = imwarp(B, RB, af, 'FillValues', 0, 'Interp', interpol);% white bg
    maskC = imwarp(maskB, RB, af, 'FillValues', 255, 'Interp', interpol);% white bg
    
    % ------ Perform translation of rotated image on the background
    tx = size(bg,1)/2 - size(C,1)/2;
    ty = size(bg,2)/2 - size(C,2)/2;
    tform=[  1    0  0
             0    1  0 
             tx  ty  1 ];
    tf = affine2d(tform);
    [D, RD] = imwarp(C, tf, 'FillValues', 0, 'Interp', interpol);
    maskD = imwarp(maskC, tf, 'FillValues', 255, 'Interp', interpol);
    
    % new implementation with use of padarray 
    % (without cropping if windmill bigger than the bg)
    padX = ceil(RD.XWorldLimits(1));
    padY = ceil(RD.YWorldLimits(1));
    I = padarray(D,[padX padY 0], 0, 'pre');
    maskI = padarray(maskD,[padX padY 0], 255, 'pre'); 
          
    padX = size(bg,1) - size(I,1);  
    padY = size(bg,2) - size(I,2);
    I = padarray(I,[ padX padY 0], 0, 'post');
    maskI = padarray(maskI,[padX padY 0], 255, 'post');
    
    images(:,:,:,i) = bg-imcomplement(maskI)+I;        
end
% ---------------------------------------
%  Displaying
% ---------------------------------------
implay(images, FPS);
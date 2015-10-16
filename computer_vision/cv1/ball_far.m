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
A = imread('ball.jpg', 'jpg');
maskA = imread('ball_mask.jpg', 'jpg');
bg = imread('beach.jpg', 'jpg');
images = uint8(zeros(size(bg,1),size(bg,2),3,numFrames));
interpol = 'linear'; %nearest/linear

% ---------------------------------------
%  Transformation
% ---------------------------------------
A = A-maskA;
% Y = zeros(1,numFrames); %debug
% X = zeros(1,numFrames); %debug
for i = 1:numFrames
    
    % ------ Scale the ball
    scf = 4+(i-1)*100/numFrames;
    sx = 1/scf;
    sy = 1/scf;
    sform=[  sx  0 0
             0  sy 0 
             0   0 1 ];
    sf = affine2d(sform);
    [B, RB] = imwarp(A, sf, 'FillValues', 0, 'Interp', interpol);    % black bg
    maskB = imwarp(maskA, sf, 'FillValues', 255, 'Interp', interpol);% white bg
    
     % ------ Perform translation of image at 0,0
    tx = - size(B,2)/2;
    ty = - size(B,1)/2;
    tform=[  1    0  0
             0    1  0 
             tx  ty  1 ];
    tf = affine2d(tform);
    [C, RC] = imwarp(B, RB, tf, 'FillValues', 0, 'Interp', interpol);
    maskC = imwarp(maskB, RB, tf, 'FillValues', 254, 'Interp', interpol);
      
    % ------ Finish one rotation in these frames
    % q = pi/4 * (i-2); $debug
    q = 2*pi * (i/numFrames)*2;
    xform=[  cos(q)  sin(q) 0
            -sin(q)  cos(q) 0 
               0      0     1 ];
    af = affine2d(xform);
    [D, RD] = imwarp(C, RC, af, 'FillValues', 0, 'Interp', interpol);% white bg
    maskD = imwarp(maskC, RC, af, 'FillValues', 255, 'Interp', interpol);% white bg
    
    % ------ Perform translation of rotated image on the background
    tx = (i-1) * size(bg,2)/numFrames;
    posCos = cos(tx/size(bg,2)*2*pi*4);
    shift = size(bg,1)-size(A,1)/4;
    ty = shift - abs(shift*posCos)/exp(tx/size(bg,2)*4);
    
    % transforming the cosine function in the projective plane in order to
    % make it go towards the horizon
    tsform = projective2d([1  1 0.0028;
                      0  1 0;
                      0 0 1]);
    [tx,ty] = transformPointsForward(tsform,tx,ty);

    % fixing the tx and ty based on the ref returned from rotation, 
    % moved the ball 300 to the right
    % X(i) = tx; %debug
    tx = tx + 300 + RD.XWorldLimits(1);
    ty = ty + RD.YWorldLimits(1);
    % Y(i) = ty; %debug

    tform=[  1    0  0
             0    1  0 
             tx  ty  1 ];
    tf = affine2d(tform);
    [F, RF] = imwarp(D, RD, tf, 'FillValues', 0, 'Interp', interpol);
    maskF = imwarp(maskD, RD, tf, 'FillValues', 254, 'Interp', interpol);
    
    
    % --------------------------------------------------------------------
    % new implementation with use of padarray and cropping 
    % cropping was implemented in order to be able to have proper
    % animations even with weird sizes!
    % --------------------------------------------------------------------
    Fsize = size(F);
    if (tx<0)
        F = F(:,-tx:Fsize(2),:);
        maskF = maskF(:,-tx:Fsize(2),:);
        tx=0;
        
    end
    if (ty<0)
        F = F(ceil(-ty):size(F,1),:,:);
        maskF = maskF(ceil(-ty):Fsize(1),:,:);
        ty=0;
    end
    
    % prepadding
    I = padarray(F,[ceil(ty) ceil(tx) 0], 0, 'pre');
    maskI = padarray(maskF,[ceil(ty) ceil(tx) 0], 255, 'pre');
    
    % crop it
    if size(I,1) > size(bg,1)
        I = I(1:size(bg,1),:,:);
        maskI = maskI(1:size(bg,1),:,:);
        padX = 0;
    else
        padX = size(bg,1)-size(I,1);
    end
    if size(I,2) > size(bg,2)
        I = I(:,1:size(bg,2),:);
        maskI = maskI(:,1:size(bg,2),:);
        padY = 0;
    else
        padY = size(bg,2)-size(I,2);
    end
       
    % postpadding 
    I = padarray(I,[padX padY 0], 0, 'post');
    maskI = padarray(maskI,[padX padY 0], 255, 'post');
    
    % adding the image in the image array
    images(:,:,:,i) = bg-imcomplement(maskI)+I;  

end

% ---------------------------------------
%  Displaying
% ---------------------------------------
implay(images, FPS);
% plot(X,-Y); %debug
fib = 2.5;
A = imread('fibonacci.jpg', 'jpg');

% crop
A = A(1:size(A,2),:);

% resize to a fibonacci number!
A = imresize(A,[233 233]);

% initialize final image
FIBO = uint8(zeros(size(A,1), 233+144));

% places
tr = [size(A,1) 0;
    size(A,1)+(5/8*size(A,1))/fib 5/8*size(A,1);
    size(A,1) 174;
    size(A,1) 5/8*size(A,1)];

% first image
FIBO(1:size(A,1), 1:size(A,2)) = A;

% loop
i = 1;
fiboend = 5;
fibostart = 8;
B = A;
while fibostart ~= fiboend
    % scale
    sx = fiboend/fibostart;
    sform=[ sx  0 0
            0  sx 0 
            0   0 1 ];
    sf = affine2d(sform);
    B = imwarp(B, sf);
    
    % translate
    tform=[  1    0  0
             0    1  0 
             tr(i,2)  tr(i,1)  1 ];
    tf = affine2d(tform);
    [B,RB] = imwarp(B, tf);
    
    FIBO(RB.XWorldLimits(1):RB.XWorldLimits(2)-1, RB.YWorldLimits(1):RB.YWorldLimits(2)-1) = B;
    
    fiboend = fibostart - fiboend;
    fibostart = fibostart - fiboend;  
    i = i+1;    
end

% show 
imshow(FIBO);
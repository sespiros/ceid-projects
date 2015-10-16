# Matrix size cuBLAS      Plain       Optimized
# 128 x64     0.367 ms    0.0364ms    0.0145ms
# 128 x128    0.0179ms    0.0485ms    0.0312ms
# 256 x128    0.0205ms    0.0487ms    0.0327ms
# 256 x256    0.0312ms    0.0931ms    0.0597ms
# 512 x256    0.0345ms    0.132 ms    0.0658ms
# 512 x512    0.0553ms    0.261 ms    0.125 ms
# 1024x512    0.0874ms    0.321 ms    0.16  ms
# 1024x1024   0.155 ms    0.671 ms    0.163 ms
# 2048x1024   0.271 ms    1.3   ms    0.233 ms
# 2048x2048   0.492 ms    2.6   ms    0.45  ms
# 4096x2048   0.954 ms    5.12  ms    0.892 ms
# 4096x4096   1.87  ms    10.3  ms    1.77  ms
# 8192x4096   3.67  ms    20.3  ms    3.59  ms
# 8192x8192   7.44  ms    40.9  ms    7.07  ms
# 9000x9000   10.9  ms    50    ms    8.64  ms
# 3342x114    0.104 ms    0.279 ms    0.0492ms
# 1025x1025   0.186 ms    0.815 ms    0.185 ms

# to work
graphics_toolkit("gnuplot")

cublas = [0.367 0.0179 0.0205 0.0312 0.0345 0.0553 0.0874 0.155 0.271 0.492 0.954 1.87 3.67 7.44 10.9 0.104 0.186];

plain = [0.0364 0.0485 0.0487 0.0931 0.132 0.261 0.321 0.671 1.3 2.6 5.12 10.3 20.3 40.9 50 0.279 0.815];

optimized = [0.0145 0.0312 0.0327 0.0597 0.0658 0.125 0.16 0.163 0.233 0.45 0.892 1.77 3.59 7.07 8.64 0.0492 0.185];

h = figure(1);
semilogy(1:length(cublas), cublas, 'LineWidth', 3, 1:length(plain), plain,'LineWidth',3)
hold on;
semilogy(1:length(optimized), optimized, 'r-','LineWidth',3)
hold off;
title('Runtimes for the three implementations')
xlabel('Test #')
ylabel('Runtime (log ms)')
legend('cuBLAS', 'Plain', 'Optimized')

fontsize=16;
set([gca; findall(gca, 'Type','text')], 'FontSize', fontsize);
print(h,'-dpng','-color','times.png')

# calculate speedup
speedup = plain ./ optimized;
figure(2)
plot(1:length(speedup), speedup, 'LineWidth', 3)
title('Speedup, plain vs optimized')

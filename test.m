function test()
    clc;clear;close all;
    format long
    % 定义符号函数
    syms x1 x2
    % sym_f = 0.5 * (x1^2 + 100 * x2^2);
    % sym_f = (1/2) * x1^2 + x1^2 * x2^2 + 2 * x2^2;
    sym_f = (x1+x2)^2 + (x1+1)^2 + (x2+3)^2;
    % sym_f = 20 + x1^2 + x2^2 - 10*cos(2*pi*x1) - 10*cos(2*pi*x2); % Rastrigin函数最小值点(0,0)
    % sym_f = (1 - x1)^2 + 100 * (x2 - x1^2)^2; % Rosenbrock函数最小值点(1,1)
    var = [x1; x2]; % 函数变量
    % 转为普通函数并求梯度和黑塞矩阵
    f = matlabFunction(sym_f, 'Vars', {var});
    grad_f = matlabFunction(gradient(sym_f), 'Vars', {var}); % 梯度
    hessian_f = matlabFunction(hessian(sym_f), 'Vars', {var}); % 黑塞矩阵
    % 测试参数
    x0 = [-1; -1];  % 初始点
    max_iter = 10000; % 最大迭代次数

    %%% 修改此处来固定收敛次数进行对比，1为固定，其他为不固定 %%%
    fix = 0;
    if fix == 1
        delta = -1; % 固定迭代次数对比
        max_iter = 20; % 固定迭代次数
    else
        delta = 1e-6; % 收敛精度
    end

    % 画图初始化
    xp1 = linspace(-2, 2, 100); % x1 范围
    xp2 = linspace(-2, 2, 100); % x2 范围
    [X1, X2] = meshgrid(xp1, xp2); % 生成网格点
    Z = arrayfun(@(xp1, xp2) f([xp1; xp2]), X1, X2); % 计算每个网格点的函数值

    % 函数的三维曲面图
    figure;
    surf(X1, X2, Z, 'EdgeColor', 'none');
    xlabel('x_1');
    ylabel('x_2');
    zlabel('f(x_1, x_2)');
    title('函数可视化');
    
    % 最速下降法
    [gd_x, gd_f, gd_iter_x, gd_iter_f, gd_iter] = steepest_descent(f, grad_f, x0, delta, max_iter);
    % 牛顿法
    [newton_x, newton_f, newton_iter_x, newton_iter_f, newton_iter] = newton(f, grad_f, hessian_f, x0, delta, max_iter);
    % 改进的牛顿法
    [mnewton_x, mnewton_f, mnewton_iter_x, mnewton_iter_f, mnewton_iter] = mod_newton(f, grad_f, hessian_f, x0, delta, max_iter);
    % 共轭梯度法
    [cg_x, cg_f, cg_iter_x, cg_iter_f, cg_iter] = fr_cg(f, grad_f, x0, delta, max_iter);
    % 拟牛顿法（BFGS）
    [bfgs_x, bfgs_f, bfgs_iter_x, bfgs_iter_f, bfgs_iter] = nnewton(f, grad_f, x0, delta, max_iter, 'BFGS');
    % 拟牛顿法（BFGS）
    [dfp_x, dfp_f, dfp_iter_x, dfp_iter_f, dfp_iter] = nnewton(f, grad_f, x0, delta, max_iter, 'DFP');
    
    % 定义输出精度
    pre = 16;
    % 输出结果汇总
    fprintf('\n算法比较:\n');
    fprintf('算法               最优解(x*)                                    最优值(f*)          迭代次数\n');
    fprintf('最速下降法        [%.*f, %.*f]     %.*f     %d\n', pre, gd_x(1), pre, gd_x(2), pre, gd_f, gd_iter);
    fprintf('牛顿法            [%.*f, %.*f]     %.*f     %d\n', pre, newton_x(1), pre, newton_x(2), pre, newton_f, newton_iter);
    fprintf('改进的牛顿法      [%.*f, %.*f]     %.*f     %d\n', pre, mnewton_x(1), pre, mnewton_x(2), pre, mnewton_f, mnewton_iter);
    fprintf('F-R共轭梯度法     [%.*f, %.*f]     %.*f     %d\n', pre, cg_x(1), pre, cg_x(2), pre, cg_f, cg_iter);
    fprintf('拟牛顿法（BFGS）  [%.*f, %.*f]     %.*f     %d\n', pre, bfgs_x(1), pre, bfgs_x(2), pre, bfgs_f, bfgs_iter);
    fprintf('拟牛顿法（DFP）   [%.*f, %.*f]     %.*f     %d\n', pre, dfp_x(1), pre, dfp_x(2), pre, dfp_f, dfp_iter);

    % 绘图
    sub = 1; %%% 决定是否将所有等高线图作为子图放入一个大图中（组合），1代表组合，其他数代表分开一一绘制 %%%
    methods = {
    '最速下降法', gd_iter_x, '-*', gd_iter_f;
    '牛顿法', newton_iter_x, '-o', newton_iter_f;
    '改进的牛顿法', mnewton_iter_x, '-+', mnewton_iter_f;
    '共轭梯度法', cg_iter_x, '-x', cg_iter_f;
    '拟牛顿法（BFGS）', bfgs_iter_x, '-v', bfgs_iter_f;
    '拟牛顿法（DFP）', dfp_iter_x, '-d', dfp_iter_f;
    };
    if sub == 1
        figure;
        sgtitle('优化算法比较'); % 设置大图总标题
    end
    num_methods = size(methods, 1);
    for i = 1:num_methods
        name = methods{i, 1};
        iter_x = methods{i, 2};
        style = methods{i, 3};
        if sub == 1
            subplot(ceil(num_methods / 3), 3, i); % 子图行列设置
        else
            figure;
        end
        contour(X1, X2, Z);
        hold on;
        plot(iter_x(:,1), iter_x(:,2), style);
        title(name);
        xlabel('x1'); ylabel('x2');
        hold off;
    end

    % 绘制对比图（将所有轨迹组合到一个图中）
    figure;
    contour(X1, X2, Z);
    hold on;
    for i = 1:num_methods
        name = methods{i, 1};
        iter_x = methods{i, 2};
        style = methods{i, 3};
        plot(iter_x(:,1), iter_x(:,2), style, 'DisplayName', name);
    end
    title('优化方法对比图');
    xlabel('x1'); ylabel('x2');
    legend('等高线', methods{:,1}) % 添加图例
    hold off;

    if delta == -1
        % 收敛值对比图（需要固定迭代次数，可将delta设为-1，固定max_iter）
        figure;
        for i = 1:num_methods
            if i == 1
                hold on;
            end
            name = methods{i, 1};
            iter_f = methods{i, 4};
            horz = 0:1:(size(iter_f) - 1); % 采用迭代函数值的长度作为横坐标
            plot(horz, iter_f, 'DisplayName', name);
        end
        % 设置图形属性
        xlabel('迭代次数'); % 横轴标签
        ylabel('算法的迭代函数值'); % 纵轴标签
        title('收敛性能对比'); % 标题
        set(gca, 'YScale', 'log'); % 设置纵轴为对数坐标
        grid on; % 显示网格
        legend(methods{:,1}); % 添加图例
        hold off;
    end
end

function [x_min, f_min] = gold618(f, a, b, tol)
    % 黄金分割法找到函数f在区间[a, b]上的最小值
    % 输入：目标函数f和其区间[a,b]，收敛容差tol
    % 输出：最小值点x_min和最小值f_min
    phi = (sqrt(5) - 1) / 2; % 黄金比例常数
    c = b - phi * (b - a); % 初始化两个内部点
    d = a + phi * (b - a);
    while abs(b - a) > tol
        if f(c) < f(d)
            b = d;
            d = c;
            c = b - phi * (b - a);
        else
            a = c;
            c = d;
            d = a + phi * (b - a);
        end
    end
    x_min = (a + b) / 2; % 取区间的中点作为近似最优解
    f_min = f(x_min);
end

function [x_opt, f_opt, x_iter, f_iter, k] = steepest_descent(f, grad_f, x0, delta, max_iter)
    % 最速下降法
    % 输入：目标函数句柄f，梯度函数句柄grad_f，初始点x0，精度要求delta，最大迭代次数max_iter
    % 输出：最优解x_opt，最优值f_opt，每次迭代的x值x_iter，每次迭代的函数值f_iter，迭代次数k
    xk = x0; k = 1; % 初始化
    x_iter(k,:) = xk;
    f_iter(k,:) = f(xk);
    while true
        dk = -grad_f(xk); % 计算当前梯度方向
        if norm(dk) <= delta || k >= max_iter % 判断是否满足停止准则
            x_opt = xk;
            f_opt = f(xk);
            k = k - 1;
            break;
        else
            phi = @(lambda) f(xk + lambda * dk); % 线搜索找到步长 λ_k
            % lambda_k = fminbnd(phi, 0, 1); % 内置函数
            lambda_k = gold618(phi, 0, 1, 1e-6); % 黄金分割法（步长在0-1之间寻找即可）
            xk = xk + lambda_k * dk; % 更新 xk
        end
        k = k + 1;
        x_iter(k,:) = xk; % 记录当前迭代的 x 值和函数值
        f_iter(k,:) = f(xk);
    end
end

function [x_opt, f_opt, x_iter, f_iter, k] = newton(f, grad_f, hess_f, x0, delta, max_iter)
    % 牛顿法
    % 输入：目标函数句柄f，梯度函数句柄grad_f，Hessian矩阵函数句柄hess_f，初始点x0，精度要求delta，最大迭代次数max_iter
    % 输出：最优解x_opt，最优值f_opt，每次迭代的x值x_iter，每次迭代的函数值f_iter，迭代次数k
    xk = x0;k = 1; % 初始化
    x_iter(k,:) = xk;
    f_iter(k,:) = f(xk);
    while true
        gk = grad_f(xk); % 计算梯度和 Hessian 矩阵
        Hk = hess_f(xk);
        if norm(gk) <= delta || k >= max_iter % 判断停止条件
            x_opt = xk;
            f_opt = f(xk);
            k = k - 1;
            break;
        end
        pk = -Hk \ gk; % 计算下降方向
        xk = xk + pk; % 更新 xk
        k = k + 1; % 增加迭代次数
        x_iter(k,:) = xk; % 记录当前迭代的 x 值和函数值
        f_iter(k,:) = f(xk);
    end
end

function [x_opt, f_opt, x_iter, f_iter, k] = mod_newton(f, grad_f, hess_f, x0, delta, max_iter)
    % 改进牛顿法
    % 输入：目标函数句柄f，梯度函数句柄grad_f，Hessian矩阵函数句柄hess_f，初始点x0，精度要求delta，最大迭代次数max_iter
    % 输出：最优解x_opt，最优值f_opt，每次迭代的x值x_iter，每次迭代的函数值f_iter，迭代次数k
    xk = x0; k = 1; % 初始化
    x_iter(k,:) = xk;
    f_iter(k,:) = f(xk);
    while true
        gk = grad_f(xk); % 计算梯度和 Hessian 矩阵
        Hk = hess_f(xk);
        if norm(gk) <= delta || k >= max_iter % 判断停止条件
            x_opt = xk;
            f_opt = f(xk);
            k =  k - 1;
            break;
        end
        dk = -Hk \ gk; % 计算下降方向
        phi = @(lambda) f(xk + lambda * dk); % 线搜索: 确定最优步长 lambda_k
        % lambda_k = fminbnd(phi, 0, 1);
        lambda_k = gold618(phi, 0, 1, 1e-6);
        xk = xk + lambda_k * dk; % 更新 xk
        k = k + 1; % 增加迭代次数
        x_iter(k,:) = xk; % 记录当前迭代的 x 值和函数值
        f_iter(k,:) = f(xk);
    end
end

function [x_opt, f_opt, x_iter, f_iter, k] = fr_cg(f, grad_f, x0, delta, max_iter)
    % F-R 共轭梯度法
    % 输入：目标函数句柄f，梯度函数句柄grad_f，初始点x0，精度要求delta，最大迭代次数max_iter
    % 输出：最优解x_opt，最优值f_opt，每次迭代的x值x_iter，每次迭代的函数值f_iter，迭代次数k
    xk = x0; % 初始化
    gk = grad_f(xk);
    dk = -gk;  % 初始方向
    k = 1;
    x_iter(k,:) = xk;
    f_iter(k,:) = f(xk);
    while norm(gk) > delta && k < max_iter
        phi = @(lambda) f(xk + lambda * dk); % 线搜索确定步长 lambda_k
        % lambda_k = fminbnd(phi, 0, 1);
        lambda_k = gold618(phi, 0, 1, 1e-6);
        xk = xk + lambda_k * dk; % 更新 xk
        gk_new = grad_f(xk); % 计算新的梯度
        beta_k = (norm(gk_new)^2) / (norm(gk)^2); % 计算共轭系数 beta_k
        dk = -gk_new + beta_k * dk; % 更新方向
        gk = gk_new; % 更新梯度
        k = k + 1; % 增加迭代次数
        x_iter(k,:) = xk;
        f_iter(k,:) = f(xk);
    end
    x_opt = xk; % 输出最优解和最优值
    f_opt = f(xk);
    k = k - 1;
end

function [x_opt, f_opt, x_iter, f_iter, k] = nnewton(f, grad_f, x0, delta, max_iter, method)
    % 拟牛顿法（BFGS & DFP）
    % 输入：目标函数句柄f，梯度函数句柄grad_f，初始点x0，精度要求delta，最大迭代次数max_iter，更新方法method ('BFGS' 或 'DFP')
    % 输出：最优解x_opt，最优值f_opt，每次迭代的x值x_iter，每次迭代的函数值f_iter，迭代次数k
    xk = x0; % 初始化
    Bk = eye(length(x0));  % 初始Hessian矩阵近似
    gk = grad_f(xk);
    k = 1;
    x_iter(k,:) = xk;
    f_iter(k,:) = f(xk);
    % 选择更新公式的函数句柄
    if strcmp(method, 'BFGS')
        update_hessian = @(Bk, sk, yk) Bk + (yk * yk') / (yk' * sk) - (Bk * (sk * sk') * Bk) / (sk' * Bk * sk);
    elseif strcmp(method, 'DFP')
        update_hessian = @(Bk, sk, yk) Bk + (sk * sk') / (sk' * yk) - (Bk * (yk * yk') * Bk) / (yk' * Bk * yk);
    else
        error('Invalid method. Use "BFGS" or "DFP".');
    end
    while norm(gk) > delta && k < max_iter
        pk = -Bk \ gk; % 确定方向 pk
        phi = @(lambda) f(xk + lambda * pk); % 线搜索确定步长 lambda_k
        % lambda_k = fminbnd(phi, 0, 1);
        lambda_k = gold618(phi, 0, 1, 1e-6);
        xk_new = xk + lambda_k * pk; % 更新 xk
        gk_new = grad_f(xk_new); % 计算梯度
        sk = xk_new - xk; % 计算 s_k 和 y_k
        yk = gk_new - gk;
        if norm(sk) == 0 || norm(yk) == 0 % 避免除以零的情况
            break;
        end
        Bk = update_hessian(Bk, sk, yk); % 使用函数句柄更新 Hessian 矩阵
        xk = xk_new; % 更新迭代变量
        gk = gk_new;
        k = k + 1;
        x_iter(k,:) = xk;
        f_iter(k,:) = f(xk);
    end
    x_opt = xk; % 输出最优解和最优值
    f_opt = f(xk);
    k = k - 1;
end

AcWing 168. 生日蛋糕 Python 暴力搜索 注意剪枝    原题链接    中等
作者：    皓首不倦 ,  2020-08-04 18:41:18 ,  阅读 15

0



'''
DFS 优化策略
假设第M层是最下面的一层, 第一层是最上面的一层

1. 层数从M层到1层枚举，先枚举体积大的层数
2. 每一个圆柱体先枚举半径， 再枚举高度，因为半径是平方级的，减枝比较快, 半径和高度都是降序枚举
3. 第i层的半径R和高度H的范围:

i <= R(i) <= R(i+1)-1
假设1层到i-1层已经累计了体积V, 那pi * R(i) * R(i) * H(i) <= pi * (N - V)
可以推导出 R(i) <= sqrt(N-V)

i <= H(i) <= H(i+1) - 1
同样的，根据pi * R(i) * R(i) * H(i) <= pi * (N - V)
可以推导出H(i) <= (N-V) / R(i) / R(i)


4. 先预处理，计算所有前i-1层的可能的侧边面积最小值和体积最小值
如果当前的累计体积加上上面所有层体积的最小和大于N，则回溯
如果当前累计面积加上上面所哟鄫的面积的最小和大于等于ans, 则回溯


5. 体积和表面积之间有关联关系
S(i) 表示1到i的圆柱体的侧面表面积和
V(i) 表示1到i的圆柱体的体积和

S(i) = 2R(1)*H(1) + 2R(2)*H(2) + ...... 2R(i)*H(i)
> (2/R(i+1)) *  [ R(1)^2 * H(1) + R(2)^2 * H(2) ...... R(i)^2 * H(i) ]

= (2/R(i+1)) * V(i)
= (2/R(i+1)) * (N - M层到i-1层的体积累计和)

也就是说如果知道了M层到i-1层的体积累计和，就能得到一个i层到1层的侧面面积总和的下界，当前累计
的表面积和加上这个下界，如果超过了ans, 则可以回溯

'''

import math

N = int(input())
M = int(input())

R = [0x7fffffff] * (M+2)     # 每一层半径
H = [0x7fffffff] * (M+2)     # 每一层高度


min_a_sum = [0] * (M+1)     # 从1层到某一层的最小侧面面积累加和
min_v_sum = [0] * (M+1)     # 从1层到某一层的最小体积的累加和

for i in range(1, M+1):
    min_a_sum[i] = min_a_sum[i-1] + 2*i*i
    min_v_sum[i] = min_v_sum[i-1] + i*i*i

ans = [0x7fffffff]
def dfs(cur_level, sum_a, sum_v):
    #print(cur_level, sum_a, sum_v)
    if cur_level == 0:
        if sum_v == N:
            ans[0] = min(ans[0], sum_a)
        return

    sum_a_lower_bound = (N - sum_v) * 2 / R[cur_level + 1]
    if sum_a + sum_a_lower_bound > ans[0]:
        return

    r_upper_bound = min(R[cur_level+1]-1, int(math.sqrt(N-sum_v)))
    for r in range(r_upper_bound, cur_level-1, -1):
        h_upper_bound = min(H[cur_level+1]-1, int( (N-sum_v) / r / r ))

        for h in range(h_upper_bound, cur_level-1, -1):
            cur_a = 2*r*h
            cur_v = r*r*h

            if sum_a + cur_a + min_a_sum[cur_level-1] >= ans[0]:
                continue
            if sum_v + cur_v + min_v_sum[cur_level-1] > N:
                continue

            old_r, old_h = R[cur_level], H[cur_level]
            R[cur_level], H[cur_level] = r, h

            if cur_level != M:
                dfs(cur_level-1, sum_a + cur_a, sum_v + cur_v)
            else:
                dfs(cur_level-1, sum_a + cur_a + r*r, sum_v + cur_v)

            R[cur_level], H[cur_level] = old_r, old_h

dfs(M, 0, 0)
if ans[0] == 0x7fffffff:
    ans[0] = 0
print(ans[0])

作者：皓首不倦
链接：https://www.acwing.com/solution/content/17731/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

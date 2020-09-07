AcWing 198. 反素数
皓首不倦的头像皓首不倦
7分钟前



'''
小于等于N的反素数就是小于等于N的所有数中，约数个数最多的一堆数中数值最小的一个
从质因数次幂乘积的形式看待1-N中没一个数字，只需要枚举每一个可能的质因数的次幂
数，就能枚举出每个数的约数个数，不断更新约数个数的最大值和约数个数是最大值的数字
的最小数值即可，题目有个特殊的限制，求的是约数个数最多的数值最小的数，而约数个数
只跟次幂数有关，根底数没关系，所以递归枚举每一个底数的次幂的时候，大底数的次幂应该
小于等于小底数的次幂，因为不是这样枚举，交换两个违反规则的底数的次幂，一定对应一个
更小的值和待枚举的值有一样的约数个数，所以递归枚举时候次幂数是非递增的，有这个约束
条件下，每个底数的次幂数最多到31，而且还是非递增的，还要保证数值累乘起来小于等于N
在这些约束下，搜索空间是很小的，直接用DFS枚举各个质数底数的次幂的组合情况
'''


n = int(input())


'''
dfs 枚举每一个质因数的次幂数
idx 表示第几个质因子
prev_cnt 表示上一个质因子的次幂数
prev_val 表示上一个状态的约数累乘值
prev_acc 表示 次幂数加1 的累乘值
'''
prime = [2, 3, 5, 7, 11, 13, 17, 19, 23]
max_cnt = [-1]              # 约数个数的最大值
min_val = [0x7fffffff]      # 约数个数最多的数值中的最小值
def dfs(idx, prev_cnt, prev_val, prev_acc):
    #print(idx, prev_cnt, prev_val, prev_acc)

    if idx >= len(prime):
        return

    base = 1
    for k in range(1, prev_cnt+1):
        base *= prime[idx]
        if prev_val * base <= n:
            if prev_acc * (k+1) > max_cnt[0]:
                max_cnt[0] = prev_acc * (k+1)
                min_val[0] = prev_val * base
            elif prev_acc * (k+1) == max_cnt[0]:
                min_val[0] = min(min_val[0], prev_val * base)

            dfs(idx+1, k, prev_val * base, prev_acc * (k+1))


dfs(0, 35, 1, 1)
print(min_val[0])

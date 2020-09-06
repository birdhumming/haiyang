AcWing 137. 雪花python3    原题链接    简单
作者：    那些花儿 ,  2019-10-16 20:37:54 ,  阅读 195

0


题目描述
参考大佬的c++思路，写了一版python的，使用桶式哈希
写python的小朋友可以一起交流哈

python 代码
import collections


MOD = 99991
Snow = collections.namedtuple('Snow', 'arms next')
snows = [None] * 100000

def check(arms, idx):
    p = snows[idx]
    while p is not None:
        for i in range(6):
            # 寻找对比起点
            if arms[i] == p.arms[0]:
                j = i
                # 正向
                for k in range(1,6):
                    j = (j + 1) % 6
                    if arms[j] != p.arms[k]:
                        break
                else:
                    return True
                j = i
                # 反向
                for k in range(5, 0, -1):
                    j = (j + 1) % 6
                    if arms[j] != p.arms[k]:
                        break
                else:
                    return True
        p = p.next
    node = Snow(arms, snows[idx])
    snows[idx] = node
    return False

def main():
    n = int(input())
    for i in range(n):
        temp = list(map(int, input().split()))
        idx = sum(temp)%MOD
        if check(temp, idx):
            print('Twin snowflakes found.')
            return
    print('No two snowflakes are alike.')


if __name__ == '__main__':
    main()

作者：那些花儿
链接：https://www.acwing.com/solution/content/5040/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

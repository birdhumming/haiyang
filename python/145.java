AcWing 145. 【java】超市（大根堆）    原题链接    简单
作者：    tt2767 ,  2019-12-21 00:42:21 ,  阅读 206

0


// 和视频的方法不太一样，因为“价值最大的物品在临期最后一天卖” == “从后向前，卖当天最大价值的物品，不然就提前卖掉”
// 所以大根堆维护当前最大值加和即可

import java.util.*;

public class Main{

    class Node{
        int x,y;
        public Node(int x, int y){ this.x= x; this.y = y;}
    }

    void run(){
        while(jin.hasNext()){
            int n = jin.nextInt();
            List<Node> list = new ArrayList<>();
            for (int i = 0 ; i < n ; i++){
                int p = jin.nextInt();
                int d = jin.nextInt();
                list.add(new Node(p, d));
            }
            System.out.println(solve(n, list));
        }

    }    

    int solve(int n, List<Node> list){

        queue.clear();
        list.sort((a, b) -> (a.y-b.y));

        int res = 0;

        // int last = list.get(list.size() - 1).y;  // 这步放在这是有问题的，因为list不一定有数据
        int last = list.size() == 0 ? 0 :list.get(list.size() - 1).y;
        for (int i = list.size() - 1; i >= 0 ; i--){
            Node node = list.get(i);
            while((!queue.isEmpty()) && node.y < last-- ) res += queue.poll().x; // last-- 可以放在判断里是因为isEmpty判断了异常情况
            queue.add(node);
            last = node.y; // 开始忘记更新last了
        }
        while (!queue.isEmpty() && 0 < last-- ) res += queue.poll().x;

        return res;
    }
    private Queue<Node> queue = new PriorityQueue<>(10000 , (a, b) -> (b.x-a.x));
    private Scanner jin = new Scanner(System.in);
    public static void main(String[] args) throws Exception {new Main().run();}
}


作者：tt2767
链接：https://www.acwing.com/solution/content/7160/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

https://www.cnblogs.com/wangchaowei/p/8288216.html

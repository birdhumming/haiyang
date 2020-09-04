/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *entryNodeOfLoop(ListNode *head) {
        auto i = head, j = head;
        while (i && j) {
            i = i->next;
            j = j->next;
            if (j) j = j->next;
            else break;
            if (i == j) {
                i = head;
                while (i != j) {
                    i = i->next;
                    j = j->next;
                }
                return i;
            }
        }
        return 0;
    }
};

作者：yxc
链接：https://www.acwing.com/video/158/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

while (j && j->next){
    j=j->next->next;
    i=i->next;
    if(i==j) break;
}

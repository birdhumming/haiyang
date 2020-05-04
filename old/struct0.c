#include <stdio.h>
#include <string.h>

/* Main */

int main(void) {

    struct buyer
    {
        char name[10];
        int price;
    } Mike, Paula, Luke, Mark, Cheryl, Al;


    struct car
    {
        int year;
        char make[20];
        char model[20];
        int price;
        char *buyers[3];
        char *vi_buyers[6];
        int vi;
    } Car1, Car2, Car3, Car4, Car5;



    strcpy(Mike.name, "Mike");
    Mike.price = 7000;

    strcpy(Paula.name, "Paula");
    Paula.price = 1999;

    strcpy(Luke.name, "Luke");
    Luke.price = 12000;

    strcpy(Mark.name, "Mark");
    Mark.price = 15000;

    strcpy(Cheryl.name, "Cheryl");
    Cheryl.price = 50000;

    strcpy(Al.name, "Al");
    Al.price = 3000;

    Car1.year = 1993;
    strcpy(Car1.make, "Toyota");
    strcpy(Car1.model, "Corolla");
    Car1.price = 2000;
    Car1.buyers[0] = "Mike";
    Car1.buyers[1] = "Paula";
    Car1.buyers[2] = "Al";
    Car1.vi = 0;

    Car2.year = 2007;
    strcpy(Car2.make, "Mercedes");
    strcpy(Car2.model, "Class E");
    Car2.price = 4000;
    Car2.buyers[0] = "Paula";
    Car2.buyers[1] = "Al";
    Car2.vi = 0;

    Car3.year = 2001;
    strcpy(Car3.make, "Audi");
    strcpy(Car3.model, "A3");
    Car3.price = 10000;
    Car3.buyers[0] = "Luke";
    Car3.buyers[1] = "Paula";
    Car3.buyers[2] = "Mark";
    Car3.vi = 0;

    Car4.year = 1999;
    strcpy(Car4.make, "Ferrari");
    strcpy(Car4.model, "F40");
    Car4.price = 30000;
    Car4.buyers[0] = "Cheryl";
    Car4.buyers[1] = "Mark";
    Car4.vi = 0;

    Car5.year = 2010;
    strcpy(Car5.make, "Chrysler");
    strcpy(Car5.model, "Voyager");
    Car5.price = 15000;
    Car5.buyers[0] = "Mark";
    Car5.buyers[1] = "Mike";
    Car5.vi = 0;

    struct buyer arr[] = {Mike, Paula, Luke, Mark, Cheryl, Al};
    struct car arr1[] = {Car1, Car2, Car3, Car4, Car5};

    for (int i = 0; i < 5; i++)
    {
        int x = arr1[i].price;
        if (i == 1 || i == 3 || i == 4)
        {
            for (int j = 0; j < 2; j++)
            {
                char *r = arr1[i].buyers[j];
                for (int h = 0; h < 6; h++)
                {
                    if (strcmp(r, arr[h].name) == 0 && arr[h].price >= x)
                    {
                        arr1[i].vi_buyers[h] = arr[h].name;
                        arr1[i].vi++;
                    }
                }
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                char *r = arr1[i].buyers[j];
                for (int h = 0; h < 6; h++)
                {
                    if (strcmp(r, arr[h].name) == 0 && arr[h].price >= x)
                    {
                        arr1[i].vi_buyers[h] = arr[h].name;
                        arr1[i].vi++;
                    }
                }
            }
        }
    }
    for (int i = 0; i < 5; i++)
    {
        printf("%s %s\n", arr1[i].make, arr1[i].model);
        printf("%d\n", arr1[i].vi);
        int y = arr1[i].vi;
        if (y == 1)
        {
            char *x = arr1[i].vi_buyers[0];
            printf("%s\n", x);
        }
        else if (y == 0)
        {
            printf("none\n");
        }
        else
        {
            for (int j = 0; j < y-1; j++)
            {
                char *x = arr1[i].vi_buyers[j];
                printf("%s\n", x);
            }
        }
    }
}

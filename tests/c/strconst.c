#include <stdio.h>

char *s = "woo";

int main(void)
{
    s = "world";
    printf("hello, %s!\n", s);
}

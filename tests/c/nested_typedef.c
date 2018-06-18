#include <stdio.h>

typedef int *string;
int main(void)
{
    typedef char *string;
    string s = "hello, world\n";
    printf("%s", s);
}

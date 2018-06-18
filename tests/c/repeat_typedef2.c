#include <stdio.h>

// can't do this in any version of C
typedef char *string;
typedef int *string;

int main(void)
{
    printf("hello, world!\n");
}

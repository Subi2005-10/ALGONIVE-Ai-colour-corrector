#include<stdio.h>
#include<string.h>
#include<ctype.h>

char production[10][10];
char first[10], follow[10];
int n;

void findfirst(char c)
{
    int i;
    
    if(!isupper(c))
        printf("%c ",c);

    for(i=0;i<n;i++)
    {
        if(production[i][0]==c)
        {
            if(islower(production[i][2]))
                printf("%c ",production[i][2]);
            else
                findfirst(production[i][2]);
        }
    }
}

void findfollow(char c)
{
    int i,j;

    if(production[0][0]==c)
        printf("$ ");

    for(i=0;i<n;i++)
    {
        for(j=2;j<strlen(production[i]);j++)
        {
            if(production[i][j]==c)
            {
                if(production[i][j+1]!='\0')
                    findfirst(production[i][j+1]);
            }
        }
    }
}

int main()
{
    int i;
    char c;

    printf("Enter number of productions: ");
    scanf("%d",&n);

    printf("Enter productions (Example: E=TR):\n");
    for(i=0;i<n;i++)
        scanf("%s",production[i]);

    printf("\nFIRST Sets:\n");
    for(i=0;i<n;i++)
    {
        c=production[i][0];
        printf("%c",c);
        findfirst(c);
      
    }

    printf("\nFOLLOW Sets:\n");
    for(i=0;i<n;i++)
    {
        c=production[i][0];
        printf("%c",c);
        findfollow(c);
      
    }

    printf("\nPredictive Parsing Table construction uses FIRST and FOLLOW sets.\n");

    return 0;
}
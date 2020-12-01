#include "stdio.h"

int main()
{   
    int n1,m1;
    int n2,m2;
    int matrix_A[300][300];
    int matrix_B[300][300];
    int matrix_C[300][300];

    printf("please enter the first matrix's row and column \n");
    scanf("%d  %d", &n1, &m1);
    if((n1 > 300) || (n1 <1 ) || (m1< 1) || ( m1 >300))
        {
            printf("the number is invalid\n");
            return 0;
        }
    printf("your row is %d and your col is %d \n",n1,m1);
    for(int i = 0; i < n1; i++ )
        for(int j = 0; j < m1; j++)
        {
            scanf("%d", &matrix_A[i][j] );
        }

    printf("please enter the second matrix's row and column:\n");
    scanf("%d  %d", &n2, &m2);
    if((n2 > 300) || (n2 <1 ) || (m2 < 1) || ( m2 >300))
        {
            printf("the number is invalid\n");
            return 0;
        }
    printf("your row is %d and your col is %d \n",n2,m2);

        if(m1 != n2 )
    {
        printf("wrong! Please enter the right matrix'row and col\n");
        return 0;
    }
    
    for(int i = 0; i < n2; i++ )
        for(int j = 0; j < m2; j++)
        {
            scanf("%d", &matrix_B[i][j] );
        }


    for(int i = 0; i < n1; i++ )
        for(int j = 0; j < m2; j++)
        {
            matrix_C[i][j] = 0;
            for (int k = 0; k < n2; k++)
            {
                matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
     printf("%d  %d\n", n1, m2);
     for(int i = 0; i < n1; i++ )
       { 
            for(int j = 0; j < m2; j++)
            {
                printf("%d ", matrix_C[i][j] );
            }
            printf("\n");
        }
    return 0;
}
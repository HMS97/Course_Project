#include<stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>    
#include <iterator>     // ostream_iterator
#include "/usr/include/mpi/mpi.h"
const int MAX_NUMBERS = 100;
using namespace std;
int prime_number(int n)
{
    int i;
    
    if(n < 2 ) return 0;
    for(i = 2; i < n; i++)
    {
        if( n % i == 0)
        return 0;
    }
    return 1;
}


int find_min_prime(int number, int rank)
{
    int temp;
    int check_number;

    temp = int(number / 8);
    if( 8 * temp + 2 * rank + 1 <= number )
            temp = temp + 1; 
    while (1)
    {
        check_number = temp * 8 + 2 * rank + 1;
        if (prime_number(check_number))
            break;
        temp += 1 ;
    }
    return check_number;

}



int find_twin_number(int number, int rank)
{
    int n;

    while(1)
    {
        n = find_min_prime(number, rank);
        if( prime_number(n+2))
            return (n,n+2);
        else
            number += 1;
    }


}



vector<int> count_twin(int number, int rank)
{
    int bottom;
    int above;
    int n;

    bottom = rank;
    above = number;
    vector< int> list;
    int * twin = new int[100]();
    int i = 0;

    while(1)
    {
        n = find_min_prime(bottom,rank);
        if (n + 2 > above)
            break;
        if (prime_number(n+2))
            {
                    list.push_back(n);
            }
        bottom += 1;
    
    }

    std::vector<int>::iterator it;
    it = std::unique (list.begin(), list.end());   // 10 20 30 20 10 ?  ?  ?  ?
    list.resize( std::distance(list.begin(),it) ); // 10 20 30 20 10

    return list;
}
    



int main(int argc, char* argv[])
{
    int a ;
    int prime;
    int state = 0;
    int number_amount;
    vector<int> list;
    vector<int> un_list;
    vector< int> list_prime_number;
    vector< int> list_twin_number;
    int numprocs, myid, source;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int number = 20;
    int message;
    if (myid == 4) 
    { 
        for(int i = 0; i<4; i++)
        { 
            MPI_Send(&number, 1, MPI_INT, i, 11,MPI_COMM_WORLD);
            cout<< "send " << number << " to processes "<< i <<" for problem 3"<< endl;
        }
        for(int i = 0; i<4; i++)
        {   
            MPI_Recv(&message, 1, MPI_INT, i, 12,MPI_COMM_WORLD,NULL);         
            list_prime_number.push_back(message);
            MPI_Recv(&message, 1, MPI_INT, i, 13,MPI_COMM_WORLD,NULL);
            list_twin_number.push_back(message);
        }
         int least_prime_number = *min_element(list_prime_number.begin(),list_prime_number.end());
         int least_twin_number = *min_element(list_twin_number.begin(),list_twin_number.end());
        cout<< "the least  prime number  is " << least_prime_number <<endl;
        cout<< "the least twin number  is (" << least_twin_number <<","<< least_twin_number +2<< ")" << endl;

        

    }

    else
    {
        MPI_Recv(&message, 1, MPI_INT, 4, 11,MPI_COMM_WORLD,NULL);
        int prime_number = find_min_prime(message,myid);
        int twin_number =  find_twin_number(message,myid);
        cout<< "the  prime number  is " << prime_number << "  at process  "<< myid<<" for problem 4 "<<endl;
        cout<< "the  twin number  is (" << twin_number <<","<< twin_number +2<< ")" << "  at process  "<< myid<<" for problem 4 "<<endl;
        MPI_Send(&prime_number, 1, MPI_INT, 4, 12,MPI_COMM_WORLD);
        MPI_Send(&twin_number, 1, MPI_INT, 4, 13,MPI_COMM_WORLD);
        list = count_twin(message,myid);

        MPI_Send(&list[0], list.size(), MPI_INT, 4, 14,MPI_COMM_WORLD);

        
    }

    vector<int> twin_list;
   

     if (myid == 4) 
     {  
        // MPI_Status status;

        for(int i = 0;i < 4;i++) 
            {        
                   vector<int> list2(MAX_NUMBERS);
                    MPI_Recv(&list2[0],  MAX_NUMBERS, MPI_INT, i, 14,MPI_COMM_WORLD,NULL);
                    for(int f = 0;f<list2.size();f++)
                   {    
                       if (list2[f] != 0)
                      {  
                        // cout<<list2[f] << "  get from process :"<< i <<endl;
                        twin_list.push_back(list2[f]);}
                    }
            }

   
        for (int i = 0;i < twin_list.size();i++)
            cout<<  twin_list[i]  <<" the twin list"<<endl;
        
        int length = twin_list.size();
        cout<<" the twin list counts number is :  "<<length <<endl;

     }

 	MPI_Finalize();
    return 0;
}



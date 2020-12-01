#include "stdio.h"

struct mynode {
	int const value;
	struct mynode *next;
	struct mynode *prev;
} ;

struct mynode* head; // global variable - pointer to head node.
//Creates a new Node and returns pointer to it. 
struct mynode* GetNewNode( ) {
	struct mynode* newNode
		= (struct mynode*)malloc(sizeof(struct mynode));
	// newNode->value = x;
    scanf("%d", &(newNode->value));
	newNode->prev = NULL;
	newNode->next = NULL;
	return newNode;
}

void Print(struct mynode* head) {
	struct mynode* temp = head;
	printf("Forward: ");
	while(temp != NULL) {
		printf("%d ",temp->value);
		temp = temp->next;
	}
	printf("\n");
}

struct mynode* insertionSort(struct mynode* current) 
{
    struct mynode* previous;
    struct mynode* result;
    printf("%d ",current->value);
		while(current->next != NULL) 
    {
        current = current->next;
         while (current->prev != NULL)
         {
              previous = current->prev;
                if (current->value < previous->value)
               {      
                    if(current->next != NULL)
                        {
                            current->next->prev = previous;
                            previous->next = current->next;
                        }
                    else
                        previous->next = NULL;
                    current->prev = previous->prev;// If previou->prev NULL, NULL
                    current->next = previous;
                    if(previous->prev != NULL)
                        previous->prev->next = current;
                    else
                        current->prev = NULL;                        
                    previous->prev = current;
                }
                else
                {
                    break;
                }   
         }
    }
     while(current->prev!= NULL)
    {
        current=current->prev;
    }
    return current;
}

int main()
{
  	head = NULL; // empty list. set head as NULL. 
    struct mynode *temp;
    struct mynode* result;
    for(int i = 0; i < 1000; i++)
    {
        struct mynode *newnode = GetNewNode();
        if(head == NULL) {
            head = newnode;
            head->prev = NULL;
            temp = head;// the first address
            continue;
            }
        if(newnode->value == 0)
             break;

        newnode->prev = head;
	    head->next = newnode; 
        head = newnode;
       
    }
    head->next = NULL;
    Print(temp);
    result = insertionSort(temp);
    Print(result);
    
    while (result!= NULL)
    {
       current = result;
       result = result->next;
       free(current);
    }
    return 0;
}




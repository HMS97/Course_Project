#include <cstddef>
#include <stdio.h> 
#include <dirent.h> 
#include <string.h>
#include <stdlib.h>
#include <string.h>

int num = 0;

void stringout(const char* sa[], const char *order)
{
    for(int i = 0; i < num - 1;i++)
         {
             for(int j = i+1; j < num; j++)
            {
                int temp = strcmp(sa[i],sa[j]);
                if (*order == 'f')
                   {
                       if(temp>0)
                        { 
                            const char *one = sa[j];
                            sa[j] = sa[i];
                            sa[i] = one;
                        }
                   }
                if (*order == 'b')
                    {
                       if(temp<0)
                        { 
                            const char *one = sa[j];
                            sa[j] = sa[i];
                            sa[i] = one;
                        }
                   }
                
            }
        }
}
int main(int argc, char *argv[]) 
{ 
    struct dirent *de;  // Pointer for directory entry 
    DIR *dr = opendir(argv[1]); 
  
    char *buf;
    char *path_string;
    char **dlist = NULL;
    int count = 0;
    char status;
    int buf_size = 1;
    /* Initial memory allocation */
    buf = (char *) malloc(1024*buf_size);

    if (dr == NULL)  // opendir returns NULL if couldn't open directory 
    { 
        printf("Could not open current directory" ); 
        return 0; 
    } 
    while ((de = readdir(dr)) != NULL) 
           {
               num += 1 ;
           }
    closedir(dr);
    dr = opendir(argv[1]); 
 
    const char* sa[num];
    int times = 0;
    while ((de = readdir(dr)) != NULL) 
           {

            path_string = (char *) malloc(strlen(de->d_name));
            strcpy (path_string, de->d_name); 
             
            for(int i = 0; i <= strlen(de->d_name) ; i++)
                {
                    
                if((count + strlen(de->d_name)) >buf_size*1024)
                {
                    buf_size += 1;
                    buf = (char *)realloc(buf, buf_size*1024);
                }
                if(i == strlen(de->d_name) )
                    {
                    buf[count + i] = '\0';
                    count += strlen(de->d_name) + 1;}
                else
                    buf[count + i] = path_string[i];
                if(i == 0) 
                      {
                          sa[times] = &buf[count + i];
                          times +=1;
                      }

                }
             
                   

            }
    if(argv[2]==NULL)
    {for(int i = 0; i < num;i++)
            printf("%s\n",sa[i]);
            return 0;   
    }
    stringout(sa, argv[2]);

    if (*argv[2] == 'b')
    printf("************ sa list by descending:************ \n");
    if (*argv[2] == 'f')
    printf("************ sa list by ascending:************ \n");
    for(int i = 0; i < num;i++)
            printf("%s\n",sa[i]);
    
    closedir(dr);     
    free(buf);
    free(path_string);
    return 0; 
} 
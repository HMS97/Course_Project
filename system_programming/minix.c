#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>    /* For O_RDWR */
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

int fd;
static char* args[512];
static char line[1024];
static int n = 0; /* number of calls to 'command' */
static void split(char* cmd);

 
#define READ  0
#define WRITE 1
#define BASE_OFFSET 1024


static void cleanup(int n)
{
	int i;
	for (i = 0; i < n; ++i)  
		wait(NULL); 
}

struct minix_super_block {
    unsigned short s_ninodes;
    unsigned short s_nzones;
    unsigned short s_imap_blocks;
    unsigned short s_zmap_blocks;
    unsigned short s_firstdatazone;
    unsigned short s_log_zone_size;
    unsigned int s_max_size;
    unsigned short s_magic;
    unsigned short s_state;
    unsigned int s_zones;
} SUPER_BLOCK;

struct minix_inode {
    unsigned short i_mode;
    unsigned short i_uid;
    unsigned int i_size;
    unsigned int i_time;
    unsigned char i_gid;
    unsigned char i_nlinks;
    unsigned short i_zone[9];
} Inode;

struct minix_dir_entry {
    unsigned short inode;
    char name[0];
} DIR_ENTRY;

void help()
{
    printf("here are the functions we support:\n");
    printf("minimount argument: mount a local minix disk, where argument is a iminix image file, e.g., imagefile.img.\n");
    printf("miniumount: umount the mounted minix disk.\n");
    printf("showsuper: to list the information of the super block.\n");
    printf("traverse [-l]: list the content in the root directory. Note that you don't have to show the entries of . and .. \n");
    printf("showzone [zone number]: show the ASCII content of the specified zone number (1024 bytes for each zone). \n");
    printf("showfile [file name]: show the hex content of the specified file name). \n");
}

void showsuper(int fd, struct minix_super_block * super_block) 
{
	if (args[0] != NULL)
     {
	
        // char temp[600];
        // fd = open("/media/minix/",O_RDONLY);
		lseek(fd, BASE_OFFSET, SEEK_SET);  
		read(fd, super_block, sizeof(super_block));
			printf(
				"number of inodes:            : %d\n"
				"number of zones:             : %d\n"
				"number of imap_blocks:       : %d\n"
				"number of zmap_blocks:       : %d\n"
				"first data zone:             : %d\n"
				"log zone size: 	             : %d\n"
				"max size:                    : %d\n"
				"magic:                       : %d\n"
				"state: 	                     : %d\n"
				"zones: 	                     : %d\n"
				,
				super_block->s_ninodes,  
				super_block->s_nzones,
				super_block->s_imap_blocks,     
				super_block->s_zmap_blocks,
				super_block->s_firstdatazone,
				super_block->s_log_zone_size,
				super_block->s_max_size,
				super_block->s_magic,
				super_block->s_state,
				super_block->s_zones);
		
	 }
}

int minimount(int fd)
{
    if (args[0] != NULL)
     {
		if (strcmp(args[0], "minimount") == 0) 
           { 
               if (args[1] == NULL)
               printf("Please enter the image file \n");
               else
               {
                //    printf("%s",args[1]);
                    char temp_string[500];
                    fd = open(args[1], O_RDONLY);
                    // printf("%d\n", fd );
                    if (fd == -1)
                    printf("open file failed, please check!\n");
                    if (fd == 3)
                    printf("open file successfully!\n");
               }
	        }
    }
    return fd;
}


int miniumount(int fd)
{
    int umount = close(fd);
     if (umount == 0) {
            // fd = -1;
            printf( "Image sucessfully unmounted\n");
     }
     else
     {
         printf("Image unmounted failed \n");
     }
     
    return fd;
}

int traverse(int fd, char l_status)
{

    struct minix_inode *root_dir_inode = (minix_inode * )malloc(sizeof(Inode));
    struct minix_dir_entry *dire_entry =  (minix_dir_entry *)malloc(sizeof(DIR_ENTRY));
    struct minix_inode *inode =  (minix_inode *)malloc(sizeof(Inode));
    // int fd;
    // int l_status = 1;
    // fd = open("/home/huimingsun/Desktop/file_system/imagefile.img", O_RDWR);
    lseek(fd, 1024*5, SEEK_SET);
    read(fd, root_dir_inode, 32);

    for (int i=0; i<7 ; i++)
    {
        // printf("%s\n",root_dir_inode->i_zone[i]);
        if (root_dir_inode->i_zone[i] == '\0')
            break;
        else
            lseek(fd, (root_dir_inode->i_zone[i] * 1024), SEEK_SET);
 
        for ( int temp = 0; temp < 1024;)
        {
            lseek(fd, ((root_dir_inode->i_zone[i] * 1024) + temp), SEEK_SET);
            temp += read(fd, dire_entry, 16);

            if(strcmp(dire_entry->name, ".") == 0 ||strcmp(dire_entry->name, "..") == 0 || strlen(dire_entry->name) == 0 )
                continue;
            else
            {   

                if(l_status)
                 {
                    int inode_number = dire_entry->inode;
                    lseek(fd, ((1024*5)+((inode_number-1)*sizeof(Inode))), SEEK_SET);
                    read(fd, inode, sizeof(Inode));
                    // printf(" %d",inode->i_mode );
                    printf( (inode->i_mode & S_IFDIR) ? "d" : "-");
                    printf( (inode->i_mode & S_IRUSR) ? "r" : "-");
                    printf( (inode->i_mode & S_IWUSR) ? "w" : "-");
                    printf( (inode->i_mode & S_IXUSR) ? "x" : "-");
                    printf( (inode->i_mode & S_IRGRP) ? "r" : "-");
                    printf( (inode->i_mode & S_IWGRP) ? "w" : "-");
                    printf( (inode->i_mode & S_IXGRP) ? "x" : "-");
                    printf( (inode->i_mode & S_IROTH) ? "r" : "-");
                    printf( (inode->i_mode & S_IWOTH) ? "w" : "-");
                    printf( (inode->i_mode & S_IXOTH) ? "x" : "-");
                    printf("   %d",inode->i_uid    );
                    printf("   %d",inode->i_size    );
                    time_t  rawtime = (time_t )inode->i_time;
                    struct tm *info;
                    char buffer[80];
                    time( &rawtime);
                    info = localtime( &rawtime );
                    strftime(buffer,80,"      %b %d %Y ", info);
                    printf("%s  " , buffer );
                    printf("   %s",dire_entry->name );
                    printf("\n");


                }
                else
                printf("%s\n",dire_entry->name );
                    // printf("****************\n");

            }
            
    }          

      }
    free(root_dir_inode);      
    free(dire_entry);
    free(inode);

}


void showzone(int fd, int zone)
{
     if (lseek(fd, 1024*zone, SEEK_SET) < 0) 
        printf("there is no such zone\n");
    else{
        char *buf = (char*)malloc(sizeof(char*));
        // char *buf = (char*)malloc(sizeof(char*));
        printf("      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n");
        for (int i = 0;i<1024;i++)
        {    lseek(fd, 1024*zone+i, SEEK_SET);
        read(fd, buf, 1);
        if (i%16 == 0)
        {
            printf("\n%X", i);
        }
        // if(i<16) printf("%X",i)

        if (isprint(*buf))
            printf("%s  ", buf);
        else
            printf(" ");

        }
        printf("\n");
    }
}

void showfile(int fd, char * file_name)
{
      struct minix_inode *root_dir_inode = (minix_inode*)malloc(sizeof(Inode));
    struct minix_dir_entry *dire_entry =  (minix_dir_entry*)malloc(sizeof(DIR_ENTRY));
    struct minix_inode *inode =  (minix_inode*)malloc(sizeof(Inode));
    // char file_name[] = "B.cla";
    int status = 0;
    lseek(fd, 1024*5, SEEK_SET);
    read(fd, root_dir_inode, 32);

    for (int i=0; i<7 ; i++)
    {
        // printf("%s\n",root_dir_inode->i_zone[i]);
        if (root_dir_inode->i_zone[i] == '\0')
            continue;
        else
            lseek(fd, (root_dir_inode->i_zone[i] * 1024), SEEK_SET);
 
        for ( int temp = 0; temp < 1024;)
        {
            lseek(fd, ((root_dir_inode->i_zone[i] * 1024) + temp), SEEK_SET);
            temp += read(fd, dire_entry, 16);

            if(strcmp(dire_entry->name, ".") == 0 ||strcmp(dire_entry->name, "..") == 0 || strlen(dire_entry->name) == 0 )
                continue;
            else
            {   
                    if (strcmp(dire_entry->name, file_name) == 0)
                    {
                        status = 1;
                        break;
                    } 
            }
    }          
      }
      if (status)
   { 
      
    lseek(fd, ((1024*5)+((dire_entry->inode-1)*sizeof(Inode))), SEEK_SET);
    read(fd, inode, sizeof(Inode));//data inode zone 
    unsigned char *data =(unsigned char *) malloc(1);

    for(int i = 0; i < 7; i++)
       { 
         if (inode->i_zone[i] == '\0')
                    break;
          else
          {
            lseek(fd, inode->i_zone[i]*1024, SEEK_SET);
            int temp_bit = 0;

            while(temp_bit< 1024)
          { 
              temp_bit += read(fd,data,1);
            //   printf("temp_bit  %d\n", temp_bit);
            if (data[0]>16)
                printf("%X ", data[0]);
            if (data[0]<16)
                printf(" %X ", data[0]);
            if (temp_bit%16 == 0)// change a new line if more than 16
                printf("\n");
                
            }

            }
           
          }
    }
    else
    {
        printf("Sorry, we can't find this file in img\n");
    }
 free(root_dir_inode);      
    free(dire_entry);
    free(inode);

}
void pipeline()
{
    if (args[0] == NULL) 
        return;
    if (strcmp(args[0], "help") == 0) 
        help();
    if (strcmp(args[0], "quit") == 0) 
			exit(0);
    if (strcmp(args[0], "minimount") == 0) 
        fd = minimount(fd);
    if (strcmp(args[0], "miniumount") == 0) 
    {
            if(fd < 3)
            printf("you need mount a file first\n");
            else
            {
                fd = miniumount(fd);
            }
    }
    if (strcmp(args[0], "showsuper") == 0) 
    {
            if(fd < 3)
            printf("you need mount a file first\n");
            else
            {
                struct minix_super_block super_block;
                showsuper(fd,&super_block);
            }
    }
     if (strcmp(args[0], "traverse") == 0) 
    {
        //  printf("I can't come in ?\n");
            if(fd < 3)
                 printf("you need mount a file first\n");
            else
            {    
                    int long_status = 0;
                if (args[1] != NULL)
               { 
                    if(strcmp(args[1], "-l") == 0)
                    {
                        long_status = 1;
                     traverse(fd,long_status);
                     }
                     else
                         printf(" you enter wrong command! Please enter traverse -l\n"); 
              }
              else traverse(fd,long_status);
            } 
    }
     if (strcmp(args[0], "showzone") == 0) 
      if(fd < 3)
                 printf("you need mount a file first\n");
            else
            {     if (args[1] != NULL)
                {
                    
                     int zone_num = atoi(args[1]); 
                     showzone(fd,zone_num);

                }
                
                 else
                 printf("please enter zone number\n");
               
            }

      if (strcmp(args[0], "showfile") == 0) 
      if(fd < 3)
                 printf("you need mount a file first\n");
            else
            {     if (args[1] != NULL)
                {
                    
                     char* file_name = args[1]; 
                     showfile(fd,file_name);

                }
                
                 else
                 printf("please enter filename\n");
               
            }
}



int main()
{
    
	printf("SIMPLE SHELL: Type 'exit' or send EOF to exit.\n");
	while (1) {
		/* Print the command prompt */
		printf("$minix: ");
		fflush(NULL);
 
		/* Read a command line */
		if (!fgets(line, 1024, stdin)) 
			return 0;

		char* cmd = line;
        split(cmd);
		pipeline();
		cleanup(n);
		n = 0;
	}
	return 0;
}
 
 
static char* skipwhite(char* s)
{
	while (isspace(*s)) ++s;
	return s;
}
 
static void split(char* cmd)
{
	cmd = skipwhite(cmd);
	char* next = strchr(cmd, ' ');
	int i = 0;
 
	while(next != NULL) {
		next[0] = '\0';
		args[i] = cmd;
		++i;
		cmd = skipwhite(next + 1);
		next = strchr(cmd, ' ');
	}
 
	if (cmd[0] != '\0') {
		args[i] = cmd;
		next = strchr(cmd, '\n');
		next[0] = '\0';
		++i; 
	}
 
	args[i] = NULL;
}


#include "fs.h"
#include "disk.h"

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

// for the 8M file system
#define TFS_MAGIC  0x345f2020

#define NUM_BLOCKS 2048
#define NUM_INODES 512 
#define NUM_DENTRIES_PER_BLOCK 128

#define INODES_PER_BLOCK   128
#define POINTERS_PER_INODE 5
#define POINTERS_PER_BLOCK 1024

// file type
#define REGULAR 1
#define DIR 2

struct tfs_superblock {
	int signature ; // Signature 
	int num_blocks; // number of blocks; same as NUM_BLOCKS in this project
	int num_inodes; // number of inodes; same as NUM_INODES in this project
	int root_inode; // inode number of root directory ; use 1
	unsigned int block_in_use[NUM_BLOCKS/sizeof(unsigned int)];
	unsigned int inode_in_use[NUM_INODES/sizeof(unsigned int)]; 
};

struct tfs_dir_entry {
	int valid; 
	char fname[24];
        int inum;
};

struct tfs_inode {
	int type;
	int size;
	int direct[POINTERS_PER_INODE];
	int indirect;
};

union tfs_block {
	struct tfs_superblock super;
	struct tfs_inode inode[INODES_PER_BLOCK];
	struct tfs_dir_entry dentry[NUM_DENTRIES_PER_BLOCK]; 
	int pointers[POINTERS_PER_BLOCK];
	char data[DISK_BLOCK_SIZE];
};

struct tfs_superblock super;
int mounted = 0;

int tfs_format()
// Creates a new file system on the disk, destroying any data already present. Sets four
// blocks for inodes, clears the inode table, and writes the super block. Returns one on success, zero
// otherwise. Note that formatting a files system does not cause it to be mounted. Also, an attempt to
// format an already-mounted disk should do nothing and return failure
{
	if (mounted == 0 )
		{
			printf("\n can not format a mounted file system! \n");
			return 0;
		}
	else
	{	
		// format super block
		union tfs_block block;
		disk_read(0,block.data);
		block.super.signature = TFS_MAGIC;
		block.super.num_blocks = NUM_BLOCKS;
		block.super.num_inodes = NUM_INODES;

		for(int i=0; i<NUM_BLOCKS; i++)
			block.super.block_in_use[i/32] = block.super.block_in_use[i/32] &0;
		for(int i=0; i<NUM_INODES; i++)
			block.super.inode_in_use[i/32] = block.super.inode_in_use[i/32] &0;
		// format inode table
		// printf("\n format the super block successfully! \n");
		disk_write(0,block.data);
		for( int i = 0; i < 5; i++)
		{
			disk_read(1+i,block.data);
			for (int in  = 0; in < INODES_PER_BLOCK; in++)
			{

				if (i == 0)
					block.inode[in].type = DIR;
				else
				{
					block.inode[in].type = REGULAR;
					block.inode[in].size = DISK_BLOCK_SIZE;
					block.inode[in].indirect = 0;

					for (int point = 0; point < POINTERS_PER_INODE; point++ )
						block.inode[in].direct[point] = 0;
				}
			}
			disk_write(1+i,block.data);

		}
		printf("\n format the file system successfully! \n");
		return 1;
	}
	
}

void tfs_debug()
{ 
        int i;
        int b_in_use = 0;
        int i_in_use = 0;

	union tfs_block block;

	disk_read(0,block.data);//super block

        // check signature
        if(block.super.signature  == TFS_MAGIC)
		printf("      signature is valid\n");
	else
		printf("      signature is invalid\n");

	for(i=0; i<NUM_BLOCKS; i++)
           if(block.super.block_in_use[i/sizeof(unsigned int)] & (1 <<(i%sizeof(unsigned int))))
		 b_in_use++ ;  
        printf("      %d blocks in use \n", b_in_use); 

        // count inodes in use 
    for(i=0; i<NUM_INODES; i++)
           if(block.super.inode_in_use[i/sizeof(unsigned int)] & (1 <<(i%sizeof(unsigned int))))
         i_in_use++ ;
        printf("      %d inodes in use \n", i_in_use);
        
	// explore root directory
    union tfs_block block1; // root inode table
    disk_read(1,block1.data);    
    union tfs_block block2; //first file dictionary block
    disk_read(block1.inode[block.super.root_inode].direct[0], block2.data);

	printf("\n%d num\n", block1.inode[block.super.root_inode].direct[0] );
    printf("root inode %d:\n      size: %d bytes\n      direct blocks: %d\n",
									block.super.root_inode,
									block1.inode[block.super.root_inode].size, 
									block1.inode[block.super.root_inode].direct[0]);

    for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++){

        if (!entry->valid) continue;
		if (!entry->inum) continue;
		union tfs_block block_inode; // root inode table
    	disk_read(1+ entry->inum/INODES_PER_BLOCK,block_inode.data);    

        printf("\n%s inode %d:\n      size: %d bytes", entry->fname, 
														entry->inum,
														block_inode.inode[entry->inum%INODES_PER_BLOCK].size);

        if (!block1.inode[entry->inum%INODES_PER_BLOCK].size) continue;

        int blocks = block_inode.inode[entry->inum%INODES_PER_BLOCK].size / DISK_BLOCK_SIZE + (block_inode.inode[entry->inum%INODES_PER_BLOCK].size % DISK_BLOCK_SIZE ? 1 : 0);
        printf("\n      direct blocks:");
        for (i = 0; i < POINTERS_PER_INODE && i < blocks; i++) {
            printf(" %d", block_inode.inode[entry->inum%INODES_PER_BLOCK].direct[i]);
        }

        if (blocks <= POINTERS_PER_INODE) continue;
        printf("\n      indirect block: %d\n      indirect data blocks:", block_inode.inode[entry->inum%INODES_PER_BLOCK].indirect);

		union tfs_block block3;
        disk_read(block_inode.inode[entry->inum%INODES_PER_BLOCK].indirect, block3.data);
        for (i = POINTERS_PER_INODE; i < blocks; i++) {
            printf(" %d", block3.pointers[i - POINTERS_PER_INODE]);
        }

		
        
    }
    putchar('\n');


}




int tfs_mount()
{

	union tfs_block block;
	disk_read(0, block.data);//super block
	super = block.super;
	if (super.signature ==  TFS_MAGIC)
		{
			mounted = 1;
			printf("\n successful mounted! \n");
			return 1;
		}
	else
	{
		printf("\n mounted! failed \n");

		return 0;
	}
	
}

int tfs_create(const char *filename )
{
	return 0;
}

int tfs_delete(const  char *filename )
{
	return 0;
}

int tfs_get_inumber(const  char *filename )
{
	return 0;
}

int tfs_getsize(const  char *filename )
{
	return -1;
}

int tfs_read( int inumber,  char *data, int length, int offset )
{
	return 0;
}

int tfs_write( int inumber, const char *data, int length, int offset )
{
	return 0;
}

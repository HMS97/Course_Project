
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

// char * data[DISK_BLOCK_SIZE] superconv;


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

union tfs_block super;

int mounted = 0;


int tfs_format()
// follow bottom-up rule, we first format the entry dirctory and then inode table, and super block.
{
	if (mounted == 0 )
		{
			printf("\n can not format a mounted file system! \n");
			return 0;
		}
	else
	{
		//format dictory entry
		union tfs_block block;
		disk_read(0,block.data);
		union tfs_block block1; // root inode table
		disk_read(1,block1.data);    
		union tfs_block block2; //first file dictionary block
		disk_read(block1.inode[block.super.root_inode].direct[0], block2.data);
    	for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++)
			{
				entry->valid = 0;
				entry->inum = 0;
			}

		// format inode table
		// printf("\n format the super block successfully! \n");
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

		// format super block
		block.super.signature = TFS_MAGIC;
		block.super.num_blocks = NUM_BLOCKS;
		block.super.num_inodes = NUM_INODES;
		block.super.root_inode = 1;
		for(int i=0; i<NUM_BLOCKS; i++)
			block.super.block_in_use[i/32] &=  0;
		for(int i=0; i<NUM_INODES; i++)
			block.super.inode_in_use[i/32] &=  0;
		for (int i =0 ; i < 4; i++)
		block.super.block_in_use[i/32] |= (1 << (i%32));
		for (int i =0 ; i < 2; i++)
		block.super.inode_in_use[i/32] |= (1 << (i%32));
		
		// format inode table
		disk_write(0,block.data);
		disk_write(block1.inode[block.super.root_inode].direct[0], block2.data);



		printf("\n format the file system successfully! \n");
		return 1;
	}
	
}

void tfs_debug()
{ 
     
	 	if (!mounted)
	{
		printf(" \n You need to mount the disk first!  \n ");
		return 0;
		
	}
	else
	{
	    int i;
        int b_in_use = 0;
        int i_in_use = 0;
    
		union tfs_block block; // root inode table
		disk_read(0, block.data);//super block
		super.super = block.super;

        // check signature
        if(super.super.signature  == TFS_MAGIC)
		printf("      signature is valid\n");
	else
		printf("      signature is invalid\n");

	for(i=0; i<NUM_BLOCKS; i++)
          { if(super.super.block_in_use[i/32] & (1 <<(i%32)))
			b_in_use++ ;}  
        printf("      %d blocks in use \n", b_in_use); 
    for(i=0; i<NUM_INODES; i++)
           {if(super.super.inode_in_use[i/32] & (1 <<(i%32)))
         	i_in_use++ ;}
        printf("      %d inodes in use \n", i_in_use);
        

	// explore root directory
    union tfs_block block1; // root inode table
    disk_read(1,block1.data);    
    union tfs_block block2; //first file dictionary block
    disk_read(block1.inode[super.super.root_inode].direct[0], block2.data);

    printf("root inode %d:\n      size: %d bytes\n      direct blocks: %d\n",
									super.super.root_inode,
									block1.inode[super.super.root_inode].size, 
									block1.inode[super.super.root_inode].direct[0]);

    for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++){

        if (!entry->valid) continue;
		if (!entry->inum) continue;
		union tfs_block block_inode; // root inode table
    	disk_read(super.super.root_inode + entry->inum/INODES_PER_BLOCK,block_inode.data);    

        printf("\n%s inode %d:\n      size: %d bytes", entry->fname, 
														entry->inum,
														block_inode.inode[entry->inum%INODES_PER_BLOCK].size);

        int blocks = block_inode.inode[entry->inum%INODES_PER_BLOCK].size / DISK_BLOCK_SIZE + (block_inode.inode[entry->inum%INODES_PER_BLOCK].size % DISK_BLOCK_SIZE ? 1 : 0);
        if (!block_inode.inode[entry->inum%INODES_PER_BLOCK].size) continue;
	    printf("\n      direct blocks:");
        for (int i = 0; i < POINTERS_PER_INODE && i < blocks; i++) {
            printf(" %d", block_inode.inode[entry->inum%INODES_PER_BLOCK].direct[i]);
        }

        if (blocks < POINTERS_PER_INODE) continue;
        printf("\n      indirect block: %d\n      indirect data blocks:", block_inode.inode[entry->inum%INODES_PER_BLOCK].indirect);

		union tfs_block block3;
        disk_read(block_inode.inode[entry->inum%INODES_PER_BLOCK].indirect, block3.data);
        for (int i = POINTERS_PER_INODE; i < blocks; i++) {
            printf(" %d", block3.pointers[i - POINTERS_PER_INODE]);
        }

		
        
    }
    putchar('\n');
	}

	

}




int tfs_mount()
{

	union tfs_block block;
	disk_read(0, block.data);//super block
	super.super = block.super;
	if (super.super.signature ==  TFS_MAGIC)
		{
			mounted = 1;
			// printf("\n successful mounted! \n");
			return 1;
		}
	else
	{
		printf("\n mounted! failed \n");

		return 0;
	}
	
}

int tfs_getfree_inode()
{

	for ( int k = 5; k<NUM_INODES; k++)
	{
		if(!(super.super.inode_in_use[k/32] & (1 << (k%32))) )
			{
				super.super.inode_in_use[k/32] |= (1 << (k%32));
				disk_write(0, super.data);
				return k;
			}
	}
	return -1;
}


int tfs_getfree_block()
{

	for ( int k = 5; k<NUM_BLOCKS; k++)
	{
		if(!(super.super.block_in_use[k/32] & (1 << (k%32))) )
			{
				super.super.block_in_use[k/32] |= (1 << (k%32));
				disk_write(0, super.data);
				return k;
			}
	}
	return -1;
}

int tfs_create(const char *filename )
{

	if (!mounted)
	{
		printf(" \n You need to mount the disk first!  \n ");
		return 0;
		
	}
	else
	{
		int free_inode = tfs_getfree_inode();
		int free_block = tfs_getfree_block();
		if (free_inode == -1)
		{
			printf("\n There are no free inode! \n");
			return 0;
		}
		if(free_block == -1 )
		{
			printf("\n There are no free block! \n");
			return 0;
		}
		union tfs_block block1; // root inode table
		disk_read(1,block1.data);    
		union tfs_block block2; //first file dictionary block
		disk_read(block1.inode[super.super.root_inode].direct[0], block2.data);
    	for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++)
		{
			// printf("\n ```````````````````````` \n");
			// printf("\n fentry->valid %d \n", entry->valid);

        	if (entry->valid) continue;

			strcpy(entry->fname,filename);
			entry->inum = free_inode;
			entry->valid = 1;
			union tfs_block block_inode; // root inode table
    		disk_read(super.super.root_inode + free_inode/INODES_PER_BLOCK, block_inode.data);    
			//  write down the size and block for the inode, pointer 0 because our file'size is 0
			block_inode.inode[free_inode%INODES_PER_BLOCK].size = 0;
			block_inode.inode[free_inode%INODES_PER_BLOCK].type = REGULAR;
			// block_inode.inode[free_inode%INODES_PER_BLOCK].direct[0] =  free_block;

			// add used inode number and save data to superblock disk
			super.super.inode_in_use[free_inode/32] |= (1 << (free_inode%32));
			// super.super.block_in_use[free_block/32] |= (1 << (free_block%32));

			disk_write(0, super.data);
			disk_write(block1.inode[super.super.root_inode].direct[0], block2.data);
			// save the data in inode table and dict entry
    		disk_write(super.super.root_inode + free_inode/INODES_PER_BLOCK,block_inode.data);    

			return entry->inum;

		}

	}
}

int tfs_delete(const  char *filename )
{
	if (!mounted)
	{
		printf(" \n You need to mount the disk first!  \n ");
		return 0;
		
	}
	else
	{

	int positive = 0;
	union tfs_block block1;
    disk_read(1,block1.data);    
    union tfs_block block2;
    disk_read(block1.inode[super.super.root_inode].direct[0],block2.data);
    for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++){
            if(strcmp(filename,entry->fname) == 0){

				positive = entry->inum;
				free_inumber(entry->inum);
					entry->inum = 0;
					entry->valid = 0;
    				disk_write(block1.inode[super.super.root_inode].direct[0],block2.data);

		
					return positive;

			}
            
    }
		printf("\n Therer is no file named %s \n", filename);
		return 0;
	}
}

int tfs_get_inumber(const  char *filename )
{
		if (!mounted)
	{
		printf(" \n You need to mount the disk first!  \n ");
		return 0;
		
	}
	else
	{
		union tfs_block block; // root inode table

		disk_read(0, block.data);//super block
		super.super = block.super;
		mounted = 1;

		union tfs_block block1;
		disk_read(1,block1.data);    
		union tfs_block block2;
		disk_read(block1.inode[super.super.root_inode].direct[0],block2.data);
		for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++){
			if(strcmp(filename,entry->fname) == 0)
			{
				return entry->inum;
			}
		}
	}
			return 0;

}

int tfs_getsize(const  char *filename )
{

		if (!mounted)
	{
		printf(" \n You need to mount the disk first!  \n ");
		return -1;
		
	}
	else
	{
		int inum = tfs_get_inumber(filename);
		union tfs_block block_inode;
		disk_read(super.super.root_inode + inum/INODES_PER_BLOCK, block_inode.data);
		return block_inode.inode[inum%INODES_PER_BLOCK].size;
	}
}

int tfs_read( int inumber,  char *data, int length, int offset )
{


		char * myfilename ;
		
		 union tfs_block block1; // root inode table
		disk_read(1,block1.data);    
		union tfs_block block2; //first file dictionary block
		disk_read(block1.inode[super.super.root_inode].direct[0], block2.data);


    	for (struct tfs_dir_entry* entry = block2.dentry; entry != block2.dentry + NUM_DENTRIES_PER_BLOCK; entry++)
			if (entry->inum == inumber)
				{
					strcpy(myfilename, entry->fname);
					break;
				}
		int file_size = tfs_getsize(myfilename) ;
		if (offset >= file_size) return -1;
		union tfs_block block_inode;
		disk_read(super.super.root_inode + inumber/INODES_PER_BLOCK, block_inode.data);

		int data_pointer = 0;
		int block_data_pointer = 0;

		if (offset < POINTERS_PER_INODE*DISK_BLOCK_SIZE)
			{
			for (int i = 0; i< (file_size-offset)/DISK_BLOCK_SIZE+1 && i< POINTERS_PER_INODE && i < length/DISK_BLOCK_SIZE ;i++)
			{	
				if(offset + data_pointer ==  POINTERS_PER_INODE*DISK_BLOCK_SIZE)
					break;
				disk_read(block_inode.inode[inumber%INODES_PER_BLOCK].direct[offset/DISK_BLOCK_SIZE + i], block2.data ); 
				block_data_pointer = 0;
				while(1)
				{

					data[data_pointer] = block2.data[block_data_pointer];
					data_pointer++;
					block_data_pointer++;
					if (offset + data_pointer == file_size || data_pointer == length || offset + data_pointer == POINTERS_PER_INODE*DISK_BLOCK_SIZE)
						return data_pointer;
					if (block_data_pointer == DISK_BLOCK_SIZE || offset + data_pointer ==  POINTERS_PER_INODE*DISK_BLOCK_SIZE)
						break;
					// if(offset + data_pointer == POINTERS_PER_INODE*DISK_BLOCK_SIZE) printf("\n wdnmd  %d \n",offset + data_pointer );
				
				}
				}
		}
			else if (offset >= POINTERS_PER_INODE*DISK_BLOCK_SIZE)
			{for (int i = 0; i< length/DISK_BLOCK_SIZE ; i++ )
			{	

				disk_read(block_inode.inode[inumber%INODES_PER_BLOCK].indirect, block2.data);
				disk_read(block2.pointers[i + (offset-POINTERS_PER_INODE*DISK_BLOCK_SIZE)/DISK_BLOCK_SIZE] ,block1.data ); 
				block_data_pointer = 0;
				while(1)
				{
					data[data_pointer] = block2.data[block_data_pointer];
					data_pointer++;
					block_data_pointer++;
					if (offset + data_pointer == file_size || data_pointer == length )
					return data_pointer;
					if (block_data_pointer == DISK_BLOCK_SIZE)
					break;
				}
			}}
			
		
	return 0;
}

void free_inumber(int inumber)
{
	union tfs_block block_inode;
    disk_read(super.super.root_inode + inumber/INODES_PER_BLOCK,block_inode.data);    
	// block_inode.inode[inumber%INODES_PER_BLOCK].size = 0;
	int blocks = block_inode.inode[inumber%INODES_PER_BLOCK].size / DISK_BLOCK_SIZE + (block_inode.inode[inumber%INODES_PER_BLOCK].size % DISK_BLOCK_SIZE ? 1 : 0);
	for (int i = 0; i< POINTERS_PER_INODE;i++)
		{
			super.super.block_in_use[block_inode.inode[inumber%INODES_PER_BLOCK].direct[i]/32] &= ~(1 << (block_inode.inode[inumber%INODES_PER_BLOCK].direct[i]%32));
			block_inode.inode[inumber%INODES_PER_BLOCK].direct[i] = 0;
		}
	if ((block_inode.inode[inumber%INODES_PER_BLOCK].size <=  POINTERS_PER_INODE*DISK_BLOCK_SIZE)  )
	{	super.super.inode_in_use[inumber/32] &= ~(1 << (inumber%32));
		disk_write(0,super.data);}


	if (block_inode.inode[inumber%INODES_PER_BLOCK].size > POINTERS_PER_INODE*DISK_BLOCK_SIZE )
		{
		union tfs_block block2;
		disk_read(block_inode.inode[inumber%INODES_PER_BLOCK].indirect, block2.data);

		for (int i = 0; i< POINTERS_PER_INODE;i++)
		{
			super.super.block_in_use[block_inode.inode[inumber%INODES_PER_BLOCK].direct[i]/32] &= ~(1 << (block_inode.inode[inumber%INODES_PER_BLOCK].direct[i]%32));
			block_inode.inode[inumber%INODES_PER_BLOCK].direct[i] = 0;
		}

		super.super.inode_in_use[inumber/32] &= ~(1 << (inumber%32));
	
		if (blocks >= POINTERS_PER_INODE)
		{
			// clear indirect point block
			union tfs_block block_indirect;
			disk_read(block_inode.inode[inumber%INODES_PER_BLOCK].indirect, block_indirect.data);
			for (int i = POINTERS_PER_INODE; i < blocks; i++ )
				{
					super.super.block_in_use[block_indirect.pointers[i - POINTERS_PER_INODE]/32] &= ~(1 << (block_indirect.pointers[i - POINTERS_PER_INODE]%32));
					block_indirect.pointers[i - POINTERS_PER_INODE] = 0;
				}
			//clear inode indirect
			super.super.inode_in_use[block_inode.inode[inumber%INODES_PER_BLOCK].indirect/32] &= ~(1 << (block_inode.inode[inumber%INODES_PER_BLOCK].indirect%32));
			block_inode.inode[inumber%INODES_PER_BLOCK].indirect = 0;
			block_inode.inode[inumber%INODES_PER_BLOCK].size = 0;
			disk_write(block_inode.inode[inumber%INODES_PER_BLOCK].indirect, block_indirect.data);

		}
		disk_write(0,super.data);

		}

}


int tfs_write( int inumber, const char *data, int length, int offset )
{
		
		if (offset == 0)
		{
			free_inumber(inumber);
			super.super.inode_in_use[inumber/32] |= (1 << (inumber%32));
			disk_write(0, super.data);
		}

		int data_pointer = 0;
		int block_data_pointer = 0;

		union tfs_block block_inode; // root inode table
		union tfs_block block2; // root inode table

    	disk_read(super.super.root_inode + inumber/INODES_PER_BLOCK,block_inode.data);    
		block_inode.inode[inumber%INODES_PER_BLOCK].size = offset+length;
		disk_write(super.super.root_inode + inumber/INODES_PER_BLOCK, block_inode.data);    

 	
		if(offset   <= POINTERS_PER_INODE*DISK_BLOCK_SIZE )
			{
        		int blocks = length / DISK_BLOCK_SIZE + (length % DISK_BLOCK_SIZE ? 1 : 0);
				for (int i = offset/DISK_BLOCK_SIZE ; i< blocks && i<POINTERS_PER_INODE ; i++)
				{
					if (i > 4) break;
					int free_block = tfs_getfree_block();
					block_inode.inode[inumber%INODES_PER_BLOCK].direct[i] = free_block;
					disk_read(block_inode.inode[inumber%INODES_PER_BLOCK].direct[i], block2.data );
					block_data_pointer = 0;
					while(1)
					{
						block2.data[block_data_pointer] = data[data_pointer];
						data_pointer++;
						block_data_pointer++;
						if (data_pointer == length)
							{
								disk_write(block_inode.inode[inumber%INODES_PER_BLOCK].direct[i], block2.data );
								return length;
							}
							
						if (block_data_pointer%DISK_BLOCK_SIZE == 0)
							{
								disk_write(block_inode.inode[inumber%INODES_PER_BLOCK].direct[i], block2.data );
								break;}
					}
				}

			}

		union tfs_block block3;
		int indirect_blocks = 0;
        // disk_read(block_inode.inode[entry->inum%INODES_PER_BLOCK].indirect, block3.data);
		if(offset<POINTERS_PER_INODE*DISK_BLOCK_SIZE)
			indirect_blocks = (length - DISK_BLOCK_SIZE )/DISK_BLOCK_SIZE ;
		else if (offset > POINTERS_PER_INODE*DISK_BLOCK_SIZE)
			indirect_blocks = length/DISK_BLOCK_SIZE ;

		int free_indirect_block = tfs_getfree_block();
		block_inode.inode[inumber%INODES_PER_BLOCK].indirect = free_indirect_block;
		disk_read(free_indirect_block, block3.data);
		disk_write(super.super.root_inode + inumber/INODES_PER_BLOCK, block_inode.data);    
		for(int i=0;i<indirect_blocks&& i< POINTERS_PER_BLOCK;i++)
		{
			int free_block = tfs_getfree_block();
			block3.pointers[i] = free_block;
			disk_read(block3.pointers[i], block2.data );
			data_pointer = 0;
			while(1)
			{
				block2.data[block_data_pointer] = data[data_pointer];
				block_data_pointer++;
				data_pointer++;
				if (data_pointer == length)
					{
						disk_write(block3.pointers[i], block3.data );
						disk_write(super.super.root_inode + inumber/INODES_PER_BLOCK, block_inode.data);    
						return length;
					}
					if (block_data_pointer%DISK_BLOCK_SIZE == 0)
					{
						disk_write(block3.pointers[i], block3.data );
						break;}
			}

		}


	return 0;
}

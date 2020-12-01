#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
//将字符串以单个字符循环输出
//1、申请锁
pthread_mutex_t mutex;
void my_print(char *str)
{
	while(*str!=0)
	{
		printf("%c", *str++);
		fflush(stdout);
		sleep(1);
	}
	printf("\n");
}
void *dealfun(void *arg)
{
	//3、上锁
	pthread_mutex_lock(&mutex);
	my_print((char *)arg);
	//4、解锁
	pthread_mutex_unlock(&mutex);
	return NULL;
}
void *dealfun2(void *arg)    
{
	sleep(2);
	//3、上锁
	pthread_mutex_lock(&mutex);
	my_print((char *)arg);
	//4、解锁
	pthread_mutex_unlock(&mutex);
	return NULL;
}
int main(int argc, char const *argv[])
{
	//2、初始化
	pthread_mutex_init(&mutex,NULL);
	void *arg=NULL;
	pthread_t tid,tid2;
	pthread_create(&tid,NULL,dealfun,"hello");
	pthread_create(&tid2,NULL,dealfun2,"world");	
	pthread_join(tid,&arg);//等待，阻塞
	pthread_join(tid2,&arg);//等待，阻塞	
	//5、销毁锁
	pthread_mutex_destroy(&mutex);
	return 0;
}
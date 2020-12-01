#include <stdio.h>
#include <string.h>
#include "tests/threads/tests.h"
#include "threads/init.h"
#include "threads/malloc.h"
#include "threads/synch.h"
#include "threads/thread.h"
#include "devices/timer.h"

#define N 10
#define BUFSIZE 3
static int buffer[BUFSIZE];
static int bufin = 0, bufout = 0;

struct semaphore empty, full;
struct lock buffer_lock;

static void producer(void *arg1 UNUSED) {
   int i, item;

   for (i = 0; i < N; i++) {
      sem_down(&empty);
      item = i*i*i*i;
      lock_acquire(&buffer_lock);
      buffer[bufin] = item;
      bufin = (bufin + 1) % BUFSIZE;
      lock_release(&buffer_lock);
      printf("p: put item %d\n", item);
      sem_up(&full);
   }
}

static void consumer(void *arg2 UNUSED) {
   int i, item;

   for (i = 0; i < N; i++) {
      sem_down(&full);
      lock_acquire(&buffer_lock);
      item = buffer[bufout];
      bufout = (bufout + 1) % BUFSIZE;
      lock_release(&buffer_lock);
      printf("c: get item %d\n",item);
      sem_up(&empty);
   }
}

void test_prd_cns(void) {
   lock_init(&buffer_lock);
   sem_init(&full, 0);
   sem_init(&empty, BUFSIZE);
   thread_create("producer_pid", PRTY_MAX, producer, NULL);
   thread_create("consumer_pid", PRTY_MIN, consumer, NULL);

   thread_exit();
}

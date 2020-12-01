/* Tests that the highest-priority thread waiting on a semaphore
   is the first to wake up. */

#include <stdio.h>
#include "tests/threads/tests.h"
#include "threads/init.h"
#include "threads/malloc.h"
#include "threads/synch.h"
#include "threads/thread.h"
#include "devices/timer.h"

static thread_func priority_sem_thread;
static struct semaphore sema;

void
test_priority_sema (void) 
{
  int i;
  
  /* This test does not work with the MLFQS. */
  ASSERT (!thread_mlfqs);

  sem_init (&sema, 0);
  thread_set_priority (PRTY_MIN);
  for (i = 0; i < 10; i++) 
    {
      int priority = PRTY_DEFAULT - (i + 3) % 10 - 1;
      char name[16];
      snprintf (name, sizeof name, "priority %d", priority);
      thread_create (name, priority, priority_sem_thread, NULL);
    }

  for (i = 0; i < 10; i++) 
    {
      sem_up (&sema);
      msg ("Back in main thread."); 
    }
}

static void
priority_sem_thread (void *aux UNUSED) 
{
  sem_down (&sema);
  msg ("Thread %s woke up.", thread_name ());
}

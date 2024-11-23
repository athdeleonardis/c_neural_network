#define WINDOWS

#ifdef WINDOWS
  #include <windows.h>
#endif
#ifdef UNIX
  #include <pthread.h>
#endif

typedef struct {
#ifdef WINDOWS
  void *(*func)(void *);
  void *data;
  HANDLE windows_handle;
#endif
#ifdef UNIX
  pthread_t unix_pthread;
#endif
} thread_wrapper_t;

typedef struct {
#ifdef WINDOWS
  HANDLE windows_handle;
#endif
#ifdef UNIX
  pthread_mutex_t unix_pthread_mutex;
#endif
} mutex_wrapper_t;

void thread_wrapper_create(thread_wrapper_t *thread_wrapper, void *(*func)(void *), void *data);
void thread_wrapper_join(thread_wrapper_t *thread_wrapper);

void mutex_wrapper_create(mutex_wrapper_t *mutex_wrapper);
void mutex_wrapper_lock(mutex_wrapper_t *mutex_wrapper);
void mutex_wrapper_unlock(mutex_wrapper_t *mutex_wrapper);
void mutex_wrapper_close(mutex_wrapper_t *mutex_wrapper);

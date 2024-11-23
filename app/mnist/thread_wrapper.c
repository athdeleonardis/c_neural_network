#include "thread_wrapper.h"
#include "../../src/error.h"

#ifdef WINDOWS
DWORD WINAPI ThreadFunctionWrapper(void *thread_wrapper_ptr) {
  thread_wrapper_t *thread_wrapper = (thread_wrapper_t *)thread_wrapper_ptr;
  thread_wrapper->func(thread_wrapper->data);
  return 0;
}
#endif

void thread_wrapper_create(thread_wrapper_t *thread_wrapper, void *(*func)(void *), void *data) {
#ifdef WINDOWS
  thread_wrapper->func = func;
  thread_wrapper->data = data;
  thread_wrapper->windows_handle = CreateThread(NULL, 0, ThreadFunctionWrapper, thread_wrapper, 0, NULL);
  cnd_make_error(thread_wrapper->windows_handle == NULL, "Failed to create thread.\n");
#endif
#ifdef UNIX
  if (pthread_create(&thread_wrapper->unix_pthread, NULL, func, data))
    make_error("Failed to create thread.\n");
#endif
}

void thread_wrapper_join(thread_wrapper_t *thread_wrapper) {
#ifdef WINDOWS
  WaitForSingleObject(thread_wrapper->windows_handle, INFINITE);
  if (!CloseHandle(thread_wrapper->windows_handle))
    make_error("Failed to close windows thread after joining.\n");
#endif
#ifdef UNIX
  if (pthread_join(thread_wrapper->unix_pthread, NULL))
    make_error("Failed to join with thread.\n");
#endif
}

void mutex_wrapper_create(mutex_wrapper_t *mutex_wrapper) {
#ifdef WINDOWS
  mutex_wrapper->windows_handle = CreateMutex(NULL, FALSE, NULL);
  cnd_make_error(mutex_wrapper->windows_handle == NULL, "Failed to initialize mutex.\n");
#endif
#ifdef UNIX
  if (pthread_mutex_init(&mutex_wrapper->unix_pthread_mutex, NULL))
    make_error("Failed to initialize mutex.\n");
#endif
}

void mutex_wrapper_lock(mutex_wrapper_t *mutex_wrapper) {
#ifdef WINDOWS
  DWORD dwWaitResult = WaitForSingleObject(mutex_wrapper->windows_handle, INFINITE);
  switch (dwWaitResult) {
    case WAIT_OBJECT_0:
      break;
    case WAIT_ABANDONED:
      make_error("Attempting to lock abandoned mutex.\n");
      break;
  }
#endif
#ifdef UNIX
  if (pthread_mutex_lock(&mutex_wrapper->unix_pthread_mutex))
    make_error("Failed to lock mutex.\n");
#endif
}

void mutex_wrapper_unlock(mutex_wrapper_t *mutex_wrapper) {
#ifdef WINDOWS
  if (!ReleaseMutex(mutex_wrapper->windows_handle))
    make_error("Failed to unlock mutex.\n");
#endif
#ifdef UNIX
  if (pthread_mutex_unlock(&mutex_wrapper->unix_pthread_mutex))
    make_error("Failed to unlock mutex.\n");
#endif
}

void mutex_wrapper_close(mutex_wrapper_t *mutex_wrapper) {
#ifdef WINDOWS
  if (!CloseHandle(mutex_wrapper->windows_handle))
    make_error("Failed to close mutex.\n");
#endif
#ifdef UNIX
  if (pthread_mutex_destroy(&mutex_wrapper->unix_pthread_mutex))
    make_error("Failed to close mutex.\n");
#endif
}
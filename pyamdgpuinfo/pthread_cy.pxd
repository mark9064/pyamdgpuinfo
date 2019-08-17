cdef extern from "pthread.h" nogil:
    int PTHREAD_CREATE_DETACHED "PTHREAD_CREATE_DETACHED"
    ctypedef int pthread_t
    ctypedef struct pthread_attr_t:
        pass
    int pthread_create(pthread_t *thread, pthread_attr_t *attr, void *start_routine (void *), void *arg)
    int pthread_attr_init(pthread_attr_t *__attr)
    int pthread_attr_setdetachstate(pthread_attr_t *__attr, int __detachstate)

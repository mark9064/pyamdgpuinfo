cdef extern from "xf86drm.h" nogil:
    # note: _drmVersion defines more fields but we only need name
    struct _drmVersion:
        char *name
    ctypedef _drmVersion *drmVersionPtr

    drmVersionPtr drmGetVersion(int fd)
    void drmFreeVersion(drmVersionPtr)

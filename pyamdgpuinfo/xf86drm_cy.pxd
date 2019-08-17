cdef extern from "xf86drm.h" nogil:
    struct _drmVersion:
        int version_major
        int version_minor
        int version_patchlevel
        int name_len
        char *name
        int date_len
        char *date
        int desc_len
        char *desc
    ctypedef _drmVersion *drmVersionPtr

    drmVersionPtr drmGetVersion(int fd)
    void drmFreeVersion(drmVersionPtr)

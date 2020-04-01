//
// Created by user on 3/25/20.
//

#ifndef SPTAGLIB_MALLOC_ALIGNED_HPP
#define SPTAGLIB_MALLOC_ALIGNED_HPP

static inline void* simde_mm_malloc (size_t size, size_t alignment)	{
    // This works on posix systems
    // For Windows users: C11 should have aligned_alloc(...) that could replace simde_mm_malloc(...), but simde requires C99
    void *ptr;
    if (alignment == 1) return malloc (size);
    if (alignment == 2 || (sizeof (void *) == 8 && alignment == 4)) alignment = sizeof (void *);
    if (posix_memalign (&ptr, alignment, size) == 0) return ptr;
    else return NULL;
}
static inline void  simde_mm_free (void * ptr) {free (ptr);}

#endif //SPTAGLIB_MALLOC_ALIGNED_HPP

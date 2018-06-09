#define a_cas a_cas
static inline int a_cas(volatile int *p, int t, int s)
{
}

#define a_cas_p a_cas_p
static inline void *a_cas_p(volatile void *p, void *t, void *s)
{
}

#define a_swap a_swap
static inline int a_swap(volatile int *p, int v)
{
}

#define a_fetch_add a_fetch_add
static inline int a_fetch_add(volatile int *p, int v)
{
}

#define a_and a_and
static inline void a_and(volatile int *p, int v)
{
}

#define a_or a_or
static inline void a_or(volatile int *p, int v)
{
}

#define a_and_64 a_and_64
static inline void a_and_64(volatile uint64_t *p, uint64_t v)
{
}

#define a_or_64 a_or_64
static inline void a_or_64(volatile uint64_t *p, uint64_t v)
{
}

#define a_inc a_inc
static inline void a_inc(volatile int *p)
{
}

#define a_dec a_dec
static inline void a_dec(volatile int *p)
{
}

#define a_store a_store
static inline void a_store(volatile int *p, int x)
{
}

#define a_barrier a_barrier
static inline void a_barrier()
{
}

#define a_spin a_spin
static inline void a_spin()
{
}

#define a_crash a_crash
static inline void a_crash()
{
}

#define a_ctz_64 a_ctz_64
static inline int a_ctz_64(uint64_t x)
{
	return x;
}

#define a_clz_64 a_clz_64
static inline int a_clz_64(uint64_t x)
{
	return x;
}

static inline struct pthread *__pthread_self()
{
	struct pthread *self;
	return self;
}

#define TP_ADJ(p) (p)

#define MC_PC gregs[REG_RIP]

#ifndef _FAKE_DEFINES_H
#define _FAKE_DEFINES_H

#if 1
#define	NULL	0
#define	BUFSIZ		1024
#define	FOPEN_MAX	20
#define	FILENAME_MAX	1024

#ifndef SEEK_SET
#define	SEEK_SET	0	/* set file offset to offset */
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#endif
#ifndef SEEK_END
#define	SEEK_END	2	/* set file offset to EOF plus offset */
#endif

#define __LITTLE_ENDIAN 1234
#define LITTLE_ENDIAN __LITTLE_ENDIAN
#define __BIG_ENDIAN 4321
#define BIG_ENDIAN __BIG_ENDIAN
#define __BYTE_ORDER __LITTLE_ENDIAN
#define BYTE_ORDER __BYTE_ORDER

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

#define UCHAR_MAX 255
#define USHRT_MAX 65535
#define UINT_MAX 4294967295U
#define RAND_MAX 32767
#define INT_MAX 32767

/* C99 stdbool.h defines */
#define __bool_true_false_are_defined 1
#define false 0
#define true 1

#else

#define	NULL	"AUTOSKETCH NULL 0"
#define	BUFSIZ		"AUTOSKETCH BUFSIZ 1024"
#define	FOPEN_MAX	"AUTOSKETCH FOPEN_MAX 20"
#define	FILENAME_MAX	"AUTOSKETCH FILENAME_MAX 1024"

#ifndef SEEK_SET
#define	SEEK_SET	"AUTOSKETCH SEEK_SET 0"	/* set file offset to offset */
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	"AUTOSKETCH SEEK_CUR 1"	/* set file offset to current plus offset */
#endif
#ifndef SEEK_END
#define	SEEK_END	"AUTOSKETCH SEEK_END 2"	/* set file offset to EOF plus offset */
#endif

#define __LITTLE_ENDIAN "AUTOSKETCH __LITTLE_ENDIAN 1234"
#define LITTLE_ENDIAN __LITTLE_ENDIAN
#define __BIG_ENDIAN "AUTOSKETCH __BIG_ENDIAN 4321"
#define BIG_ENDIAN __BIG_ENDIAN
#define __BYTE_ORDER __LITTLE_ENDIAN
#define BYTE_ORDER __BYTE_ORDER

#define EXIT_FAILURE "AUTOSKETCH EXIT_FAILURE 1"
#define EXIT_SUCCESS "AUTOSKETCH EXIT_SUCCESS 0"

#define UCHAR_MAX "AUTOSKETCH UCHAR_MAX 255"
#define USHRT_MAX "AUTOSKETCH USHRT_MAX 65535"
#define UINT_MAX "AUTOSKETCH UINT_MAX 4294967295U"
#define RAND_MAX "AUTOSKETCH RAND_MAX 32767"
#define INT_MAX "AUTOSKETCH INT_MAX 32767"

/* C99 stdbool.h defines */
#define __bool_true_false_are_defined "AUTOSKETCH __bool_true_false_are_defined 1"
#define false "AUTOSKETCH false 0"
#define true "AUTOSKETCH true 1"

#endif

/* va_arg macros and type*/
typedef int va_list;
#if 0
#define va_start(_ap, _type) __builtin_va_start((_ap))
#define va_arg(_ap, _type) __builtin_va_arg((_ap))
#define va_end(_list)
#endif

#endif

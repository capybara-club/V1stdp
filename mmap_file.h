#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static inline void
mmap_file_open_ro( const char* file_name, uint8_t** mapped_data, size_t* mapped_size )
{
  int fd = open( file_name, O_RDONLY );
  if ( fd < 0 )
  {
    printf("Can't open file %s\n",file_name);
    exit(1);
  }

  struct stat statbuf;
  int err = fstat( fd, &statbuf );
  if (err < 0)
  {
    printf("Can't fstat file %s\n",file_name);
    exit(1);
  }

  *mapped_size = statbuf.st_size;
  *mapped_data = (uint8_t*)mmap( NULL, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0 ); 
  if ( *mapped_data == MAP_FAILED )
  {
    printf("Can't mmap file %s\n",file_name);
    exit(1);
  }
  close( fd );
}

static inline void
mmap_file_close( uint8_t* mapped_data, size_t mapped_size )
{
  int err = munmap( mapped_data, mapped_size );
  if ( err != 0 )
  {
    printf("Could not munmap at %p\n",mapped_data);
    exit(1);
  }
}

